"""Machine-learning verification stage.

Two-stage design, as in modern AFIS:

1. The minutiae matcher (matcher.py) finds the best alignment between the two prints.
2. A small convolutional network looks at the *ridge images* of both prints inside their
   overlapping area, after alignment, and outputs how well the ridges agree (a logit).
   This uses ridge detail between the minutiae, which helps most on partial latents
   with few minutiae.
3. A logistic-regression fusion combines the network output with the minutiae evidence
   (score, supported/paired minutiae, overlap) into one ML match score 0-100.

Training code lives in training/. The app only runs inference, through OpenCV's DNN
module (no PyTorch needed). Model files: fpmatch/models/verifier.onnx + verifier.json.

The ML score is a similarity score from a model trained on the data described in
verifier.json. It is not a probability of identity and not an identification.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CROP = 192          # window around the overlap, px at 500 ppi
NET = 96            # network input size (CROP downsampled by 2)
MIN_OVERLAP = 1500  # px; smaller overlaps carry too little ridge detail


# ---------------------------------------------------------------------------- pair input
def pair_input(q_enh: np.ndarray, q_mask: np.ndarray, r_enh: np.ndarray, r_mask: np.ndarray, res):
    """Align the query onto the reference using the matcher's transform and cut the overlap window.

    Returns (x, overlap_fraction): x is (3, NET, NET) float32 = [query ridges, reference ridges,
    overlap mask], or (None, 0.0) when there is no usable alignment/overlap.
    """
    if res is None or res.n_paired < 3:
        return None, 0.0
    rot = np.radians(res.rotation_deg)
    c, s = np.cos(rot), np.sin(rot)
    tx, ty = res.translation
    M = np.array([[c, -s, tx], [s, c, ty]], np.float32)
    h, w = r_enh.shape
    qw = cv2.warpAffine(q_enh, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    qm = cv2.warpAffine(q_mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    ov = qm & r_mask
    area = int(ov.sum())
    if area < MIN_OVERLAP:
        return None, 0.0
    frac = area / max(1, min(int(q_mask.sum()), int(r_mask.sum())))
    ys, xs = np.nonzero(ov)
    cy, cx = int(ys.mean()), int(xs.mean())
    p = CROP // 2
    pad = lambda a, v: cv2.copyMakeBorder(a, p, p, p, p, cv2.BORDER_CONSTANT, value=v)
    win = lambda a: a[cy:cy + CROP, cx:cx + CROP]            # (cy, cx) shifted by the pad
    qc = win(pad(qw, 255)).astype(np.float32)
    rc = win(pad(r_enh, 255)).astype(np.float32)
    mc = win(pad(ov.astype(np.uint8), 0)).astype(np.float32)
    small = lambda a: cv2.resize(a, (NET, NET), interpolation=cv2.INTER_AREA)
    qc, rc, mc = small(qc), small(rc), small(mc)
    m = (mc > 0.5).astype(np.float32)
    x = np.stack([(1 - qc / 127.5) * m, (1 - rc / 127.5) * m, m])  # ridges -> positive
    return x.astype(np.float32), float(min(frac, 1.0))


def features(res, cnn_logit: float, overlap: float) -> np.ndarray:
    """Fusion features for a pair that has an overlap window. Order must match verifier.json.

    Only evidence that should push one way is used (more agreement -> higher score). Counts such as
    'number of minutiae' are left out on purpose: in training data they mostly reflect how pairs were
    sampled, not whether two prints share a source."""
    return np.array([res.score / 100.0, res.n_supported / 10.0, cnn_logit, overlap], np.float32)


FEATURE_NAMES = ["minutiae_score", "supported", "cnn_logit", "overlap"]


# ---------------------------------------------------------------------------- model
class Verifier:
    """Loads the ONNX network + fusion parameters. `available` is False if the files are missing."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.available = False
        self.error = ""
        self._lock = threading.Lock()   # cv2.dnn.Net is not thread-safe
        onnx_path = os.path.join(model_dir, "verifier.onnx")
        json_path = os.path.join(model_dir, "verifier.json")
        try:
            with open(json_path, encoding="utf-8") as fh:
                self.meta = json.load(fh)
            with open(onnx_path, "rb") as fh:
                blob = fh.read()
            self.sha256 = hashlib.sha256(blob).hexdigest()
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            f = self.meta["fusion"]
            assert f["features"] == FEATURE_NAMES, "fusion features do not match this version of FP-Match"
            self.mean = np.array(f["mean"], np.float32)
            self.std = np.array(f["std"], np.float32)
            self.coef = np.array(f["coef"], np.float32)
            self.bias = float(f["bias"])
            self.t_strong = float(self.meta["thresholds"]["strong"])
            self.t_possible = float(self.meta["thresholds"]["possible"])
            self.available = True
        except Exception as e:  # missing or broken model: the app falls back to minutiae only
            self.error = f"{type(e).__name__}: {e}"

    @property
    def label(self) -> str:
        return f"{self.meta.get('name', 'verifier')} v{self.meta.get('version', '?')}" if self.available else "not loaded"

    def cnn_logits(self, xs: list[np.ndarray], batch: int = 64) -> np.ndarray:
        out = []
        for k in range(0, len(xs), batch):
            blob = np.ascontiguousarray(np.stack(xs[k:k + batch]))
            with self._lock:
                self.net.setInput(blob)
                out.append(self.net.forward().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def score(self, pairs: list[tuple]) -> list[dict]:
        """pairs: [(q_enh, q_mask, r_enh, r_mask, MatchResult), ...] -> [{ml, cnn, overlap, assessed}, ...]

        ml is 0-100. Pairs without a usable alignment/overlap carry no ridge evidence: they are not
        assessed and get ml = 0 (never a borderline score)."""
        inputs = [pair_input(*p) for p in pairs]
        idx = [k for k, (x, _) in enumerate(inputs) if x is not None]
        logits = np.zeros(len(pairs), np.float32)
        if idx:
            logits[idx] = self.cnn_logits([inputs[k][0] for k in idx])
        out = []
        for (q, qm, r, rm, res), (x, ov), lg in zip(pairs, inputs, logits):
            if x is None:
                out.append(dict(ml=0.0, cnn=None, overlap=0.0, assessed=False))
                continue
            f = (features(res, float(lg), ov) - self.mean) / self.std
            z = float(f @ self.coef + self.bias)
            out.append(dict(ml=100.0 / (1.0 + np.exp(-z)), cnn=float(lg), overlap=ov, assessed=True))
        return out

    def verdict(self, s: dict) -> tuple[str, str]:
        """s: one item from score(). 'Strong' also needs the network itself to see ridge agreement
        (logit > 0), so minutiae evidence alone can never make an ML 'Strong'."""
        if not s.get("assessed"):
            return "Not supported - no usable overlap for the ML stage", "gray"
        if s["ml"] >= 100 * self.t_strong and s["cnn"] > 0:
            return "Strong candidate - verify (ACE-V)", "green"
        if s["ml"] >= 100 * self.t_possible:
            return "Possible candidate - examiner review", "orange"
        return "Not supported", "gray"


_VERIFIER = None


def get_verifier() -> Verifier:
    global _VERIFIER
    if _VERIFIER is None:
        _VERIFIER = Verifier()
    return _VERIFIER
