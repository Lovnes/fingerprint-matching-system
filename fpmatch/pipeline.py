"""One-call processing: image -> template, plus drawing helpers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from .enhance import EnhanceParams, EnhanceResult, enhance, load_image
from .minutiae import BIFURCATION, extract
from .matcher import MatchResult, local_structures


@dataclass
class Processed:
    template: np.ndarray        # (N,5) x, y, angle, type, quality  (500 ppi coordinates)
    enh: EnhanceResult
    skeleton: np.ndarray
    sha256: str
    quality: float              # 0-100 overall image quality estimate
    local: np.ndarray = None

    @property
    def n(self):
        return len(self.template)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quality_score(enh: EnhanceResult, n_minutiae: int) -> float:
    """Rough 0-100 quality: usable area x mean coherence x minutiae sufficiency."""
    if not enh.mask.any():
        return 0.0
    area = min(1.0, enh.mask.sum() / 60000.0)
    coh = float(enh.coherence[enh.mask].mean())
    suff = min(1.0, n_minutiae / 25.0)
    return round(100 * (0.35 * area + 0.4 * min(1, coh / 0.6) + 0.25 * suff), 1)


def process(image, params: EnhanceParams | None = None, raw_bytes: bytes | None = None,
            roi: np.ndarray | None = None) -> Processed:
    """image: path, bytes or numpy array."""
    if isinstance(image, (bytes, bytearray)):
        raw_bytes = bytes(image)
        gray = load_image(raw_bytes)
    elif isinstance(image, np.ndarray):
        gray = image
    else:
        with open(image, "rb") as fh:
            raw_bytes = fh.read()
        gray = load_image(raw_bytes)
    params = params or EnhanceParams()
    enh = enhance(gray, params, roi=roi)
    tpl, skel = extract(enh.skeleton, enh.mask, enh.coherence, latent=params.latent)
    digest = sha256_bytes(raw_bytes) if raw_bytes is not None else sha256_bytes(gray.tobytes())
    return Processed(tpl, enh, skel, digest, quality_score(enh, len(tpl)), local_structures(tpl))


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
END_COL = (40, 40, 230)      # BGR red
BIF_COL = (220, 120, 20)     # BGR blue


def draw_template(gray, template, mask=None, skeleton=None, highlight=None) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.75, np.full_like(img, 255), 0.25, 0)
    if skeleton is not None:
        img[skeleton] = (60, 160, 60)
    if mask is not None:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 200, 255), 1)
    hl = set(highlight or [])
    for idx, (x, y, a, t, q) in enumerate(template):
        col = BIF_COL if t == BIFURCATION else END_COL
        if highlight is not None and idx in hl:
            col = (20, 170, 20)
        c = (int(round(x)), int(round(y)))
        if t == BIFURCATION:
            cv2.rectangle(img, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), col, 1)
        else:
            cv2.circle(img, c, 4, col, 1)
        e = (int(round(x + 12 * np.cos(a))), int(round(y + 12 * np.sin(a))))
        cv2.line(img, c, e, col, 1, cv2.LINE_AA)
    return img


def draw_minutiae(p: Processed, show_skeleton=False, highlight=None) -> np.ndarray:
    return draw_template(p.enh.gray, p.template, p.enh.mask,
                         p.skeleton if show_skeleton else None, highlight)


def draw_match_arrays(q_gray, q_tpl, r_gray, r_tpl, res: MatchResult) -> np.ndarray:
    """Side-by-side view; supported pairs drawn green, other pairs yellow."""
    qi = draw_template(q_gray, q_tpl, highlight=[i for i, _ in res.pairs])
    ri = draw_template(r_gray, r_tpl, highlight=[j for _, j in res.pairs])
    h = max(qi.shape[0], ri.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gap = np.full((h, 24, 3), 255, np.uint8)
    canvas = np.hstack([pad(qi), gap, pad(ri)])
    off = qi.shape[1] + 24
    sup = res.pair_support or [1.0] * len(res.pairs)
    from .matcher import SUPPORT_MIN
    for (i, j), s in zip(res.pairs, sup):
        col = (40, 170, 40) if s >= SUPPORT_MIN else (0, 190, 230)
        p1 = tuple(int(v) for v in q_tpl[i, :2])
        p2 = (int(r_tpl[j, 0]) + off, int(r_tpl[j, 1]))
        cv2.line(canvas, p1, p2, col, 1, cv2.LINE_AA)
    return canvas


def draw_match(q: Processed, r: Processed, res: MatchResult) -> np.ndarray:
    return draw_match_arrays(q.enh.gray, q.template, r.enh.gray, r.template, res)
