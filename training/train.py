"""Train the ML verifier and write fpmatch/models/verifier.onnx + verifier.json.

    pip install -r training/requirements-train.txt
    python training/train.py --train training/work/train.npz --val training/work/val.npz \
                             --test training/work/test.npz --out fpmatch/models

* The CNN is trained on the train split only (pairs that have an overlap window).
* The fusion (logistic regression) and the decision thresholds are fitted on the val split.
* The test split is used once, at the end, for the numbers written to verifier.json.
The exported ONNX file is checked against OpenCV's DNN module, which is what the app uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpmatch import __version__ as APP_VERSION  # noqa: E402
from fpmatch.ml import FEATURE_NAMES, NET, features  # noqa: E402
from fpmatch.matcher import MatchResult, verdict  # noqa: E402

MODEL_VERSION = "1.0"


# ---------------------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class PairNet(nn.Module):
    """Input (N, 3, 96, 96): aligned query ridges, reference ridges, overlap mask. Output: logit (N, 1)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(3, 32), block(32, 32), nn.MaxPool2d(2),
                                      block(32, 64), block(64, 64), nn.MaxPool2d(2),
                                      block(64, 128), block(128, 128), nn.MaxPool2d(2),
                                      block(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------- data
def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def to_float(X):
    return X.astype(np.float32) / 127.0


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Same random rotation/flip on all channels (keeps alignment); random query/reference swap."""
    k = int(torch.randint(0, 4, (1,)))
    xb = torch.rot90(xb, k, (2, 3))
    if torch.rand(1) < 0.5:
        xb = torch.flip(xb, (3,))
    swap = torch.rand(xb.shape[0], device=xb.device) < 0.5
    xb[swap] = xb[swap][:, [1, 0, 2]]
    return xb


def auc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def eer(scores, y):
    ths = np.unique(scores)
    g, i = scores[y == 1], scores[y == 0]
    best = (1.0, 0.0)
    for t in ths[:: max(1, len(ths) // 2000)]:
        fnmr, fmr = (g < t).mean(), (i >= t).mean()
        if abs(fnmr - fmr) < abs(best[0] - best[1]):
            best = (fnmr, fmr)
    return 100 * (best[0] + best[1]) / 2


# ---------------------------------------------------------------------------- fusion
def feature_matrix(d, logits):
    """Features for every pair; rows without an overlap window are filler (they are never fused)."""
    rows = []
    for ev, lg in zip(d["ev"], logits):
        res = SimpleNamespace(score=float(ev[0]), n_supported=int(ev[1]), n_paired=int(ev[2]),
                              n_query=int(ev[3]), n_ref=int(ev[4]))
        rows.append(features(res, float(lg), float(ev[5])))
    return np.stack(rows)


def ml_scores(d, logits, mean, std, coef, bias):
    """Fused probability; 0 for pairs without an overlap window (not assessed), as in the app."""
    p = fused(feature_matrix(d, logits), mean, std, coef, bias)
    return np.where(d["has_x"], p, 0.0)


def fit_logreg(F, y, l2=1e-2, iters=50):
    """Logistic regression (Newton / IRLS) on standardised features with non-negative weights:
    a feature whose weight comes out negative is fixed at 0 and the rest are refitted, so more
    agreement can never lower the score."""
    mean, std = F.mean(0), F.std(0) + 1e-6
    Zall = (F - mean) / std
    active = np.ones(F.shape[1], bool)
    while True:
        Z = np.hstack([Zall[:, active], np.ones((len(F), 1))])
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + l2 * np.r_[w[:-1], 0]
            H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        coef = np.zeros(F.shape[1]); coef[active] = w[:-1]
        if (coef[active] >= 0).all():
            return mean, std, coef, w[-1]
        active[np.flatnonzero(active)[np.argmin(w[:-1])]] = False


def fused(F, mean, std, coef, bias):
    return 1 / (1 + np.exp(-(((F - mean) / std) @ coef + bias)))


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "fpmatch", "models"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion-only", action="store_true",
                    help="keep the existing network in --out; refit only the fusion, thresholds and test metrics")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = load(a.train), load(a.val), load(a.test)
    onnx_path = os.path.join(a.out, "verifier.onnx")
    if a.fusion_only:
        old = json.load(open(os.path.join(a.out, "verifier.json"), encoding="utf-8"))
        train_info = old["training"]
    else:
        train_info = train_network(a, tr, va, onnx_path, dev)
    fit_and_write(a, tr, va, te, onnx_path, train_info)


def train_network(a, tr, va, onnx_path, dev):
    # keep pairs as int8 (4x smaller) and convert batch by batch - the full set as float32 would need ~3 GB
    Xtr = torch.from_numpy(tr["X"][tr["has_x"]])
    ytr = torch.from_numpy(tr["y"][tr["has_x"]].astype(np.float32))
    Xva = torch.from_numpy(va["X"][va["has_x"]])
    yva = va["y"][va["has_x"]]
    print(f"device {dev}; train {len(ytr)} pairs ({int(ytr.sum())} genuine); val {len(yva)}", flush=True)

    net = PairNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(ytr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=((len(ytr) - ytr.sum()) / ytr.sum()).detach().clone().to(dev))

    def predict(X):
        net.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(X), 512):
                out.append(net(X[k:k + 512].to(dev).float() / 127.0).float().cpu().numpy().ravel())
        return np.concatenate(out)

    best, best_state = -1, None
    for ep in range(a.epochs):
        net.train(); t = time.time(); perm = torch.randperm(len(ytr)); tot = 0.0
        for k in range(0, len(perm), a.batch):
            idx = perm[k:k + a.batch]
            xb, yb = augment(Xtr[idx].to(dev).float() / 127.0), ytr[idx].to(dev)
            loss = lossf(net(xb).squeeze(1), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx)
        v = auc(predict(Xva), yva)
        print(f"epoch {ep + 1:2d}  loss {tot / len(perm):.4f}  val AUC {v:.4f}  ({time.time() - t:.0f}s)", flush=True)
        if v > best:
            best, best_state = v, {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    net = net.cpu().eval()

    # ---- export + check with OpenCV (what the app runs)
    import cv2
    os.makedirs(a.out, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 3, NET, NET), onnx_path, input_names=["pair"], output_names=["logit"],
                      dynamic_axes={"pair": {0: "n"}, "logit": {0: "n"}}, opset_version=13, dynamo=False)
    cvnet = cv2.dnn.readNetFromONNX(onnx_path)
    idx = np.flatnonzero(va["has_x"])[:256]
    cvnet.setInput(np.ascontiguousarray(to_float(va["X"][idx])))
    with torch.no_grad():
        tv = net(torch.from_numpy(to_float(va["X"][idx]))).numpy().ravel()
    diff = float(np.abs(tv - cvnet.forward().ravel()).max())
    print(f"ONNX via OpenCV {cv2.__version__}: max |diff| vs PyTorch = {diff:.2e}")
    assert diff < 1e-3, "OpenCV output differs from PyTorch"
    return dict(train=str(tr["source"]), train_pairs=int(len(tr["y"])), epochs=a.epochs,
                best_val_auc=round(best, 4), device=dev, torch=torch.__version__)


def fit_and_write(a, tr, va, te, onnx_path, train_info):
    import cv2
    cvnet = cv2.dnn.readNetFromONNX(onnx_path)

    def cv_logits(d):
        out = np.full(len(d["y"]), -8.0, np.float32)
        idx = np.flatnonzero(d["has_x"])
        for k in range(0, len(idx), 256):
            cvnet.setInput(np.ascontiguousarray(to_float(d["X"][idx[k:k + 256]])))
            out[idx[k:k + 256]] = cvnet.forward().ravel()
        return out

    lv = cv_logits(va)

    # ---- fusion (fitted on val pairs that have an overlap window) + thresholds on val
    hv = va["has_x"]
    mean, std, coef, bias = fit_logreg(feature_matrix(va, lv)[hv], va["y"][hv].astype(np.float64))
    print("fusion weights:", dict(zip(FEATURE_NAMES, np.round(coef, 2))), "bias", round(float(bias), 2))
    if (coef == 0).any():
        print("note: features with weight 0 (no independent evidence):", [n for n, c in zip(FEATURE_NAMES, coef) if c == 0])
    pv = ml_scores(va, lv, mean, std, coef, bias)
    imp_v = np.sort(pv[va["y"] == 0])
    t_strong = float(min(0.999, imp_v[-1] + 1e-3))           # above every val impostor (hard negatives)
    t_possible = float(min(np.quantile(imp_v, 0.99), t_strong - 0.05))

    # ---- test (used once)
    lt = cv_logits(te)
    pt = ml_scores(te, lt, mean, std, coef, bias)
    y = te["y"].astype(int)
    lat = te["query_latent"].astype(bool)
    minut = te["ev"][:, 0]
    classic = np.array([{"green": 2, "orange": 1, "gray": 0}[verdict(MatchResult(float(e[0]), int(e[2]), int(e[3]),
                        int(e[4]), n_supported=int(e[1])))[1]] for e in te["ev"]])
    strong = (pt >= t_strong) & (lt > 0) & te["has_x"]          # same rule as ml.Verifier.verdict
    mlband = np.where(strong, 2, np.where(pt >= t_possible, 1, 0))

    def rates(band, sel):
        g, i = band[sel & (y == 1)], band[sel & (y == 0)]
        return dict(genuine=int(len(g)), impostor=int(len(i)),
                    strong_genuine_pct=round(100 * float((g == 2).mean()), 1),
                    flagged_genuine_pct=round(100 * float((g >= 1).mean()), 1),
                    strong_impostor_pct=round(100 * float((i == 2).mean()), 3),
                    flagged_impostor_pct=round(100 * float((i >= 1).mean()), 3))

    metrics = {}
    for name, sel in (("all", np.ones(len(y), bool)), ("plain_queries", ~lat), ("latent_queries", lat)):
        metrics[name] = dict(
            auc=dict(minutiae=round(auc(minut[sel], y[sel]), 4), cnn=round(auc(lt[sel], y[sel]), 4),
                     fused=round(auc(pt[sel], y[sel]), 4)),
            eer_pct=dict(minutiae=round(eer(minut[sel], y[sel]), 2), fused=round(eer(pt[sel], y[sel]), 2)),
            bands_minutiae_only=rates(classic, sel), bands_ml=rates(mlband, sel))

    blob = open(onnx_path, "rb").read()
    meta = dict(
        name="FP-Match pair verifier", version=MODEL_VERSION, app_version=APP_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        onnx_sha256=hashlib.sha256(blob).hexdigest(),
        architecture="CNN, 7 conv layers (32-64-128 ch) + 2 FC, 443k parameters, input 3x96x96 "
                     "(aligned query ridges, reference ridges, overlap mask; 192 px window at 500 ppi)",
        training=dict(train_info, val=str(va["source"]), test=str(te["source"]),
                      val_pairs=int(len(va["y"])), test_pairs=int(len(te["y"]))),
        fusion=dict(features=FEATURE_NAMES, mean=mean.tolist(), std=std.tolist(), coef=coef.tolist(), bias=float(bias)),
        thresholds=dict(strong=t_strong, possible=t_possible,
                        rule="strong: above every validation impostor AND network logit > 0; "
                             "possible: 99th pct of validation impostors "
                             "(hard impostors: the 3 best-scoring of 40 random other fingers per query); "
                             "pairs without a usable overlap are not assessed (score 0)"),
        test_metrics=metrics,
        caveat="Trained and tested on synthetic fingerprints only. Re-train / re-validate on real prints "
               "(training/build_dataset.py --folder) before operational use.")
    with open(os.path.join(a.out, "verifier.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(dict(thresholds=meta["thresholds"], test_metrics=metrics), indent=2))
    print(f"wrote {onnx_path} ({len(blob) / 1e6:.1f} MB) and verifier.json")


if __name__ == "__main__":
    main()
