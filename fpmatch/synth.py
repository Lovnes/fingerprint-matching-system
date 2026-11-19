"""Synthetic fingerprint generator (SFinGe-style) for testing and demos.

A 'master' fingerprint is grown from random seeds by repeated Gabor filtering
along an orientation field defined by core/delta singular points
(Sherlock & Monro model). Impressions add rotation, translation, elastic
skin distortion, pressure (ridge thickness), noise and partial cropping.
Latent impressions are small, low-contrast and sit on a textured background.
"""
from __future__ import annotations

import cv2
import numpy as np

H, W = 400, 320


def _orientation(rng, h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = xx + 1j * yy
    kind = rng.choice(["loop", "whorl", "arch", "loop_r"], p=[0.35, 0.3, 0.1, 0.25])
    cx, cy = w / 2 + rng.uniform(-30, 30), h * 0.42 + rng.uniform(-30, 30)
    if kind == "loop":
        cores = [cx + 1j * cy]; deltas = [cx - 90 + 1j * (cy + 120)]
    elif kind == "loop_r":
        cores = [cx + 1j * cy]; deltas = [cx + 90 + 1j * (cy + 120)]
    elif kind == "whorl":
        cores = [cx + 1j * (cy - 25), cx + 1j * (cy + 25)]
        deltas = [cx - 120 + 1j * (cy + 130), cx + 120 + 1j * (cy + 130)]
    else:
        cores = [cx + 1j * (cy + 260)]; deltas = [cx + 1j * (cy + 300)]
    th = np.zeros((h, w), np.float32)
    for d in deltas:
        th -= np.angle(z - d)
    for c in cores:
        th += np.angle(z - c)
    th = 0.5 * th + rng.uniform(-0.2, 0.2)
    return (th % np.pi).astype(np.float32)


def master(seed: int) -> np.ndarray:
    """Return a float image in [-1,1] with ridges (negative = ridge)."""
    rng = np.random.default_rng(seed)
    ridge_theta = _orientation(rng)
    grad_theta = ridge_theta + np.pi / 2            # intensity varies across ridges
    period = rng.uniform(8.5, 10.5)
    img = np.zeros((H, W), np.float32)
    n = 60
    img[rng.integers(0, H, n), rng.integers(0, W, n)] = 1
    img = cv2.GaussianBlur(img, (0, 0), 2)
    sigma = 0.5 * period
    k = int(sigma * 6) | 1
    nang = 24
    angles = np.linspace(0, np.pi, nang, endpoint=False)
    idx = np.round(((grad_theta) % np.pi) / (np.pi / nang)).astype(int) % nang
    kernels = []
    for a in angles:
        g = cv2.getGaborKernel((k, k), sigma, float(a), period, 1.0, 0, ktype=cv2.CV_32F)
        kernels.append(g - g.mean())
    for _ in range(28):
        out = np.zeros_like(img)
        for i, g in enumerate(kernels):
            sel = idx == i
            if sel.any():
                out[sel] = cv2.filter2D(img, cv2.CV_32F, g)[sel]
        img = np.tanh(out / (np.abs(out).mean() + 1e-6) * 1.5)
    return img


def _elastic(rng, h, w, amp):
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 40)
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 40)
    dx *= amp / (np.abs(dx).max() + 1e-6)
    dy *= amp / (np.abs(dy).max() + 1e-6)
    return dx, dy


def impression(m: np.ndarray, seed: int, latent: bool = False) -> np.ndarray:
    """Render one uint8 impression (dark ridges on light background)."""
    rng = np.random.default_rng(seed)
    h, w = m.shape
    rot = rng.uniform(-25, 25) if not latent else rng.uniform(-60, 60)
    tx, ty = rng.uniform(-25, 25, 2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    dx, dy = _elastic(rng, h, w, amp=6 if not latent else 8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=1.0)

    # pressure -> ridge thickness
    img = img + rng.uniform(-0.35, 0.35)
    ridges = np.clip(0.5 - img, 0, 1)                 # 1 on ridge

    # finger-shaped contact area
    cy, cx = h / 2 + rng.uniform(-20, 20), w / 2 + rng.uniform(-15, 15)
    ay, ax = rng.uniform(150, 185), rng.uniform(115, 140)
    contact = (((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2) < 1
    if latent:
        # a partial blob somewhere on the finger
        py, px = cy + rng.uniform(-60, 60), cx + rng.uniform(-50, 50)
        r = rng.uniform(70, 100)
        blob = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 25)
        blob = blob / np.abs(blob).max() * 25
        contact &= np.hypot(yy - py, xx - px) + blob < r
    contact = cv2.GaussianBlur(contact.astype(np.float32), (0, 0), 3)
    ridges *= contact

    # pores / dryness breaks
    holes = cv2.GaussianBlur((rng.random((h, w)) > 0.985).astype(np.float32), (0, 0), 1.2)
    ridges *= 1 - np.clip(holes * (6 if latent else 3), 0, 0.9)

    if latent:
        bg = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 15)
        bg = 0.6 + 0.15 * bg / (np.abs(bg).max() + 1e-6)
        streaks = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (1, 41), 0)
        img = bg - 0.28 * ridges + 0.05 * streaks / (np.abs(streaks).max() + 1e-6)
        img += rng.normal(0, 0.06, (h, w))
    else:
        img = 1 - 0.85 * ridges + rng.normal(0, 0.05, (h, w))
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(0.6, 1.1))
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _save(path, img):
    """cv2.imwrite cannot write non-ASCII paths on Windows, so encode and write the bytes ourselves."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"could not encode {path}")
    buf.tofile(path)


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Generate a synthetic fingerprint test set")
    ap.add_argument("out")
    ap.add_argument("--fingers", type=int, default=20)
    ap.add_argument("--impressions", type=int, default=3)
    ap.add_argument("--latents", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in range(a.fingers):
        m = master(1000 + f)
        for i in range(a.impressions):
            _save(os.path.join(a.out, f"S{f:03d}_{i + 1}.png"), impression(m, f * 100 + i))
        for i in range(a.latents):
            _save(os.path.join(a.out, f"S{f:03d}_L{i + 1}.png"), impression(m, f * 100 + 50 + i, latent=True))
    print("done")
