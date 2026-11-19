"""Fingerprint image enhancement.

Pipeline (Hong, Wan & Jain 1998 style):
    resample to 500 ppi -> normalise -> segment (ROI mask) -> ridge orientation
    -> ridge frequency -> oriented Gabor filtering -> binarise -> thin

A separate *latent* path adds background subtraction, CLAHE and heavier
orientation smoothing, because crime-scene lifts are low-contrast,
partial and noisy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

TARGET_PPI = 500


@dataclass
class EnhanceParams:
    ppi: int = 500                  # resolution of the input image
    latent: bool = False            # latent / crime-scene mode
    invert: bool = False            # light ridges on a dark background
    orient_sigma: float = 7.0       # orientation smoothing (pixels at 500 ppi)
    block: int = 16                 # block size for segmentation / frequency
    seg_threshold: float = 0.10     # relative std-dev threshold for the ROI

    def effective(self) -> "EnhanceParams":
        p = EnhanceParams(**self.__dict__)
        if p.latent:
            p.orient_sigma = max(p.orient_sigma, 11.0)
        return p


@dataclass
class EnhanceResult:
    gray: np.ndarray            # resampled grayscale input (uint8)
    norm: np.ndarray            # normalised image (float, ~zero mean / unit var)
    mask: np.ndarray            # bool ROI mask
    orientation: np.ndarray     # ridge-normal (gradient) angle in radians, per pixel
    coherence: np.ndarray       # 0..1 orientation reliability
    frequency: float            # ridges per pixel
    enhanced: np.ndarray        # Gabor output (uint8, ridges dark)
    binary: np.ndarray          # bool, True = ridge
    skeleton: np.ndarray        # bool, 1-px wide ridges
    scale: float = 1.0          # factor applied to reach 500 ppi
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def load_image(path_or_bytes) -> np.ndarray:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    else:
        # Read the bytes ourselves: cv2.imread cannot open non-ASCII paths on Windows.
        with open(path_or_bytes, "rb") as fh:
            img = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image")
    return to_gray(img)


def _normalise(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    f = img.astype(np.float32)
    ref = f[mask] if mask is not None and mask.any() else f
    return (f - ref.mean()) / (ref.std() + 1e-6)


# ----------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------
def latent_preprocess(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven background and boost local contrast."""
    f = gray.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), 20)
    flat = cv2.normalize(f - bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = clahe.apply(flat)
    return cv2.fastNlMeansDenoising(out, None, h=7, templateWindowSize=7, searchWindowSize=21)


def orientation_field(norm: np.ndarray, sigma: float):
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    # dominant gradient direction (perpendicular to ridges)
    num, den = 2 * gxy, gxx - gyy
    coh = np.sqrt(num ** 2 + den ** 2) / (gxx + gyy + 1e-6)
    # smooth again in the doubled-angle domain
    c = cv2.GaussianBlur(den, (0, 0), sigma / 2)
    s = cv2.GaussianBlur(num, (0, 0), sigma / 2)
    theta = 0.5 * np.arctan2(s, c)
    return theta.astype(np.float32), np.clip(coh, 0, 1).astype(np.float32)


def segment(norm: np.ndarray, coh: np.ndarray, block: int, thr: float, latent: bool) -> np.ndarray:
    """ROI = blocks with ridge texture (variance) and, for latents, coherent flow."""
    h, w = norm.shape
    var = cv2.blur(norm ** 2, (block, block)) - cv2.blur(norm, (block, block)) ** 2
    std = np.sqrt(np.clip(var, 0, None))
    mask = std > thr * std.max() * 3  # relative threshold
    if latent:
        mask &= cv2.blur(coh, (block, block)) > 0.35
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block * 2 + 1, block * 2 + 1))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = ndi.binary_fill_holes(m)
    # keep components bigger than 5% of the largest
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.where(sizes >= 0.05 * sizes.max())[0])
        m = keep
    # do not trust the outermost rim of the image
    m[:8, :] = m[-8:, :] = 0
    m[:, :8] = m[:, -8:] = 0
    return m.astype(bool)


def ridge_frequency(norm, theta, mask, block=32) -> float:
    """Median ridge frequency using the x-signature method on oriented blocks."""
    h, w = norm.shape
    freqs = []
    for y in range(0, h - block, block // 2):
        for x in range(0, w - block, block // 2):
            if not mask[y + block // 2, x + block // 2]:
                continue
            blk = norm[y:y + block, x:x + block]
            ang = np.degrees(theta[y + block // 2, x + block // 2])
            # rotate so the gradient direction is horizontal -> ridges vertical
            M = cv2.getRotationMatrix2D((block / 2, block / 2), ang, 1)
            rot = cv2.warpAffine(blk, M, (block, block), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            c = block // 4
            sig = rot[c:-c, :].mean(axis=0)
            sig = cv2.GaussianBlur(sig.reshape(1, -1), (0, 0), 1).ravel()
            peaks = np.where((sig[1:-1] > sig[:-2]) & (sig[1:-1] > sig[2:]))[0]
            if len(peaks) >= 2:
                period = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                if 4 <= period <= 16:
                    freqs.append(1.0 / period)
    if not freqs:
        return 1 / 9.0
    return float(np.clip(np.median(freqs), 1 / 15, 1 / 5))


def gabor_enhance(norm, theta, freq, mask, n_angles=16):
    period = 1.0 / freq
    sigma = 0.55 * period
    ksize = int(round(sigma * 6)) | 1
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_angles, endpoint=False)
    out = np.zeros_like(norm)
    # quantise orientation and pick each pixel's response from its own filter
    idx = np.round((theta + np.pi / 2) / (np.pi / n_angles)).astype(int) % n_angles
    for i, a in enumerate(angles):
        k = cv2.getGaborKernel((ksize, ksize), sigma, float(a), period, 1.0, 0, ktype=cv2.CV_32F)
        k -= k.mean()
        resp = cv2.filter2D(norm, cv2.CV_32F, k)
        sel = idx == i
        out[sel] = resp[sel]
    out[~mask] = out[mask].max() if mask.any() else 0
    return out


def remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = ndi.label(binary, structure=np.ones((3, 3)))
    if n == 0:
        return binary
    sizes = ndi.sum(binary, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.where(sizes >= min_size)[0])


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------
def enhance(gray: np.ndarray, params: EnhanceParams | None = None,
            roi: np.ndarray | None = None) -> EnhanceResult:
    p = (params or EnhanceParams()).effective()
    gray = to_gray(gray)
    notes = []

    scale = TARGET_PPI / float(p.ppi)
    if abs(scale - 1) > 0.02:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        notes.append(f"resampled x{scale:.2f} to 500 ppi")
    if roi is not None and roi.shape != gray.shape:
        roi = cv2.resize(roi.astype(np.uint8), gray.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0

    work = 255 - gray if p.invert else gray
    if p.latent:
        work = latent_preprocess(work)
        notes.append("latent preprocessing (background flattening, CLAHE, NL-means)")

    norm0 = _normalise(work)
    theta, coh = orientation_field(norm0, p.orient_sigma)
    mask = segment(norm0, coh, p.block, p.seg_threshold, p.latent)
    if roi is not None:
        mask &= roi.astype(bool)
    if mask.sum() < 1000:
        notes.append("WARNING: very small usable ridge area")
    norm = _normalise(work, mask)
    freq = ridge_frequency(norm, theta, mask)
    g = gabor_enhance(norm, theta, freq, mask)

    binary = (g < 0) & mask            # ridges are dark -> negative response
    binary = remove_small(binary, 30)
    binary = ndi.binary_opening(binary, np.ones((2, 2)))
    skel = skeletonize(binary)

    enhanced = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return EnhanceResult(gray=gray, norm=norm, mask=mask, orientation=theta, coherence=coh,
                         frequency=freq, enhanced=enhanced, binary=binary, skeleton=skel,
                         scale=scale, notes=notes)
