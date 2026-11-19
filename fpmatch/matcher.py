"""Minutiae matching, rotation- and translation-invariant.

1. Each minutia gets a local descriptor built from its k nearest neighbours
   (distance, relative bearing, relative direction). These do not change when
   the whole print is rotated or shifted.
2. Pairs of minutiae with similar descriptors are candidate correspondences.
3. For the best candidates, the print is aligned on that pair, every minutia
   is paired one-to-one within distance/angle tolerances, then the alignment
   is refined by least squares and re-paired.
4. The alignment with the most paired minutiae wins.

Score (0-100) = paired minutiae relative to the smaller template that lies
inside the overlapping area. The paired count itself is also reported,
because examiners reason in terms of corresponding points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

K = 6               # neighbours per local structure
DIST_TOL = 12.0     # px at 500 ppi
ANG_TOL = np.radians(20)
TOP_CANDIDATES = 25
PAIR_BASE = 0.0     # paired minutiae count only as far as their neighbourhoods agree
MIN_DENOM = 12
SUPPORT_MIN = 0.35  # neighbourhood agreement needed to call a pair 'supported'


@dataclass
class MatchResult:
    score: float
    n_paired: int
    n_query: int
    n_ref: int
    pairs: list = field(default_factory=list)          # [(i_query, i_ref), ...]
    n_supported: int = 0       # paired minutiae whose neighbourhoods also agree
    pair_support: list = field(default_factory=list)  # per-pair neighbourhood agreement 0..1
    rotation_deg: float = 0.0
    translation: tuple = (0.0, 0.0)

    def as_dict(self):
        return dict(score=round(self.score, 1), paired=self.n_paired, supported=self.n_supported, query_minutiae=self.n_query,
                    reference_minutiae=self.n_ref, rotation_deg=round(self.rotation_deg, 1),
                    translation_px=tuple(round(t, 1) for t in self.translation))


def _adiff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def local_structures(m: np.ndarray, k: int = K) -> np.ndarray:
    """(N, k, 3): distance, bearing relative to own direction, neighbour direction relative to own."""
    n = len(m)
    if n < 2:
        return np.zeros((n, k, 3), np.float32) + np.nan
    xy, th = m[:, :2], m[:, 2]
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    nn = np.argsort(d, axis=1)[:, :kk]
    out = np.full((n, k, 3), np.nan, np.float32)
    rows = np.arange(n)[:, None]
    dx = xy[nn, 0] - xy[rows, 0]
    dy = xy[nn, 1] - xy[rows, 1]
    out[:, :kk, 0] = d[rows, nn]
    out[:, :kk, 1] = _adiff(np.arctan2(dy, dx), th[:, None])
    out[:, :kk, 2] = _adiff(th[nn], th[:, None])
    return out


def _structure_similarity(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """(nA, nB) similarity in 0..1: fraction of neighbours with a compatible counterpart."""
    a = la[:, None, :, None, :]   # nA,1,k,1,3
    b = lb[None, :, None, :, :]   # 1,nB,1,k,3
    dd = np.abs(a[..., 0] - b[..., 0])
    db = np.abs(_adiff(a[..., 1], b[..., 1]))
    dt = np.abs(_adiff(a[..., 2], b[..., 2]))
    with np.errstate(invalid="ignore"):
        ok = (dd < 10) & (db < np.radians(20)) & (dt < np.radians(25))
        cost = dd / 10 + db / np.radians(20) + dt / np.radians(25)
    s = np.where(ok, 1 - cost / 3, 0)
    s = np.nan_to_num(s)
    return s.max(axis=3).mean(axis=2)


def _transform(m, rot, tx, ty):
    c, s = np.cos(rot), np.sin(rot)
    x = c * m[:, 0] - s * m[:, 1] + tx
    y = s * m[:, 0] + c * m[:, 1] + ty
    return np.stack([x, y, (m[:, 2] + rot) % (2 * np.pi)], 1)


def _pair(ta: np.ndarray, b: np.ndarray):
    """Greedy one-to-one pairing of transformed query ta (N,3) with reference b."""
    d = np.sqrt(((ta[:, None, :2] - b[None, :, :2]) ** 2).sum(-1))
    da = np.abs(_adiff(ta[:, None, 2], b[None, :, 2]))
    ok = (d < DIST_TOL) & (da < ANG_TOL)
    if not ok.any():
        return []
    cost = np.where(ok, d / DIST_TOL + da / ANG_TOL, np.inf)
    order = np.argsort(cost, axis=None)
    used_a, used_b, pairs = set(), set(), []
    for flat in order:
        if not np.isfinite(cost.flat[flat]):
            break
        i, j = divmod(int(flat), cost.shape[1])
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); pairs.append((i, j))
    return pairs


def _fit(a, b, pairs):
    """Least-squares rigid transform from paired points (Kabsch in 2-D)."""
    P = a[[i for i, _ in pairs], :2]
    Q = b[[j for _, j in pairs], :2]
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    rot = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(rot), np.sin(rot)
    t = qc - np.array([[c, -s], [s, c]]) @ pc
    return rot, t[0], t[1]


def _overlap_count(t, other, margin=10):
    """How many points of t fall inside the bounding hull of other (approximate overlap)."""
    if len(other) < 3:
        return len(t)
    lo, hi = other[:, :2].min(0) - margin, other[:, :2].max(0) + margin
    inside = np.all((t[:, :2] >= lo) & (t[:, :2] <= hi), axis=1)
    return int(inside.sum())


def match(a: np.ndarray, b: np.ndarray, la=None, lb=None) -> MatchResult:
    """Compare query template a with reference template b."""
    na, nb = len(a), len(b)
    empty = MatchResult(0.0, 0, na, nb)
    if na < 3 or nb < 3:
        return empty
    la = local_structures(a) if la is None else la
    lb = local_structures(b) if lb is None else lb
    sim = _structure_similarity(la, lb)
    flat = np.argsort(sim, axis=None)[::-1][:TOP_CANDIDATES]

    best = (0, [], 0.0, 0.0, 0.0)
    for f in flat:
        i, j = divmod(int(f), nb)
        if sim[i, j] <= 0.05:
            break
        rot = _adiff(b[j, 2], a[i, 2])
        c, s = np.cos(rot), np.sin(rot)
        tx = b[j, 0] - (c * a[i, 0] - s * a[i, 1])
        ty = b[j, 1] - (s * a[i, 0] + c * a[i, 1])
        pairs = _pair(_transform(a, rot, tx, ty), b)
        if len(pairs) >= 3:
            for _ in range(2):  # refine
                rot, tx, ty = _fit(a, b, pairs)
                new = _pair(_transform(a, rot, tx, ty), b)
                if len(new) <= len(pairs):
                    break
                pairs = new
        w = sum(PAIR_BASE + (1 - PAIR_BASE) * sim[p, q] for p, q in pairs)
        if w > best[0]:
            best = (w, pairs, rot, tx, ty)

    w, pairs, rot, tx, ty = best
    n = len(pairs)
    if n == 0:
        return empty
    ta = _transform(a, rot, tx, ty)
    ov_a = max(_overlap_count(ta, b), 1)
    ov_b = max(_overlap_count(b, ta), 1)
    denom = max(min(ov_a, ov_b), MIN_DENOM)    # do not reward tiny templates
    score = 100.0 * min(1.0, w / denom) * min(1.0, n / 8)  # need >=8 pairs for full credit
    support = [float(sim[p, q]) for p, q in pairs]
    n_sup = int(sum(v >= SUPPORT_MIN for v in support))
    return MatchResult(score=score, n_paired=n, n_query=na, n_ref=nb, pairs=pairs,
                       n_supported=n_sup, pair_support=support,
                       rotation_deg=float(np.degrees(rot)), translation=(float(tx), float(ty)))


# Decision bands, calibrated on 40 synthetic fingers (3 plain impressions + 1
# latent each; 7,000+ impostor comparisons): highest impostor score was 17 and
# highest impostor supported-pair count was 6. Re-calibrate on your own data.
STRONG_SCORE, STRONG_SUPPORT = 25.0, 7
POSSIBLE_SCORE = 15.0


def verdict(res: MatchResult) -> tuple[str, str]:
    """Return (label, colour) for display. Never a conclusion of identity."""
    if res.score >= STRONG_SCORE and res.n_supported >= STRONG_SUPPORT:
        return "Strong candidate - verify (ACE-V)", "green"
    if res.score >= POSSIBLE_SCORE or res.n_supported >= STRONG_SUPPORT:
        return "Possible candidate - examiner review", "orange"
    return "Not supported by minutiae", "gray"
