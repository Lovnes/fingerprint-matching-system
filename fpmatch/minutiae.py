"""Minutiae extraction from a thinned ridge skeleton.

* Crossing-number method: CN=1 -> ridge ending, CN=3 -> bifurcation.
* Direction is found by tracing the ridge ~12 px away from the point.
* False minutiae are removed: skeleton spurs, points near the ROI border,
  broken-ridge pairs, bridges / short ridges, and low-quality areas.

A template is an (N, 5) float32 array: x, y, angle (rad, 0..2pi), type
(1 = ending, 3 = bifurcation) and quality (0..1).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

ENDING, BIFURCATION = 1, 3
NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]  # clockwise


def crossing_number(skel: np.ndarray) -> np.ndarray:
    s = skel.astype(np.uint8)
    p = np.pad(s, 1)
    h, w = s.shape
    ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in NB]
    cn = np.zeros((h, w), np.int32)
    for i in range(8):
        cn += np.abs(ring[i].astype(np.int32) - ring[(i + 1) % 8])
    cn //= 2
    cn[~skel] = 0
    return cn


def _neighbours(skel, y, x):
    h, w = skel.shape
    for dy, dx in NB:
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and skel[yy, xx]:
            yield yy, xx


def trace(skel, y, x, length, first=None, stop=None):
    """Follow a ridge from (y, x). Returns list of (y, x) visited (excluding start)."""
    start = (y, x)
    visited = {start}
    path, prev_pt, cur = [], start, start
    if first is not None:
        # leaving a junction: the other branches' first pixels are off-limits
        visited.update(_neighbours(skel, y, x))
        path.append(first)
        prev_pt, cur = start, first
    while len(path) < length:
        cand = [n for n in _neighbours(skel, *cur) if n not in visited]
        # discard corner-cutting pixels that also touch the previous pixel
        fwd = [n for n in cand if max(abs(n[0] - prev_pt[0]), abs(n[1] - prev_pt[1])) > 1]
        cand = fwd or cand
        if not cand:
            break
        if len(cand) > 1:
            adjacent = all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for a in cand for b in cand)
            if not adjacent:
                break  # real junction
            cand = [min(cand, key=lambda n: abs(n[0] - cur[0]) + abs(n[1] - cur[1]))]
        visited.add(cur)
        visited.update(n for n in _neighbours(skel, *cur) if n != cand[0] and n not in path)
        prev_pt, cur = cur, cand[0]
        path.append(cur)
        if stop is not None and stop[cur]:
            break
    return path


def prune_spurs(skel: np.ndarray, max_len: int = 10, iterations: int = 2) -> np.ndarray:
    skel = skel.copy()
    for _ in range(iterations):
        cn = crossing_number(skel)
        junction = cn >= 3
        removed = False
        for y, x in zip(*np.where(cn == 1)):
            path = trace(skel, y, x, max_len + 1, stop=junction)
            if path and len(path) <= max_len and junction[path[-1]]:
                skel[y, x] = False
                for pt in path[:-1]:
                    skel[pt] = False
                removed = True
        if not removed:
            break
    # drop tiny isolated fragments
    lab, n = ndi.label(skel, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(skel, lab, range(1, n + 1))
        skel = np.isin(lab, 1 + np.where(sizes >= 15)[0])
    return skel


def _angle(y0, x0, y1, x1):
    return float(np.arctan2(y0 - y1, x0 - x1) % (2 * np.pi))  # points from (x1,y1) towards (x0,y0)


def extract(skel: np.ndarray, mask: np.ndarray, coherence: np.ndarray | None = None,
            border: int = 14, trace_len: int = 12, latent: bool = False):
    """Return (minutiae array, cleaned skeleton)."""
    skel = prune_spurs(skel)
    cn = crossing_number(skel)
    # distance to the edge of the ROI
    dist = ndi.distance_transform_edt(mask)
    border = border + (8 if latent else 0)

    pts = []
    for y, x in zip(*np.where((cn == 1) | (cn == 3))):
        if dist[y, x] < border:
            continue
        t = cn[y, x]
        if t == ENDING:
            path = trace(skel, y, x, trace_len)
            if len(path) < trace_len // 2:
                continue
            ang = _angle(y, x, *path[-1])
        else:
            branches = []
            for n in _neighbours(skel, y, x):
                p = trace(skel, y, x, trace_len, first=n)
                if len(p) >= 3:
                    branches.append(_angle(*p[-1], y, x))  # direction out along the branch
            if len(branches) < 3:
                continue
            branches = branches[:3]
            # the stem is the branch most different from the other two
            diff = []
            for i in range(3):
                d = [abs((branches[i] - branches[j] + np.pi) % (2 * np.pi) - np.pi)
                     for j in range(3) if j != i]
                diff.append(sum(d))
            stem = branches[int(np.argmax(diff))]
            ang = (stem + np.pi) % (2 * np.pi)  # points from stem into the fork
        q = float(coherence[y, x]) if coherence is not None else 1.0
        pts.append((x, y, ang, t, q))

    m = np.array(pts, np.float32).reshape(-1, 5)
    m = _remove_false(m, latent)
    return m, skel


def _remove_false(m: np.ndarray, latent: bool) -> np.ndarray:
    if len(m) < 2:
        return m
    xy = m[:, :2]
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    drop = np.zeros(len(m), bool)
    ang = m[:, 2]
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if d[i, j] > 16:
                continue
            ti, tj = m[i, 3], m[j, 3]
            dang = abs((ang[i] - ang[j] + np.pi) % (2 * np.pi) - np.pi)
            if ti == ENDING and tj == ENDING and dang > 2.3 and d[i, j] < 16:
                drop[[i, j]] = True        # broken ridge: two endings facing each other
            elif ti == BIFURCATION and tj == BIFURCATION and d[i, j] < 10:
                drop[[i, j]] = True        # bridge / hole
            elif ti != tj and d[i, j] < 7:
                drop[[i, j]] = True        # spur next to a fork
            elif d[i, j] < 5:
                drop[[i, j]] = True
    # clusters: more than 3 minutiae within 20 px is almost always noise
    crowd = (d < 20).sum(1)
    drop |= crowd > 3
    if latent:
        drop |= m[:, 4] < 0.3
    return m[~drop]
