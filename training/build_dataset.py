"""Build a pair dataset for training / validating / testing the ML verifier.

Synthetic prints (fingers are generated from master seeds, so splits never share a finger):
    python training/build_dataset.py --synthetic 600 --seed-base 200000 --out training/work/train.npz
Real prints (files named <subject>_<impression>.<ext>; impressions starting with L are latents):
    python training/build_dataset.py --folder /path/to/FVC2002/DB1_B --ppi 500 --out training/work/real.npz

For every print used as a query:
  * genuine pairs  - against every other plain print of the same finger
  * impostor pairs - against `--impostor-pool` random plain prints of other fingers; the
                     `--hard` highest-scoring ones (the cases that fool the minutiae matcher)
                     plus one random one are kept.
Each pair stores the aligned overlap window (ml.pair_input) and the minutiae-matcher evidence.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _init():
    import cv2
    cv2.setNumThreads(1)


# ---------------------------------------------------------------------------- phase 1: images -> cache
def _save_record(cache, name, subject, latent, p):
    np.savez_compressed(os.path.join(cache, name + ".npz"), template=p.template.astype(np.float32),
                        enhanced=p.enh.enhanced, mask=np.packbits(p.enh.mask), shape=np.array(p.enh.mask.shape),
                        subject=subject, latent=latent)


def _synthetic_finger(args):
    f, seed_base, cache, n_imp, n_lat = args
    from fpmatch import synth
    from fpmatch.enhance import EnhanceParams
    from fpmatch.pipeline import process
    m = synth.master(seed_base + f)
    out = []
    for i in range(n_imp + n_lat):
        latent = i >= n_imp
        name = f"F{seed_base + f}_{'L' + str(i - n_imp + 1) if latent else str(i + 1)}"
        img = synth.impression(m, (seed_base + f) * 100 + (50 if latent else 0) + i, latent=latent)
        p = process(img, EnhanceParams(latent=latent))
        _save_record(cache, name, str(seed_base + f), latent, p)
        out.append(name)
    return out


def _real_image(args):
    path, cache, ppi = args
    from fpmatch.enhance import EnhanceParams
    from fpmatch.pipeline import process
    base = os.path.splitext(os.path.basename(path))[0]
    subject, imp = base.split("_")[0], base.split("_")[-1]
    latent = imp.upper().startswith("L")
    try:
        p = process(path, EnhanceParams(ppi=ppi, latent=latent))
    except Exception as e:
        print(f"skip {path}: {e}", flush=True)
        return []
    _save_record(cache, base, subject, latent, p)
    return [base]


# ---------------------------------------------------------------------------- phase 2: pairs
_T = {}


def _init_pairs(cache, names):
    _init()
    from fpmatch.matcher import local_structures
    for n in names:
        d = np.load(os.path.join(cache, n + ".npz"))
        t = d["template"]
        _T[n] = (str(d["subject"]), bool(d["latent"]), t, local_structures(t))
    _T["__cache__"] = cache


def _load_img(name):
    d = np.load(os.path.join(_T["__cache__"], name + ".npz"))
    shape = tuple(d["shape"])
    mask = np.unpackbits(d["mask"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    return d["enhanced"], mask


def _pairs_for(args):
    q, pool_size, hard, seed = args
    from fpmatch.matcher import match
    from fpmatch.ml import pair_input
    rng = np.random.default_rng(seed)
    qs, qlat, qt, ql = _T[q]
    plains = [n for n, v in _T.items() if n != "__cache__" and not v[1] and n != q]
    gen = [n for n in plains if _T[n][0] == qs]
    others = [n for n in plains if _T[n][0] != qs]
    pool = list(rng.choice(others, size=min(pool_size, len(others)), replace=False)) if others else []
    scored = [(match(qt, _T[r][2], ql, _T[r][3]), r) for r in pool]
    scored.sort(key=lambda t: (t[0].score, t[0].n_supported), reverse=True)
    imp = scored[:hard] + ([scored[int(rng.integers(hard, len(scored)))]] if len(scored) > hard else [])
    chosen = [(match(qt, _T[r][2], ql, _T[r][3]), r, 1) for r in gen] + [(res, r, 0) for res, r in imp]
    qe, qm = _load_img(q)
    rows = []
    for res, r, y in chosen:
        re_, rm = _load_img(r)
        x, ov = pair_input(qe, qm, re_, rm, res)
        xi = None if x is None else np.concatenate([np.clip(np.round(x[:2] * 127), -127, 127),
                                                    x[2:] * 127]).astype(np.int8)
        rows.append((xi, np.array([res.score, res.n_supported, res.n_paired, res.n_query, res.n_ref, ov], np.float32),
                     y, q, r, qlat))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int, metavar="FINGERS")
    src.add_argument("--folder")
    ap.add_argument("--seed-base", type=int, default=200000, help="first master seed (synthetic)")
    ap.add_argument("--impressions", type=int, default=3)
    ap.add_argument("--latents", type=int, default=2)
    ap.add_argument("--ppi", type=int, default=500)
    ap.add_argument("--impostor-pool", type=int, default=40)
    ap.add_argument("--hard", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="folder for processed prints (default: next to --out)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    cache = a.cache or os.path.splitext(a.out)[0] + "_cache"
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ctx = get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init) as pool:
        if a.synthetic:
            tasks = [(f, a.seed_base, cache, a.impressions, a.latents) for f in range(a.synthetic)]
            names = [n for ns in pool.imap(_synthetic_finger, tasks, chunksize=2) for n in ns]
        else:
            files = sorted(f for f in glob.glob(os.path.join(a.folder, "*")) if f.lower().endswith(EXT))
            names = [n for ns in pool.imap(_real_image, [(f, cache, a.ppi) for f in files], chunksize=4) for n in ns]
    print(f"phase 1: {len(names)} prints processed in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    with ctx.Pool(a.workers, initializer=_init_pairs, initargs=(cache, names)) as pool:
        tasks = [(q, a.impostor_pool, a.hard, k) for k, q in enumerate(names)]
        rows = [r for rs in pool.imap(_pairs_for, tasks, chunksize=4) for r in rs]
    print(f"phase 2: {len(rows)} pairs in {time.time() - t1:.0f}s", flush=True)

    X = np.zeros((len(rows), 3, 96, 96), np.int8)
    has = np.zeros(len(rows), bool)
    for k, (xi, *_rest) in enumerate(rows):
        if xi is not None:
            X[k] = xi; has[k] = True
    np.savez(a.out, X=X, has_x=has, ev=np.stack([r[1] for r in rows]), y=np.array([r[2] for r in rows], np.int8),
             query=np.array([r[3] for r in rows]), ref=np.array([r[4] for r in rows]),
             query_latent=np.array([r[5] for r in rows]),
             source=np.array(f"synthetic:{a.synthetic}@{a.seed_base}" if a.synthetic else f"folder:{a.folder}"))
    y = np.array([r[2] for r in rows])
    print(f"wrote {a.out}: {int(y.sum())} genuine, {int((1 - y).sum())} impostor, "
          f"{int(has.sum())} with overlap window ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
