"""Measure accuracy on a labelled folder and check the decision thresholds.

Files must be named <subject>_<impression>.<ext>, e.g. FVC2002 '101_1.tif' or
'S003_2.png'. Files whose impression part starts with 'L' (S003_L1.png) are
treated as latent queries searched against impression 1 of every subject.

    python evaluate.py /path/to/FVC2002/DB1_B --ppi 500
    python evaluate.py samples/synthetic --csv results.csv
"""
import argparse
import glob
import itertools
import os
import sys
import time

import numpy as np

from fpmatch.enhance import EnhanceParams
from fpmatch.matcher import POSSIBLE_SCORE, STRONG_SCORE, STRONG_SUPPORT, match, verdict
from fpmatch.pipeline import process

EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--ppi", type=int, default=500)
    ap.add_argument("--csv", help="write every comparison to this CSV")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.folder, "*")) if f.lower().endswith(EXT))
    if not files:
        sys.exit("no images found")
    sid = lambda f: os.path.basename(f).split("_")[0]
    imp = lambda f: os.path.splitext(os.path.basename(f))[0].split("_")[-1]
    is_lat = lambda f: imp(f).upper().startswith("L")

    t = time.time()
    P = {f: process(f, EnhanceParams(ppi=a.ppi, latent=is_lat(f))) for f in files}
    print(f"{len(files)} images processed in {time.time() - t:.1f}s")
    weak = [os.path.basename(f) for f, p in P.items() if p.n < 12]
    if weak:
        more = f" and {len(weak) - 8} more" if len(weak) > 8 else ""
        print(f"  {len(weak)} with <12 minutiae (failure-to-extract risk): {', '.join(weak[:8])}{more}")

    plain = [f for f in files if not is_lat(f)]
    gen, imps, rows = [], [], []
    for x, y in itertools.combinations(plain, 2):
        r = match(P[x].template, P[y].template, P[x].local, P[y].local)
        same = sid(x) == sid(y)
        (gen if same else imps).append((r.score, r.n_supported))
        rows.append((os.path.basename(x), os.path.basename(y), int(same), round(r.score, 2), r.n_supported, r.n_paired))
    gen, imps = np.array(gen), np.array(imps)

    if len(gen) and len(imps):
        ths = np.linspace(0, 100, 401)
        fnmr = np.array([(gen[:, 0] < th).mean() for th in ths])
        fmr = np.array([(imps[:, 0] >= th).mean() for th in ths])
        k = np.argmin(np.abs(fnmr - fmr))
        print(f"\nPlain-to-plain: {len(gen)} genuine, {len(imps)} impostor comparisons")
        print(f"  EER ~ {100 * (fnmr[k] + fmr[k]) / 2:.2f}% at score {ths[k]:.1f}")
        print(f"  genuine score   5th pct {np.percentile(gen[:, 0], 5):.1f}, median {np.median(gen[:, 0]):.1f}")
        print(f"  impostor score  99th pct {np.percentile(imps[:, 0], 99):.1f}, max {imps[:, 0].max():.1f}")
        print(f"  impostor supported pairs max {int(imps[:, 1].max())}")
        for name, ok in (("strong", lambda s, n: s >= STRONG_SCORE and n >= STRONG_SUPPORT),
                         ("possible+", lambda s, n: s >= POSSIBLE_SCORE or n >= STRONG_SUPPORT)):
            g = np.mean([ok(s, n) for s, n in gen]); i = np.mean([ok(s, n) for s, n in imps])
            print(f"  band '{name}': hits {100 * g:.1f}% of genuine, {100 * i:.3f}% of impostors")

    lats = [f for f in files if is_lat(f)]
    gallery = {sid(f): f for f in plain if imp(f) == "1"}
    if lats and gallery:
        ranks = []
        for l in lats:
            sc = sorted(((match(P[l].template, P[g].template, P[l].local, P[g].local).score, s)
                         for s, g in gallery.items()), reverse=True)
            ids = [s for _, s in sc]
            ranks.append(ids.index(sid(l)) + 1 if sid(l) in ids else 10 ** 6)
        ranks = np.array(ranks)
        print(f"\nLatent 1:N ({len(lats)} queries vs {len(gallery)} references)")
        for r in (1, 5, 10):
            print(f"  rank-{r}: {100 * (ranks <= r).mean():.1f}%")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "same_subject", "score", "supported", "paired"])
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
