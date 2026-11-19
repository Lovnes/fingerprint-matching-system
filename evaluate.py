"""Measure accuracy on a labelled folder and compare minutiae-only with the ML verifier.

Files must be named <subject>_<impression>.<ext>, e.g. FVC2002 '101_1.tif' or
'S003_2.png'. Files whose impression part starts with 'L' (S003_L1.png) are latent
queries: they are compared with every plain print, and searched 1:N against
impression 1 of every subject.

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
from fpmatch.matcher import match, verdict
from fpmatch.ml import Verifier
from fpmatch.pipeline import process

EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
BAND = {"green": 2, "orange": 1, "gray": 0}


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
    V = Verifier()
    print(f"ML verifier: {V.label}" if V.available else f"ML verifier not available ({V.error}) - minutiae only")

    t = time.time()
    P = {f: process(f, EnhanceParams(ppi=a.ppi, latent=is_lat(f))) for f in files}
    print(f"{len(files)} images processed in {time.time() - t:.1f}s")
    weak = [os.path.basename(f) for f, p in P.items() if p.n < 12]
    if weak:
        more = f" and {len(weak) - 8} more" if len(weak) > 8 else ""
        print(f"  {len(weak)} with <12 minutiae (failure-to-extract risk): {', '.join(weak[:8])}{more}")

    def compare(pairs):
        """-> list of (x, y, same, minutiae_score, supported, paired, minutiae_band, ml_score, ml_band)"""
        res = [match(P[x].template, P[y].template, P[x].local, P[y].local) for x, y in pairs]
        mls = (V.score([(P[x].enh.enhanced, P[x].enh.mask, P[y].enh.enhanced, P[y].enh.mask, r)
                        for (x, y), r in zip(pairs, res)]) if V.available else [None] * len(pairs))
        return [(x, y, sid(x) == sid(y), r.score, r.n_supported, r.n_paired, BAND[verdict(r)[1]],
                 m["ml"] if m else np.nan, BAND[V.verdict(m)[1]] if m else -1)
                for (x, y), r, m in zip(pairs, res, mls)]

    def report(title, rows):
        g = [r for r in rows if r[2]]; i = [r for r in rows if not r[2]]
        if not g or not i:
            return
        print(f"\n{title}: {len(g)} genuine, {len(i)} impostor comparisons")
        print(f"  {'':22}{'strong':>16}{'possible+':>16}{'impostors flagged':>20}")
        for name, k in (("minutiae only", 6), ("ML verifier", 8)):
            if k == 8 and not V.available:
                continue
            gs = 100 * np.mean([r[k] == 2 for r in g]); gp = 100 * np.mean([r[k] >= 1 for r in g])
            ip = 100 * np.mean([r[k] >= 1 for r in i])
            print(f"  {name:22}{gs:15.1f}%{gp:15.1f}%{ip:19.3f}%")
        for name, k in (("minutiae score", 3), ("ML score", 7)):
            if k == 7 and not V.available:
                continue
            print(f"  {name:15} genuine median {np.median([r[k] for r in g]):5.1f} · impostor max {max(r[k] for r in i):5.1f}")

    plain = [f for f in files if not is_lat(f)]
    lats = [f for f in files if is_lat(f)]
    rows_pp = compare(list(itertools.combinations(plain, 2)))
    report("Plain-to-plain", rows_pp)
    rows_lp = compare([(l, p) for l in lats for p in plain])
    report("Latent-to-plain", rows_lp)

    gallery = {sid(f): f for f in plain if imp(f) == "1"}
    if lats and gallery:
        print(f"\nLatent 1:N ({len(lats)} queries vs {len(gallery)} references)")
        by = {(r[0], r[1]): r for r in rows_lp}
        for name, key in (("minutiae only", lambda r: (r[3], r[4])), ("ML verifier", lambda r: (r[7], r[3]))):
            if name == "ML verifier" and not V.available:
                continue
            ranks = []
            for l in lats:
                order = sorted(gallery.items(), key=lambda kv: key(by[(l, kv[1])]), reverse=True)
                ids = [s for s, _ in order]
                ranks.append(ids.index(sid(l)) + 1 if sid(l) in ids else 10 ** 6)
            ranks = np.array(ranks)
            print(f"  {name:14} rank-1 {100 * (ranks <= 1).mean():5.1f}%  rank-5 {100 * (ranks <= 5).mean():5.1f}%"
                  f"  rank-10 {100 * (ranks <= 10).mean():5.1f}%")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["a", "b", "same_subject", "minutiae_score", "supported", "paired", "minutiae_band",
                        "ml_score", "ml_band"])
            w.writerows([(os.path.basename(r[0]), os.path.basename(r[1]), int(r[2]), round(r[3], 2), *r[4:7],
                          round(r[7], 2), r[8]) for r in rows_pp + rows_lp])
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
