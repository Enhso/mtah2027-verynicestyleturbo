"""Per-match label statistics for the external matches, next to the published public window.

Usage: python -m harness.label_stats [notes/harness-label-stats.txt]   (prints, and writes the table)
"""

import glob
import os
import sys

import numpy as np

from evaluation import possession_metrics as pm
from possession.match import CACHE_DIR, load

PUBLIC_POSS = [0.8, 1.3, 2.8, 3.2, 3.5, 3.6, 6.6, 7.4, 7.6, 8.1, 11.8, 16.0, 20.0, 28.6, 35.7,
               41.6, 42.4]
PUBLIC_STOPS = [5.5, 14.5, 18.3, 20.4]
COLS = ["match", "alive_min", "poss/min", "mean_s", "median_s", "<2s", "<4s", "n_stops",
        "med_stop_s", "ball_share", "home_share"]


def stats(poss_s, stops_s, alive_min, ball_share, home_share, name):
    p = np.asarray(poss_s)
    return [name, f"{alive_min:.1f}", f"{len(p) / alive_min:.2f}", f"{p.mean():.1f}",
            f"{np.median(p):.1f}", f"{(p < 2).mean():.2f}", f"{(p < 4).mean():.2f}",
            str(len(stops_s)), f"{np.median(stops_s):.1f}", ball_share, home_share]


def match_row(m):
    runs = pm.valid_runs(m.labels)
    segs = pm.segments(m.labels, runs)
    poss = [(b - a) / m.fps for a, b, k in segs if k in (pm.H, pm.A)]
    stops = [(b - a) / m.fps for a, b, k in segs if k == pm.D]
    graded = m.labels != pm.X
    alive = int(np.isin(m.labels, (pm.H, pm.A)).sum())
    ball = f"{np.isfinite(m.ball[graded, 0]).mean():.3f}"
    home = f"{(m.labels == pm.H).sum() / alive:.2f}"
    return stats(poss, stops, alive / m.fps / 60, ball, home, m.name)


def table():
    ext = os.path.join(CACHE_DIR, "ext")
    rows = [match_row(load(os.path.basename(p)[:-4], ext))
            for p in sorted(glob.glob(os.path.join(ext, "*.npz")))]
    rows.append(stats(PUBLIC_POSS, PUBLIC_STOPS, 5.0, "-", "-", "public window"))
    w = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(COLS)]
    fmt = lambda r: "  ".join(v.rjust(w[i]) for i, v in enumerate(r))
    return "\n".join([fmt(COLS)] + [fmt(r) for r in rows])


if __name__ == "__main__":
    t = table()
    print(t)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(t + "\n")
