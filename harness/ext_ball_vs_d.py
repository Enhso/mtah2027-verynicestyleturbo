import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.match import CACHE_DIR, load, D, X
ext = os.path.join(CACHE_DIR, "ext")
for name in ["metrica_game1", "metrica_game2", "metrica_game3", "dfl_J03WMX"]:
    m = load(name, ext)
    g = m.labels != X
    miss = np.isnan(m.ball[:, 0]) & g
    dead = (m.labels == D) & g
    # gap lengths of missing ball inside graded frames
    e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    lens = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
    print(f"{name}: graded {g.sum()} miss {miss.mean()/g.mean():.3f} dead {dead.mean()/g.mean():.3f} "
          f"P(dead|miss) {dead[miss].mean() if miss.any() else float('nan'):.3f} P(miss|dead) {miss[dead].mean():.3f} "
          f"P(dead|vis) {dead[g & ~miss].mean():.3f} gaps {len(lens)} median gap {np.median(lens) if len(lens) else 0:.0f} >125: {(lens>125).sum()}")
