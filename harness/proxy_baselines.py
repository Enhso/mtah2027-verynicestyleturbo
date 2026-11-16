"""Proxy check. Score the four published baselines on each external match; compare with the full-match board."""
import os, sys, numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [R, R + "/starter", R + "/evaluation"]
from possession.match import CACHE_DIR, load, X
from baseline import smooth, HOME, AWAY, NOBODY
import possession_metrics as pm

ext = os.path.join(CACHE_DIR, "ext")
names = ["metrica_game1", "metrica_game2", "metrica_game3"] + ["dfl_" + i for i in
         ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]]
print("full match [doc]: raw .049/.653  s25 .200/.675  s50 .243/.672  s125 .224/.636")
for name in names:
    m = load(name, ext)
    d = np.linalg.norm(m.players - m.ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=1)
    codes = np.where(np.isfinite(d.min(axis=1)), m.team[nearest], NOBODY).astype(np.int8)
    row = []
    for w in (0, 25, 50, 125):
        lab = smooth(codes, w)
        pred = pm.label_codes(lab)
        pred[m.labels == X] = X
        r = pm.evaluate(m.labels, pred)
        row.append(f"{r['spell_f1_mean']:.3f}/{r['macro_f1']:.3f}")
    print(f"{name:14s} raw {row[0]}  s25 {row[1]}  s50 {row[2]}  s125 {row[3]}")
