"""Score a predicted code array (H=0 A=1 D=2 X=3, one per frame) against an external match's labels."""

import numpy as np

from evaluation import possession_metrics as pm


def score(match, pred_codes):
    """pm.evaluate(match.labels, pred_codes); keys include spell_f1_mean and macro_f1."""
    return pm.evaluate(match.labels, np.asarray(pred_codes))


def self_check(match):
    r = score(match, match.labels)
    assert r["spell_f1_mean"] == 1.0 and r["macro_f1"] == 1.0, match.name
    runs = pm.valid_runs(match.labels)
    assert [tuple(p) for p in match.periods.tolist()] == runs, match.name + ": periods != valid_runs"
    return r["spell_f1_mean"], r["macro_f1"]


if __name__ == "__main__":
    import glob
    import os
    from possession.match import CACHE_DIR, load
    ext = os.path.join(CACHE_DIR, "ext")
    for p in sorted(glob.glob(os.path.join(ext, "*.npz"))):
        m = load(os.path.basename(p)[:-4], ext)
        print(m.name, "spellF1, macroF1 =", self_check(m))
