"""The four published baselines and the current params on each proxy variant (sanity check of the
proxy, not a fit)."""
import sys
import numpy as np
sys.path.insert(0, "starter")
from baseline import smooth, NOBODY          # noqa: E402
from evaluation import possession_metrics as pm
from calibration import common as c, variants
from possession.match import X
from possession.params import Params


def board(m):
    d = np.linalg.norm(m.players - m.ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    codes = np.where(np.isfinite(d.min(axis=1)), m.team[np.argmin(d, axis=1)], NOBODY).astype(np.int8)
    row = []
    for w in (0, 25, 50, 125):
        pred = pm.label_codes(smooth(codes, w))
        pred[m.labels == X] = X
        r = pm.evaluate(m.labels, pred)
        row.append("{:.3f}/{:.3f}".format(r["spell_f1_mean"], r["macro_f1"]))
    return row


if __name__ == "__main__":
    vs = (sys.argv[1] if len(sys.argv) > 1 else "clean,atd,atd+mask").split(",")
    names = sys.argv[2:] or c.METRICA
    print("spellF1/macroF1. full match [doc]: raw .049/.653  s25 .200/.675  s50 .243/.672  s125 .224/.636")
    for v in vs:
        tot = []
        for nm in names:
            m = variants.build(nm, v)
            c.register((nm, v), m)
            r = c.result((nm, v), Params())
            row = board(m)
            tot.append([float(x.split("/")[0]) for x in row] + [r[0]])
            print("{:<14} {:<10} raw {}  s25 {}  s50 {}  s125 {}  params {:.3f}/{:.3f} n_pred {}".format(
                nm, v, *row, *r))
        print("{:<14} {:<10} mean spellF1: raw {:.3f} s25 {:.3f} s50 {:.3f} s125 {:.3f} params {:.3f}".format(
            "mean", v, *np.mean(tot, axis=0)))
