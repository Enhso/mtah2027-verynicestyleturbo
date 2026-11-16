"""Proxy scores of parameter sets: python -m calibration.report name=path.json ... (path 'default' = params.py)

Metrica per game on clean, atd and atd+mask; DFL (clean, speed in m/s) per match and mean.
"""
import sys

from calibration import common as c, run_search, search
from possession.params import Params

if __name__ == "__main__":
    sets = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=")
        sets.append((name, Params() if path == "default" else run_search.load(path)))
    variants_ = ("clean", "atd", "atd+mask")
    keys = search.setup(c.METRICA, variants_) + search.setup(c.DFL, ("clean",))
    res = search.evaluate([p for _, p in sets], keys)
    for (name, _), r in zip(sets, res):
        print(name)
        for v in variants_:
            ks = [k for k in keys if k[1] == v and k[0] in c.METRICA]
            print("  metrica {:<9} {}  mean {:.3f}/{:.3f}".format(
                v, " ".join("{:.3f}/{:.3f}".format(*r[k][:2]) for k in ks), *c.mean_score(r, ks)))
        ks = [k for k in keys if k[0] in c.DFL]
        print("  dfl clean         {}  mean {:.3f}/{:.3f}".format(
            " ".join("{:.3f}".format(r[k][0]) for k in ks), *c.mean_score(r, ks)))
