"""Coordinate descent on coarse grids over the proxy objective.

Objective: mean spellF1 over the Metrica games on the `atd` and `atd+mask` variants (they bracket
this match, see notes/calibration-progress.md), macroF1 as tie-break. A value replaces the current
one only when it gains at least MIN_GAIN, so noise does not walk the parameters around.
Every grid row is appended to notes/calibration-log.tsv.
"""

import dataclasses
import multiprocessing as mp
import os

from calibration import common as c, variants
from possession.params import Params

VARIANTS = ("atd", "atd+mask")
MIN_GAIN = 0.003
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes",
                   "calibration-log.tsv")
_POOL = None


def _job(args):
    return c.result(*args)


def setup(names=c.METRICA, vs=VARIANTS, workers=7):
    """Register the variants, compute their features, then fork the pool so workers share them."""
    global _POOL
    keys = variants.register(names, vs)
    for k in keys:
        c.feats(k, Params())
    _POOL = mp.get_context("fork").Pool(workers)
    return keys


def evaluate(param_list, keys):
    """[{key: (spellF1, macroF1, n_pred)}], one dict per Params."""
    out = _POOL.map(_job, [(k, p) for p in param_list for k in keys], chunksize=1)
    n = len(keys)
    return [dict(zip(keys, out[i * n:(i + 1) * n])) for i in range(len(param_list))]


def objective(res, keys):
    return c.mean_score(res, keys)


def log(step, what, why, evidence, result):
    with open(LOG, "a") as fh:
        fh.write("\t".join(str(x) for x in (step, what, why, evidence, result)) + "\n")


def by_variant(res, keys):
    vs = sorted({k[1] for k in keys})
    return " ".join("{}={:.3f}".format(v, c.mean_score(res, [k for k in keys if k[1] == v])[0])
                    for v in vs)


def descend(base, grids, keys, step, passes=2, verbose=True):
    """Two-pass coordinate descent. grids: [(name, [values])] in order. Returns (params, score)."""
    cur = base
    best = objective(evaluate([cur], keys)[0], keys)
    if verbose:
        print("{} start {:.4f}/{:.4f}".format(step, *best))
    for p in range(passes):
        for name, values in grids:
            values = sorted(set(list(values) + [getattr(cur, name)]))
            cands = [dataclasses.replace(cur, **{name: v}) for v in values]
            res = evaluate(cands, keys)
            scores = [objective(r, keys) for r in res]
            i = max(range(len(values)), key=lambda j: scores[j])
            row = " ".join("{}:{:.4f}".format(v, s[0]) for v, s in zip(values, scores))
            chosen = getattr(cur, name)
            if values[i] != chosen and scores[i][0] >= best[0] + MIN_GAIN:
                chosen, cur, best = values[i], cands[i], scores[i]
            if verbose:
                print("  pass {} {:<16} {} -> {} [{}]".format(p + 1, name, row, chosen,
                                                            by_variant(res[values.index(chosen)], keys)))
            log(step, "pass {} grid {}".format(p + 1, name), "coordinate descent", row,
                "{}={} J={:.4f}".format(name, chosen, best[0]))
    return cur, best


def changes(params):
    """{name: value} where params differs from the defaults in params.py."""
    d = Params()
    return {f.name: getattr(params, f.name) for f in dataclasses.fields(Params)
            if getattr(params, f.name) != getattr(d, f.name)}
