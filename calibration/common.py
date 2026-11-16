"""Shared calibration plumbing: load matches, cache features per period, decode, score.

`decode` is predict() with the features passed in, so a grid over evidence weights and costs
recomputes only evidence and Viterbi. `check_same_as_predict` asserts the two agree.
Scripts that touch BAR-ATH print counts, rates, quantiles and scores only.
"""

import dataclasses
import os
import multiprocessing as mp

import numpy as np

from harness.score import score
from possession import config as cfg, decode as dec, evidence, features, guards
from possession.match import CACHE_DIR, load
from possession.params import Params
from possession.predict import predict

EXT = os.path.join(CACHE_DIR, "ext")
METRICA = ["metrica_game1", "metrica_game2", "metrica_game3"]
DFL = ["dfl_" + i for i in ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]]
FEATURE_FIELDS = ("interp_max_gap", "vel_half", "still_half", "touch_span", "player_vmax",
                  "offpitch_margin")

_MATCH = {}      # key -> Match
_FEATS = {}      # (key, feature params) -> [Features per period]


def load_match(name):
    return load(name, None if name == "BAR-ATH" else EXT)


def register(key, match):
    _MATCH[key] = match


def feats(key, params):
    fkey = (key,) + tuple(getattr(params, f) for f in FEATURE_FIELDS)
    if fkey not in _FEATS:
        m = _MATCH[key]
        _FEATS[fkey] = [features.compute(m.ball[a:b], m.players[a:b], m.speed[a:b], m.team,
                                         m.pitch, m.fps, params) for a, b in m.periods.tolist()]
    return _FEATS[fkey]


def decode(key, params):
    """Same codes as predict(match, params), from cached features."""
    m = _MATCH[key]
    config = cfg.for_match(m.name)
    costs = dec.transition_costs(params)
    paths = []
    for (a, b), f in zip(m.periods.tolist(), feats(key, params)):
        allowed = config.allowed(a, b)
        tags = evidence.localize_tags(m.tags, config, a, b)
        path = dec.viterbi(evidence.evidence(f, tags, params), costs, allowed)
        paths.append(guards.merge_short_dead(path, params.min_dead_frames, allowed)[0])
    return guards.assemble(m.n_frames, m.periods, paths)


def check_same_as_predict(key, params):
    assert np.array_equal(decode(key, params), predict(_MATCH[key], params)), "decode != predict"


def result(key, params):
    r = score(_MATCH[key], decode(key, params))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]


def _job(args):
    key, changes = args
    return result(key, dataclasses.replace(_BASE, **changes))


_BASE = Params()
_POOL = None


def evaluate_many(base, change_list, keys, workers=7):
    """[{key: (spellF1, macroF1, n_pred)}] for base params with each dict of changes applied.

    Features for the base feature params are computed before the pool forks, so workers share them.
    """
    global _BASE, _POOL
    for k in keys:
        feats(k, base)
    if _POOL is None or base != _BASE:
        if _POOL is not None:
            _POOL.close()
        _BASE = base
        _POOL = mp.get_context("fork").Pool(workers)
    jobs = [(k, ch) for ch in change_list for k in keys]
    out = _POOL.map(_job, jobs, chunksize=1)
    n = len(keys)
    return [dict(zip(keys, out[i * n:(i + 1) * n])) for i in range(len(change_list))]


def mean_score(res, keys):
    """(mean spellF1, mean macroF1) over keys."""
    return (float(np.mean([res[k][0] for k in keys])), float(np.mean([res[k][1] for k in keys])))
