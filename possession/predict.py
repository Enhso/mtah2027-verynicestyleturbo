"""predict(match, params) -> codes: the one entry point every match runs through."""

import numpy as np

from . import config as cfg, decode, evidence, features, guards
from .params import Params


def predict(match, params=None, config=None, info=None):
    """(N,) int8 codes (H=0, A=1, D=2, X=3) for a Match.

    Each period is decoded on its own. `config` defaults to the per-match constraints in config.py
    (none for a match it does not know). `info`, when a dict, receives the guard counts.
    """
    params = params or Params()
    config = config if config is not None else cfg.for_match(match.name)
    costs = decode.transition_costs(params)
    paths, merged = [], 0
    for a, b in match.periods:
        a, b = int(a), int(b)
        f = features.compute(match.ball[a:b], match.players[a:b], match.speed[a:b], match.team,
                             match.pitch, match.fps, params)
        tags = evidence.localize_tags(match.tags, config, a, b)
        allowed = config.allowed(a, b)
        path = decode.viterbi(evidence.evidence(f, tags, params), costs, allowed)
        path, k = guards.merge_short_dead(path, params.min_dead_frames, allowed)
        if not allowed[np.arange(b - a), path].all():
            raise AssertionError("a guard broke a hard constraint")
        paths.append(path)
        merged += k
    if info is not None:
        info["merged_dead_runs"] = merged
    return guards.assemble(match.n_frames, match.periods, paths)
