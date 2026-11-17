"""Planted-tag simulation of the set-piece rules on the Metrica games (they have no Smart Tagging).

For a seeded sample of each game's true restarts (a D run of the labels followed by an alive frame)
a SET PIECES window of WINDOW_S seconds is planted, starting at the restart minus an offset drawn
from BAR-ATH's launch-offset distribution (the first-launch candidate of possession.evidence on this
match's own set-piece windows; quantiles only are printed). This is a planted-tag simulation: the
window offsets are real, the windows' relation to the true restarts is assumed (every planted
window contains its restart; real tags may be off).

    python -m calibration.restarts_proxy [share ...]     # default: 0.59

Prints mean spellF1 over the three games on `atd` and `atd+mask` (objective J of
notes/calibration-progress.md) for: no tags, the v1 touch rule, the launch rule, each at several
`setpiece_w`, over several seeds. Only aggregates are printed.
"""

import dataclasses
import multiprocessing as mp
import sys
import zlib

import numpy as np

from calibration import common as c, variants
from harness.score import score
from possession import config as cfg, decode as dec, evidence, guards
from possession.features import runs
from possession.params import Params

WINDOW_S = 35.0
SEEDS = range(5)
WEIGHTS = (1.5, 3.0, 6.0, 12.0)
VARIANTS = ("atd", "atd+mask")
SHARE = 0.59       # 59 tagged windows of about 100 restarts, see notes/restarts-progress.md

_BASE = {}         # key -> Match, features cached in common._FEATS


def true_restarts(m):
    """Global frame of each first alive frame that follows a D run inside a period."""
    out = []
    for a, b in m.periods.tolist():
        st, en = runs(m.labels[a:b] == 2)
        out += [a + e for s, e in zip(st, en) if s > 0 and e < b - a]
    return np.array(out, dtype=np.int64)


def bar_ath_offsets():
    """Launch offsets (s from the window start) in BAR-ATH's SET PIECES windows that have one."""
    m = c.load_match("BAR-ATH")
    c.register("BAR-ATH", m)
    P = Params()
    offs = []
    for (a, b), f in zip(m.periods.tolist(), c.feats("BAR-ATH", P)):
        still = evidence.still_ball(f, P)
        for code, s, e, _ in evidence.localize_tags(m.tags, cfg.for_match(m.name), a, b):
            if code == evidence.SET_PIECE_CODE:
                k = evidence.launch_frame(f, still, P, s, e)
                if k is not None:
                    offs.append((k - s) / f.fps)
    return np.array(offs)


def plant(m, share, seed, offsets):
    """Smart Tagging-shaped dicts for a seeded share of the true restarts of m."""
    rng = np.random.default_rng(zlib.crc32(m.name.encode()) + 1000 * seed)
    rs = true_restarts(m)
    pick = np.sort(rng.choice(len(rs), int(round(share * len(rs))), replace=False))
    tags = []
    for r in rs[pick]:
        a, b = [p for p in m.periods.tolist() if p[0] <= r < p[1]][0]
        start = int(r - round(rng.choice(offsets) * m.fps))
        start = max(start, a)
        end = min(start + int(WINDOW_S * m.fps), b)
        tags.append({"code": "SET PIECES", "start_frame": start, "end_frame": end, "start_s": start / m.fps,
                     "end_s": end / m.fps, "tags": {"Type": "Throw-in"}})
    return tags, rs[pick]


def decode_tagged(key, tags, params):
    """c.decode(key, params) with the given tags in place of the match's own."""
    m = _BASE[key]
    config = cfg.for_match(m.name)
    costs = dec.transition_costs(params)
    paths = []
    for (a, b), f in zip(m.periods.tolist(), c.feats(key, params)):
        allowed = config.allowed(a, b)
        path = dec.viterbi(evidence.evidence(f, evidence.localize_tags(tags, config, a, b), params),
                           costs, allowed)
        paths.append(guards.merge_short_dead(path, params.min_dead_frames, allowed)[0])
    return guards.assemble(m.n_frames, m.periods, paths)


def _job(args):
    key, tags, params = args
    return score(_BASE[key], decode_tagged(key, tags, params))["spell_f1_mean"]


def main(shares):
    offsets = bar_ath_offsets()
    print("BAR-ATH launch offsets (s): n={} q5/25/50/75/95 {}".format(
        len(offsets), np.percentile(offsets, [5, 25, 50, 75, 95]).round(1).tolist()))
    keys = []
    for nm in c.METRICA:
        for v in VARIANTS:
            m = variants.build(nm, v)
            _BASE[(nm, v)] = m
            c.register((nm, v), m)
            c.feats((nm, v), Params())
            keys.append((nm, v))
    for k in keys:   # the planted decode must equal the library decode when the tags are the match's own
        assert np.array_equal(decode_tagged(k, _BASE[k].tags, Params()), c.decode(k, Params()))
    rs = {k: true_restarts(_BASE[k]) for k in keys}
    print("true restarts per game: {}".format([len(rs[(nm, "atd")]) for nm in c.METRICA]))

    rules = [("none", None, 0.0)] + [(r, launch, w) for r, launch in (("touch", False), ("launch", True))
                                      for w in WEIGHTS]
    pool = mp.get_context("fork").Pool(7)
    for share in shares:
        planted = {(k, s): plant(_BASE[k], share, s, offsets) for k in keys for s in SEEDS}
        n_win = np.mean([len(planted[(k, 0)][0]) for k in keys])
        # how well the launch candidate finds the planted restart on the proxy
        P = Params(setpiece_launch=True)
        hit, err = [], []
        for k in keys:
            tags, rr = planted[(k, 0)]
            m = _BASE[k]
            for t, r in zip(tags, rr):
                a, b = [p for p in m.periods.tolist() if p[0] <= r < p[1]][0]
                f = c.feats(k, P)[m.periods.tolist().index([a, b])]
                still = evidence.still_ball(f, P)
                cand = evidence.launch_frame(f, still, P, t["start_frame"] - a, t["end_frame"] - a)
                hit.append(cand is not None)
                if cand is not None:
                    err.append((cand + a - r) / m.fps)
        print("\nshare {:.2f}: {:.0f} windows per game; launch candidate present {:.0f}%, error vs true restart "
              "q10/50/90 s {}".format(share, n_win, 100 * np.mean(hit),
                                      np.percentile(err, [10, 50, 90]).round(1).tolist()))
        jobs = [(k, [] if launch is None else planted[(k, s)][0],
                 Params(setpiece_launch=bool(launch), setpiece_w=w))
                for rule, launch, w in rules for s in SEEDS for k in keys]
        out = np.array(pool.map(_job, jobs, chunksize=1)).reshape(len(rules), len(SEEDS), len(keys))
        # out[rule, seed, key]; keys ordered game-major then variant
        print("{:<14} {:>6} {:>8} {:>8} {:>8}   per game (atd+mask) {}".format(
            "rule", "w", "J", "atd", "atd+mask", "seed-sd of J"))
        for i, (rule, launch, w) in enumerate(rules):
            per_seed = out[i].mean(axis=1)
            a_ = out[i][:, 0::2].mean()
            am = out[i][:, 1::2].mean()
            games = out[i].mean(axis=0).reshape(3, 2)
            print("{:<14} {:>6} {:>8.4f} {:>8.4f} {:>8.4f}   sd {:.4f}   per game atd {} atd+mask {}".format(
                rule, w, per_seed.mean(), a_, am, per_seed.std(),
                games[:, 0].round(3).tolist(), games[:, 1].round(3).tolist()))
        for w in WEIGHTS:
            i_t, i_l = rules.index(("touch", False, w)), rules.index(("launch", True, w))
            d = out[i_l] - out[i_t]                     # (seed, key)
            print("launch minus touch at w={}: J diff {:+.4f}; atd {:+.4f}, atd+mask {:+.4f}; seeds with launch >= touch "
                  "(J): {}/{}; games x variants with launch >= touch (seed mean): {}/6".format(
                      w, d.mean(), d[:, 0::2].mean(), d[:, 1::2].mean(),
                      int((d.mean(axis=1) >= 0).sum()), len(SEEDS), int((d.mean(axis=0) >= 0).sum())))


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [SHARE])
