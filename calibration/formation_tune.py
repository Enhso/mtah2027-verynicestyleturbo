"""Held-out check and tuning of the team-shape channel (run 2): python -m calibration.formation_tune

Specified before running (notes/formation-progress.md, run 2). The classifier is the shipped one,
trained on DFL only, so every Metrica score is held out from it.
  0. Gate check: with formation_w_vis = 0, DFL clean codes must equal the weight-0 codes.
  1. For each Metrica game held out: (w_vis, w_blk) chosen on J over the other two games (atd and
     atd+mask), scored on the held-out game against the current params. Adopt only if the gain
     is positive on all three games on both variants.
  2. If adopted: weights chosen on all three games, one descent pass over the costs, carry_w and
     touch_w; the descent is kept only if its leave-one-game-out diffs stay within 0.04.
Output: calibration/results/formation_loo.json and lines in notes/calibration-log.tsv.
"""
import dataclasses
import itertools
import json
import os

import numpy as np

from calibration import common as c, run_search, search, variants
from possession.params import Params

W_VIS = [0.0, 0.01, 0.02]
W_BLK = [0.01, 0.02, 0.05, 0.1]
DESCENT = [("stop_cost", [2.0, 4.0, 6.0, 8.0]), ("restart_cost", [2.0, 4.0, 6.0, 8.0]),
           ("carry_w", [0.05, 0.1, 0.2, 0.4, 0.8]), ("touch_w", [0.5, 1.0, 2.0, 4.0]),
           ("turnover_cost", [2.0, 3.0, 4.0, 6.0])]
LOO_TOL = 0.04


def grid(base):
    return [dataclasses.replace(base, formation_w_vis=v, formation_w_blk=b)
            for v, b in itertools.product(W_VIS, W_BLK)]


def label(p):
    return "w_vis={} w_blk={}".format(p.formation_w_vis, p.formation_w_blk)


def best(cands, res, keys):
    i = max(range(len(cands)), key=lambda j: search.objective(res[j], keys))
    return cands[i], res[i]


def held(res, g):
    return float(np.mean([res[(g, v)][0] for v in search.VARIANTS]))


def dfl_clean(params_list):
    keys = [(n, "clean") for n in c.DFL]
    for k in keys:
        if k not in c._MATCH:
            c.register(k, variants.build(*k))
    return keys, [[c.decode(k, p) for k in keys] for p in params_list]


if __name__ == "__main__":
    base = Params()
    keys = search.setup()
    out = {"base": search.changes(base)}

    dkeys, codes = dfl_clean([base] + [dataclasses.replace(base, formation_w_blk=b) for b in W_BLK])
    same = all(np.array_equal(a, b) for cs in codes[1:] for a, b in zip(codes[0], cs))
    print("gate check, w_vis=0: DFL clean codes identical to weight 0:", same)
    out["gate_check_dfl_identical"] = same
    assert same, "the visible/blackout gating is broken"

    v1 = search.evaluate([base], keys)[0]
    cands = grid(base)
    res = search.evaluate(cands, keys)
    ok, out["folds"] = True, {}
    for g in c.METRICA:
        train = [k for k in keys if k[0] != g]
        p, r = best(cands, res, train)
        row = {v: {"base": round(v1[(g, v)][0], 4), "channel": round(r[(g, v)][0], 4),
                   "gain": round(r[(g, v)][0] - v1[(g, v)][0], 4)} for v in search.VARIANTS}
        ok = ok and all(row[v]["channel"] > row[v]["base"] for v in search.VARIANTS)
        out["folds"][g] = dict(choice=label(p), train_J_base=search.objective(v1, train)[0],
                               train_J=search.objective(r, train)[0], held_out=row)
        print("held out {}: chose {} (train J {:.4f} vs base {:.4f}) | {}".format(
            g, label(p), out["folds"][g]["train_J"], out["folds"][g]["train_J_base"],
            " ".join("{} {:.3f}->{:.3f} ({:+.3f})".format(v, row[v]["base"], row[v]["channel"],
                                                       row[v]["gain"]) for v in search.VARIANTS)))
        search.log("formation run 2", "held out " + g, "adoption rule", label(p), json.dumps(row))
    out["adopt"] = ok
    print("adoption rule:", "PASS" if ok else "FAIL")
    print("grid on all three games, base J {:.4f}/{:.4f} [{}]".format(
        *search.objective(v1, keys), search.by_variant(v1, keys)))
    for p, r in zip(cands, res):
        print("  {:<22} J {:.4f}/{:.4f} [{}]".format(label(p), *search.objective(r, keys),
                                                    search.by_variant(r, keys)))

    if ok:
        chosen, r = best(cands, res, keys)
        out["chosen"] = dict(choice=label(chosen), J=search.objective(r, keys))
        tuned, j = search.descend(chosen, DESCENT, keys, "formation descent", passes=1)
        rt = search.evaluate([tuned], keys)[0]
        loo, keep = {}, True
        for g in c.METRICA:
            train = [k for k in keys if k[0] != g]
            pg, _ = search.descend(chosen, DESCENT, train, "formation descent loo " + g, passes=1,
                                   verbose=False)
            rg = search.evaluate([pg], keys)[0]
            d = held(rg, g) - held(rt, g)
            keep = keep and abs(d) <= LOO_TOL
            loo[g] = dict(leave_out=held(rg, g), all_three=held(rt, g), diff=d,
                          changes=search.changes(pg))
            print("descent loo {}: leave-out {:.4f} all-three {:.4f} diff {:+.4f} {}".format(
                g, held(rg, g), held(rt, g), d, search.changes(pg)))
        final = tuned if keep else chosen
        rf = rt if keep else r
        out["descent"] = dict(changes=search.changes(tuned), J=j, loo=loo, kept=keep)
        dk, (c0, c1) = dfl_clean([base, final])
        from harness.score import score
        dfl = [float(np.mean([score(c._MATCH[k], cs[i])["spell_f1_mean"] for i, k in enumerate(dk)]))
               for cs in (c0, c1)]
        out["final"] = dict(changes=search.changes(final), J=search.objective(rf, keys),
                            per_key=run_search.per_key(rf), dfl_clean_mean=dfl)
        print("descent kept:", keep, "| final", search.changes(final))
        print("final J {:.4f}/{:.4f} [{}] | DFL clean mean base {:.4f} final {:.4f}".format(
            *search.objective(rf, keys), search.by_variant(rf, keys), *dfl))
        print("per cell base -> final:", " ".join("{}|{} {:.3f}->{:.3f}".format(
            k[0][-5:], k[1], v1[k][0], rf[k][0]) for k in keys))
    os.makedirs(run_search.OUT, exist_ok=True)
    with open(os.path.join(run_search.OUT, "formation_loo.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
