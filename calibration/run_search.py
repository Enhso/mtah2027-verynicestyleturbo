"""Search stages. python -m calibration.run_search control | full [start.json] | loo [start.json]

  control  the three control definitions (carry only, touch only, both), each tuned over its own
           thresholds and weights from the v0 values
  full     two-pass coordinate descent over every tuned parameter, from a start file
  loo      the full descent on each pair of Metrica games, scored on the game left out
Results go to calibration/results/<stage>.json as {name: value} differences from params.py.

The recorded search (commit 9ea8c7e, results/*.json, notes/calibration-log.tsv) started from the v0
params.py and also searched three experiments: restart_at_end (adopted, now the only behaviour),
entry_w and move_w/move_speed (not adopted, removed). Since params.py now holds the v1 values, a
rerun starts from v1, not v0.
"""
import dataclasses
import json
import os
import sys

from calibration import common as c, search
from possession.params import Params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CARRY = [("carry_dv", [2.0, 4.0, 8.0, 100.0]), ("carry_radius", [1.5, 2.0, 3.0]),
         ("carry_margin", [0.0, 0.5, 1.0, 2.0]), ("carry_w", [0.05, 0.1, 0.2, 0.4, 0.8])]
TOUCH = [("touch_dv", [1.5, 2.0, 3.0, 4.0]), ("touch_angle", [35.0, 60.0, 400.0]),
         ("touch_radius", [1.5, 2.5, 4.0]), ("touch_w", [0.5, 1.0, 2.0, 4.0])]
RESTART = [("restart_w", [1.0, 2.0, 4.0, 8.0])]
DEAD = [("stop_cost", [2.0, 4.0, 6.0, 8.0]), ("restart_cost", [2.0, 4.0, 6.0, 8.0]),
        ("low_speed", [1.0, 1.2, 1.5]), ("low_w", [0.02, 0.05, 0.1, 0.2]), ("low_s", [1.0, 3.0, 6.0]),
        ("long_gap_s", [2.0, 5.0, 10.0, 30.0]), ("long_gap_w", [0.0, 2.0, 4.0, 8.0, 16.0]),
        ("exit_w", [0.0, 4.0, 8.0, 16.0]),
        ("still_w", [0.0, 0.1, 0.3]), ("offpitch_w", [1.0, 3.0, 10.0]),
        ("min_dead_frames", [0, 25, 50, 75])]


def save(name, params, extra=None):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(dict(params=search.changes(params), **(extra or {})), fh, indent=1, sort_keys=True)


def load(path):
    with open(path) as fh:
        known = {f.name for f in dataclasses.fields(Params)}   # removed experiments are skipped
        return dataclasses.replace(Params(), **{k: v for k, v in json.load(fh)["params"].items()
                                                if k in known})


def per_key(res):
    return {"{}|{}".format(*k): round(v[0], 4) for k, v in res.items()}


def control(keys):
    base = Params()
    out = {}
    for name, start, grids in (
            ("carry only", dataclasses.replace(base, touch_w=0.0), CARRY + RESTART),
            ("touch only", dataclasses.replace(base, carry_w=0.0), TOUCH + RESTART),
            ("both", base, CARRY + TOUCH + RESTART)):
        p, best = search.descend(start, grids, keys, "control: " + name)
        res = search.evaluate([p], keys)[0]
        out[name] = (p, best, res)
        print("{:<10} J {:.4f}/{:.4f} [{}] {}".format(name, *best, search.by_variant(res, keys),
                                                     search.changes(p)))
        search.log("control: " + name, "tuned control definition", "PLAN Phase 3",
                   json.dumps(search.changes(p)), "J={:.4f} macro={:.4f} {}".format(
                       *best, search.by_variant(res, keys)))
        save("control_" + name.replace(" ", "_"), p, dict(J=best, per_key=per_key(res)))
    return out


def full(keys, start, step="full"):
    grids = CARRY + TOUCH + RESTART + DEAD
    grids = [g for g in grids if not (g[0].startswith("carry_") and start.carry_w == 0)
             and not (g[0].startswith("touch_") and start.touch_w == 0)]
    return search.descend(start, grids, keys, step)


if __name__ == "__main__":
    stage = sys.argv[1]
    keys = search.setup()
    if stage == "control":
        control(keys)
    elif stage == "full":
        start = load(sys.argv[2]) if len(sys.argv) > 2 else Params()
        p, best = full(keys, start)
        res = search.evaluate([p], keys)[0]
        print("full J {:.4f}/{:.4f} [{}] {}".format(*best, search.by_variant(res, keys), search.changes(p)))
        save(sys.argv[3] if len(sys.argv) > 3 else "full", p, dict(J=best, per_key=per_key(res)))
    elif stage == "loo":
        start = load(sys.argv[2]) if len(sys.argv) > 2 else Params()
        allp = load(sys.argv[3])
        res_all = search.evaluate([allp], keys)[0]
        for held in c.METRICA:
            train = [k for k in keys if k[0] != held]
            test = [k for k in keys if k[0] == held]
            p, best = full(train, start, "loo without " + held)
            r = search.evaluate([p], keys)[0]
            a, b = c.mean_score(r, test)[0], c.mean_score(res_all, test)[0]
            print("held out {}: leave-out choice {:.4f}, all-three choice {:.4f}, diff {:+.4f} | {}".format(
                held, a, b, a - b, search.changes(p)))
            search.log("loo", "held out " + held, "PLAN Phase 3 done-when", json.dumps(search.changes(p)),
                       "leave-out {:.4f} all-three {:.4f} diff {:+.4f}".format(a, b, a - b))
            save("loo_" + held, p, dict(held_out=a, all_three=b))
