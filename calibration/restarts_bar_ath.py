"""BAR-ATH change table for the set-piece launch rule and the goal rule (aggregates only).

    python -m calibration.restarts_bar_ath

For each setting: share of graded frames whose label differs from v1, how many of those lie inside
the SET PIECES windows and inside the goal-to-kick-off stretches (GOALS tag start to the end of the
next kick-off-after-goal window), change in possession count, and the two release gates.
"""

import numpy as np

from calibration import common as c
from evaluation import possession_metrics as pm
from possession import config as cfg, evidence, gates
from possession.params import Params

B = dict(goal_after_setpiece=True)   # goal D starts at the end of the last overlapped set-piece window
SETTINGS = [
    ("v1", {}),
    ("launch", dict(setpiece_launch=True)),
    ("goal@tag w=1", dict(goal_dead_w=1.0)),
    ("goal@tag w=10", dict(goal_dead_w=10.0)),
    ("goal@sp w=0.3", dict(goal_dead_w=0.3, **B)),
    ("goal@sp w=1", dict(goal_dead_w=1.0, **B)),
    ("goal@sp w=10", dict(goal_dead_w=10.0, **B)),
    ("launch + goal@tag w=1", dict(setpiece_launch=True, goal_dead_w=1.0)),
    ("launch + goal@sp w=1", dict(setpiece_launch=True, goal_dead_w=1.0, **B)),
]


def stretches(m):
    """(GOALS tag start, end of the next kick-off-after-goal window) frames, global."""
    ko = sorted((t["start_frame"], t["end_frame"]) for t in m.tags
                if (t["code"] or "").strip().lower() == evidence.SET_PIECE_CODE
                and "after goal" in (t["tags"].get("Type") or "").lower())
    return [(g["start_frame"], next(e for s, e in ko if s >= g["start_frame"]))
            for g in m.tags if (g["code"] or "").strip().lower() == evidence.GOAL_CODE]


def main():
    m = c.load_match("BAR-ATH")
    c.register("BAR-ATH", m)
    graded = np.zeros(m.n_frames, bool)
    for a, b in m.periods.tolist():
        graded[a:b] = True
    in_sp = np.zeros(m.n_frames, bool)
    for t in m.tags:
        if (t["code"] or "").strip().lower() == evidence.SET_PIECE_CODE:
            in_sp[t["start_frame"]:t["end_frame"]] = True
    in_goal = np.zeros(m.n_frames, bool)
    for a, b in stretches(m):
        in_goal[a:b] = True
    print("graded frames {}; inside SET PIECES windows {:.1%}; inside goal stretches {:.2%}".format(
        int(graded.sum()), in_sp[graded].mean(), in_goal[graded].mean()))
    base = None
    for name, ch in SETTINGS:
        codes = c.decode("BAR-ATH", Params(**ch))
        base = codes if base is None else base
        diff = (codes != base) & graded
        gate = {r[0].split(" (")[0]: r[1] for r in gates.compute(m, codes, True, smoke=False)}
        sp = next(v for k, v in gate.items() if k.startswith("set-piece windows"))
        ko = next(v for k, v in gate.items() if k.startswith("D before"))
        print("{:<22} changed {:.2%} (in set-piece windows {:.0%}, in goal stretches {:.0%}, elsewhere {} frames) "
              "spells {} ({:+d}) D share {:.3f} | set-piece gate {} | kick-offs {}".format(
                  name, diff.sum() / graded.sum(),
                  (diff & in_sp).sum() / max(diff.sum(), 1), (diff & in_goal).sum() / max(diff.sum(), 1),
                  int((diff & ~in_sp & ~in_goal).sum()),
                  len(pm.possession_spells(codes, m.periods.tolist())),
                  len(pm.possession_spells(codes, m.periods.tolist())) - len(pm.possession_spells(base, m.periods.tolist())),
                  float((codes[graded] == 2).mean()), sp, ko))


if __name__ == "__main__":
    main()
