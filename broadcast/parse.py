"""Parse the broadcast ATD to float32 arrays and cache them. Prints aggregates only."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import games  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

OUT = os.environ.get("BC_CACHE", "/tmp/bc_cache.npz")


def parse(raw_path, n_slots):
    ids, ball, players = [], [], []
    for line in open(raw_path):
        p = line.rstrip("\n").split(":")
        ids.append(int(p[0]))
        sl = p[1].split(";")
        assert len(sl) == n_slots
        players.append([[float(v) for v in s.split(",")[:2]] for s in sl])
        ball.append([float(v) for v in p[2].split(",")[:2]])
    return np.array(ids), np.array(ball, np.float32), np.array(players, np.float32)


if __name__ == "__main__":
    P = games.asset_paths(source="broadcast")
    team = team_of_each_track(P["atd_meta"])
    ids, ball, pl = parse(P["atd_raw"], len(team))
    np.savez(OUT, ids=ids, ball=ball, players=pl, team=team)
    print("frames", len(ids), "ids contiguous", bool((np.diff(ids) == 1).all()), "first", ids[0])
    print("slots", len(team), "home", int((team == 0).sum()), "away", int((team == 1).sum()))
    print("ball present", float(np.isfinite(ball[:, 0]).mean()))
    pv = np.isfinite(pl[:, :, 0])
    print("players per frame quantiles", np.quantile(pv.sum(1), [0, .1, .5, .9, 1]))
    print("x quantiles", np.nanquantile(pl[:, :, 0], [0, .01, .99, 1]))
    print("y quantiles", np.nanquantile(pl[:, :, 1], [0, .01, .99, 1]))
