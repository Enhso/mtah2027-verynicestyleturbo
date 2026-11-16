"""Phase 0 check: the PLAN figures, reproduced from the raw files and the cache.

    python -m possession.check_phase0

Prints and asserts counts only. Exits non-zero on the first figure that disagrees.
"""

import os
import subprocess
import sys

import numpy as np

from . import ingest, match as mt

ROOT = ingest.ROOT
import games  # noqa: E402
from load_atd import check_alignment_raw  # noqa: E402
from baseline import team_names  # noqa: E402


def ball_group_field_counts(raw_path):
    """{fields in the ball group: number of records}. A count, never a value."""
    counts = {}
    with open(raw_path) as handle:
        for line in handle:
            n = line.rstrip("\n").rsplit(":", 1)[1].count(",") + 1
            counts[n] = counts.get(n, 0) + 1
    return counts


def main():
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"])
    report = check_alignment_raw()
    print("records {}  frame_id range {}..{}  contiguous {}".format(
        report["n_records"], report["frame_id_range"][0], report["frame_id_range"][1],
        report["contiguous"]))
    assert report["n_records"] == 199319 and report["frame_id_range"] == (1, 199319)
    assert report["contiguous"] and report["agrees"]

    m = ingest.ensure()
    present = ~np.isnan(m.ball).any(axis=1)
    graded = np.zeros(m.n_frames, bool)
    for a, b in m.periods:
        graded[a:b] = True
    print("ball present: {} frames overall; {} of {} graded frames".format(
        int(present.sum()), int(present[graded].sum()), int(graded.sum())))
    assert m.n_frames == 199319
    assert int(present.sum()) == 113488
    assert int(graded.sum()) == 158857 and int(present[graded].sum()) == 94214

    home, away = team_names(meta_path)
    print("first team in the metadata: {} (home), second: {}".format(home, away))
    assert "barcelona" in home.lower()
    print("slots {}  home slots {}  away slots {}".format(
        len(m.team), int((m.team == mt.HOME).sum()), int((m.team == mt.AWAY).sum())))

    counts = ball_group_field_counts(raw_path)
    print("ball group field counts (fields: records): {}".format(dict(sorted(counts.items()))))
    print("ball group has a third field on {} records".format(
        sum(n for f, n in counts.items() if f >= 3)))

    print("periods (half-open): {}".format(m.periods.tolist()))
    print("smart tagging instances in cache: {}".format(len(m.tags)))

    out = subprocess.run([sys.executable, os.path.join(ROOT, "starter", "load_smart_tagging.py")],
                         capture_output=True, text=True, check=True).stdout
    ok_lines = [l.strip() for l in out.splitlines() if l.strip().startswith("OK")]
    print("starter/load_smart_tagging.py OK line: {}".format(ok_lines[0] if ok_lines else "MISSING"))
    assert ok_lines, "load_smart_tagging.py printed no OK line"
    print("\nPHASE 0 CHECK PASSED")


if __name__ == "__main__":
    main()
