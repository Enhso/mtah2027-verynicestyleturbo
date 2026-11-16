"""DFL open Bundesliga tracking (Bassek et al., CC-BY 4.0) to the shared Match schema.

Parses the raw XML with iterparse (kloppy builds a full Python object per frame and is too
heavy for 8 GB). Source files: HuggingFace pysport/idsse-data, the same files kloppy downloads.

Label rule: D where the ball's BallStatus is 0 (dead), else H for BallPossession 1 (home) and A for
2 (away). A frame with a missing BallPossession is not read as away (kloppy does): it takes the
previous frame's label, and the count is printed so a match with many such frames can be flagged.

Timeline: frame index = DFL frame number N minus the first half's first N, so the halftime gap is
the real one (X labels, NaN tracking); periods are the two halves' [first, last+1) ranges.
Coordinates: DFL is centred (-52.5..52.5, -34..34); shifted to a corner origin.

Usage: python -m harness.dfl J03WMX <dir containing the two raw XML files>
"""

import sys
import xml.etree.ElementTree as ET
from glob import glob

import numpy as np

from possession.match import A, D, H, X, Match, save

FPS = 25
BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
COMPETITION = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001"}  # others: DFL-COM-000002


def urls(match_id):
    comp = COMPETITION.get(match_id, "DFL-COM-000002")
    meta = f"DFL_02_01_matchinformation_{comp}_DFL-MAT-{match_id}.xml"
    track = f"DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{match_id}.xml"
    return BASE + meta, BASE + track


def _teams(meta_path):
    """({team id: 0 home / 1 guest}, [pitch x, pitch y], {person id: team index})."""
    root = ET.parse(meta_path).getroot()
    env = root.find(".//Environment").attrib
    role = {t.get("TeamId"): (0 if t.get("Role") == "home" else 1) for t in root.iter("Team")}
    person = {p.get("PersonId"): role[t.get("TeamId")]
              for t in root.iter("Team") for p in t.iter("Player")}
    return role, (float(env["PitchX"]), float(env["PitchY"])), person


def parse(meta_path, track_path):
    role, (L, W), _ = _teams(meta_path)
    slots, team = {}, []  # PersonId -> slot index
    half_n = {"firstHalf": [], "secondHalf": []}  # frame numbers of the ball per half
    pos = {}  # (section, person) -> (N array, xy array, speed array)
    ball = {}
    cur = None
    for ev, el in ET.iterparse(track_path, events=("start", "end")):
        if ev == "start":
            if el.tag == "FrameSet":
                cur = (el.get("GameSection"), el.get("PersonId"), el.get("TeamId"), [])
            continue
        if el.tag == "Frame":
            a = el.attrib
            row = (int(a["N"]), float(a["X"]), float(a["Y"]), float(a["S"]),
                   int(a["BallPossession"]) if "BallPossession" in a else 0,
                   int(a["BallStatus"]) if "BallStatus" in a else -1)
            cur[3].append(row)
            el.clear()
        elif el.tag == "FrameSet":
            sec, pid, tid, rows = cur
            arr = np.array(rows, dtype=np.float64)
            if tid == "BALL":
                ball[sec] = arr
            elif tid in role:
                if pid not in slots:
                    slots[pid] = len(slots)
                    team.append(role[tid])
                pos[(sec, pid)] = arr
            el.clear()

    n0 = int(ball["firstHalf"][0, 0])
    spans = [(int(ball[s][0, 0]) - n0, int(ball[s][-1, 0]) - n0 + 1)
             for s in ("firstHalf", "secondHalf")]
    n = spans[1][1]
    K = len(slots)
    b = np.full((n, 2), np.nan, np.float32)
    pl = np.full((n, K, 2), np.nan, np.float32)
    sp = np.full((n, K), np.nan, np.float32)
    lab = np.full(n, X, np.int8)
    stats = {"missing_possession": 0}
    for sec, (a0, a1) in zip(("firstHalf", "secondHalf"), spans):
        arr = ball[sec]
        i = arr[:, 0].astype(int) - n0
        b[i, 0], b[i, 1] = arr[:, 1] + L / 2, arr[:, 2] + W / 2
        poss, status = arr[:, 4], arr[:, 5]
        stats["missing_possession"] += int(((poss == 0) & (status != 0)).sum())
        code = np.where(status == 0, D, np.where(poss == 1, H, np.where(poss == 2, A, -1)))
        # a frame with no usable possession takes the previous frame's label (first: D)
        last = D
        for j in range(len(code)):
            if code[j] < 0:
                code[j] = last
            last = code[j]
        lab[i] = code
        # fill frames absent from the ball record (none expected) with the previous label
        seg = lab[a0:a1]
        seen = np.zeros(a1 - a0, bool)
        seen[i - a0] = True
        for j in range(1, a1 - a0):
            if not seen[j]:
                seg[j] = seg[j - 1]
    for (sec, pid), arr in pos.items():
        i = arr[:, 0].astype(int) - n0
        k = slots[pid]
        pl[i, k, 0], pl[i, k, 1] = arr[:, 1] + L / 2, arr[:, 2] + W / 2
        sp[i, k] = arr[:, 3]
    return Match(name="", fps=FPS, pitch=(L, W), ball=b, players=pl, speed=sp,
                 team=np.array(team, np.int8), periods=np.array(spans, np.int64),
                 labels=lab), stats


def export(match_id, raw_dir, cache_dir):
    meta = glob(f"{raw_dir}/DFL_02_01_matchinformation_*{match_id}.xml")[0]
    track = glob(f"{raw_dir}/DFL_04_03_positions_raw_observed_*{match_id}.xml")[0]
    m, stats = parse(meta, track)
    m = Match(**{**m.__dict__, "name": f"dfl_{match_id}"})
    save(m, cache_dir=cache_dir)
    return m, stats


if __name__ == "__main__":
    import os
    cache = os.path.join(os.environ["MTAH_DATA_DIR"], "cache", "ext")
    m, stats = export(sys.argv[1], sys.argv[2], cache)
    print(m.name, m.n_frames, m.players.shape, m.periods.tolist(), stats)
