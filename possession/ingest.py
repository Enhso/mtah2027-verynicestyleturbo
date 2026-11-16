"""Phase 0: parse this match's ATD text, metadata XML and Smart Tagging XML into one `Match`.

    python -m possession.ingest          # builds the cache if it is missing

The ATD is already on the tactical video timeline (video frame = frame_id - 1). Coordinates are
normalised 0..1 and are scaled to metres here with the pitch of games.py.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (first Team of the metadata is home)

from . import match as mt  # noqa: E402

NAME = games.GAME
FIELD_SEP, GROUP_SEP = ":", ";"


def _group(text):
    """(x, y, speed) of one 'x,y,speed' group; NaN for whatever is absent."""
    fields = text.split(",")
    out = [float(f) for f in fields[:3]]
    return out + [np.nan] * (3 - len(out))


def parse_atd(raw_path, n_frames, n_slots):
    """Stream the raw ATD once. Returns normalised ball (N,3), players (N,K,3), ball field count.

    The third field of the ball group is counted rather than assumed: the second returned value is
    the number of ball groups that carry three comma-separated fields.
    """
    ball = np.full((n_frames, 3), np.nan)
    players = np.full((n_frames, n_slots, 3), np.nan)
    ball_with_third = 0
    expected_id = 1
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError("ATD line {} has {} parts".format(expected_id, len(parts)))
            if int(parts[0]) != expected_id:
                raise ValueError("ATD frame ids are not 1..N at record {}".format(expected_id))
            slots = parts[1].split(GROUP_SEP)
            if len(slots) != n_slots:
                raise ValueError("ATD record {} has {} slots, expected {}".format(
                    expected_id, len(slots), n_slots))
            row = expected_id - 1
            for k, text in enumerate(slots):
                players[row, k] = _group(text)
            ball[row] = _group(parts[2])
            if len(parts[2].split(",")) >= 3:
                ball_with_third += 1
            expected_id += 1
    if expected_id - 1 != n_frames:
        raise ValueError("ATD has {} records, expected {}".format(expected_id - 1, n_frames))
    return ball, players, ball_with_third


def build(game=NAME, data_dir=None):
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = games.source_frames("tactical", game)
    length, width = games.GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    ball, players, _ = parse_atd(raw_path, n_frames, len(team))

    scale = np.array([length, width])
    periods = np.array([[first, last + 1] for _, first, last in games.half_video_blocks(game)],
                       dtype=np.int64)
    return mt.Match(
        name=game, fps=games.GAMES[game]["fps"], pitch=(length, width),
        ball=(ball[:, :2] * scale).astype(np.float32),
        players=(players[:, :, :2] * scale).astype(np.float32),
        speed=players[:, :, 2].astype(np.float32),
        team=team.astype(np.int8), periods=periods, tags=load_tags(game, data_dir))


def ensure(game=NAME, data_dir=None, cache_dir=None):
    """The cached Match, built and saved first only when the cache file is missing."""
    if not os.path.isfile(mt.path_for(game, cache_dir)):
        mt.save(build(game, data_dir), cache_dir)
    return mt.load(game, cache_dir)


if __name__ == "__main__":
    m = ensure()
    print("cache ready: {} frames, {} slots, {} tags".format(m.n_frames, len(m.team), len(m.tags)))
