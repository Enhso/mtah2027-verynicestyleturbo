"""Metrica Sports public sample games (3 games, normalised 0..1 coordinates) to the Match schema.

Source: the metrica-sports sample-data repository on GitHub. No licence file; the README's "Legal
stuff" asks for responsible use and acknowledgement of the source in anything public.

Label derivation (from the event stream, by code; games 1 and 2 come from the CSV events, game 3
from the JSON events, which differ only by also having CARRY):
  * Events are walked in order of start frame, inside each half (the half's frame range comes from
    the tracking file's Period column, or from the metadata for game 3).
  * On-ball events: PASS, SHOT, RECOVERY, SET PIECE, CARRY. At its start frame the possession
    passes to the event's team (H for Home / Team A, A for Away / Team B) and the ball is alive.
    BALL LOST, CHALLENGE and CARD do not change possession: the opponent's RECOVERY follows.
  * Stoppage events: BALL OUT, FAULT RECEIVED, a BALL LOST whose subtype says OFFSIDE or HAND
    BALL, and a SHOT whose subtype says GOAL (not GOAL KICK). At their start frame the label
    becomes D. It stays D until the restart: the next SET PIECE, or the next PASS of subtype
    GOAL KICK (goal kicks have no SET PIECE row). Other on-ball events in between (the CARRY that
    game 3 logs at the same frame as BALL OUT) are ignored while dead.
  * Each half starts as D until its first on-ball event (the kick-off) and the last label holds to
    the end of the half.
Frame f of the source (1-based) is index f - first frame of half 1 on the new timeline; half 2
starts after the end of half 1 plus GAP X frames with NaN tracking, so valid_runs sees two halves.
Pitch 105 x 68 m. Speed is computed from positions (m/s, forward difference inside a half, NaN
where it exceeds 12 m/s, a tracking jump).

Usage: python -m harness.metrica <dir with the sample-data files>
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from possession.match import A, D, H, X, Match, save

FPS = 25
L, W = 105.0, 68.0
GAP = 22500          # 15 minutes of X between the halves
MAX_SPEED = 12.0
ON_BALL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "CARRY"}


def labels_from_events(events, halves, n):
    """events: (team 0/1, type, subtype, start_frame, period) in file order; halves: [(start, stop)]
    new-timeline index ranges with the source frame offset: [(src_first, src_last, new_first)]."""
    lab = np.full(n, X, np.int8)
    for p, (f0, f1, new0) in enumerate(halves, start=1):
        ev = sorted((e for e in events if e[4] == p and f0 <= e[3] <= f1), key=lambda e: e[3])
        state, last = D, f0
        for team, typ, sub, f, _ in ev:
            restart = typ == "SET PIECE" or (typ == "PASS" and "GOAL KICK" in sub)
            if typ in ("BALL OUT", "FAULT RECEIVED") or (
                    typ == "SHOT" and "GOAL" in sub.replace("GOAL KICK", "")) or (
                    typ == "BALL LOST" and ("OFFSIDE" in sub or "HAND BALL" in sub)):
                new = D
            elif typ in ON_BALL and (state != D or restart):
                new = team
            else:
                continue
            lab[new0 + last - f0:new0 + f - f0] = state
            state, last = new, f
        lab[new0 + last - f0:new0 + f1 - f0 + 1] = state
    return lab


def _finish(name, halves, ball, players, team, events):
    """ball/players are on the source frame axis (index = frame - 1)."""
    n = halves[1][2] + halves[1][1] - halves[1][0] + 1
    K = players.shape[1]
    B = np.full((n, 2), np.nan, np.float32)
    P = np.full((n, K, 2), np.nan, np.float32)
    S = np.full((n, K), np.nan, np.float32)
    periods = []
    for f0, f1, new0 in halves:
        sl = slice(new0, new0 + f1 - f0 + 1)
        src = slice(f0 - 1, f1)
        B[sl] = ball[src] * (L, W)
        P[sl] = players[src] * (L, W)
        v = np.full((f1 - f0 + 1, K), np.nan, np.float32)
        v[1:] = np.linalg.norm(np.diff(P[sl], axis=0), axis=2) * FPS
        S[sl] = np.where(v > MAX_SPEED, np.nan, v)
        periods.append((new0, new0 + f1 - f0 + 1))
    lab = labels_from_events(events, halves, n)
    return Match(name=name, fps=FPS, pitch=(L, W), ball=B, players=P, speed=S,
                 team=np.array(team, np.int8), periods=np.array(periods, np.int64), labels=lab)


def _halves(frames, period):
    out, new0 = [], 0
    for p in (1, 2):
        f = frames[period == p]
        out.append((int(f.min()), int(f.max()), new0))
        new0 += int(f.max() - f.min()) + 1 + GAP
    return out


def game_csv(d, g):
    t = {}
    for side in ("Home", "Away"):
        t[side] = pd.read_csv(f"{d}/Sample_Game_{g}_RawTrackingData_{side}_Team.csv", header=2)
    home, away = t["Home"], t["Away"]
    frames, period = home["Frame"].to_numpy(), home["Period"].to_numpy()
    assert (frames == np.arange(1, len(frames) + 1)).all()
    xy = lambda df: df.iloc[:, 3:-2].to_numpy(np.float32).reshape(len(df), -1, 2)
    hx, ax = xy(home), xy(away)
    players = np.concatenate([hx, ax], axis=1)
    ball = home.iloc[:, -2:].to_numpy(np.float32)
    team = [0] * hx.shape[1] + [1] * ax.shape[1]
    ev = pd.read_csv(f"{d}/Sample_Game_{g}_RawEventsData.csv")
    events = [(0 if r.Team == "Home" else 1, r.Type, "" if pd.isna(r.Subtype) else r.Subtype,
               int(r["Start Frame"]), int(r.Period)) for _, r in ev.iterrows()]
    return _finish(f"metrica_game{g}", _halves(frames, period), ball, players, team, events)


def game3(d):
    rows = open(f"{d}/Sample_Game_3_tracking.txt").read().splitlines()
    n = len(rows)
    players = np.full((n, 22, 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    for i, line in enumerate(rows):
        _, rest = line.split(":", 1)
        pl, b = rest.split(":")[:2]
        players[i] = [[float(v) for v in p.split(",")] for p in pl.split(";")]
        ball[i] = [float(v) for v in b.split(";")[0].split(",")]
    meta = open(f"{d}/Sample_Game_3_metadata.xml").read()
    par = lambda k: int(meta.split(f"<Name>{k}</Name>")[1].split("<Value>")[1].split("<")[0])
    halves = [(par("first_half_start"), par("first_half_end"), 0)]
    halves.append((par("second_half_start"), par("second_half_end"),
                   halves[0][1] - halves[0][0] + 1 + GAP))
    ev = json.load(open(f"{d}/Sample_Game_3_events.json"))["data"]
    sub = lambda e: (e["subtypes"] or {}).get("name", "") if isinstance(e["subtypes"], dict) else \
        " ".join(s["name"] for s in e["subtypes"] or [])
    events = [(0 if e["team"]["id"] == "FIFATMA" else 1, e["type"]["name"], sub(e),
               int(e["start"]["frame"]), int(e["period"])) for e in ev]
    return _finish("metrica_game3", halves, ball, players, [0] * 11 + [1] * 11, events)


if __name__ == "__main__":
    d = sys.argv[1]
    cache = os.path.join(os.environ["MTAH_DATA_DIR"], "cache", "ext")
    for m in (game_csv(d, 1), game_csv(d, 2), game3(d)):
        save(m, cache_dir=cache)
        print(m.name, m.n_frames, m.players.shape, m.periods.tolist())
