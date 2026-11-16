"""PLAN section 6: the six categorical questions, answered on the public window.

    python -m possession.semantics

Prints counts, offsets in frames and shares. No frame index and no coordinate is printed. The
answers decide channel definitions, never a number; nothing here selects a parameter.
"""

import glob
import os

import numpy as np

from . import evidence, features, gates, ingest  # gates puts evaluation/ on the path
from .params import Params

import possession_metrics as pm  # noqa: E402

WINDOW = (11185, 18685)       # public ground truth, half-open [DATA.md]
SEARCH = 125                  # frames either side of a ground-truth transition


def _q(values):
    if not len(values):
        return "none"
    v = np.asarray(values, float)
    return "n={} median {:+.0f} min {:+.0f} max {:+.0f}".format(len(v), np.median(v), v.min(), v.max())


def _bridge(mask, gap):
    """mask with False runs of at most `gap` frames between True frames filled in."""
    out = mask.copy()
    for a, b in zip(*features.runs(~mask)):
        if 0 < a and b < len(mask) and b - a <= gap:
            out[a:b] = True
    return out


def _first_offset(frames, t, lo, hi):
    """Offset of the first element of frames inside [t+lo, t+hi], else None."""
    inside = frames[(frames >= t + lo) & (frames <= t + hi)]
    return int(inside[0] - t) if len(inside) else None


def _nearest_offset(frames, t, span):
    if not len(frames):
        return None
    d = frames - t
    d = d[np.abs(d) <= span]
    return int(d[np.argmin(np.abs(d))]) if len(d) else None


def main():
    p = Params()
    med = lambda v: np.median(np.abs(v)) if len(v) else np.inf
    match = ingest.ensure()
    a, b = WINDOW
    f = features.compute(match.ball[a:b], match.players[a:b], match.speed[a:b], match.team,
                         match.pitch, match.fps, p)
    gt_path = glob.glob(os.path.join(ingest.ROOT, "evaluation", "gt", "*_possession_gt_public.csv"))[0]
    gt = pm.label_codes(pm.read_label_csv(gt_path))[a:b]
    n = len(gt)
    assert (gt != 3).all(), "public window has X"
    runs = [(0, n)]
    trans = pm.transitions(gt, runs)
    spells = pm.possession_spells(gt, runs)
    print("window: {} frames, {} GT possessions, {} transitions, labels H/A/D share {}".format(
        n, len(spells), len(trans), [round(float((gt == k).mean()), 3) for k in range(3)]))

    carry = evidence.carry_events(f, p)
    t_frames, t_team = evidence.touch_events(f, p)
    print("detected in window: {} carry frames, {} touch events".format(
        int((carry >= 0).sum()), len(t_frames)))

    # Q1. first touch of the new team, or sustained control?
    print("\nQ1  possession start: first touch or sustained control?")
    onset, touch_near, missing = [], [], 0
    for t, old, new in trans:
        if new == 2:
            continue
        spans = [(int(s), int(e)) for s, e in zip(*features.runs(_bridge(carry == new, 10)))]
        dist = [0 if s <= t < e else min(abs(s - t), abs(e - t)) for s, e in spans]
        if dist and min(dist) <= SEARCH:
            onset.append(spans[int(np.argmin(dist))][0] - t)
        else:
            missing += 1
        o = _nearest_offset(t_frames[t_team == new], t, SEARCH)
        if o is not None:
            touch_near.append(o)
    print("  transitions into a possession: {} ({} from D); no carry run within +-{} frames: {}".format(
        sum(1 for _, _, new in trans if new != 2),
        sum(1 for _, old, new in trans if new != 2 and old == 2), SEARCH, missing))
    print("  new team's nearest carry run starts at (offset from GT start):", _q(onset))
    print("  nearest new-team touch event (offset)                        :", _q(touch_near))
    lag = np.median(onset) if len(onset) else np.nan
    print("  ANSWER Q1:", "sustained control (GT starts >= 12 frames after carry onset)"
          if lag <= -12 else "first touch (GT start within 12 frames of carry onset, or before it)")

    # Q2. does the owner stay through a pass in flight or a loose ball?
    print("\nQ2  owner through flight and loose ball")
    control = carry.copy()
    control[t_frames] = t_team
    last = np.maximum.accumulate(np.where(control >= 0, np.arange(n), -1))
    latest = np.where(last >= 0, control[np.maximum(last, 0)], -1)
    alive = gt < 2
    for name, far in (("no player within 3 m", 3.0), ("no player within 6 m (flight)", 6.0)):
        sel = alive & f.visible & (f.near_dist > far) & (latest >= 0)
        if sel.sum():
            print("  {:<32} frames {:>5}  GT == latest controlling team {:.3f}  GT == nearest team {:.3f}".format(
                name, int(sel.sum()), float((gt[sel] == latest[sel]).mean()),
                float((gt[sel] == f.near_team[sel]).mean())))
    sel = alive & f.visible & (f.near_dist > 3.0) & (latest >= 0)
    print("  ANSWER Q2:", "owner unchanged (latch)"
          if (gt[sel] == latest[sel]).mean() > (gt[sel] == f.near_team[sel]).mean()
          else "owner follows the nearest player")

    # Q3. does D start when the ball crosses the line or when tracking is lost?
    print("\nQ3  D start: line crossing or tracking lost?")
    cross_off, lost_off = [], []
    for t, old, new in trans:
        if new != 2:
            continue
        off = np.flatnonzero(f.offpitch)
        gap_starts = np.array([g for g, _ in f.gaps], dtype=np.int64)
        c, l = _first_offset(off, t, -SEARCH, SEARCH), _first_offset(gap_starts, t, -SEARCH, SEARCH)
        near_line = _first_offset(np.flatnonzero(f.visible & (f.line_dist <= 0.5)), t, -SEARCH, SEARCH)
        print("  stoppage: first off-pitch offset {}, first blackout start offset {}, within 0.5 m of a line {}".format(
            c, l, near_line))
        if c is not None:
            cross_off.append(c)
        if l is not None:
            lost_off.append(l)
    print("  off-pitch offsets", _q(cross_off), "| blackout-start offsets", _q(lost_off))
    n_cross = sum(abs(o) <= 25 for o in cross_off)
    n_lost = sum(abs(o) <= 25 for o in lost_off)
    print("  stoppages with the event within 25 frames of the GT start: off-pitch {}, blackout start {}".format(
        n_cross, n_lost))
    print("  ANSWER Q3:", "line crossing" if n_cross > n_lost else "tracking lost" if n_lost > n_cross
          else "neither alone: D starts at the stop, which the line crossing and the blackout only sometimes coincide with")

    # Q4. does D end on the strike? Strike = the ball leaving rest: forward speed over 3 frames
    # rises from under 1 m/s to over 2 m/s.
    print("\nQ4  D end: ball struck or thrown?")
    for name, sel in (("GT D", gt == 2), ("GT alive", gt < 2)):
        v = f.still_speed[sel & f.visible]
        print("  ball speed over 10 frames on visible {} frames, quantiles 5/25/50/75/95: {}  (n={})".format(
            name, np.round(np.nanquantile(v, [.05, .25, .5, .75, .95]), 2).tolist() if len(v) else "none", len(v)))
    carry_off, touch_off = [], []
    for t, old, new in trans:
        if old == 2 and new != 2:
            spans = [(int(s0), int(e0)) for s0, e0 in zip(*features.runs(_bridge(carry == new, 10)))]
            dist = [0 if s0 <= t < e0 else min(abs(s0 - t), abs(e0 - t)) for s0, e0 in spans]
            c = spans[int(np.argmin(dist))][0] - t if dist and min(dist) <= SEARCH else None
            o = _nearest_offset(t_frames[t_team == new], t, SEARCH)
            print("  restart: nearest restarting-team carry run starts at {}, nearest touch event {}".format(c, o))
            carry_off += [] if c is None else [c]
            touch_off += [] if o is None else [o]
    print("  carry-run onset offsets:", _q(carry_off), "| touch offsets:", _q(touch_off))
    print("  ANSWER Q4:", "undetermined: no carry onset near the restarts" if not len(carry_off)
          else "D ends at the strike (carry onset within 12 frames)" if med(carry_off) <= 12
          else "D ends {} the first carry".format("before" if np.median(carry_off) > 0 else "after"))

    # Q5. possessions under 1.5 s
    print("\nQ5  possessions under 1.5 s: did the ball change velocity next to the owner?")
    change = np.linalg.norm(f.v_after - f.v_before, axis=1)
    short = with_owner_touch = 0
    for s, e, team in spells:
        if (e - s) / match.fps >= 1.5:
            continue
        short += 1
        with_owner_touch += bool(((t_frames >= s - 5) & (t_frames < e + 5) & (t_team == team)).any())
        seg = np.nan_to_num(change[max(s - 5, 0):e + 5])
        k = max(s - 5, 0) + int(np.argmax(seg))
        print("  spell {:.1f} s: peak velocity change {:.1f} m/s, nearest team at peak == owner {}, "
              "nearest dist {:.1f} m, touch events by owner {}, by other {}, carry frames by owner {}".format(
                  (e - s) / match.fps, seg.max(), bool(f.near_team[k] == team),
                  f.near_dist[k] if np.isfinite(f.near_dist[k]) else float("nan"),
                  int(((t_frames >= s - 5) & (t_frames < e + 5) & (t_team == team)).sum()),
                  int(((t_frames >= s - 5) & (t_frames < e + 5) & (t_team != team)).sum()),
                  int((carry[s:e] == team).sum())))

    print("  ANSWER Q5: {} of {} short possessions have a detected ball-velocity change next to the owner{}".format(
        with_owner_touch, short, "" if with_owner_touch else " (no: the possession flips with no touch signal)"))

    # Q6. the keeper holding the ball
    print("\nQ6  keeper holding the ball: alive or dead?")
    L, W = match.pitch
    still = evidence.still_ball(f, p) & f.visible & (f.near_dist <= 1.5)
    x, y = f.ball[:, 0], f.ball[:, 1]
    box = ((x <= 16.5) | (x >= L - 16.5)) & (np.abs(y - W / 2) <= 20.16)
    for name, sel in (("still ball, player within 1.5 m, inside a penalty area", still & box),
                      ("still ball, player within 1.5 m, outside the areas", still & ~box)):
        k = int(sel.sum())
        print("  {:<58} frames {:>4} runs {:>2}  GT alive {:.2f}  GT D {:.2f}".format(
            name, k, len(features.runs(sel)[0]), float((gt[sel] < 2).mean()) if k else float("nan"),
            float((gt[sel] == 2).mean()) if k else float("nan")))
    sel = still & box
    print("  ANSWER Q6:", "undetermined: no still-ball-in-box case in the window" if not sel.sum()
          else ("alive" if (gt[sel] < 2).mean() >= 0.5 else "dead"))


if __name__ == "__main__":
    main()
