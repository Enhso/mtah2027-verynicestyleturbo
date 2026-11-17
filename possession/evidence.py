"""Turn the features of one period into an (n,3) float64 score in nats for H, A, D.

Each channel returns its own (n,3) array and contributes zero wherever its input is absent, so a
blackout needs no separate code path. Per-event and per-gap weights are counted once and spread
over the frames of the event, never summed per frame.
"""

import numpy as np

from .features import keep_long, runs

H, A, D = 0, 1, 2

# Smart Tagging codes that carry a team-tagged attacking phase [doc: PLAN section 8]. The four
# DEFENDING codes are mirrors of four of these and are not used.
ATTACK_CODES = frozenset([
    "build up", "progression", "creating chances", "attacking transition", "long ball",
    "ball in final third", "ball in the box", "players in the box"])
SET_PIECE_CODE = "set pieces"
GOAL_CODE = "goals"
# Added by localize_tags next to a "set pieces" tag whose Type is a kick-off after a goal.
KICKOFF_AFTER_GOAL_CODE = "kick off after goal"


def localize_tags(tags, config, start, stop):
    """[(code, a, b, side)] of the tags that touch [start, stop), with frames local to the period.

    side is HOME, AWAY or None, from the Team tag through the per-match config.
    """
    out = []
    for tag in tags:
        a, b = max(tag["start_frame"], start), min(tag["end_frame"], stop)
        if a < b:
            code = (tag["code"] or "").strip().lower()
            out.append((code, a - start, b - start, config.tag_side(tag["tags"].get("Team"))))
            if code == SET_PIECE_CODE and "after goal" in (tag["tags"].get("Type") or "").lower():
                out.append((KICKOFF_AFTER_GOAL_CODE, a - start, b - start, None))
    return out


def cluster_peaks(candidate, strength, gap):
    """Frames of the strongest candidate in each cluster of candidates at most `gap` frames apart."""
    frames = np.flatnonzero(candidate)
    if not len(frames):
        return frames
    cluster = np.concatenate(([0], np.cumsum(np.diff(frames) > gap)))
    peaks = []
    for c in range(cluster[-1] + 1):
        members = frames[cluster == c]
        peaks.append(members[np.argmax(strength[members])])
    return np.array(peaks, dtype=np.int64)


def _team_column(score, mask, team, weight):
    """score[frame, team[frame]] += weight on the frames in mask."""
    for side in (0, 1):
        score[mask & (team == side), side] += weight


def carry_events(f, params):
    """(n,) int8: team of the carrying player on frames that meet the carry rule, else -1."""
    with np.errstate(invalid="ignore"):
        dv = np.linalg.norm(f.ball_v - f.near_v, axis=1)
        ok = (f.visible & (f.near_team >= 0) & (f.near_dist <= params.carry_radius)
              & (f.opp_dist - f.near_dist >= params.carry_margin) & (dv <= params.carry_dv))
    return np.where(ok, f.near_team, -1).astype(np.int8)


def touch_events(f, params):
    """Frames where the ball changed velocity next to a player (one per cluster), and their teams."""
    with np.errstate(invalid="ignore", divide="ignore"):
        change = np.linalg.norm(f.v_after - f.v_before, axis=1)
        sb, sa = np.linalg.norm(f.v_before, axis=1), np.linalg.norm(f.v_after, axis=1)
        cos = np.einsum("ij,ij->i", f.v_before, f.v_after) / (sb * sa)
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        angle[~((sb >= params.touch_min_speed) & (sa >= params.touch_min_speed))] = np.nan
        big = (change >= params.touch_dv) | (angle >= params.touch_angle)
        near = (f.near_team >= 0) & (f.near_dist <= params.touch_radius)
        candidate = f.visible & near & big
        strength = np.fmax(change / params.touch_dv, angle / params.touch_angle)
    frames = cluster_peaks(candidate, np.nan_to_num(strength), params.touch_cluster)
    return frames, f.near_team[frames]


def still_ball(f, params):
    with np.errstate(invalid="ignore"):
        return keep_long(f.still_speed < params.still_speed, int(round(params.still_s * f.fps)))


def launch_frame(f, still, params, a, b):
    """First frame of [a, b) where the ball is struck, or None: centred speed above
    setpiece_launch_speed after a hidden or still frame inside the window, else the first visible
    frame after a hidden one (notes/labelfree-answers.md, proposal 2)."""
    dead = ~f.visible[a:b] | still[a:b]
    after_dead = np.concatenate(([False], np.maximum.accumulate(dead)[:-1]))
    with np.errstate(invalid="ignore"):
        fast = np.linalg.norm(f.ball_v[a:b], axis=1) > params.setpiece_launch_speed
    hit = np.flatnonzero(fast & after_dead)
    if len(hit):
        return a + int(hit[0])
    hidden = np.flatnonzero(~f.visible[a:b])
    back = np.flatnonzero(f.visible[a + hidden[0]:b]) if len(hidden) else []
    return a + int(hidden[0] + back[0]) if len(back) else None


def channels(f, tags, params):
    """{name: (n,3) array} for every evidence channel of PLAN section 3 except the hard constraints."""
    n = f.n
    out = {name: np.zeros((n, 3)) for name in (
        "carry", "touch", "restart", "phase", "offpitch", "exit", "long_gap", "still", "low",
        "setpiece", "latch", "goal", "formation")}

    carry = carry_events(f, params)
    _team_column(out["carry"], carry >= 0, carry, params.carry_w)

    t_frames, t_team = touch_events(f, params)
    for side in (0, 1):
        out["touch"][t_frames[t_team == side], side] += params.touch_w

    # A control definition with zero weight is switched off: it does not feed restart or latch.
    control = carry.copy() if params.carry_w > 0 else np.full(n, -1, dtype=np.int8)
    if params.touch_w > 0:
        control[t_frames] = t_team
    control_frames = np.flatnonzero(control >= 0)

    still = still_ball(f, params)
    deadish = ~f.visible | f.offpitch | still
    window = int(round(params.restart_window_s * f.fps))
    for a, b in zip(*runs(deadish)):
        if b - a >= params.restart_min_s * f.fps and b < n:
            k = np.searchsorted(control_frames, b)
            if k < len(control_frames) and control_frames[k] - b <= window:
                frame = control_frames[k]
                # D ends when the ball is struck, which is where the dead-looking stretch ends;
                # the first control event after it only names the team.
                out["restart"][b, control[frame]] += params.restart_w

    for code, a, b, side in tags:
        if code in ATTACK_CODES and side is not None:
            out["phase"][a:b, side] += params.phase_w / (b - a)

    out["offpitch"][f.offpitch, D] = params.offpitch_w

    for a, b in (f.gaps if f.visible.any() else []):   # no ball at all: nothing to read
        if a > 0:
            i = a - 1
            with np.errstate(invalid="ignore"):
                outward = np.dot(f.v_before[i], f.out_normal[i])
            if f.line_dist[i] <= params.exit_dist and outward >= params.exit_out_speed:
                out["exit"][a:b, D] += params.exit_w / (b - a)
        if b - a > params.long_gap_s * f.fps:
            out["long_gap"][a:b, D] += params.long_gap_w / (b - a)

    out["still"][still, D] = params.still_w
    with np.errstate(invalid="ignore"):
        low = keep_long(f.median_speed < params.low_speed, int(round(params.low_s * f.fps)))
    out["low"][low, D] = params.low_w

    for code, a, b, _ in tags:
        if code == SET_PIECE_CODE:
            if params.setpiece_launch:
                c = launch_frame(f, still, params, a, b)
                if c is None:   # nothing to anchor on: D for the median launch offset, then silence
                    e = min(a + int(round(params.setpiece_fallback_s * f.fps)), b)
                    out["setpiece"][a:e, D] += params.setpiece_w / (e - a)
                    continue
            else:
                inside = t_frames[(t_frames >= a) & (t_frames < b)]
                if not len(inside):
                    continue
                c = inside[0]
            if c > a:
                out["setpiece"][a:c, D] += params.setpiece_w / (c - a)
            out["setpiece"][c:b, :2] += params.setpiece_w / (b - c)

    # After a goal the ball is dead until the kick-off (law of the game): D from the goal to the
    # restart candidate of the next kick-off-after-goal window, or to that window's start without
    # one. A GOALS tag opens at the start of the scoring sequence, which here begins with a set
    # piece and live play, so goal_after_setpiece starts the D at the end of the last other
    # set-piece window the tag overlaps (and skips a goal whose tag overlaps none).
    kickoffs = sorted((a, b) for code, a, b, _ in tags if code == KICKOFF_AFTER_GOAL_CODE)
    if params.goal_dead_w > 0:
        others = [(a, b) for code, a, b, _ in tags
                  if code == SET_PIECE_CODE and (a, b) not in kickoffs]
        for code, a, b, _ in tags:
            if code == GOAL_CODE:
                nxt = next(((ka, kb) for ka, kb in kickoffs if ka >= a), None)
                c = None if nxt is None else launch_frame(f, still, params, *nxt)
                end = b if nxt is None else (nxt[0] if c is None else c)
                if params.goal_after_setpiece:
                    a = max([wb for wa, wb in others if wa < end and wb > a], default=end)
                out["goal"][a:end, D] += params.goal_dead_w

    last = np.maximum.accumulate(np.where(control >= 0, np.arange(n), -1))
    latest = np.where(last >= 0, control[np.maximum(last, 0)], -1)
    _team_column(out["latch"], latest >= 0, latest, params.latch_w)

    if params.formation_w_vis > 0 or params.formation_w_blk > 0:
        # Only the disfavoured team pays, so the best alive score against D does not move.
        z = f.shape @ np.asarray(params.formation_coef)
        w = np.where(f.visible, params.formation_w_vis, params.formation_w_blk)
        out["formation"][:, A] -= w * np.maximum(z, 0.0)
        out["formation"][:, H] -= w * np.maximum(-z, 0.0)
    return out


def evidence(f, tags, params):
    """(n,3) float64 score in nats for H, A, D: the sum of every channel."""
    total = np.zeros((f.n, 3))
    for part in channels(f, tags, params).values():
        total += part
    return total
