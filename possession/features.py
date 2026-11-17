"""Per-frame geometry and kinematics for one period of a match.

Every function takes arrays for one decoded period, so nothing here can look across halftime. The
ball is interpolated over gaps of `interp_max_gap` frames or fewer. Every ball velocity is taken
inside one visible run: a window that touches a longer blackout gives NaN, never a number.
"""

import warnings
from dataclasses import dataclass

import numpy as np

HOME, AWAY = 0, 1


def runs(mask):
    """(starts, stops) of the maximal True runs of a bool vector, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def keep_long(mask, min_len):
    """mask with every True run shorter than min_len frames cleared."""
    out = np.zeros_like(mask)
    for a, b in zip(*runs(mask)):
        if b - a >= min_len:
            out[a:b] = True
    return out


def _shift(a, k, fill):
    """out[t] = a[t + k], `fill` where t + k falls outside."""
    out = np.full_like(a, fill)
    n = len(a)
    if k >= 0:
        out[:n - k] = a[k:]
    else:
        out[-k:] = a[:n + k]
    return out


def fill_short_gaps(ball, max_gap):
    """Linear interpolation of NaN gaps of at most max_gap frames that have a visible frame on both sides."""
    out = ball.copy()
    visible = ~np.isnan(ball).any(axis=1)
    for a, b in zip(*runs(~visible)):
        if 0 < a and b < len(ball) and b - a <= max_gap:
            w = (np.arange(a, b) - (a - 1)) / float(b - (a - 1))
            out[a:b] = ball[a - 1] + w[:, None] * (ball[b] - ball[a - 1])
    return out


def run_ids(visible):
    """Id of the visible run each frame belongs to; -1 on frames without a ball."""
    starts = visible & ~np.concatenate(([False], visible[:-1]))
    return np.where(visible, np.cumsum(starts) - 1, -1)


def centred_velocity(pos, run_id, half, fps):
    """(pos[t+half] - pos[t-half]) * fps / (2 half) where both ends lie in one visible run, else NaN."""
    a, b = _shift(run_id, -half, -1), _shift(run_id, half, -2)
    ok = (a >= 0) & (a == b)
    v = (_shift(pos, half, np.nan) - _shift(pos, -half, np.nan)) * (fps / (2.0 * half))
    v[~ok] = np.nan
    return v


def span_velocity(pos, run_id, k, fps):
    """Velocity between frame t and frame t+k, forward for k > 0 and backward for k < 0, inside one
    visible run only (NaN when t and t+k lie in different runs or in a blackout)."""
    other = _shift(run_id, k, -2)
    ok = (run_id >= 0) & (run_id == other)
    d = _shift(pos, k, np.nan) - pos
    v = d * (fps / float(k))
    v[~ok] = np.nan
    return v


@dataclass
class Features:
    n: int
    fps: int
    visible: np.ndarray          # (n,) ball present after interpolation
    ball: np.ndarray             # (n,2) interpolated ball, NaN where not visible
    run_id: np.ndarray           # (n,)
    gaps: list                   # [(start, stop)] blackouts after interpolation
    v_before: np.ndarray         # (n,2) ball velocity over the touch_span frames ending at t
    v_after: np.ndarray          # (n,2) ball velocity over the touch_span frames starting at t
    ball_v: np.ndarray           # (n,2) centred ball velocity
    still_speed: np.ndarray      # (n,) centred ball speed over a longer span
    line_dist: np.ndarray        # (n,) metres to the nearest line, negative outside
    out_normal: np.ndarray       # (n,2) unit outward normal of that line
    offpitch: np.ndarray         # (n,) bool
    near_team: np.ndarray        # (n,) int8 team of the nearest player, -1 if none or tied
    near_dist: np.ndarray        # (n,) distance to that player
    opp_dist: np.ndarray         # (n,) distance to the nearest player of the other team
    near_v: np.ndarray           # (n,2) that player's centred velocity
    median_speed: np.ndarray     # (n,) median ATD speed over present slots
    shape: np.ndarray            # (n,len(SHAPE_NAMES)) home-minus-away team shape, see team_shape


SHAPE_NAMES = ("advance", "forward_speed", "length", "width", "in_opp_half")


def team_shape(players, pv, team, pitch):
    """(n,5) home minus away of per-team shape features, each in that team's own attack frame.

    The attack direction is label-free, per period: home attacks towards the away team's side
    (sign of the median over frames of away minus home mean x). Per team: mean x, mean forward
    velocity, std x, std y, share of players in the opponent half. Coordinates are centred and the
    sign is applied before any reduction, so reflecting both axes gives bit-identical features and
    swapping the teams negates them exactly. Rows where a team has fewer than 2 players are 0.
    """
    length, width = pitch
    xc, yc = players[:, :, 0] - length / 2.0, players[:, :, 1] - width / 2.0
    present = np.isfinite(xc)
    cols = [np.flatnonzero(team == side) for side in (HOME, AWAY)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.sign(np.nanmedian(np.nanmean(xc[:, cols[1]], axis=1)
                                 - np.nanmean(xc[:, cols[0]], axis=1)))
        s = 0.0 if not np.isfinite(s) else float(s)
        per = []
        for side, c in zip((HOME, AWAY), cols):
            d = s if side == HOME else -s
            xa = d * xc[:, c]
            per.append(np.stack([
                np.nanmean(xa, axis=1), np.nanmean(d * pv[:, c, 0], axis=1),
                np.nanstd(xa, axis=1), np.nanstd(yc[:, c], axis=1),
                (xa > 0).sum(axis=1) / present[:, c].sum(axis=1)], axis=1))
    out = np.nan_to_num(per[0] - per[1])
    out[(present[:, cols[0]].sum(axis=1) < 2) | (present[:, cols[1]].sum(axis=1) < 2)] = 0.0
    return out


def compute(ball, players, speed, team, pitch, fps, params):
    """Features of one period. ball (n,2), players (n,K,2), speed (n,K) in metres, any float dtype."""
    ball = np.asarray(ball, dtype=np.float64)
    players = np.asarray(players, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    team = np.asarray(team)
    n = len(ball)
    length, width = pitch

    ball = fill_short_gaps(ball, params.interp_max_gap)
    visible = ~np.isnan(ball).any(axis=1)
    rid = run_ids(visible)
    gaps = list(zip(*[a.tolist() for a in runs(~visible)]))

    s = params.touch_span
    v_before = span_velocity(ball, rid, -s, fps)
    v_after = span_velocity(ball, rid, s, fps)
    ball_v = centred_velocity(ball, rid, params.vel_half, fps)
    still = np.linalg.norm(centred_velocity(ball, rid, params.still_half, fps), axis=1)

    x, y = ball[:, 0], ball[:, 1]
    margins = np.stack([x, length - x, y, width - y], axis=1)
    side = np.argmin(margins, axis=1)
    line_dist = margins[np.arange(n), side]
    normals = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    offpitch = visible & (line_dist < -params.offpitch_margin)

    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    best = []
    for side_id in (HOME, AWAY):
        cols = np.flatnonzero(team == side_id)
        sub = d[:, cols]
        k = np.argmin(sub, axis=1) if len(cols) else np.zeros(n, dtype=int)
        dist = sub[np.arange(n), k] if len(cols) else np.full(n, np.inf)
        best.append((dist, cols[k] if len(cols) else np.full(n, -1)))
    (d_home, i_home), (d_away, i_away) = best
    home_nearer, away_nearer = d_home < d_away, d_away < d_home
    near_team = np.where(home_nearer, HOME, np.where(away_nearer, AWAY, -1)).astype(np.int8)
    near_team[~np.isfinite(np.minimum(d_home, d_away))] = -1
    near_dist = np.where(home_nearer, d_home, np.where(away_nearer, d_away, np.nan))
    opp_dist = np.where(home_nearer, d_away, np.where(away_nearer, d_home, np.nan))
    opp_dist[~np.isfinite(opp_dist)] = np.inf   # no opponent tracked: nothing contests the ball
    near_idx = np.where(home_nearer, i_home, np.where(away_nearer, i_away, -1))

    # Per-slot centred velocity with a jump guard: a slot that changes person teleports.
    rows = np.arange(n)
    pv = np.full_like(players, np.nan)
    h = params.vel_half
    if n > 2 * h:
        pv[h:n - h] = (players[2 * h:] - players[:n - 2 * h]) * (fps / (2.0 * h))
    pv[np.linalg.norm(pv, axis=2) > params.player_vmax] = np.nan
    near_v = pv[rows, np.maximum(near_idx, 0)]
    near_v[near_idx < 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows give NaN, which is intended
        median_speed = np.nanmedian(speed, axis=1)

    return Features(
        n=n, fps=fps, visible=visible, ball=ball, run_id=rid, gaps=gaps, v_before=v_before,
        v_after=v_after, ball_v=ball_v, still_speed=still, line_dist=line_dist,
        out_normal=normals[side], offpitch=offpitch, near_team=near_team, near_dist=near_dist,
        opp_dist=opp_dist, near_v=near_v, median_speed=median_speed,
        shape=team_shape(players, pv, team, pitch))
