"""A synthetic match built from a known label sequence, on a dyadic pitch.

Coordinates sit on a 1/8 m grid and the pitch is 104 x 68 m, so reflecting both axes is exact in
float32 and float64. The away team is the point mirror of the home team, so team shape alone favours
neither. The team on the ball pushes SHIFT m up and spreads WIDEN wider, the other drops back and
narrows (ramped over 1 s), so home players stay in x in [1, 51], away players in x in [53, 103], and
the two teams are always at least 8 m apart in x: the holder's team is the only one near the ball. An alive segment is a chain of dribbles and
passes ending in a flight to the next segment's first holder (a turnover) or to a spot outside the
touchline (a stoppage). Every synthetic channel input is a stand-in for what the real channels read.
"""

import numpy as np

from possession import match as mt

PITCH = (104.0, 68.0)
FPS = 25
K = 22
HOME, AWAY = 0, 1
H, A, D = 0, 1, 2
MIN_ALIVE, MIN_DEAD = 300, 150      # frames; 12 s and 6 s
SHIFT, WIDEN = 3.0, 0.1             # m forward and relative width change of the team on the ball


def _grid(a):
    return np.round(a * 8.0) / 8.0


class _Players:
    def __init__(self, rng):
        self.team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
        hx, hy = rng.uniform(8, 44, 11), rng.uniform(8, 60, 11)
        self.base = np.stack([np.concatenate([hx, PITCH[0] - hx]),
                              np.concatenate([hy, PITCH[1] - hy])], axis=1)
        self.amp = np.tile(rng.uniform(1.0, 4.0, 11), 2)
        self.period = np.tile(rng.uniform(300, 600, 11), 2)
        self.phase = np.tile(rng.uniform(0, 2 * np.pi, (11, 2)), (2, 1))
        self.sign = np.where(self.team == HOME, 1.0, -1.0)   # home attacks +x; away is mirrored
        self.lead = None    # (2, n) per team: +1 on the ball, -1 defending, 0 dead, ramped

    def follow(self, truth):
        """Make each team's shape follow possession, given the frame labels."""
        ramp = np.ones(FPS) / FPS
        own = [np.convolve((truth == side).astype(float), ramp, mode="same") for side in (H, A)]
        self.lead = np.stack([own[0] - own[1], own[1] - own[0]])

    def at(self, i, t):
        """Position of player i at frame(s) t, shape (len(t), 2)."""
        t = np.asarray(t)
        w = 2 * np.pi * t.astype(float) / self.period[i]
        pos = self.base[i] + self.sign[i] * self.amp[i] * np.stack(
            [np.sin(w + self.phase[i, 0]), np.cos(w + self.phase[i, 1])], axis=-1)
        if self.lead is not None:
            lead = self.lead[self.team[i]][t]
            pos = pos + np.stack([SHIFT * lead * self.sign[i],
                                  WIDEN * lead * (pos[..., 1] - PITCH[1] / 2)], axis=-1)
        return pos

    def all_at(self, t):
        return np.stack([self.at(i, t) for i in range(K)], axis=1)


def _holder(rng, players, label):
    return int(rng.choice(np.flatnonzero(players.team == label)))


def make_match(segments, dead_visible, seed, tags=None, name="SYNTH"):
    """segments: [(label, frames)] with label H/A/D, no two neighbours equal, alive >= MIN_ALIVE
    and D >= MIN_DEAD frames. Returns (Match, truth codes). The graded period is the whole array."""
    rng = np.random.default_rng(seed)
    players = _Players(rng)
    n = sum(f for _, f in segments)
    ball = np.full((n, 2), np.nan)
    truth = np.repeat([lab for lab, _ in segments], [f for _, f in segments]).astype(np.int8)
    players.follow(truth)
    offset = np.array([0.5, 0.0])

    starts = np.cumsum([0] + [f for _, f in segments])
    first_holder = [(_holder(rng, players, lab) if lab != D else -1) for lab, _ in segments]
    # The ball goes out in the last holder's half, so its flight never passes the other team.
    spot_x = {i: float(rng.uniform(20, 80) if i == 0 else
                       rng.uniform(8, 44) if segments[i - 1][0] == H else rng.uniform(60, 96))
              for i, (lab, _) in enumerate(segments) if lab == D}

    for i, (lab, frames) in enumerate(segments):
        t0, t1 = int(starts[i]), int(starts[i + 1])
        truth[t0:t1] = lab
        if lab == D:
            if dead_visible:
                ball[t0:t1] = (spot_x[i], -2.0)
            continue

        # Where the final flight goes: next segment's first holder or the stoppage spot.
        if i + 1 < len(segments):
            nxt_lab = segments[i + 1][0]
            if nxt_lab == D:
                target = lambda t, x=spot_x[i + 1]: np.array([x, -2.0])
            else:
                target = lambda t, h=first_holder[i + 1]: players.at(h, [t])[0] + offset
        else:
            target = None
        t = t0
        holder = first_holder[i]
        end_flight = 0
        if target is not None:
            far = float(np.linalg.norm(target(t1) - (players.at(holder, [t1 - 1])[0] + offset)))
            end_flight = int(max(20, np.ceil(far / 0.8)))
        t_end_hold = t1 - end_flight
        while t < t_end_hold:
            hold = int(rng.integers(50, 91))
            stop = min(t + hold, t_end_hold)
            tt = np.arange(t, stop)
            ball[tt] = players.at(holder, tt) + offset
            t = stop
            if t >= t_end_hold:
                break
            mates = [p for p in np.flatnonzero(players.team == lab) if p != holder]
            near = [p for p in mates if np.linalg.norm(
                players.at(p, [t])[0] - players.at(holder, [t])[0]) <= 30.0]
            nxt = int(rng.choice(near)) if near else holder
            flight = 20
            if t + flight >= t_end_hold or nxt == holder:
                continue
            a = players.at(holder, [t])[0] + offset
            b = players.at(nxt, [t + flight])[0] + offset
            w = (np.arange(1, flight + 1) / float(flight))[:, None]
            ball[t:t + flight] = a + w * (b - a)
            t, holder = t + flight, nxt
        if target is not None and end_flight:
            a = ball[t_end_hold - 1] if t_end_hold > t0 else players.at(holder, [t0])[0] + offset
            b = target(t1)
            w = (np.arange(1, end_flight + 1) / float(end_flight))[:, None]
            ball[t_end_hold:t1] = a + w * (b - a)

    frames = np.arange(n)
    pos = players.all_at(frames) + rng.normal(0.0, 0.05, (n, K, 2))
    speed = np.where((truth == D)[:, None], rng.uniform(0.2, 1.0, (n, K)),
                     rng.uniform(2.0, 4.0, (n, K)))
    m = mt.Match(
        name=name, fps=FPS, pitch=PITCH, ball=_grid(ball).astype(np.float32),
        players=_grid(pos).astype(np.float32), speed=speed.astype(np.float32),
        team=players.team.copy(), periods=np.array([[0, n]], dtype=np.int64),
        tags=list(tags or []))
    return m, truth


def reflect(m):
    """The same match with both pitch axes reflected."""
    L, W = m.pitch

    def flip(a):
        return np.stack([L - a[..., 0], W - a[..., 1]], axis=-1).astype(np.float32)

    return mt.Match(m.name, m.fps, m.pitch, flip(m.ball), flip(m.players), m.speed, m.team,
                    m.periods, m.tags)


def swap_teams(m):
    """The same match with the two sides exchanged, in the slot teams and in the Team tags."""
    names = {"barcelona": "Atletic Club", "atletic club": "Barcelona"}
    tags = []
    for t in m.tags:
        t = dict(t, tags=dict(t["tags"]))
        team = (t["tags"].get("Team") or "").strip().lower()
        if team in names:
            t["tags"]["Team"] = names[team]
        tags.append(t)
    return mt.Match(m.name, m.fps, m.pitch, m.ball, m.players, m.speed, 1 - m.team, m.periods, tags)


def tag(code, start, stop, team=None, kind=None):
    labels = {}
    if team:
        labels["Team"] = team
    if kind:
        labels["Type"] = kind
    return {"code": code, "start_s": start / FPS, "end_s": stop / FPS, "start_frame": start,
            "end_frame": stop, "tags": labels}
