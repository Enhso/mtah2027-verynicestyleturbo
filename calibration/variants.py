"""Proxy variants of an external match, built from label-free properties of this match.

  clean   the export as it is (DFL speed converted from km/h to m/s)
  atd     ball low-passed, shifted by a slow offset and jittered, and player slots dropped, so that
          three label-free statistics match this match: the 3-frame ball velocity change, the
          distance from the ball to the nearest player, and the number of tracked players per team
  +mask   this match's ball-visibility mask (per half) OR-ed onto the ball (harness.mask)

The knobs (SMOOTH, OFFSET, JITTER) were chosen by `python -m calibration.variants fit`, which prints
only quantiles. Nothing here reads a label of this match. Everything is seeded.
"""

import dataclasses
import sys
import zlib

import numpy as np

from calibration import common as c
from harness.mask import apply_mask

SMOOTH = 6.0      # frames, sigma of the Gaussian low-pass on the ball
OFFSET = 0.7      # m per axis, slow ball offset (correlation about 1 s)
JITTER = 0.02     # m per axis, white ball jitter
OFFSET_CORR = 25  # frames

_REF = {}


def reference():
    """This match's per-half ball-visibility masks and per-team tracked-player counts (in memory only)."""
    if not _REF:
        m = c.load_match("BAR-ATH")
        per = m.periods.tolist()
        _REF["mask"] = [np.isfinite(m.ball[a:b, 0]) for a, b in per]
        _REF["count"] = [[np.isfinite(m.players[a:b][:, m.team == s, 0]).sum(axis=1) for s in (0, 1)]
                         for a, b in per]
    return _REF


def _gauss(sigma):
    r = int(np.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def _smooth_nan(x, sigma):
    """Gaussian low-pass of the columns of x that ignores NaN and keeps NaN where x was NaN."""
    if sigma <= 0:
        return x
    k = _gauss(sigma)
    ok = np.isfinite(x[:, 0])
    den = np.convolve(ok.astype(float), k, mode="same")
    out = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        num = np.convolve(np.where(ok, x[:, j], 0.0), k, mode="same")
        out[ok, j] = num[ok] / den[ok]
    return out


def atd_like(match, smooth=None, offset=None, jitter=None, dropout=True):
    smooth = SMOOTH if smooth is None else smooth
    offset = OFFSET if offset is None else offset
    jitter = JITTER if jitter is None else jitter
    rng = np.random.default_rng(zlib.crc32(match.name.encode()))
    ball = match.ball.astype(np.float64).copy()
    players = match.players.copy()
    speed = match.speed.copy()
    ref = reference()
    for p, (a, b) in enumerate(match.periods.tolist()):
        n = b - a
        slow = np.stack([np.convolve(rng.standard_normal(n), _gauss(OFFSET_CORR), mode="same")
                         for _ in range(2)], axis=1)
        slow *= offset / max(slow.std(), 1e-9)
        seg = _smooth_nan(ball[a:b] + slow, smooth) + jitter * rng.standard_normal((n, 2))
        ball[a:b] = seg
        if dropout:
            for side in (0, 1):
                cols = np.flatnonzero(match.team == side)
                target = np.resize(ref["count"][p % len(ref["count"])][side], n)
                cuts = np.flatnonzero(np.diff(target)) + 1
                for s, e in zip(np.concatenate(([0], cuts)), np.concatenate((cuts, [n]))):
                    here = cols[np.isfinite(players[a + s, cols, 0])]
                    extra = len(here) - int(target[s])
                    if extra > 0:
                        drop = rng.choice(here, extra, replace=False)
                        players[a + s:a + e, drop] = np.nan
                        speed[a + s:a + e, drop] = np.nan
    return dataclasses.replace(match, ball=ball.astype(np.float32), players=players, speed=speed)


def build(name, variant):
    """variant: 'clean', 'atd', 'clean+mask' or 'atd+mask'."""
    m = c.load_match(name)
    if name.startswith("dfl_"):
        m = dataclasses.replace(m, speed=m.speed / 3.6)   # the DFL export carries km/h
    kind, _, mask = variant.partition("+")
    if kind == "atd":
        m = atd_like(m)
    if mask:
        m = apply_mask(m, reference()["mask"])
    return m


def register(names, variants):
    keys = []
    for nm in names:
        for v in variants:
            c.register((nm, v), build(nm, v))
            keys.append((nm, v))
    return keys


if __name__ == "__main__":
    from calibration.labelfree_compare import stats
    want = ["ball missing share", "nearest dist m (visible)", "touch change m/s q50/90/95/99/99.9",
            "nearest-team flips per visible min", "present home slots q5/25/50/75/95",
            "[labels] P(dead|missing), P(missing|alive), P(dead|visible)",
            "[labels] nearest team right on alive visible"]
    if sys.argv[1:2] == ["fit"]:
        ref = stats(c.load_match("BAR-ATH"))
        for k in want[:5]:
            print("BAR-ATH", k, ref[k])
        for nm in ("metrica_game1", "metrica_game3"):
            m = c.load_match(nm)
            for sm in (2.0, 3.0, 4.0, 6.0):
                for off in (0.5, 0.7, 1.0):
                    s = stats(atd_like(m, smooth=sm, offset=off))
                    print(nm, "smooth", sm, "offset", off, "| near", s[want[1]], "| touch", s[want[2]],
                          "| flips", s[want[3]], "| right", s[want[6]])
    else:
        for nm in sys.argv[2:] or c.METRICA:
            for v in sys.argv[1].split(","):
                s = stats(build(nm, v))
                print(nm, v)
                for k in want:
                    print("    ", k, s.get(k, "-"))
