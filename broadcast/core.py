"""Shared pieces of the broadcast probe: loading both tracks and the offset score."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import match as mt  # noqa: E402

CACHE = mt.CACHE_DIR
SCR = CACHE
PITCH = np.array([105.3, 68.0], np.float32)
FLIPS = {"id": (1, 1), "fx": (-1, 1), "fy": (1, -1), "fxy": (-1, -1)}


def load_tactical():
    m = mt.load("BAR-ATH", CACHE)
    return m.players, m.team, m.ball, m.periods


def load_broadcast():
    z = np.load(os.path.join(SCR, "bc.npz"))
    return z["players"], z["team"], z["ball"]


def to_metres(norm, flip="id"):
    """Normalised (…,2) -> metres, with an optional axis flip applied in normalised units."""
    s = np.array(FLIPS[flip], np.float32)
    o = np.where(s < 0, 1.0, 0.0).astype(np.float32)
    return (norm * s + o) * PITCH


def score(bc_p, bc_t, tac_p, tac_t, frames, offsets, same_team=True, min_players=4):
    """Mean over `frames` of the mean nearest-neighbour distance (m) from visible broadcast
    players to tactical players, at each offset (tactical frame = broadcast frame + offset).

    bc_p (Nb,K,2) and tac_p (Nt,L,2) are in metres. Frames with fewer than `min_players` visible
    broadcast players are skipped. Returns (len(offsets),) float32; NaN if nothing was usable.
    Offsets that push a frame outside the tactical range count that frame as unusable.
    """
    bp = bc_p[frames]                                   # (n,K,2)
    bvis = np.isfinite(bp[:, :, 0])
    use = bvis.sum(1) >= min_players
    bp, bvis, fr = bp[use], bvis[use], frames[use]
    n = len(fr)
    out = np.full(len(offsets), np.nan, np.float32)
    if n == 0:
        return out
    bp = np.where(bvis[..., None], bp, 0).astype(np.float32)
    nt = len(tac_p)
    batch = max(1, int(3e7 // (n * bp.shape[1] * tac_p.shape[1])))
    for s in range(0, len(offsets), batch):
        off = offsets[s:s + batch]
        T = fr[None, :] + off[:, None]                  # (k,n)
        ok = (T >= 0) & (T < nt)
        Tc = np.clip(T, 0, nt - 1)
        tp = tac_p[Tc]                                  # (k,n,L,2)
        tvis = np.isfinite(tp[..., 0])
        d2 = ((bp[None, :, :, None, :] - np.where(tvis[..., None], tp, 1e6)[:, :, None, :, :]) ** 2).sum(-1)
        if same_team:
            d2 = np.where(bc_t[None, None, :, None] == tac_t[None, None, None, :], d2, 1e12)
        d = np.sqrt(d2.min(-1))                         # (k,n,K)
        d = np.where(bvis[None] & (d < 1e5), d, np.nan)
        per_frame = np.nanmean(d, axis=2)               # (k,n)
        per_frame = np.where(ok, per_frame, np.nan)
        with np.errstate(all="ignore"):
            out[s:s + batch] = np.nanmean(per_frame, axis=1)
    return out
