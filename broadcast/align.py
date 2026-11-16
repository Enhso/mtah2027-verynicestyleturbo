"""Per one-minute-chunk offset search over the whole broadcast. Saves a chunk table; prints counts.

    python broadcast/align.py        # writes SCR/chunks.npz

tactical frame = broadcast frame + offset. Orientation: identity (checked by coarse.py).
"""
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from broadcast import core  # noqa: E402

CHUNK = 1500
LO, HI = 0, 12000          # coarse window around the ~4010 offset seen on probe chunks
tp, tt, _, _ = core.load_tactical()
bp, bt, _ = core.load_broadcast()
bpm = core.to_metres(bp, "id")
nb = len(bpm)
starts = np.arange(0, nb - CHUNK // 2, CHUNK)


def one(start):
    fr = np.arange(start, min(start + CHUNK, nb), 50)
    off = np.arange(LO, HI, 3)
    s = core.score(bpm, bt, tp, tt, fr, off)
    if not np.isfinite(s).any():
        return start, -1, np.nan, np.nan
    i = int(np.nanargmin(s))
    fine = np.arange(off[i] - 4, off[i] + 5)
    fr2 = np.arange(start, min(start + CHUNK, nb), 15)
    s2 = core.score(bpm, bt, tp, tt, fr2, fine)
    j = int(np.nanargmin(s2))
    # runner-up: best coarse score farther than 60 frames from the winner
    far = np.abs(off - off[i]) > 60
    return start, int(fine[j]), float(s2[j]), float(np.nanmin(np.where(far, s, np.nan))) if np.isfinite(s[far]).any() else np.nan


if __name__ == "__main__":
    with Pool(3) as p:
        rows = p.map(one, starts, chunksize=1)
    tab = np.array(rows, dtype=np.float64)
    np.savez(core.SCR + "/chunks.npz", tab=tab)
    good = np.isfinite(tab[:, 2]) & (tab[:, 2] < 2.5)
    print("chunks", len(tab), "with fit under 2.5 m", int(good.sum()))
    print("offset quantiles (good)", np.quantile(tab[good, 1], [0, .1, .5, .9, 1]))
    print("score quantiles (good)", np.round(np.quantile(tab[good, 2], [0, .5, .9, 1]), 2))
