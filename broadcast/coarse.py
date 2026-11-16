"""Coarse global offset search for one broadcast chunk. Prints the best offsets and score quantiles.

    python broadcast/coarse.py <chunk_start_frame> [flip] [same_team 0/1]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from broadcast import core  # noqa: E402

start = int(sys.argv[1])
flip = sys.argv[2] if len(sys.argv) > 2 else "id"
same = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"

tp, tt, _, _ = core.load_tactical()
bp, bt, _ = core.load_broadcast()
tpm = tp  # cache is already in metres
bpm = core.to_metres(bp, flip)
frames = np.arange(start, start + 1500, 60)
offsets = np.arange(-start, len(tp) - start, 6)
s = core.score(bpm, bt, tpm, tt, frames, offsets, same_team=same)
ok = np.isfinite(s)
order = np.argsort(np.where(ok, s, np.inf))[:5]
print("chunk", start, "flip", flip, "same_team", same, "usable offsets", int(ok.sum()))
print("score quantiles (m)", np.round(np.nanquantile(s, [0, .01, .1, .5]), 2))
print("best 5 offsets", offsets[order].tolist(), "scores", np.round(s[order], 2).tolist())
