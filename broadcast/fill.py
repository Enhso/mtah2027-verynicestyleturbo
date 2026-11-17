"""Map the broadcast ball onto the tactical timeline. Prints aggregates only.

    MTAH_DATA_DIR=... python broadcast/fill.py     # needs bc.npz (parse.py) and chunks.npz (align.py)

tactical frame = broadcast frame + offset, offset piecewise constant per continuous stretch.
A broadcast frame is accepted when its same-team player fit to the mapped tactical frame is under
FIT metres; this rejects replays and close-ups. Writes $MTAH_DATA_DIR/cache/bc_ball.npz:
ball (N, 2) metres on the tactical timeline, NaN where nothing is accepted.
"""
import os
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from broadcast import core  # noqa: E402

FIT = 2.5          # m, per-frame acceptance and chunk acceptance
MIN_PLAYERS = 4    # visible broadcast players needed for a fit
SPAN = 6           # candidate offsets per cluster: mode +- SPAN
MIN_MINUTE = 200   # accepted frames a one-minute window needs to give its own offset
WIN = 250          # frames (10 s) per stretch-detection window
OUT = os.path.join(core.CACHE, "bc_ball.npz")


def per_frame_fit(bpm, bt, tp, tt, offsets, batch=4000):
    """(Nb, len(offsets)) mean same-team nearest-neighbour distance (m); NaN if unusable."""
    nb, nt = len(bpm), len(tp)
    out = np.full((nb, len(offsets)), np.nan, np.float32)
    same = bt[:, None] == tt[None, :]                                  # (K, L)
    for s in range(0, nb, batch):
        b = np.arange(s, min(s + batch, nb))
        bp = bpm[b]
        vis = np.isfinite(bp[:, :, 0])
        use = vis.sum(1) >= MIN_PLAYERS
        bp0 = np.where(vis[..., None], bp, 0)
        for j, o in enumerate(offsets):
            T = b + o
            ok = use & (T >= 0) & (T < nt)
            tpf = tp[np.clip(T, 0, nt - 1)]                           # (n, L, 2)
            tvis = np.isfinite(tpf[..., 0])
            tpf = np.where(tvis[..., None], tpf, 1e6)
            d2 = ((bp0[:, :, None, :] - tpf[:, None, :, :]) ** 2).sum(-1)   # (n, K, L)
            d2 = np.where(same[None], d2, 1e12)
            d = np.sqrt(d2.min(-1))                                    # (n, K)
            with np.errstate(all="ignore"):
                f = np.nanmean(np.where(vis & (d < 1e5), d, np.nan), axis=1)
            out[b, j] = np.where(ok, f, np.nan)
    return out


def clusters(tab):
    """Coarse offsets of the well-fitting chunks, grouped by gaps over 20 frames; returns modes."""
    g = np.sort(tab[np.isfinite(tab[:, 2]) & (tab[:, 2] < FIT), 1].astype(int))
    cut = np.flatnonzero(np.diff(g) > 20) + 1
    return [int(np.median(c)) for c in np.split(g, cut)]


def _best(G):
    """Index of the offset column with the lowest median fit (the fit is flat near the optimum in
    accepted-frame count, so the median is the sharper statistic)."""
    with np.errstate(all="ignore"):
        m = np.nanmedian(G, 0)
    return int(np.argmin(np.where(np.isfinite(m), m, np.inf)))


def _null(bpm, bt, tp, tt, start, off):
    """Share of 3000 broadcast frames from `start` accepted at tactical = frame + off."""
    f = per_frame_fit(bpm[start:start + 3000], bt, tp, tt, np.array([off + start]))[:, 0]
    return float(np.mean(f < FIT))


def stretches(F, modes, nb):
    """Window labels -> [(start, end, cluster)] broadcast-frame stretches, split at gap midpoints."""
    nw = nb // WIN
    med = np.full((len(modes), nw), np.inf)
    for c in range(len(modes)):
        G = F[:nw * WIN, c * (2 * SPAN + 1):(c + 1) * (2 * SPAN + 1)].reshape(nw, WIN, -1)
        with np.errstate(all="ignore"):
            m = np.nanmedian(G, 1)
        med[c] = np.where(np.isfinite(m), m, np.inf).min(1)
    lab = np.where(med.min(0) < FIT, med.argmin(0), -1)
    idx = np.flatnonzero(lab >= 0)
    runs = np.split(idx, np.flatnonzero(np.diff(lab[idx]) != 0) + 1)
    out = []
    for r in runs:
        out.append([r[0] * WIN, (r[-1] + 1) * WIN, int(lab[r[0]])])
    out[0][0], out[-1][1] = 0, nb
    for a, b in zip(out[:-1], out[1:]):
        a[1] = b[0] = (a[1] + b[0]) // 2
    return out


def main():
    tp, tt, tball, periods = core.load_tactical()
    bp, bt, bball = core.load_broadcast()
    bpm = core.to_metres(bp, "id")
    bballm = core.to_metres(bball, "id")
    nb, nt = len(bpm), len(tp)
    tab = np.load(os.path.join(core.CACHE, "chunks.npz"))["tab"]
    modes = clusters(tab)
    offs = np.array([m + d for m in modes for d in range(-SPAN, SPAN + 1)])
    cache = os.path.join(core.CACHE, "bc_fit.npz")
    if os.path.exists(cache) and np.array_equal(np.load(cache)["offs"], offs):
        F = np.load(cache)["F"]
    else:
        F = per_frame_fit(bpm, bt, tp, tt, offs)
        np.savez(cache, offs=offs, F=F)
    print("offset clusters (frames)", modes)

    null = []
    ok_frame = np.zeros(nb, bool)
    off_of = np.zeros(nb, np.int64)
    n = 2 * SPAN + 1
    print("stretches: broadcast frames [start,end) -> offset, tactical frames, minutes with fit,"
          " per-minute offset (min..max, share within +-1 of stretch offset)")
    for a, b, c in stretches(F, modes, nb):
        G = F[a:b, c * n:(c + 1) * n]
        k = c * n + _best(G)
        o = int(offs[k])
        mins = []
        for m in range(a, b - MIN_MINUTE + 1, 1500):
            g = F[m:m + 1500, c * n:(c + 1) * n]
            if (g.min(1) < FIT).sum() >= MIN_MINUTE:
                mins.append(int(offs[c * n + _best(g)]))
        mins = np.array(mins)
        print(f"  [{a},{b}) -> {o}; tactical [{a + o},{b + o}); minutes {len(mins)}; "
              f"per-minute {mins.min()}..{mins.max()} (range {mins.max() - mins.min()}), "
              f"within +-1: {np.mean(abs(mins - o) <= 1):.2f}, within +-2: {np.mean(abs(mins - o) <= 2):.2f}")
        ok_frame[a:b] = F[a:b, k] < FIT
        off_of[a:b] = o
        null += [_null(bpm, bt, tp, tt, x, o + 500) for x in range(a, b - 3000, 20000)]
    print(f"null control: share of frames accepted at offset +500 (wrong): {np.nanmean(null):.4f} "
          f"vs {ok_frame.mean():.3f} at the stretch offsets")

    T = np.arange(nb) + off_of
    keep = ok_frame & (T < nt) & np.isfinite(bballm[:, 0])
    ball = np.full((nt, 2), np.nan, np.float32)
    ball[T[keep]] = bballm[keep]
    frame_ok = np.zeros(nt, bool)
    frame_ok[T[ok_frame & (T < nt)]] = True
    np.savez(OUT, ball=ball, frame_ok=frame_ok)
    seen = np.zeros(nt, bool)       # broadcast ball present at the mapped frame, acceptance ignored
    ok = (T < nt) & np.isfinite(bballm[:, 0])
    seen[T[ok]] = True
    report(tball, ball, frame_ok, seen, periods)


def report(tball, ball, frame_ok, seen, periods, fps=25):
    tv, bv = np.isfinite(tball[:, 0]), np.isfinite(ball[:, 0])
    for h, (a, b) in enumerate(periods, 1):
        print(f"half {h}: tactical frames [{a},{b}) = {b - a}")
        print(f"  graded frames with an accepted broadcast frame {frame_ok[a:b].mean():.3f}; "
              f"with an accepted broadcast ball {bv[a:b].mean():.3f}; tactical ball {tv[a:b].mean():.3f}")
        d = np.diff(np.concatenate([[0], (~tv[a:b]).astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        ln = ends - starts
        allb = ~tv[a:b]
        for name, sel in (("<5 s", ln < 5 * fps), ("5-30 s", (ln >= 5 * fps) & (ln <= 30 * fps)),
                          (">30 s", ln > 30 * fps)):
            cov = [bv[a + s:a + e].sum() for s, e in zip(starts[sel], ends[sel])]
            tot = int(ln[sel].sum())
            print(f"  blackouts {name}: n={int(sel.sum())}, frames={tot} ({tot / (b - a):.3f} of half), "
                  f"covered frames {sum(cov) / max(tot, 1):.3f}, blackouts with any coverage {int(np.sum(np.array(cov) > 0))}")
        print(f"  in blackout frames: accepted broadcast frame {frame_ok[a:b][allb].mean():.3f}, "
              f"broadcast ball present ignoring acceptance {seen[a:b][allb].mean():.3f}")
        print(f"  all blackout frames covered {bv[a:b][allb].mean():.3f} (blackout {allb.mean():.3f} of half)")
        both = tv[a:b] & bv[a:b]
        dist = np.linalg.norm(tball[a:b][both] - ball[a:b][both], axis=1)
        print(f"  agreement where both see the ball: n={int(both.sum())}, median {np.median(dist):.2f} m, "
              f"within 2 m {np.mean(dist < 2):.3f}, within 5 m {np.mean(dist < 5):.3f}")


if __name__ == "__main__":
    main()
