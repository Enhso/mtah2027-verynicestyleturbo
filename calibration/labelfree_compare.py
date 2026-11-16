"""Label-free statistics of this match next to the external matches (aggregates only).

Decides which proxy variant is closest to this match and feeds the quantile transfer.
"""
import sys
import numpy as np
from calibration import common as c
from possession import features
from possession.match import D, X
from possession.params import Params

Q = [5, 25, 50, 75, 95]


def q(v, qs=Q):
    v = v[np.isfinite(v)]
    return " ".join("{:.2f}".format(x) for x in np.percentile(v, qs)) if len(v) else "n/a"


def stats(m):
    p = Params()
    out = {}
    fs = [features.compute(m.ball[a:b], m.players[a:b], m.speed[a:b], m.team, m.pitch, m.fps, p)
          for a, b in m.periods.tolist()]
    cat = lambda name: np.concatenate([getattr(f, name) for f in fs])
    vis = cat("visible")
    n = len(vis)
    gaps = np.array([b - a for f in fs for a, b in f.gaps])
    vruns = np.concatenate([np.subtract(*features.runs(f.visible)[::-1]) for f in fs])
    out["graded min"] = "{:.1f}".format(n / 25 / 60)
    out["ball missing share"] = "{:.3f}".format(1 - vis.mean())
    out["gaps (n, per 90 graded min)"] = "{} {:.0f}".format(len(gaps), len(gaps) / (n / 25 / 60) * 90)
    out["gap s q5/25/50/75/95"] = q(gaps / 25.0)
    out["gaps > 30 s, share of frames"] = "{} {:.3f}".format(int((gaps > 750).sum()), gaps[gaps > 750].sum() / n)
    out["visible run s"] = q(vruns / 25.0)
    present = np.concatenate([np.isfinite(m.players[a:b, :, 0]) for a, b in m.periods.tolist()])
    for side, nm in ((0, "home"), (1, "away")):
        out["present " + nm + " slots q5/25/50/75/95"] = q(present[:, m.team == side].sum(axis=1).astype(float))
    out["nearest dist m (visible)"] = q(cat("near_dist")[vis])
    opp = cat("opp_dist") - cat("near_dist")
    out["opp margin m (visible)"] = q(opp[vis & np.isfinite(opp)])
    out["ball speed m/s (visible)"] = q(np.linalg.norm(cat("ball_v"), axis=1))
    out["touch change m/s q50/90/95/99/99.9"] = q(np.linalg.norm(cat("v_after") - cat("v_before"), axis=1), [50, 90, 95, 99, 99.9])
    out["median player speed (visible)"] = q(cat("median_speed")[vis])
    out["median player speed (missing)"] = q(cat("median_speed")[~vis])
    out["line dist m (visible)"] = q(cat("line_dist")[vis])
    out["off pitch share of visible"] = "{:.4f}".format(cat("offpitch")[vis].mean())
    nt = cat("near_team")
    flips = 0
    for f in fs:
        t = f.near_team[f.visible & (f.near_team >= 0)]
        flips += int((t[1:] != t[:-1]).sum())
    out["nearest-team flips per visible min"] = "{:.1f}".format(flips / (vis.sum() / 25 / 60))
    # player jitter: second difference of position, metres per frame^2, all present slots
    acc = []
    for a, b in m.periods.tolist():
        pl = m.players[a:b].astype(np.float64)
        d2 = np.linalg.norm(pl[2:] - 2 * pl[1:-1] + pl[:-2], axis=2)
        acc.append(d2[np.isfinite(d2)][::7])
    out["player 2nd diff m q50/90/95/99/99.9"] = q(np.concatenate(acc), [50, 90, 95, 99, 99.9])
    if m.labels is not None:
        lab = np.concatenate([m.labels[a:b] for a, b in m.periods.tolist()])
        alive = lab != D
        ok = vis & alive & (nt >= 0)
        out["[labels] dead share"] = "{:.3f}".format(1 - alive.mean())
        out["[labels] P(dead|missing), P(missing|alive), P(dead|visible)"] = "{:.3f} {:.3f} {:.3f}".format(
            (~alive)[~vis].mean() if (~vis).any() else float("nan"), (~vis)[alive].mean(), (~alive)[vis].mean())
        out["[labels] nearest team right on alive visible"] = "{:.3f}".format((nt[ok] == lab[ok]).mean())
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["BAR-ATH"] + c.METRICA + c.DFL[:2]
    rows = [stats(c.load_match(nm)) for nm in names]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    for k in keys:
        print(k)
        for nm, r in zip(names, rows):
            print("    {:<14} {}".format(nm, r.get(k, "-")))
