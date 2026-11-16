"""How well each control definition witnesses the labelled owner, before any decoding.

Per definition, pooled over the Metrica games on one variant:
  coverage   share of alive visible frames (or count per possession, for events) with a signal
  precision  share of signals whose team is the labelled owner
  onset      for each true possession start: offset (frames) of the first signal of the new team
             in [start-12, start+75], and the share of starts that have one
"""
import dataclasses
import sys
import numpy as np
from calibration import common as c, variants
from possession import evidence
from possession.match import D
from possession.params import Params


def starts_of(lab):
    """[(frame, team)] where a possession starts inside one period's labels."""
    prev = np.concatenate(([D], lab[:-1]))
    idx = np.flatnonzero((lab < 2) & (prev != lab))
    return idx, lab[idx]


def study(keys, params, kind):
    cov_n = cov_d = prec_n = prec_d = 0
    offs, n_starts, n_poss = [], 0, 0
    for key in keys:
        m = c._MATCH[key]
        for (a, b), f in zip(m.periods.tolist(), c.feats(key, params)):
            lab = m.labels[a:b]
            alive = lab < 2
            if kind == "carry":
                team = evidence.carry_events(f, params)
            else:
                team = np.full(f.n, -1, np.int8)
                fr, tt = evidence.touch_events(f, params)
                team[fr] = tt
            sig = team >= 0
            cov_n += (sig & alive).sum()
            cov_d += (alive & f.visible).sum()
            prec_n += (sig & alive & (team == lab)).sum()
            prec_d += (sig & alive).sum()
            st, tm = starts_of(lab)
            n_poss += len(st)
            for s, t in zip(st, tm):
                lo, hi = max(s - 12, 0), min(s + 76, f.n)
                hit = np.flatnonzero(team[lo:hi] == t)
                n_starts += 1
                if len(hit):
                    offs.append(lo + hit[0] - s)
    offs = np.array(offs)
    cov = cov_n / cov_d if kind == "carry" else cov_n / n_poss
    return "cov {:.3f} prec {:.3f} | starts with signal {:.2f}, offset q10/25/50/75/90 {}".format(
        cov, prec_n / max(prec_d, 1), len(offs) / n_starts,
        " ".join("{:.0f}".format(v) for v in np.percentile(offs, [10, 25, 50, 75, 90])))


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "atd"
    keys = variants.register(c.METRICA, [v])
    base = Params()
    print("variant", v)
    print("carry (cov = share of alive visible frames)")
    for r in (1.5, 2.0, 3.0, 4.0):
        for mg in (0.0, 0.5, 1.0, 2.0):
            for dv in (2.0, 4.0, 100.0):
                p = dataclasses.replace(base, carry_radius=r, carry_margin=mg, carry_dv=dv)
                print("  radius {} margin {} dv {:<5}: {}".format(r, mg, dv, study(keys, p, "carry")))
    print("touch (cov = events per true possession)")
    for dvv in (1.5, 2.0, 3.0, 4.0):
        for ang in (25.0, 35.0, 60.0, 400.0):
            for rad in (1.5, 2.5, 4.0):
                p = dataclasses.replace(base, touch_dv=dvv, touch_angle=ang, touch_radius=rad)
                print("  dv {} angle {:<5} radius {}: {}".format(dvv, ang, rad, study(keys, p, "touch")))
