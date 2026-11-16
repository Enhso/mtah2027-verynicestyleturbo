"""Where the current params lose spellF1 on a proxy variant: stoppages or teams, and by how much.

  oracleD     true D frames, predicted team elsewhere: what is left is turnover error
  oracleTeam  predicted D frames, true team elsewhere: what is left is stoppage error
  aliveF1     spellF1 with both teams merged: stoppage boundaries only
Also the latency of matched transitions by kind, and what blackouts look like under the labels.
"""
import sys
import numpy as np
from evaluation import possession_metrics as pm
from calibration import common as c, variants
from possession.match import D, X
from possession.params import Params


def _ffill_team(team_or_minus1):
    """Carry the last value >= 0 forward, then the first one backward."""
    idx = np.where(team_or_minus1 >= 0, np.arange(len(team_or_minus1)), -1)
    last = np.maximum.accumulate(idx)
    first = np.flatnonzero(team_or_minus1 >= 0)
    out = team_or_minus1[np.maximum(last, first[0] if len(first) else 0)]
    return out


def oracles(m, pred):
    lab = m.labels
    out = {}
    od, ot = pred.copy(), pred.copy()
    for a, b in m.periods.tolist():
        l, p = lab[a:b], pred[a:b]
        pteam = _ffill_team(np.where(p < 2, p, -1).astype(np.int8))
        od[a:b] = np.where(l == D, D, pteam)
        lteam = _ffill_team(np.where(l < 2, l, -1).astype(np.int8))
        ot[a:b] = np.where(p == D, D, lteam)
    merge = lambda x: np.where(x == 1, 0, x)
    out["oracleD"] = pm.evaluate(lab, od)["spell_f1_mean"]
    out["oracleTeam"] = pm.evaluate(lab, ot)["spell_f1_mean"]
    out["aliveF1"] = pm.evaluate(merge(lab), merge(pred))["spell_f1_mean"]
    return out


def transitions(m, pred, tol=50):
    """Per kind (turnover, stop, restart): n true, n pred, matched within tol, median signed latency."""
    out = {}
    lab = m.labels
    for kind in ("turnover", "stop", "restart"):
        def frames(x):
            fr = []
            for a, b in m.periods.tolist():
                s = x[a:b]
                prev, cur = s[:-1], s[1:]
                if kind == "turnover":
                    hit = (prev < 2) & (cur < 2) & (prev != cur)
                elif kind == "stop":
                    hit = (prev < 2) & (cur == D)
                else:
                    hit = (prev == D) & (cur < 2)
                fr.append(np.flatnonzero(hit) + 1 + a)
            return np.concatenate(fr)
        t, p = frames(lab), frames(pred)
        lat = []
        if len(p):
            for f in t:
                k = np.argmin(np.abs(p - f))
                if abs(p[k] - f) <= tol:
                    lat.append(p[k] - f)
        lat = np.array(lat)
        out[kind] = "true {} pred {} matched {} lat q25/50/75 {}".format(
            len(t), len(p), len(lat),
            " ".join("{:.0f}".format(v) for v in np.percentile(lat, [25, 50, 75])) if len(lat) else "n/a")
    return out


def gap_table(key, params):
    """Blackouts under the labels: by length class, count, share fully dead, share with any dead."""
    m = c._MATCH[key]
    rows = {}
    for (a, b), f in zip(m.periods.tolist(), c.feats(key, params)):
        lab = m.labels[a:b]
        for g0, g1 in f.gaps:
            n = g1 - g0
            cls = "<=1s" if n <= 25 else "1-5s" if n <= 125 else "5-30s" if n <= 750 else ">30s"
            dead = (lab[g0:g1] == D).mean()
            r = rows.setdefault(cls, [0, 0, 0, 0, 0.0])
            r[0] += 1
            r[1] += dead > 0.8
            r[2] += dead < 0.2
            r[3] += n
            r[4] += dead * n
    return {k: "n {} mostly dead {} mostly alive {} dead frame share {:.2f}".format(
        r[0], r[1], r[2], r[4] / r[3]) for k, r in rows.items()}


if __name__ == "__main__":
    vs = (sys.argv[1] if len(sys.argv) > 1 else "atd,atd+mask").split(",")
    p = Params()
    for v in vs:
        for nm in sys.argv[2:] or c.METRICA:
            key = (nm, v)
            c.register(key, variants.build(nm, v))
            m = c._MATCH[key]
            pred = c.decode(key, p)
            r = pm.evaluate(m.labels, pred)
            o = oracles(m, pred)
            print("{} {}: spellF1 {:.3f} macroF1 {:.3f} n_pred {} | oracleD {:.3f} oracleTeam {:.3f} aliveF1 {:.3f}".format(
                nm, v, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], o["oracleD"],
                o["oracleTeam"], o["aliveF1"]))
            for k, s in transitions(m, pred).items():
                print("    {:<9} {}".format(k, s))
            for k, s in sorted(gap_table(key, p).items()):
                print("    gaps {:<6} {}".format(k, s))
