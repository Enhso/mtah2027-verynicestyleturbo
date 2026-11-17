"""Train the team-shape classifier: python -m calibration.formation_train [train | check]

  train  logistic regression, H against A, on alive frames of the `atd` variant of the 7 DFL
         matches (Metrica stays out, so every Metrica score is held out), every STRIDE-th frame, plus a team-swapped copy of each row. No intercept
         and features scaled by their std only (never centred), so the model is antisymmetric:
         swapping the teams negates the log-odds. Writes possession/formation_model.json.
  check  label-free: attack sign per period and feature quantiles on BAR-ATH and the proxies,
         then frame accuracy of the DFL model on each Metrica game (counts and rates only).

Deterministic: fixed stride, fixed iteration count, no randomness.
"""
import json
import sys

import numpy as np

from calibration import common as c, variants
from possession.features import SHAPE_NAMES
from possession.match import H, A
from possession.params import MODEL_PATH, Params

STRIDE = 5
ITERS = 25
RIDGE = 1e-3     # on the scaled features, per row; only keeps IRLS well posed


def feats(name, variant="atd"):
    key = (name, variant)
    if key not in c._MATCH:
        c.register(key, variants.build(name, variant))
    return c.feats(key, Params())


def rows(name, variant="atd", stride=STRIDE):
    """(X, y, blackout) on alive frames of every period, every stride-th frame. y=1 for H."""
    fs = feats(name, variant)
    m = c._MATCH[(name, variant)]
    xs, ys, bs = [], [], []
    for (a, b), f in zip(m.periods.tolist(), fs):
        lab = m.labels[a:b]
        keep = (lab == H) | (lab == A)
        keep[np.arange(b - a) % stride != 0] = False
        xs.append(f.shape[keep])
        ys.append((lab[keep] == H).astype(float))
        bs.append(~f.visible[keep])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(bs)


def fit(X, y):
    """No-intercept logistic regression by IRLS on swap-augmented rows. Returns raw-unit coefs."""
    X = np.concatenate([X, -X])
    y = np.concatenate([y, 1.0 - y])
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = X / scale
    w = np.zeros(Z.shape[1])
    for _ in range(ITERS):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        g = Z.T @ (y - p) - RIDGE * len(y) * w
        Hm = (Z * (p * (1 - p))[:, None]).T @ Z + RIDGE * len(y) * np.eye(len(w))
        w = w + np.linalg.solve(Hm, g)
    return w / scale, scale


def train(names):
    data = [rows(n) for n in names]
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([d[1] for d in data])
    coef, scale = fit(X, y)
    return coef, scale, len(y)


def accuracy(coef, X, y):
    z = X @ coef
    decided = z != 0
    return float(((z > 0) == (y == 1))[decided].mean()) if decided.any() else float("nan"), int(decided.sum())


def check():
    ref = c.load_match("BAR-ATH")
    from possession import features
    print("attack sign per period (+1: home attacks +x), feature quantiles q5/25/50/75/95")
    def describe(name, m):
        signs, shapes = [], []
        for a, b in m.periods.tolist():
            f = features.compute(m.ball[a:b], m.players[a:b], m.speed[a:b], m.team, m.pitch,
                                 m.fps, Params())
            xc = m.players[a:b, :, 0] - m.pitch[0] / 2.0
            with np.errstate(invalid="ignore"):
                d = np.nanmean(xc[:, m.team == 1], axis=1) - np.nanmean(xc[:, m.team == 0], axis=1)
            signs.append("{:+.0f} (frames agreeing {:.2f})".format(
                np.sign(np.nanmedian(d)), np.mean(np.sign(d[np.isfinite(d)]) == np.sign(np.nanmedian(d)))))
            shapes.append(f.shape)
        sh = np.concatenate(shapes)
        print(name, "signs", signs, "zero rows {:.3f}".format(float((sh == 0).all(axis=1).mean())))
        for j, nm in enumerate(SHAPE_NAMES):
            print("    {:<14} {}".format(nm, " ".join("{:.2f}".format(v) for v in
                                                    np.percentile(np.abs(sh[:, j]), [5, 25, 50, 75, 95]))))
    describe("BAR-ATH", ref)
    for nm in c.METRICA + c.DFL[:2]:
        describe(nm + " atd", variants.build(nm, "atd"))


def heldout_accuracy(coef, n):
    """Frame accuracy of the model on each Metrica game: alive frames (atd), alive blackout frames
    (atd+mask). {game: (acc, n, acc blackout, n blackout, n_train)}."""
    out = {}
    for g in c.METRICA:
        X, y, _ = rows(g, "atd", stride=1)
        Xm, ym, bm = rows(g, "atd+mask", stride=1)
        acc, k = accuracy(coef, X, y)
        accb, kb = accuracy(coef, Xm[bm], ym[bm])
        out[g] = (acc, k, accb, kb, n)
        print("{}: n_train {} | alive frames acc {:.3f} (n {}) | atd+mask alive blackout "
              "frames acc {:.3f} (n {})".format(g, n, acc, k, accb, kb))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    coef, scale, n = train(c.DFL)
    if mode == "check":
        check()
        heldout_accuracy(coef, n)
    else:
        with open(MODEL_PATH, "w") as fh:
            json.dump({"features": list(SHAPE_NAMES), "coef": [round(float(v), 6) for v in coef],
                       "scale": [round(float(v), 6) for v in scale], "n_train_rows": n,
                       "n_train_rows_with_swap": 2 * n, "matches": c.DFL,
                       "variant": "atd", "stride": STRIDE, "iters": ITERS, "ridge": RIDGE},
                      fh, indent=1)
        print("coef", dict(zip(SHAPE_NAMES, np.round(coef, 4).tolist())), "rows", n)
