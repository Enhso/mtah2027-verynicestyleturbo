"""Quantile transfer of thresholds from the proxy to this match (PLAN section 4).

A threshold at quantile q of a feature on the proxy (the three Metrica games, `atd` variant, pooled)
becomes the value at quantile q of the same feature on this match, both on visible-ball frames.
Prints quantiles and transferred values only. A threshold outside the middle 99.8% of the proxy
feature (a rule that is in effect switched off) is kept as it is.

python -m calibration.transfer <params.json>   writes calibration/results/transfer.json
"""
import json
import os
import sys

import numpy as np

from calibration import common as c, run_search, variants

# threshold -> feature it cuts
FEATURE = {"carry_radius": "near_dist", "touch_radius": "near_dist", "carry_margin": "margin",
           "carry_dv": "ball_minus_player_v", "touch_dv": "v_change", "touch_angle": "angle",
           "still_speed": "still_speed", "low_speed": "median_speed", "move_speed": "ball_speed"}


def feature_values(fs, params):
    """{feature: 1-d array over the visible-ball frames of every period, finite values only}."""
    out = {}

    def put(name, per_period):
        v = np.concatenate(per_period)
        out[name] = v[np.isfinite(v)]

    with np.errstate(invalid="ignore", divide="ignore"):
        put("near_dist", [f.near_dist[f.visible] for f in fs])
        put("margin", [(f.opp_dist - f.near_dist)[f.visible] for f in fs])
        put("ball_minus_player_v", [np.linalg.norm(f.ball_v - f.near_v, axis=1)[f.visible] for f in fs])
        put("v_change", [np.linalg.norm(f.v_after - f.v_before, axis=1)[f.visible] for f in fs])
        ang = []
        for f in fs:
            sb, sa = np.linalg.norm(f.v_before, axis=1), np.linalg.norm(f.v_after, axis=1)
            cos = np.einsum("ij,ij->i", f.v_before, f.v_after) / (sb * sa)
            a = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            a[~((sb >= params.touch_min_speed) & (sa >= params.touch_min_speed))] = np.nan
            ang.append(a[f.visible])
        put("angle", ang)
        put("still_speed", [f.still_speed[f.visible] for f in fs])
        put("median_speed", [f.median_speed[f.visible] for f in fs])
        put("ball_speed", [np.linalg.norm(f.ball_v, axis=1)[f.visible] for f in fs])
    return out


def transfer(params):
    keys = variants.register(c.METRICA, ["atd"])
    proxy = [feature_values(c.feats(k, params), params) for k in keys]
    c.register("BAR-ATH", c.load_match("BAR-ATH"))
    here = feature_values(c.feats("BAR-ATH", params), params)
    rows = {}
    for name, feat in FEATURE.items():
        value = float(getattr(params, name))
        pooled = np.concatenate([p[feat] for p in proxy])
        q = float((pooled <= value).mean())
        new = float(np.quantile(here[feat], q)) if 0.001 < q < 0.999 else value
        rows[name] = dict(feature=feat, tuned=value, quantile=round(q, 4), transferred=round(new, 2))
    return rows


if __name__ == "__main__":
    params = run_search.load(sys.argv[1])
    rows = transfer(params)
    print("{:<14} {:<20} {:>8} {:>9} {:>12}".format("threshold", "feature", "tuned", "quantile", "transferred"))
    for name, r in rows.items():
        print("{:<14} {:<20} {:>8.2f} {:>9.4f} {:>12.2f}".format(name, r["feature"], r["tuned"],
                                                                r["quantile"], r["transferred"]))
    with open(os.path.join(run_search.OUT, "transfer.json"), "w") as fh:
        json.dump(rows, fh, indent=1, sort_keys=True)
