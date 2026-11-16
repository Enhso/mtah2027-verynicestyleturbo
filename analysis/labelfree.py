"""PLAN section 6, label-free questions on the full BAR-ATH match.

    .venv/bin/python analysis/labelfree.py > notes/labelfree-output.txt

Prints counts, rates, quantiles and scores only: no frame index, coordinate or per-tag row. Uses no
ground truth. Reads the v0 feature and evidence code without changing it.
"""

import dataclasses
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import config as cfg, evidence, features, ingest, match as mt, predict  # noqa: E402
from possession.params import Params  # noqa: E402

QS = (1, 5, 25, 50, 75, 95, 99)
CLASSES = (("6-25f (<1s)", 6, 26), ("26-125f (1-5s)", 26, 126),
           ("126-750f (5-30s)", 126, 751), (">750f (>30s)", 751, 10 ** 9))
PRE_KICKOFF = (110808, 113742)      # published dead block, half-open [PLAN section 8]
ATTACK_PAIR = {"defending in attacking third": "build up", "defending in middle third": "progression",
               "defending in defensive third": "creating chances",
               "defensive transition": "attacking transition"}


def h(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


def qs(v, qlist=QS, fmt="{:.2f}"):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "n=0"
    return "n={} ".format(len(v)) + " ".join("q{}={}".format(q, fmt.format(np.percentile(v, q))) for q in qlist)


def share(mask):
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - r, c + r


def logodds(k, n):
    """Laplace-smoothed log-odds of agreement k of n."""
    p = (k + 1.0) / (n + 2.0)
    return math.log(p / (1 - p))


def tag_side(m, tag):
    return cfg.for_match(m.name).tag_side(tag["tags"].get("Team"))


def main():
    m = ingest.ensure()
    L, W = m.pitch
    fps = m.fps
    P = Params()
    conf = cfg.for_match(m.name)
    periods = [(int(a), int(b)) for a, b in m.periods]
    graded = np.zeros(m.n_frames, bool)
    for a, b in periods:
        graded[a:b] = True
    print("match {} frames {} fps {} pitch {}x{} slots {} (home {}, away {}) tags {} graded frames {}".format(
        m.name, m.n_frames, fps, L, W, len(m.team), int((m.team == 0).sum()), int((m.team == 1).sum()),
        len(m.tags), int(graded.sum())))

    # Per-period features and channels (v0 code, default params).
    F, CH = [], []
    for a, b in periods:
        f = features.compute(m.ball[a:b], m.players[a:b], m.speed[a:b], m.team, m.pitch, fps, P)
        tags = evidence.localize_tags(m.tags, conf, a, b)
        F.append(f)
        CH.append(evidence.channels(f, tags, P))

    def gather(get):
        """Concatenate a per-period (n,...) array into a full-match array (NaN/False elsewhere)."""
        out = None
        for (a, b), f in zip(periods, F):
            v = np.asarray(get(f))
            if out is None:
                out = np.full((m.n_frames,) + v.shape[1:], np.nan if v.dtype.kind == "f" else 0, v.dtype)
            out[a:b] = v
        return out

    visible = gather(lambda f: f.visible).astype(bool)

    # ------------------------------------------------------------------------------------------
    h("Q1 Do positions leave the 0 to 1 range?")
    nb = m.ball / np.array([L, W])
    npl = m.players / np.array([L, W])
    for name, arr, mask in (("ball all frames", nb, np.ones(m.n_frames, bool)),
                            ("ball graded", nb, graded),
                            ("players all frames", npl, np.ones(m.n_frames, bool)),
                            ("players graded", npl, graded)):
        v = arr[mask]
        v = v[~np.isnan(v).any(axis=-1)]
        x, y = v[..., 0], v[..., 1]
        out = (x < 0) | (x > 1) | (y < 0) | (y > 1)
        print("{:<20} n={} x[min {:.4f} max {:.4f}] y[min {:.4f} max {:.4f}] share outside {:.5f} "
              "(x {:.5f}, y {:.5f})".format(name, len(v), x.min(), x.max(), y.min(), y.max(), out.mean(),
                                             ((x < 0) | (x > 1)).mean(), ((y < 0) | (y > 1)).mean()))
    bg = nb[graded]
    bg = bg[~np.isnan(bg).any(axis=1)]
    d_out = np.minimum.reduce([bg[:, 0] * L, (1 - bg[:, 0]) * L, bg[:, 1] * W, (1 - bg[:, 1]) * W])
    print("ball graded, metres beyond nearest line when outside: " + qs(-d_out[d_out < 0], (50, 90, 99)))
    print("ball graded visible share {:.4f}; offpitch frames (v0 rule) share of visible {:.5f}".format(
        share(visible[graded]), share(gather(lambda f: f.offpitch)[graded & visible])))

    # ------------------------------------------------------------------------------------------
    for lim in (0.25, 0.5, 1.0, 2.0):
        print("ball graded outside the lines by more than {:.2f} m: share of outside frames {:.3f}".format(
            lim, share(-d_out[d_out < 0] > lim)))
    runs_out = []
    for (a, b), f in zip(periods, F):
        s_, e_ = features.runs(f.offpitch)
        runs_out += list((e_ - s_) / fps)
    print("off-pitch runs: count {} duration s: {}".format(len(runs_out), qs(runs_out, (5, 25, 50, 75, 95), "{:.1f}")))

    h("Q2 Blackout anatomy per gap class (gaps after v0 interpolation of <=5 frame holes)")
    restart_spots = np.array([[L / 2, W / 2], [0, 0], [0, W], [L, 0], [L, W],
                              [6.0, W / 2], [L - 6.0, W / 2]])
    rows = {c[0]: [] for c in CLASSES}
    interior, boundary = 0, 0
    for (a0, _), f, ch in zip(periods, F, CH):
        n = f.n
        for a, b in f.gaps:
            if a == 0 or b == n:
                boundary += 1
                continue
            interior += 1
            dur = b - a
            cls = next(c[0] for c in CLASSES if c[1] <= dur < c[2]) if dur >= 6 else None
            if cls is None:
                cls = CLASSES[0][0]     # 1-5 frame survivors cannot exist (interpolated); safety
            i = a - 1
            outward = float(np.dot(f.v_before[i], f.out_normal[i])) if np.isfinite(f.v_before[i]).all() else np.nan
            end_ball = f.ball[b]
            end_line = f.line_dist[b]
            spot = np.min(np.hypot(*(restart_spots - end_ball).T)) if np.isfinite(end_ball).all() else np.inf
            rows[cls].append(dict(
                dur=dur / fps, start_line=f.line_dist[i], outward=outward, end_line=end_line, spot=spot,
                exit_fire=bool(ch["exit"][a:b, 2].any()), long_fire=bool(ch["long_gap"][a:b, 2].any()),
                v_before=float(np.linalg.norm(f.v_before[i])), v_after=float(np.linalg.norm(f.v_after[b])),
                ball_before_vis=np.isfinite(f.v_before[i]).all()))
    print("interior gaps {} (gaps touching a period edge, skipped: {})".format(interior, boundary))
    tot_frames = sum(int(round(r["dur"] * fps)) for rs in rows.values() for r in rs)
    print("graded frames inside interior gaps: {} ({:.3f} of graded)".format(
        tot_frames, tot_frames / graded.sum()))
    for name, lo, hi in CLASSES:
        rs = rows[name]
        if not rs:
            print("{:<18} count 0".format(name))
            continue
        g = lambda k: np.array([r[k] for r in rs], float)
        sl, ob, el, sp = g("start_line"), g("outward"), g("end_line"), g("spot")
        print("{:<18} count {:>4} dur_s: {}".format(name, len(rs), qs(g("dur"), (5, 25, 50, 75, 95))))
        print("   start within 3m of line {:.3f}; ... and outward>=0.5 m/s {:.3f}; outward>=0 {:.3f}; "
              "exit channel fires {:.3f}; long-gap channel fires {:.3f}; either {:.3f}".format(
                  share(sl <= 3), share((sl <= 3) & (ob >= 0.5)), share((sl <= 3) & (ob >= 0)),
                  share(g("exit_fire")), share(g("long_fire")),
                  share((g("exit_fire") > 0) | (g("long_fire") > 0))))
        print("   end within 3m of a line {:.3f}; end within 3m of a restart spot (centre/corner/goal-kick) {:.3f}; "
              "end within 3m of line or spot {:.3f}".format(share(el <= 3), share(sp <= 3),
                                                            share((el <= 3) | (sp <= 3))))
        print("   ball speed last frame before: {}".format(qs(g("v_before"), (25, 50, 75, 95))))
        print("   ball speed first frame after: {}".format(qs(g("v_after"), (25, 50, 75, 95))))

    # ------------------------------------------------------------------------------------------
    print("exit-rule sweep, share of gaps (classes >=5 s) whose start fires: dist<=D and outward speed>=V")
    big = rows["126-750f (5-30s)"] + rows[">750f (>30s)"]
    sl_ = np.array([r["start_line"] for r in big])
    ob_ = np.array([r["outward"] for r in big])
    el_ = np.array([r["end_line"] for r in big])
    sp_ = np.array([r["spot"] for r in big])
    for D_ in (3, 5, 8):
        print("  D={}m: ".format(D_) + "  ".join("V>={}: {:.3f}".format(V_, share((sl_ <= D_) & (ob_ >= V_)))
                                               for V_ in (0.5, 0.0, -9.9)) +
              "   start-or-end near line/spot (<=D, any velocity): {:.3f}".format(
                  share((sl_ <= D_) | (el_ <= D_) | (sp_ <= D_))))
    nbig = sum(int(round(r["dur"] * fps)) for r in big)
    exit_frames = sum(int(round(r["dur"] * fps)) for r in big if r["exit_fire"] or r["long_fire"])
    print("share of >=5 s blackout frames inside a gap where exit or long-gap fires: {:.3f}".format(exit_frames / nbig))

    h("Q3 Median player speed (m/s) in visible frames, blackouts by class, pre-kickoff block")
    med = gather(lambda f: f.median_speed)
    in_gap = np.zeros(m.n_frames, bool)
    gap_cls = {}
    for (a0, _), f in zip(periods, F):
        for a, b in f.gaps:
            cls = next((c[0] for c in CLASSES if c[1] <= b - a < c[2]), CLASSES[0][0])
            in_gap[a0 + a:a0 + b] = True
            gap_cls.setdefault(cls, []).append(np.arange(a0 + a, a0 + b))
    pre = np.zeros(m.n_frames, bool)
    pre[PRE_KICKOFF[0]:PRE_KICKOFF[1]] = True
    groups = [("visible, graded, excl pre-kickoff", graded & visible & ~pre),
              ("pre-kickoff block (dead)", pre)]
    for name, _, _ in CLASSES:
        if name in gap_cls:
            mk = np.zeros(m.n_frames, bool)
            mk[np.concatenate(gap_cls[name])] = True
            groups.append(("blackout " + name, mk & graded & ~pre))
    for name, mk in groups:
        print("{:<36} {}".format(name, qs(med[mk], (5, 10, 25, 50, 75, 90, 95))))
    print("pre-kickoff block ball visible share {:.3f}".format(share(visible[pre])))
    # Threshold table: threshold = quantile of visible-frame distribution.
    vis_med = med[graded & visible & ~pre]
    vis_med = vis_med[np.isfinite(vis_med)]
    low_run = int(round(P.low_s * fps))

    def low_mask(thr):
        out = np.zeros(m.n_frames, bool)
        for (a, b) in periods:
            out[a:b] = features.keep_long(np.nan_to_num(med[a:b], nan=1e9) < thr, low_run)
        return out

    print("low-activity (median speed < thr for >= {:.0f} s): share of frames firing".format(P.low_s))
    print("{:>8} {:>7} | {:>8} {:>10} {:>10} {:>14} | ratio pre/visible  logodds(pre vs visible)".format(
        "vis-q", "thr", "visible", "pre-kick", "blackout", "bk>30s"))
    long_mask = np.zeros(m.n_frames, bool)
    if ">750f (>30s)" in gap_cls:
        long_mask[np.concatenate(gap_cls[">750f (>30s)"])] = True
    for q in (2, 5, 10, 15, 20, 25, 35, 50):
        thr = float(np.percentile(vis_med, q))
        lm = low_mask(thr)
        sv, sp_, sb = share(lm[graded & visible & ~pre]), share(lm[pre]), share(lm[graded & in_gap & ~pre])
        sl = share(lm[graded & long_mask & ~pre]) if long_mask.any() else float("nan")
        print("{:>7}% {:>7.3f} | {:>8.3f} {:>10.3f} {:>10.3f} {:>14.3f} | {:>6.1f}  {:+.2f}".format(
            q, thr, sv, sp_, sb, sl, sp_ / max(sv, 1e-9),
            math.log(max(sp_, 1e-6) / max(1 - sp_, 1e-6)) - math.log(max(sv, 1e-6) / max(1 - sv, 1e-6))))
    thr_v0 = P.low_speed
    print("v0 first value 1.5 m/s sits at visible-frame quantile {:.1f}%".format(100 * share(vis_med < thr_v0)))

    # ------------------------------------------------------------------------------------------
    h("Q4 Which slots are not players?")
    team = m.team
    pl = m.players[graded]
    sp = m.speed[graded]
    present = ~np.isnan(pl).any(axis=2)
    x, y = pl[..., 0], pl[..., 1]
    with np.errstate(invalid="ignore"):
        edge = present & ((y < 1.0) | (y > W - 1.0))
        outside = present & ((x < 0) | (x > L) | (y < 0) | (y > W))
    print("{:>4} {:>4} {:>9} {:>11} {:>10} {:>10}".format("slot", "team", "presence", "touchline1m", "outside", "meanspeed"))
    flagged = []
    for k in range(pl.shape[1]):
        pk = present[:, k].sum()
        if pk == 0:
            print("{:>4} {:>4} {:>9.4f}".format(k, int(team[k]), 0.0))
            continue
        e, o = edge[:, k].sum() / pk, outside[:, k].sum() / pk
        print("{:>4} {:>4} {:>9.4f} {:>11.4f} {:>10.4f} {:>10.2f}".format(
            k, int(team[k]), pk / len(pl), e, o, float(np.nanmean(sp[:, k]))))
        if e > 0.5:
            flagged.append(k)
    print("slots with >50% of present frames within 1 m of a touchline or beyond: {}".format(flagged))
    print("present slots per graded frame: " + qs(present.sum(axis=1), (5, 25, 50, 75, 95), "{:.0f}"))
    print("present slots per frame excluding flagged: " + qs(
        np.delete(present, flagged, axis=1).sum(axis=1), (5, 25, 50, 75, 95), "{:.0f}"))
    print("same-team slot pairs closer than 0.5 m (both present), share of frames where both present:")
    pairs = []
    for side in (0, 1):
        cols = np.flatnonzero(team == side)
        for i, ci in enumerate(cols):
            for cj in cols[i + 1:]:
                both = present[:, ci] & present[:, cj]
                if both.sum() < 250:
                    continue
                d = np.hypot(x[both, ci] - x[both, cj], y[both, ci] - y[both, cj])
                pairs.append((float((d < 0.5).mean()), int(both.sum()), int(ci), int(cj)))
    pairs.sort(reverse=True)
    print("  pairs tested {}; pairs with share>0.01: {}; share>0.05: {}; share>0.20: {}".format(
        len(pairs), sum(p[0] > 0.01 for p in pairs), sum(p[0] > 0.05 for p in pairs),
        sum(p[0] > 0.2 for p in pairs)))
    for s_, n_, ci, cj in pairs[:8]:
        print("  slots {:>2},{:>2}: both present {:>6} frames, within 0.5 m on {:.3f}".format(ci, cj, n_, s_))

    # ------------------------------------------------------------------------------------------
    h("Q5 Which way does each team attack? mean x (normalised, 0 = low-x end) in the first minute")
    minute = 60 * fps
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES
    for label, start in (("1st half", KICKOFF_VIDEO_FRAMES["1st Half"]), ("2nd half", KICKOFF_VIDEO_FRAMES["2nd Half"])):
        sl = slice(start, start + minute)
        for side, nm in ((0, "home"), (1, "away")):
            cols = np.flatnonzero(team == side)
            xs = m.players[sl][:, cols, 0] / L
            ball_vis = share(~np.isnan(m.ball[sl, 0]))
            print("{} {:<5} mean x {:.3f} (all frames of first minute; ball visible {:.2f})".format(
                label, nm, float(np.nanmean(xs)), ball_vis))
        xs0 = m.players[start][np.flatnonzero(team == 0), 0] / L
        xs1 = m.players[start][np.flatnonzero(team == 1), 0] / L
        print("{} kickoff frame itself: home mean x {:.3f}, away mean x {:.3f}".format(
            label, float(np.nanmean(xs0)), float(np.nanmean(xs1))))

    # ------------------------------------------------------------------------------------------
    for label, start in (("1st half", KICKOFF_VIDEO_FRAMES["1st Half"]), ("2nd half", KICKOFF_VIDEO_FRAMES["2nd Half"])):
        stop = periods[0][1] if label == "1st half" else periods[1][1]
        for side, nm in ((0, "home"), (1, "away")):
            cols = [k for k in np.flatnonzero(team == side) if (~np.isnan(m.players[start:stop, k, 0])).mean() > 0.5]
            k = min(cols, key=lambda c: float(np.nanmean(m.speed[start:stop, c])))
            print("{} {} keeper (slot of lowest mean speed with presence>0.5): mean x {:.3f}".format(
                label, nm, float(np.nanmean(m.players[start:stop, k, 0])) / L))

    h("Q6 Tag coverage and DEFENDING mirrors")
    att_codes = evidence.ATTACK_CODES
    cover = {0: np.zeros(m.n_frames, bool), 1: np.zeros(m.n_frames, bool)}
    cover_def = {0: np.zeros(m.n_frames, bool), 1: np.zeros(m.n_frames, bool)}
    bycode = {}
    for t in m.tags:
        code = (t["code"] or "").strip().lower()
        s = tag_side(m, t)
        bycode.setdefault(code, [0, 0, 0])
        bycode[code][0] += 1
        bycode[code][1] += s is not None
        bycode[code][2] += t["end_frame"] - t["start_frame"]
        if s is not None and code in att_codes:
            cover[s][t["start_frame"]:t["end_frame"]] = True
        if s is not None and code in ATTACK_PAIR:
            cover_def[s][t["start_frame"]:t["end_frame"]] = True
    for code, (n, ns, fr) in sorted(bycode.items()):
        print("{:<32} tags {:>3} with team {:>3} total frames {:>7} (mean {:.1f}s)".format(code, n, ns, fr, fr / n / fps))
    anyc = cover[0] | cover[1]
    print("share of graded frames under a team-tagged attacking phase: any {:.3f}; home {:.3f}; away {:.3f}; "
          "both teams at once {:.4f}".format(share(anyc[graded]), share(cover[0][graded]), share(cover[1][graded]),
                                           share((cover[0] & cover[1])[graded])))
    print("share of graded frames under a team-tagged DEFENDING phase: any {:.3f}".format(
        share((cover_def[0] | cover_def[1])[graded])))
    print("share of attacking-covered graded frames where the ball is visible: {:.3f} (graded overall {:.3f})".format(
        share(visible[graded & anyc]), share(visible[graded])))
    # mirror test
    att = [(t["start_frame"], t["end_frame"], (t["code"] or "").strip().lower(), tag_side(m, t)) for t in m.tags
           if (t["code"] or "").strip().lower() in att_codes and tag_side(m, t) is not None]
    dfn = [(t["start_frame"], t["end_frame"], (t["code"] or "").strip().lower(), tag_side(m, t)) for t in m.tags
           if (t["code"] or "").strip().lower() in ATTACK_PAIR]

    def iou(a, b):
        i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        return i / float((a[1] - a[0]) + (b[1] - b[0]) - i)

    best_opp, best_same, best_pair_opp, no_team = [], [], [], 0
    for d in dfn:
        if d[3] is None:
            no_team += 1
            continue
        best_opp.append(max([iou(d, a) for a in att if a[3] != d[3]] or [0]))
        best_same.append(max([iou(d, a) for a in att if a[3] == d[3]] or [0]))
        best_pair_opp.append(max([iou(d, a) for a in att if a[3] != d[3] and a[2] == ATTACK_PAIR[d[2]]] or [0]))
    best_opp, best_same, best_pair_opp = map(np.array, (best_opp, best_same, best_pair_opp))
    print("DEFENDING tags {} (without team {}); best IoU with an attacking tag of the OTHER team: {}".format(
        len(dfn), no_team, qs(best_opp, (5, 25, 50, 75, 95))))
    print("  share with IoU>0.8 vs other team: {:.3f}; vs paired code of other team: {:.3f}; vs SAME team {:.3f}".format(
        share(best_opp > 0.8), share(best_pair_opp > 0.8), share(best_same > 0.8)))
    print("  share with exact interval match to any attacking tag of the other team: {:.3f}".format(share(best_opp > 0.999)))
    both_att = share((cover[0] & cover[1])[graded])

    # ------------------------------------------------------------------------------------------
    h("Q7 Per-code tag reliability: agreement with local carry evidence and the v0 decoder")
    p_notag = dataclasses.replace(P, phase_w=0.0, setpiece_w=0.0)
    codes_nt = predict.predict(m, p_notag)
    carry = np.full(m.n_frames, -1, np.int8)
    for (a, b), f in zip(periods, F):
        carry[a:b] = evidence.carry_events(f, P)
    half = 25
    cs = {s: np.concatenate(([0], np.cumsum(carry == s))) for s in (0, 1)}
    cnt = {s: cs[s][np.minimum(np.arange(m.n_frames) + half + 1, m.n_frames)] -
           cs[s][np.maximum(np.arange(m.n_frames) - half, 0)] for s in (0, 1)}
    strong = np.full(m.n_frames, -1, np.int8)
    strong[(cnt[0] >= 10) & (cnt[1] == 0)] = 0
    strong[(cnt[1] >= 10) & (cnt[0] == 0)] = 1
    strong[~graded] = -1
    print("strong-carry frames (>=10 carry frames of one team and none of the other within +-1 s): "
          "{:.3f} of graded; decoder (no tag channels) agrees with strong carry team on alive strong frames: {:.3f}; "
          "decoder is D on strong frames: {:.3f}".format(
              share(strong[graded] >= 0), share((codes_nt == strong)[(strong >= 0) & (codes_nt < 2)]),
              share(codes_nt[strong >= 0] == 2)))
    stats = {}
    ALL = "ALL ATTACK CODES"
    for t in m.tags:
        code = (t["code"] or "").strip().lower()
        s = tag_side(m, t)
        if s is None or not (code in att_codes or code in ATTACK_PAIR):
            continue
        a_, b_ = t["start_frame"], t["end_frame"]
        sl = slice(a_, b_)
        sf = strong[sl]
        sel = sf >= 0
        dec = codes_nt[sl]
        for key in ([code, ALL] if code in att_codes else [code]):
            d = stats.setdefault(key, dict(tags=0, ev_tags=0, agree_tags=0, frames=0, agree_frames=0,
                                          dec_frames=0, dec_agree=0, dec_tags_ev=0, dec_tags_agree=0))
            d["tags"] += 1
            d["frames"] += int(sel.sum())
            d["agree_frames"] += int((sf[sel] == s).sum())
            alive = (dec < 2)
            d["dec_frames"] += int(alive.sum())
            d["dec_agree"] += int((dec[alive] == s).sum())
            if sel.sum() >= 25:
                d["ev_tags"] += 1
                d["agree_tags"] += int((sf[sel] == s).mean() > 0.5)
            if alive.sum() >= 25:
                d["dec_tags_ev"] += 1
                d["dec_tags_agree"] += int((dec[alive] == s).mean() > 0.5)
    print("{:<30} {:>4} | strong-carry frames agree (n, share, log-odds) | tags with >=25 strong frames: majority agrees "
          "(n, share, Wilson95, log-odds) | decoder alive frames agree (n, share) | tag majority vs decoder (n, share)".format("code", "tags"))
    for key in sorted(stats, key=lambda k: (k != ALL, k)):
        d = stats[key]
        lo, hi = wilson(d["agree_tags"], d["ev_tags"])
        print("{:<30} {:>4} | {:>7} {:.3f} {:+.2f} | {:>3} {:.3f} [{:.2f},{:.2f}] {:+.2f} | {:>7} {:.3f} | {:>3} {:.3f}".format(
            key, d["tags"], d["frames"], d["agree_frames"] / max(d["frames"], 1),
            logodds(d["agree_frames"], d["frames"]), d["ev_tags"], d["agree_tags"] / max(d["ev_tags"], 1), lo, hi,
            logodds(d["agree_tags"], d["ev_tags"]), d["dec_frames"], d["dec_agree"] / max(d["dec_frames"], 1),
            d["dec_tags_ev"], d["dec_tags_agree"] / max(d["dec_tags_ev"], 1)))
    pool = stats[ALL]
    p0 = pool["agree_tags"] / pool["ev_tags"]
    print("log-odds shrunk to the pooled rate with 10 pseudo-tags (tag-level majority agreement; floor 0):")
    for key in sorted(k for k in stats if k in att_codes):
        d = stats[key]
        p = (d["agree_tags"] + 10 * p0) / (d["ev_tags"] + 10)
        print("  {:<24} tags {:>3} raw p {:.3f} shrunk p {:.3f} -> w {:.2f}".format(
            key, d["ev_tags"], d["agree_tags"] / max(d["ev_tags"], 1), p, max(0.0, math.log(p / (1 - p)))))
    print("(DEFENDING rows use the tag's own Team as the side; a share below 0.5 means Team names the attacker.)")
    # position inside the tag: does agreement concentrate in the middle?
    rel = []
    for t in m.tags:
        code = (t["code"] or "").strip().lower()
        s = tag_side(m, t)
        if s is None or code not in att_codes:
            continue
        a_, b_ = t["start_frame"], t["end_frame"]
        if b_ - a_ < 100:
            continue
        sf = strong[a_:b_]
        thirds = np.array_split(np.arange(b_ - a_), 3)
        rel.append([share(sf[i][sf[i] >= 0] == s) if (sf[i] >= 0).sum() >= 10 else np.nan for i in thirds])
    rel = np.array(rel, float)
    print("agreement by third of the tag (tags >=4 s): first {:.3f} middle {:.3f} last {:.3f} (n tags with data {})".format(
        np.nanmean(rel[:, 0]), np.nanmean(rel[:, 1]), np.nanmean(rel[:, 2]), int(np.isfinite(rel).all(axis=1).sum())))

    # ------------------------------------------------------------------------------------------
    h("Q8 Restart timing inside the SET PIECES windows")
    t_all = []
    sp_tags = [t for t in m.tags if (t["code"] or "").strip().lower() == evidence.SET_PIECE_CODE]
    touch_g = np.zeros(m.n_frames, bool)
    for (a, b), f in zip(periods, F):
        fr, _ = evidence.touch_events(f, P)
        touch_g[a + fr] = True
    still_g = np.zeros(m.n_frames, bool)
    ball_speed = np.full(m.n_frames, np.nan)
    for (a, b), f in zip(periods, F):
        still_g[a:b] = evidence.still_ball(f, P)
        ball_speed[a:b] = np.linalg.norm(f.ball_v, axis=1)
    carry_g = carry >= 0
    cand_defs = {"touch event": touch_g, "first carry frame": carry_g}
    res = {k: [] for k in ("touch event", "first carry frame", "ball reappears", "ball speed >2 m/s after still/hidden")}
    types = []
    lens, n_in_graded = [], 0
    for t in sp_tags:
        a_, b_ = t["start_frame"], t["end_frame"]
        types.append((t["tags"].get("Type") or "?").strip().lower())
        lens.append((b_ - a_) / fps)
        ing = graded[a_:b_].all()
        n_in_graded += ing
        for k, mask in cand_defs.items():
            idx = np.flatnonzero(mask[a_:b_])
            res[k].append((idx[0] / fps) if len(idx) else np.nan)
        vis = visible[a_:b_]
        dead = (~vis) | still_g[a_:b_]
        # ball reappears: first visible frame that follows a hidden frame inside the window
        k0 = np.flatnonzero(~vis)
        r = np.nan
        if len(k0):
            nxt = np.flatnonzero(vis[k0[0]:])
            r = ((k0[0] + nxt[0]) / fps) if len(nxt) else np.nan
        res["ball reappears"].append(r)
        sp_ok = np.nan_to_num(ball_speed[a_:b_]) > 2.0
        first = np.nan
        for i in np.flatnonzero(sp_ok):
            if i > 0 and dead[:i].any():
                first = i / fps
                break
        res["ball speed >2 m/s after still/hidden"].append(first)
    types = np.array(types)
    lens = np.array(lens)
    print("windows {} ({} fully inside graded periods); window length s: {}".format(
        len(sp_tags), n_in_graded, qs(lens, (5, 25, 50, 75, 95), "{:.1f}")))
    print("windows by type: " + ", ".join("{} {}".format(k, int((types == k).sum())) for k in sorted(set(types))))
    for k, v in res.items():
        v = np.array(v, float)
        print("candidate '{}': windows with candidate {} of {} ({:.2f}); offset from window start s: {}".format(
            k, int(np.isfinite(v).sum()), len(v), np.isfinite(v).mean(), qs(v, (5, 25, 50, 75, 95), "{:.1f}")))
        for ty in sorted(set(types)):
            vv = v[types == ty]
            print("     {:<22} with candidate {:>2}/{:<2} offset s: {}".format(
                ty, int(np.isfinite(vv).sum()), len(vv), qs(vv, (25, 50, 75), "{:.1f}")))
    ends = []
    v = np.array(res["touch event"], float)
    print("touch candidate distance to window end (s): " + qs(lens[np.isfinite(v)] - v[np.isfinite(v)], (5, 25, 50, 75, 95), "{:.1f}"))
    print("windows whose ball is hidden for the whole window: {:.2f}; hidden share of window frames: {:.2f}".format(
        np.mean([not visible[t["start_frame"]:t["end_frame"]].any() for t in sp_tags]),
        np.mean([1 - visible[t["start_frame"]:t["end_frame"]].mean() for t in sp_tags])))
    print("D share of no-tag-channel decoder inside windows before first touch event / after: " + "{:.2f} / {:.2f}".format(
        np.mean([share(codes_nt[t["start_frame"]:t["start_frame"] + max(1, int(res['touch event'][i] * fps))] == 2)
                 for i, t in enumerate(sp_tags) if np.isfinite(res['touch event'][i])]),
        np.mean([share(codes_nt[t["start_frame"] + int(res['touch event'][i] * fps):t["end_frame"]] == 2)
                 for i, t in enumerate(sp_tags) if np.isfinite(res['touch event'][i]) and
                 t["start_frame"] + int(res['touch event'][i] * fps) < t["end_frame"]])))

    # ------------------------------------------------------------------------------------------
    for name_, cand in res.items():
        v = np.array(cand, float)
        pre_d, post_d = [], []
        for i, t in enumerate(sp_tags):
            if not np.isfinite(v[i]):
                continue
            c = t["start_frame"] + int(v[i] * fps)
            if t["start_frame"] < c < t["end_frame"]:
                pre_d.append(share(codes_nt[t["start_frame"]:c] == 2))
                post_d.append(share(codes_nt[c:t["end_frame"]] == 2))
        print("decoder D share (no tag channels) before / after candidate '{}': {:.2f} / {:.2f} over {} windows".format(
            name_, np.mean(pre_d), np.mean(post_d), len(pre_d)))

    h("Q9 Is the ball speed usable?")
    spd = ball_speed[graded & visible]
    print("centred ball speed (m/s), graded visible frames: " + qs(spd, (1, 5, 25, 50, 75, 95, 99, 99.9)))
    print("share of visible graded frames with NaN centred velocity (touches a blackout): {:.3f}".format(
        share(~np.isfinite(ball_speed[graded & visible]))))
    print("share above 35 m/s {:.5f}, above 45 m/s {:.5f}".format(share(np.nan_to_num(spd) > 35), share(np.nan_to_num(spd) > 45)))
    dv = []
    for f in F:
        c = np.linalg.norm(f.v_after - f.v_before, axis=1)
        dv.append(c[f.visible])
    print("3-frame velocity change (m/s) on visible frames: " + qs(np.concatenate(dv), (50, 90, 99, 99.9), "{:.2f}"))
    print("share of visible frames with 3-frame velocity change >= 4 m/s: {:.4f}".format(share(np.concatenate(dv) >= 4.0)))
    n_still, dur_still, still_frames = 0, [], 0
    for (a, b) in periods:
        s_, e_ = features.runs(still_g[a:b])
        n_still += len(s_)
        dur_still += list((e_ - s_) / fps)
    print("still-ball runs (speed<{} for >={} s): count {} ; duration s: {}; share of graded frames {:.3f}".format(
        P.still_speed, P.still_s, n_still, qs(dur_still, (5, 25, 50, 75, 95), "{:.1f}"), share(still_g[graded])))
    for thr in (0.2, 0.5, 1.0, 1.5):
        tot, cnt_ = 0, 0
        for (a, b), f in zip(periods, F):
            with np.errstate(invalid="ignore"):
                mk = features.keep_long(f.still_speed < thr, int(round(P.still_s * fps)))
            tot += int(mk.sum())
            cnt_ += len(features.runs(mk)[0])
        print("   still threshold {:.1f} m/s: runs {} frames {:.3f} of graded".format(thr, cnt_, tot / graded.sum()))
    try:
        import games
        meta_path, raw_path = games.require(["atd_meta", "atd_raw"], m.name, None)
        n_third = 0
        with open(raw_path) as fh:
            for i, line in enumerate(fh):
                if i % 50 == 0:
                    n_third += len(line.rstrip("\n").split(":")[2].split(",")) >= 3
        print("ATD ball group carries a third field on {:.3f} of every 50th record".format(n_third / (i // 50 + 1)))
    except Exception as e:   # no data interpolated into the message
        print("ATD third-field check skipped: {}".format(type(e).__name__))


if __name__ == "__main__":
    main()
