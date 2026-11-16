"""PLAN section 7 release gates. Printed for a frozen version; a gate outside its range is a reason
to investigate, never a target to tune toward."""

import glob
import os
import sys

import numpy as np

from . import ingest

sys.path.insert(0, os.path.join(ingest.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
SET_PIECE_TAG = "set pieces"
RESTART_SLACK_S = 3.0          # a D to alive transition may follow the tag window by this long
KICKOFF_DEAD_MIN_S = 20.0
SMOKE_SPELL_F1_MIN = 0.42
SMOKE_LATENCY_MAX = 8
SMOKE_TAU = 25                 # transition tolerance (frames) whose median latency is gated


def _d_to_alive(codes, periods):
    """Frames where D gives way to H or A inside a period."""
    out = []
    for a, b in periods:
        c = codes[a:b]
        out.extend((np.flatnonzero((c[1:] != D) & (c[:-1] == D)) + 1 + a).tolist())
    return np.array(out, dtype=np.int64)


def _dead_run_before(codes, frame):
    n = 0
    while frame - 1 - n >= 0 and codes[frame - 1 - n] == D:
        n += 1
    return n


def compute(match, codes, validator_ok, smoke=True):
    """[(gate, value text, range text, verdict)] for one submission."""
    fps = match.fps
    periods = [(int(a), int(b)) for a, b in match.periods]
    graded = np.zeros(match.n_frames, bool)
    for a, b in periods:
        graded[a:b] = True
    alive = (codes == H) | (codes == A)
    spells = pm.possession_spells(codes, periods)
    minutes = graded.sum() / fps / 60.0
    rows = []

    def add(name, value, lo, hi, fmt="{:.3f}"):
        ok = lo <= value <= hi
        rows.append((name, fmt.format(value), "{} to {}".format(lo, hi), "ok" if ok else "OUT"))

    rows.append(("validate_submission.py", "OK" if validator_ok else "FAIL", "OK",
                 "ok" if validator_ok else "OUT"))
    add("share of D on graded frames", float((codes[graded] == D).mean()), 0.30, 0.55)
    add("possessions per graded minute", len(spells) / minutes, 1.5, 5)
    add("mean possession (s)", float(alive.sum()) / fps / max(len(spells), 1), 6, 20, "{:.1f}")
    add("home share of alive frames", float((codes == H).sum()) / max(int(alive.sum()), 1),
        0.45, 0.75)

    windows = [t for t in match.tags if (t["code"] or "").strip().lower() == SET_PIECE_TAG]
    trans = _d_to_alive(codes, periods)
    slack = int(RESTART_SLACK_S * fps)
    hit = sum(bool(((trans >= t["start_frame"]) & (trans < t["end_frame"] + slack)).any())
              for t in windows)
    rows.append(("set-piece windows with D to alive within {:.0f} s".format(RESTART_SLACK_S),
                 "{} of {} ({:.0f}%)".format(hit, len(windows), 100.0 * hit / max(len(windows), 1)),
                 "80% or more", "ok" if windows and hit >= 0.8 * len(windows) else "OUT"))

    after_goal = [t for t in windows if "after goal" in (t["tags"].get("Type") or "").lower()]
    # Per kick-off: the D run that ends at the first D to alive transition inside the window plus
    # slack, or at the window end when the decode found no restart there.
    lengths, found = [], []
    for t in after_goal:
        near = trans[(trans >= t["start_frame"]) & (trans < t["end_frame"] + slack)]
        found.append(len(near) > 0)
        lengths.append(_dead_run_before(codes, int(near[0]) if len(near) else t["end_frame"]) / fps)
    rows.append(("D before the {} kick-offs after a goal".format(len(after_goal)),
                 ", ".join("{:.0f} s{}".format(n, "" if f else " (no restart)")
                           for n, f in zip(lengths, found)) or "none tagged",
                 "{:.0f} s or more each".format(KICKOFF_DEAD_MIN_S),
                 "ok" if lengths and min(lengths) >= KICKOFF_DEAD_MIN_S else "OUT"))

    if smoke:
        gt_path = glob.glob(os.path.join(ingest.ROOT, "evaluation", "gt",
                                         "*_possession_gt_public.csv"))[0]
        gt = pm.label_codes(pm.read_label_csv(gt_path))
        result = pm.evaluate(gt, codes)
        spell = result["spell_f1_mean"]
        lat = result["transitions"][SMOKE_TAU]["median_latency"]
        ok = spell > SMOKE_SPELL_F1_MIN and lat is not None and abs(lat) <= SMOKE_LATENCY_MAX
        rows.append(("public-window smoke (once per frozen version)",
                     "spellF1 {:.3f}, median latency {} frames".format(
                         spell, "n/a" if lat is None else "{:.1f}".format(lat)),
                     "spellF1 > {}, latency <= {}".format(SMOKE_SPELL_F1_MIN, SMOKE_LATENCY_MAX),
                     "ok" if ok else "OUT"))
    else:
        rows.append(("public-window smoke", "not run", "-", "skipped"))
    rows.append(("external harness spellF1 (python -m calibration.report)", "notes/calibration-progress.md s6", "recorded", "-"))
    return rows


def format_rows(rows):
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
