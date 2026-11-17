"""One test per property of PLAN section 7 (see tests/PROPERTIES.md).

Failures on real data are reported as counts only; the synthetic matches are free to echo.
"""

import dataclasses
import itertools
import os

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

import synth
from baseline import write_submission
import possession_metrics as pm
from possession import config as cfg, decode, evidence, features, ingest, match as mt, params as prm
from possession.guards import merge_short_dead
from possession.params import Params
from possession.predict import predict

H, A, D, X = 0, 1, 2, 3
SLOW = settings(deadline=None, database=None, max_examples=12,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
FAST = settings(deadline=None, database=None, max_examples=200)
TAGGED = cfg.MatchConfig(tag_team=(("barcelona", cfg.HOME), ("atletic club", cfg.AWAY)))


@st.composite
def segment_lists(draw):
    labels, prev = [], None
    for _ in range(draw(st.integers(2, 6))):
        prev = draw(st.sampled_from([lab for lab in (H, A, D) if lab != prev]))
        labels.append(prev)
    return [(lab, draw(st.integers(synth.MIN_DEAD, 400)) if lab == D
             else draw(st.integers(synth.MIN_ALIVE, 500))) for lab in labels]


def tags_for(segments, rng_seed):
    """Attack-phase and set-piece tags laid over a synthetic match, some deliberately wrong."""
    rng = np.random.default_rng(rng_seed)
    n = sum(f for _, f in segments)
    out = []
    for _ in range(4):
        a = int(rng.integers(0, n - 100))
        out.append(synth.tag("PROGRESSION", a, a + int(rng.integers(50, 300)),
                             team=str(rng.choice(["Barcelona", "Atletic Club"]))))
    a = int(rng.integers(0, n - 150))
    out.append(synth.tag("SET PIECES", a, a + 125, kind="Throw-in"))
    return out


# --- 1 to 3: the file --------------------------------------------------------------------------

@given(segment_lists(), st.booleans(), st.integers(0, 3))
@SLOW
def test_p1_file_shape(tmp_path_factory, segments, dead_visible, seed):
    m, _ = synth.make_match(segments, dead_visible, seed)
    codes = predict(m)
    path = str(tmp_path_factory.mktemp("p1") / "s.csv")
    write_submission(path, mt.LETTERS[codes])
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, m.n_frames)
    assert reason is None, "validator rejected the synthetic file"
    assert set(table["label"]) <= {"H", "A", "D", "X"}


@given(segment_lists(), st.integers(0, 3), st.data())
@SLOW
def test_p2_x_only_on_ungraded_frames(segments, seed, data):
    m, _ = synth.make_match(segments, True, seed)
    n = m.n_frames
    a = data.draw(st.integers(0, n // 3))
    b = data.draw(st.integers(a + 200, n))
    mid = (a + b) // 2
    m = dataclasses.replace(m, periods=np.array([[a, mid], [mid + 50, b]], dtype=np.int64)
                            if mid + 100 < b else np.array([[a, b]], dtype=np.int64))
    codes = predict(m)
    graded = np.zeros(n, bool)
    for p, q in m.periods:
        graded[p:q] = True
    assert not (codes[~graded] != X).any(), "ungraded frame without X"
    assert not (codes[graded] == X).any(), "graded frame with X"


@given(segment_lists(), st.integers(0, 3), st.data())
@SLOW
def test_p3_forced_dead_block_is_dead(segments, seed, data):
    m, _ = synth.make_match(segments, True, seed)
    a = data.draw(st.integers(1, m.n_frames - 60))
    b = data.draw(st.integers(a + 25, min(m.n_frames, a + 400)))
    codes = predict(m, config=cfg.MatchConfig(forced_dead=((a, b),), forced_alive=(0,)))
    assert (codes[a:b] == D).all(), "forced block is not all D"
    assert codes[0] != D, "frame 0 must be alive"


def test_p3_bar_ath_block_bounds():
    c = cfg.bar_ath()
    assert c.forced_dead == ((110808, 113742),)
    assert c.forced_alive == (11185,)
    mask = c.allowed(110000, 114000)
    assert not mask[110808 - 110000:113742 - 110000, :2].any()
    assert mask[110808 - 110000:113742 - 110000, 2].all()
    assert mask[113742 - 110000].all()


# --- 4, 5: symmetry ----------------------------------------------------------------------------

@given(segment_lists(), st.booleans(), st.integers(0, 3))
@SLOW
def test_p4_team_swap(segments, dead_visible, seed):
    m, _ = synth.make_match(segments, dead_visible, seed, tags=tags_for(segments, seed))
    swapped = synth.swap_teams(m)
    forced = cfg.MatchConfig(tag_team=TAGGED.tag_team, forced_dead=((100, 160),))
    base, other = predict(m, config=forced), predict(swapped, config=forced)
    expected = np.where(base == H, A, np.where(base == A, H, base))
    assert np.array_equal(other, expected), "swap did not swap H and A only"


@given(segment_lists(), st.booleans(), st.integers(0, 3))
@SLOW
def test_p5_reflection(segments, dead_visible, seed):
    m, _ = synth.make_match(segments, dead_visible, seed, tags=tags_for(segments, seed))
    assert np.array_equal(predict(m, config=TAGGED), predict(synth.reflect(m), config=TAGGED))


# --- 6: the decoder ----------------------------------------------------------------------------

@st.composite
def decode_cases(draw):
    n = draw(st.integers(1, 8))
    score = np.array(draw(st.lists(st.lists(st.floats(-10, 10), min_size=3, max_size=3),
                                   min_size=n, max_size=n)))
    off = st.floats(0, 6)
    c = [[0.0, draw(off), draw(off)], [draw(off), 0.0, draw(off)], [draw(off), draw(off), 0.0]]
    allowed = np.array(draw(st.lists(st.lists(st.booleans(), min_size=3, max_size=3),
                                     min_size=n, max_size=n)))
    for row in allowed:
        if not row.any():
            row[draw(st.integers(0, 2))] = True
    return score, c, allowed


@given(decode_cases())
@FAST
def test_p6_viterbi_matches_enumeration(case):
    score, c, allowed = case

    def total(path):
        s = sum(score[t, st_] for t, st_ in enumerate(path))
        return s - sum(c[p][q] for p, q in zip(path[:-1], path[1:]))

    best = max(total(p) for p in itertools.product(range(3), repeat=len(score))
               if all(allowed[t, st_] for t, st_ in enumerate(p)))
    path = decode.viterbi(score, c, allowed)
    assert all(allowed[t, int(st_)] for t, st_ in enumerate(path)), "path breaks a constraint"
    assert np.isclose(total([int(s) for s in path]), best), "path is not optimal"


# --- 7: no velocity across a blackout ----------------------------------------------------------

@given(st.lists(st.booleans(), min_size=12, max_size=90), st.integers(0, 2 ** 31 - 1))
@FAST
def test_p7_no_velocity_across_blackout(visible, seed):
    rng = np.random.default_rng(seed)
    n = len(visible)
    ball = rng.uniform(0, 100, (n, 2))
    ball[~np.array(visible)] = np.nan
    players = np.full((n, 2, 2), np.nan)
    p = Params()
    f = features.compute(ball, players, np.full((n, 2), np.nan), np.array([0, 1]),
                         (104.0, 68.0), 25, p)

    # Gaps of more than interp_max_gap frames stay NaN; shorter inner ones are filled.
    gap_runs = list(zip(*features.runs(np.isnan(ball).any(axis=1))))
    for a, b in gap_runs:
        stays_nan = b - a > p.interp_max_gap or a == 0 or b == n
        assert np.isnan(f.ball[a:b]).all() == stays_nan, "gap fill is wrong"

    finite = ~np.isnan(f.ball).any(axis=1)
    for name, v, lo, hi in (("ball_v", f.ball_v, p.vel_half, p.vel_half),
                            ("before", f.v_before, p.touch_span, 0),
                            ("after", f.v_after, 0, p.touch_span),
                            ("still", f.still_speed, p.still_half, p.still_half)):
        for t in np.flatnonzero(~np.isnan(v).reshape(n, -1).any(axis=1)):
            window = finite[t - lo:t + hi + 1]
            assert t - lo >= 0 and t + hi < n and window.all(), name + " velocity crosses a blackout"

    # Moving everything after the first long blackout changes no velocity: nothing reads across it.
    long_gaps = [(a, b) for a, b in gap_runs if b - a > p.interp_max_gap and 0 < a and b < n]
    if long_gaps:
        moved = ball.copy()
        moved[long_gaps[0][1]:] += 50.0
        g = features.compute(moved, players, np.full((n, 2), np.nan), np.array([0, 1]),
                             (104.0, 68.0), 25, p)
        for name in ("ball_v", "v_before", "v_after", "still_speed"):
            assert np.allclose(getattr(f, name), getattr(g, name), equal_nan=True), name


# --- 8: recovery -------------------------------------------------------------------------------

@given(segment_lists(), st.booleans(), st.integers(0, 3))
@SLOW
def test_p8_synthetic_recovery(segments, dead_visible, seed):
    m, truth = synth.make_match(segments, dead_visible, seed)
    result = pm.evaluate(truth, predict(m))
    assert result["spell_f1_mean"] >= 0.95, "spellF1 below 0.95: {:.3f}".format(
        result["spell_f1_mean"])


# --- 9: determinism ----------------------------------------------------------------------------

@given(segment_lists(), st.booleans(), st.integers(0, 3))
@SLOW
def test_p9_same_bytes(tmp_path_factory, segments, dead_visible, seed):
    m, _ = synth.make_match(segments, dead_visible, seed, tags=tags_for(segments, seed))
    folder = tmp_path_factory.mktemp("p9")
    blobs = []
    for i in (0, 1):
        path = str(folder / "{}.csv".format(i))
        write_submission(path, mt.LETTERS[predict(m, config=TAGGED)])
        with open(path, "rb") as handle:
            blobs.append(handle.read())
    assert blobs[0] == blobs[1], "two runs wrote different bytes"


# --- 10: restart tags stay where they were laid --------------------------------------------------

@given(segment_lists(), st.booleans(), st.integers(0, 3), st.booleans(), st.booleans())
@SLOW
def test_p10_restart_channels_stay_inside_their_tags(segments, dead_visible, seed, launch, after_sp):
    n = sum(f for _, f in segments)
    assume(n > 700)
    tags = [synth.tag("GOALS", 100, n - 200, team="Barcelona"),
            synth.tag("SET PIECES", 120, 120 + 125, kind="Throw-in"),
            synth.tag("SET PIECES", n - 250, n - 200 + 125, kind="Kick Off (After Goal)")]
    m, _ = synth.make_match(segments, dead_visible, seed, tags)
    p = Params(setpiece_launch=launch, goal_dead_w=1.0, goal_after_setpiece=after_sp)
    f = features.compute(m.ball, m.players, m.speed, m.team, m.pitch, m.fps, p)
    out = evidence.channels(f, evidence.localize_tags(m.tags, cfg.EMPTY, 0, n), p)
    goal, sp = out["goal"], out["setpiece"]
    assert not goal[:, :2].any(), "the goal rule put weight on H or A"
    assert not goal[:100].any() and not goal[n - 200 + 125:].any(), "goal weight outside the stretch"
    outside = np.ones(n, bool)
    for a, b in ((120, 245), (n - 250, n - 75)):
        outside[a:b] = False
    assert not sp[outside].any(), "set-piece weight outside every set-piece window"


# --- team-shape channel ------------------------------------------------------------------------

@given(segment_lists(), st.booleans(), st.integers(0, 3), st.floats(0.0, 0.2), st.floats(0.0, 0.2))
@SLOW
def test_formation_channel_symmetry(segments, dead_visible, seed, w_vis, w_blk):
    m, _ = synth.make_match(segments, dead_visible, seed)
    p = Params(formation_w_vis=w_vis, formation_w_blk=w_blk)

    def run(match):
        f = features.compute(match.ball, match.players, match.speed, match.team, match.pitch,
                             match.fps, p)
        return f.shape, evidence.channels(f, [], p)["formation"]

    shape, ch = run(m)
    s_shape, s_ch = run(synth.swap_teams(m))
    r_shape, r_ch = run(synth.reflect(m))
    assert np.array_equal(s_shape, -shape) and np.array_equal(s_ch[:, [1, 0, 2]], ch), "swap"
    assert np.array_equal(r_shape, shape) and np.array_equal(r_ch, ch), "reflection"
    assert not ch[:, D].any() and (ch[:, :2].max(axis=1) == 0).all(), "favoured team or D moved"


# --- supporting checks -------------------------------------------------------------------------

def test_params_have_sources():
    assert set(prm.SOURCES) == {f.name for f in dataclasses.fields(Params)}
    assert all(prm.SOURCES.values())


def test_absent_inputs_contribute_zero():
    n = 200
    rng = np.random.default_rng(0)
    m = mt.Match(
        name="NOBALL", fps=25, pitch=synth.PITCH, ball=np.full((n, 2), np.nan, np.float32),
        players=rng.uniform(0, 60, (n, 4, 2)).astype(np.float32),
        speed=np.full((n, 4), np.nan, np.float32), team=np.array([0, 0, 1, 1], np.int8),
        periods=np.array([[0, n]], dtype=np.int64))
    f = features.compute(m.ball, m.players, m.speed, m.team, m.pitch, m.fps, Params())
    # Players are present here, so the team-shape channel (which reads only players) may fire.
    parts = evidence.channels(f, [], Params())
    assert not any(v.any() for k, v in parts.items() if k != "formation"), \
        "a channel fired with no ball, no speed and no tags"
    assert predict(m).shape == (n,)


def test_guard_merges_short_dead_runs():
    path = np.array([H] * 30 + [D] * 4 + [A] * 30 + [D] * 40 + [H] * 5, dtype=np.int8)
    allowed = np.ones((len(path), 3), bool)
    out, merged = merge_short_dead(path, 25, allowed)
    assert merged == 1 and (out[30:34] != D).all() and (out[64:104] == D).all()
    allowed[31, :2] = False                      # a forced-dead frame blocks the merge
    out, merged = merge_short_dead(path, 25, allowed)
    assert merged == 0 and np.array_equal(out, path)


# --- real data, only when the cache and a submission exist -------------------------------------

REAL_CSV = os.path.join(ingest.ROOT, "evaluation", "results", "Verynicestyleturbo.csv")
needs_real = pytest.mark.skipif(
    not (os.path.isfile(REAL_CSV) and os.path.isfile(mt.path_for("BAR-ATH"))),
    reason="needs the BAR-ATH cache and a written submission")


@needs_real
def test_real_file_properties():
    codes = pm.label_codes(pm.read_label_csv(REAL_CSV))
    n_rows = len(codes)
    m = mt.load("BAR-ATH")
    graded = np.zeros(n_rows, bool)
    for a, b in m.periods:
        graded[a:b] = True
    bad_x = int((codes[~graded] != X).sum()) + int((codes[graded] == X).sum())
    bad_block = int((codes[110808:113742] != D).sum())
    assert n_rows == 199319, "row count"
    assert bad_x == 0, "frames with X misplaced"
    assert bad_block == 0, "frames of the pre-kickoff block that are not D"
    assert int(codes[11185]) != D, "first frame is not alive"
