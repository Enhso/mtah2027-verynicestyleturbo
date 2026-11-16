"""Apply a per-frame ball-visibility mask to an external match's ball (blackout transplant)."""

import dataclasses

import numpy as np


def apply_mask(match, masks):
    """Copy of match whose ball is NaN where the mask is False.

    masks: one bool array per half (True = ball visible), same order as match.periods. A mask
    shorter than its half is tiled to length, a longer one is cropped to its start.
    """
    assert len(masks) == len(match.periods), "need one mask per period"
    ball = match.ball.copy()
    for (a, b), m in zip(match.periods.tolist(), masks):
        m = np.asarray(m, bool)
        assert m.size > 0, "empty mask"
        keep = np.resize(m, b - a)  # np.resize repeats, then crops
        ball[a:b][~keep] = np.nan
    return dataclasses.replace(match, ball=ball)


def test_apply_mask():
    from possession.match import Match
    n = 20
    ball = np.ones((n, 2), np.float32)
    m = Match("t", 25, (105, 68), ball, np.zeros((n, 1, 2), np.float32), np.zeros((n, 1), np.float32),
              np.zeros(1, np.int8), np.array([[0, 8], [12, 20]]))
    out = apply_mask(m, [np.array([1, 0, 0], bool), np.array([1] * 3 + [0] * 20, bool)])
    vis = np.isfinite(out.ball[:, 0])
    assert vis[:8].tolist() == [1, 0, 0, 1, 0, 0, 1, 0]   # tiled
    assert vis[8:12].all()                                # between halves untouched
    assert vis[12:].tolist() == [1, 1, 1] + [0] * 5       # cropped
    assert np.isfinite(m.ball).all()                      # input not modified


if __name__ == "__main__":
    test_apply_mask()
    print("ok")
