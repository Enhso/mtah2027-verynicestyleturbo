# restarts-progress (set-piece launch candidate and goal-to-kick-off rule)

Difficulty: medium-high. The evidence channel is under a byte-exact hash, and the proxy test needs a
planted-tag simulation on Metrica games that have no tagging.

Run from the worktree root: `PY=.venv/bin/python`,
`MTAH_DATA_DIR=$MTAH_DATA_DIR`. No public-window smoke was run.

## Status
- [x] step 1 (commits 97568c5, 727f021): both rules behind parameters, off by default; sha256 still
      e0913342...f1f88, 15 tests pass
- [x] step 2: planted-tag proxy, `python -m calibration.restarts_proxy [share ...]`
- [x] step 3: BAR-ATH change table, `python -m calibration.restarts_bar_ath`
- [x] step 4: adopted launch rule and the goal rule anchored after the last set-piece window; the goal
      rule as first specified (D from the GOALS tag start) is not adopted. New sha256
      6ddeef442c942ec314f3ebf3fe30112b2adc8d764f80233dc578a4239b9675d7

## What the code does
- `setpiece_launch`: restart candidate of a SET PIECES window = first frame with centred ball speed
  above `setpiece_launch_speed` (2 m/s) after a hidden or still frame inside the window; else the first
  visible frame after a hidden one; else D for the first `setpiece_fallback_s` (6.4 s) and nothing
  after. D before the candidate, alive after it, `setpiece_w` 3 nats each side, spread. False keeps the
  v1 first-touch rule.
- `goal_dead_w` (nats per frame, 0 = off): D from a goal to the restart candidate of the next
  kick-off-after-goal window (to that window's start when it has no candidate). It is evidence, not a
  hard constraint: the law of the game fixes the state (dead until the kick-off) but not the frame the
  stretch starts on, because the start comes from a tag whose boundaries are second-accurate and, as
  shown below, are not the goal instant. A hard constraint would have needed `config.py` (outside this
  task's files); weights 0.3 to 10 give the same result on BAR-ATH, so 10 already acts as one.
- `goal_after_setpiece`: the stretch starts where the last other set-piece window overlapping the GOALS
  tag ends (a goal whose tag overlaps none is skipped); False starts it at the tag start.
- `localize_tags` appends a `kick off after goal` entry next to a SET PIECES tag whose Type contains
  "after goal" (case-insensitive), so the channels can find the kick-off windows without a new tuple
  shape.

## Finding: the GOALS tag does not start at the goal
BAR-ATH has 4 GOALS tags, each one to a few minutes long, and 4 short kick-off-after-goal windows;
each GOALS tag ends just after its kick-off window starts. In 3 of 4 tags a free kick, corner or
throw-in window starts a few seconds after the tag start and is followed by about 20 s of visible
live play (the fourth tag overlaps a free-kick window from its start). So a goal tag opens with the set piece of the scoring sequence and about 20 s of live play,
then the goal, then the dead time to the kick-off. D from the tag start therefore forces D over
live play and over a real restart in 3 of 4 goals.

## Step 2: planted-tag proxy (planted, not measured on tags)
Method (`calibration/restarts_proxy.py`): on Metrica games 1 to 3, variants `atd` and `atd+mask`, a
seeded share of each game's true restarts (labels: a D run followed by an alive frame; 70, 73, 79 per
game) gets a 35 s SET PIECES window that starts at the restart minus an offset resampled from BAR-ATH's
own launch offsets (n = 48, q5/25/50/75/95 = 2.3/4.6/6.6/9.3/18.8 s). Seeds 0 to 4; windows are clipped
to the period; every window contains its restart, which real tags may not.

Share: 59 tagged windows of about 100 restarts, so 0.59. Basis: the v1 decode of BAR-ATH has 98 D runs
inside periods (label-free), 59 / 98 = 0.60; the Metrica labels have 70 to 79 restarts per game, so
0.59 gives 41 to 47 windows per game. Sensitivity at 0.80 (59 windows, the same count as this match) and
1.00.

J = mean spellF1 over games 1 to 3, mean over 5 seeds (sd of J across seeds in brackets), share 0.59:

| rule | w | J | atd | atd+mask |
|---|---|---|---|---|
| no tags | - | 0.5105 | 0.5748 | 0.4462 |
| v1 touch | 1.5 | 0.5099 (0.0020) | 0.5728 | 0.4471 |
| v1 touch | 3 | 0.5089 (0.0024) | 0.5717 | 0.4460 |
| v1 touch | 6 | 0.5070 (0.0023) | 0.5702 | 0.4439 |
| v1 touch | 12 | 0.5033 (0.0028) | 0.5659 | 0.4407 |
| launch | 1.5 | 0.5113 (0.0009) | 0.5755 | 0.4472 |
| launch | 3 | 0.5114 (0.0011) | 0.5753 | 0.4474 |
| launch | 6 | 0.5105 (0.0015) | 0.5741 | 0.4468 |
| launch | 12 | 0.5098 (0.0015) | 0.5726 | 0.4470 |

Launch minus touch (J): +0.0014, +0.0025, +0.0034, +0.0066 at w 1.5, 3, 6, 12; launch is at or above
touch in 5 of 5 seeds at every w; per game and variant, 6 of 6 at w 3 to 12 (5 of 6 at w 1.5). At share
0.80 the differences are +0.0028, +0.0039, +0.0052, +0.0090 and at 1.00 +0.0032, +0.0051, +0.0067,
+0.0110 (5 of 5 seeds each). Per game (atd+mask) the launch rule is within 0.002 of no tags. Reading:
the touch rule loses to no tags at every weight (about -0.001 at w 3, more at larger w); the launch
rule is level with no tags (+0.001 at w 3, inside the 0.003 noise floor), best at w 3 and 1.5. On the
proxy the launch candidate exists in 94% of windows (80% on BAR-ATH, where the ball is hidden more)
and lands a median 0.1 s from the true restart. The Metrica decoder already finds most restarts
(D to alive transitions), so tags have little to add; the test shows "does not hurt", not "helps".
`setpiece_w` stays 3.0 (best of the grid for launch, flat 1.5 to 6).

## Step 3: BAR-ATH (aggregates; v1 is the reference)
| setting | frames changed | in set-piece windows / goal stretches / elsewhere | possessions | set-piece gate | kick-off gate (D before, s) |
|---|---|---|---|---|---|
| v1 | 0 | - | 326 | 49 of 59 (83%) | 43, 19 (no restart), 131, 221: OUT |
| launch | 0.38% | 100% / 14% / 0 frames | 326 (+0) | 50 of 59 (85%) | 43, 19 (no restart), 131, 221: OUT |
| goal at tag start, w 1 | 1.11% | 85% / 100% / 0 | 318 (-8) | 47 of 59 (80%, 79.7%: OUT) | 168, 217 (no restart), 131, 253: ok |
| goal at tag start, w 10 | 1.13% | 86% / 100% / 0 | 318 (-8) | 47 of 59 | same |
| goal after set piece, w 0.3 | 0.10% | 0% / 100% / 0 | 324 (-2) | 49 of 59 | 122, 163 (no restart), 131, 221: ok |
| goal after set piece, w 1 | 0.12% | 17% / 100% / 0 | 324 (-2) | 49 of 59 | 123, 163 (no restart), 131, 221: ok |
| goal after set piece, w 10 | 0.13% | 26% / 100% / 0 | 324 (-2) | 49 of 59 | 123, 163 (no restart), 131, 221: ok |
| launch + goal after set piece, w 1 (adopted) | 0.47% | 80% / 31% / 0 | 324 (-2) | 50 of 59 (85%) | 122, 163 (no restart), 131, 221: ok |

D share graded: 0.421 for v1, 0.422 adopted. "Goal stretch" is the GOALS tag start to the end of
the next kick-off window (10.6% of graded frames). Goal at the tag start loses the set-piece gate because
the stretches contain a corner and a throw-in window whose restarts it turns to D.

## Decisions
- Launch rule: adopted (`setpiece_launch` True). Proxy: at or above the touch rule in both variants,
  5 of 5 seeds; BAR-ATH: set-piece gate 49 to 50 of 59, kick-off gate unchanged, no other changes.
- Goal rule as specified (D from the GOALS tag start): not adopted. The kick-off gate passes and nothing
  changes outside the stretches, but the tag start is before the goal in 3 of 4 goals (finding above),
  the set-piece gate falls to 47 of 59 (below 80%), and 8 possessions disappear.
- Goal rule anchored after the last overlapped set-piece window: adopted at w 1
  (`goal_dead_w` 1.0, `goal_after_setpiece` True). The kick-off gate passes (122, 163, 131, 221 s), the
  set-piece gate is not worse, no frame changes outside the stretches (0 frames), 0.12% of frames
  change. Caveat: the anchor was found by looking at the 4 stretches the gate is graded on, so the
  gate pass is not independent evidence; the structural argument (a tag opens with its set piece) is
  what carries it. It touches nothing in a match without GOALS tags (Metrica, DFL).
- Not checked: the "no restart" on the second kick-off (no D to alive transition inside its window;
  the window has no launch candidate, so D runs over the whole 5 s window). The gate still passes by
  its 20 s rule.
