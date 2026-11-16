# calibration-progress (Phase 3)

Difficulty: hard. Evidence weights and transition costs trade off on one scale, the proxy (Metrica)
is easier than this match so the objective itself has to be built and justified, and only three
proxy matches exist, so over-fitting is the main risk rather than compute.

Branch: worktree-agent-a4c68f2d217e26b43. Decision trail: `notes/calibration-log.tsv`.
Python: `.venv/bin/python`, run from the worktree root.

## Status (checkpoint 2)
- [x] 1 run time and search budget
- [x] 2 proxy objective (Metrica variants `atd` and `atd+mask`)
- [x] 3 control definition (both), two-pass coordinate descent (`calibration/results/full.json`)
- [x] 4 quantile transfer (`calibration/results/transfer.json`)
- [x] 5 leave-one-game-out (section 6 below)
- [x] 6 params.py written with sources, tests pass, v1 gates printed (section 7). Not frozen.
- [ ] not done: masked re-tune of the blackout channels as a separate step (the objective already
      includes the masked variant), DFL-with-mask scores, a second descent from a different start.

## 1. Run time
`predict` takes 1.2 to 1.3 s per external match (features 0.5 to 0.7 s, Viterbi 0.2 s). With
features cached (`calibration/common.py: decode`, same codes as `predict`) one evaluation is about
0.4 s per match, so a full two-pass descent over 25 parameters on 6 match-variants runs in minutes
on 7 workers. Compute is not the limit; three matches are.

## 2. Proxy objective
`python -m calibration.labelfree_compare` (quantiles only). Clean Metrica already matches this
match on every blackout statistic (missing share 0.32 to 0.41 vs 0.407, gap count, gap length,
visible-run length), on median player speed (visible and missing), on the opponent margin and on
nearest-team flips per visible minute. It differs in three label-free ways:

| statistic | this match | Metrica clean | DFL |
|---|---|---|---|
| ball to nearest player, q25/q50 (m) | 0.89 / 1.91 | 0.10 / 0.28 (games 1, 2), 0.17 / 1.27 (game 3) | 0.95 / 2.6 |
| 3-frame ball velocity change, q95/q99 (m/s) | 2.75 / 4.57 | 7.6 to 8.1 / 18 to 21 | 5.5 to 7.3 / 16 |
| tracked players per team, q5/q25 | 6 to 7 / 10 | 11 / 11 | 11 / 11 |

The Metrica sample ball is glued to the player on the ball and kinks sharply; the ATD ball is a
smoothed independent track, and slots drop out. `calibration/variants.py` builds:
- `atd`: Gaussian low-pass on the ball (sigma 6 frames), a slow ball offset (0.7 m per axis,
  correlation about 1 s), 0.02 m jitter, and player slots dropped so the per-team tracked count
  follows this match's count series. The three knobs were set by `python -m calibration.variants
  fit` to match the velocity-change q99 (4.1 to 4.6 vs 4.57) and the nearest-distance q25/q50
  (0.81 to 0.97 / 1.5 to 1.9 vs 0.89 / 1.91). No label of this match was read.
- `+mask`: this match's per-half visibility mask OR-ed on (harness.mask). Missing share rises to
  0.57 to 0.64, above this match's 0.407, and P(dead | missing) falls from 0.78 to about 0.5.

Baseline board, mean spellF1 over the three games (`python -m calibration.baselines`):

| variant | raw | smooth25 | smooth50 | smooth125 | v0 params |
|---|---|---|---|---|---|
| clean | 0.089 | 0.360 | 0.417 | 0.308 | 0.619 |
| atd | 0.079 | 0.287 | 0.348 | 0.283 | 0.498 |
| clean+mask | 0.078 | 0.280 | 0.320 | 0.279 | 0.466 |
| atd+mask | 0.068 | 0.230 | 0.269 | 0.246 | 0.370 |
| this match, published | 0.049 | 0.200 | 0.243 | 0.224 | - |

Raw macroF1: atd 0.70 to 0.74, atd+mask 0.56 to 0.58, published 0.653. Qualitatively this match
sits between `atd` and `atd+mask` on frame-level D behaviour and at or below `atd+mask` on spells.
Decision: objective J = mean spellF1 over the three games on both `atd` and `atd+mask` (six
match-variants), macroF1 as tie-break. The two bracket this match; `atd` alone would teach
"blackout means dead" too strongly, `atd+mask` alone has too many blackouts. No knob was fitted to
the published row.

What neither variant reproduces: dead ball with the ball visible and moving. Label-free hints say
this match has more of it (ball within 0.54 m of a line at q5 vs 1.4 to 1.9 on Metrica, off-pitch
share of visible frames 0.042 vs 0.007 to 0.027; the public-window semantics say the ball moves at
a median 2.6 m/s on GT D frames). So the weights of the still-ball, off-pitch and low-activity
channels on visible-ball frames are weakly identified by Metrica.

DFL: the export's speed field is km/h (median player speed 5 m/s as exported); `variants.build`
divides DFL speed by 3.6. DFL is reported, not optimised.

## 3. Findings so far
`python -m calibration.diagnose atd,atd+mask` on v0 params:
- v0 finds about 80 turnovers per game against 205 to 248 true ones. With true D and predicted
  teams spellF1 is 0.50 to 0.59 (atd); with predicted D and true teams 0.79 to 0.80. Team/turnover
  error is the larger loss.
- Predicted restarts lag the true ones by a median 27 to 32 frames (stops: 1 to 8 frames).
- Blackouts on `atd`: 5 to 30 s gaps are mostly dead in 47 to 58 of 57 to 72; over 30 s, 14 to 21
  of 19 to 27; under 1 s mostly alive.

`python -m calibration.control_study atd` (signal quality before decoding): the carry rule's
velocity test (`carry_dv` 2 m/s) makes it late. With it, 59% of true possession starts get a carry
of the new team within 3 s, median 27 frames after the start; without it 86%, median 2 frames,
precision 0.92 against 0.98. Touch events at 2 m/s within 2.5 m: 5 to 6 per possession, precision
0.87, median 5 to 7 frames after the start.

## 4. Control definition and descent
Objective J (six Metrica match-variants). Each definition was tuned over its own thresholds and
weights from the v0 values (`python -m calibration.run_search control`):

| definition | J | atd | atd+mask | macroF1 |
|---|---|---|---|---|
| v0 (both, first values) | 0.434 | 0.498 | 0.370 | 0.712 |
| carry only | 0.484 | 0.547 | 0.421 | 0.736 |
| touch only | 0.454 | 0.513 | 0.395 | 0.726 |
| both | 0.502 | 0.568 | 0.437 | 0.747 |

Chosen: both. The margin over carry only (0.018) is under the 0.04 noise floor but has the same
sign on both variants and on macroF1.

Full two-pass descent from there over 25 parameters (`run_search full`): J 0.521 (atd 0.584,
atd+mask 0.458, macroF1 0.757). Values that moved: carry_radius 2.0 to 3.0, carry_margin 1.0 to
0.5, carry_dv 2 to 8, touch_dv 4 to 1.5, low_w 0.1 to 0.05, min_dead_frames 25 to 75, and the
restart anchored at the end of the dead-looking stretch. Every weight and cost with a real slope
peaked at its first value (carry_w, touch_w, restart_w, stop_cost, restart_cost, exit_w, low_s,
long_gap_s 30 s). Flat grids, left at first value: long_gap_w, still_w, offpitch_w, touch_angle.
turnover_cost stayed 3.0 as the scale anchor, so the tag weights keep their meaning.
carry_radius 3.0, touch_dv 1.5 and min_dead_frames 75 sit on the edge of their grids.

Tested and not adopted (gain under 0.003 or negative): long-gap threshold of 2, 5 or 10 s (0.511
to 0.513 against 0.521 at 30 s), exit rule extended to a ball that reappears near a line
(0.518 to 0.520 against 0.521), a moving-ball alive channel (0.514 to 0.517), low_speed 1.2
(0.506) and 1.0 (0.492). These differences are all inside the noise floor, so Metrica does not
separate them; they were left at the simpler setting.

Code changes kept (possession/evidence.py): a control definition with zero weight no longer feeds
restart and latch; the restart evidence sits on the first frame after the dead-looking stretch
(the strike), with the team still named by the first control event within 3 s. The experimental
parameters were removed.

## 5. Quantile transfer (`python -m calibration.transfer calibration/results/tuned_proxy.json`)
Proxy = three Metrica games, `atd` variant, pooled, visible-ball frames.

| threshold | feature | Metrica value | quantile | this match |
|---|---|---|---|---|
| carry_radius | ball to nearest player (m) | 3.0 | 0.690 | 3.37 |
| touch_radius | ball to nearest player (m) | 2.5 | 0.631 | 2.80 |
| carry_margin | nearest opponent margin (m) | 0.5 | 0.105 | 0.61 |
| carry_dv | ball minus player velocity (m/s) | 8.0 | 0.726 | 7.67 |
| touch_dv | 3-frame ball velocity change (m/s) | 1.5 | 0.756 | 1.43 |
| touch_angle | ball turning angle (deg) | 35 | 0.977 | 28.93 |
| low_speed | median player speed (m/s) | 1.5 | 0.324 | 1.39 |
| still_speed | 10-frame ball speed (m/s) | 0.5 | 0.023 | 0.18, not applied |

still_speed stays at 0.5: the proxy ball is missing when dead, so its low quantiles say nothing
about a dead ball that stays tracked. low_speed 1.39 is at about the 30th percentile of this
match's visible frames; the label-free worker proposes 1.21 (q20). Metrica prefers the higher
value by 0.015, inside noise, so the lead can take either.

## 6. Scores before and after (spellF1/macroF1, `python -m calibration.report`)

| set | Metrica clean g1 g2 g3 | atd g1 g2 g3 | atd+mask g1 g2 g3 | DFL mean (7) |
|---|---|---|---|---|
| v0 | 0.599 0.646 0.611 | 0.487 0.539 0.469 | 0.370 0.391 0.349 | 0.512/0.900 |
| tuned, Metrica thresholds | 0.650 0.687 0.655 | 0.560 0.595 0.597 | 0.461 0.464 0.449 | 0.573/0.913 |
| v1 as shipped (transferred thresholds) | 0.657 0.694 0.646 | 0.549 0.576 0.600 | 0.452 0.445 0.443 | 0.573/0.915 |

DFL was never optimised and improves on all seven matches (0.463 to 0.587 becomes 0.532 to 0.631).

Leave one game out (`run_search loo`, full descent on the other two games, both variants):

| held out | leave-out choice | all-three choice | diff | what the leave-out descent chose differently |
|---|---|---|---|---|
| game 1 | 0.482 | 0.510 | -0.029 | carry_radius stayed 2.0, touch_dv 2.0, touch_radius 4.0, exit_w 16, min_dead_frames 25 |
| game 2 | 0.502 | 0.530 | -0.028 | low_speed 1.2 |
| game 3 | 0.523 | 0.523 | 0.000 | nothing |

Scores are the mean of the held-out game's two variants. All three are within 0.04, so the
done-when holds. The control definition and the experiments' fate were decided on all three games
before this check, so it tests the descent, not those two choices. The carry thresholds
(carry_margin 0.5, carry_dv 8), low_w 0.05 and the restart anchor were chosen in every fold.

## 7. v1 release gates (`python -m possession`, sha256 e0913342...; smoke scored once, nothing selected on it)

| gate | v1 | v0 | range | |
|---|---|---|---|---|
| validator | OK | OK | OK | ok |
| D share, graded | 0.421 | 0.553 | 0.30 to 0.55 | ok |
| possessions per graded minute | 3.08 | 1.58 | 1.5 to 5 | ok |
| mean possession | 11.3 s | 17.0 s | 6 to 20 | ok |
| home share of alive | 0.569 | 0.630 | 0.45 to 0.75 | ok |
| set-piece windows with D to alive within 3 s | 49 of 59 (83%) | 51 of 59 | 80% | ok |
| D before the 4 kick-offs after a goal | 43, 19 (no restart), 131, 221 s | 124, 19, 133, 230 s | 20 s each | OUT |
| public smoke | spellF1 0.422, median latency -12 frames | 0.384, +11 | > 0.42, within 8 | OUT on latency |

The latency flipped sign (transitions now early by a median 12 frames at tau 25, on a handful of
matched transitions). The restart anchor moved to the strike in this version; whether that is the
cause was not investigated, because the smoke selects nothing.


## 8. Wrap-up: current best, what is left, commands to finish
Current best = the v1 values in `possession/params.py` (section 4 and 5 tables). Scores behind them:
J 0.521 with Metrica thresholds (atd 0.584, atd+mask 0.458), 0.511 as shipped after transfer
(atd 0.575, atd+mask 0.446); DFL mean 0.573; leave-one-out diffs -0.029, -0.028, 0.000.
No job is running. Nothing is frozen.

Left (all optional, none started): a second descent to confirm the local optimum, a separate
masked re-tune, DFL-with-mask scores.

Commands, from the worktree root, with PY=.venv/bin/python
and MTAH_DATA_DIR set as in this session:
- Second descent from the tuned set, about 12 min on 8 cores:
  `$PY -u -m calibration.run_search full calibration/results/tuned_proxy.json full2 > full2.log`
  Output: `calibration/results/full2.json`, grid rows appended to `notes/calibration-log.tsv`.
- Leave one game out for it, about 35 min:
  `$PY -u -m calibration.run_search loo calibration/results/tuned_proxy.json calibration/results/full2.json > loo2.log`
  Output: `calibration/results/loo_metrica_game*.json` (overwrites) and `held out ...` lines in the log.
- Transfer table for a result, about 20 s: `$PY -m calibration.transfer calibration/results/full2.json`
  Output: printed table and `calibration/results/transfer.json`. Values go into params.py by hand.
- Proxy scores of params.py, about 1 min: `$PY -m calibration.report v1=default`
- v1 file and gates, about 15 s: `$PY -m possession` (add `--no-smoke` unless freezing).
