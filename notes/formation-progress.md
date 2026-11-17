# formation-progress (team-shape evidence channel)

Difficulty: hard. A new learned channel, validated held-out by Metrica game against a strict
adoption rule (gain on all three games on both `atd` and `atd+mask`).

Started 2026-11-16. Baseline: tests 13 passed 1 skipped; `python -m possession --no-smoke`
sha256 e09133425dce...1f88, D share 0.421, 3.078 possessions/min, mean 11.3 s, home share 0.569.

## Plan
1. Features (label-free attack frame per period), channel at weight 0, commit (sha unchanged).
2. Training script `calibration/formation_train.py` (DFL + Metrica, atd variant, swap-augmented).
3. Held-out by Metrica game: classifier and weight chosen without the held-out game.
4. Adopt only if gain > 0 on all 3 games on both atd and atd+mask; then descent on costs.

## Status
- [x] 1 features and weight-0 channel (commit 44980ef; sha unchanged, 15 tests pass)
- [x] 2 training (`python -m calibration.formation_train`, deterministic: two runs same md5)
- [x] 3 held-out table, run 1: rule FAILS on game 3 atd (section 4)
- [x] 4 run 1 not adopted (section 5); run 2 adopted w_vis 0.02, w_blk 0.1, costs unchanged (section 6)

## 1. Features and why (features current; the model and channel described here are run 1, superseded by section 6)
`possession/features.py: team_shape`, five home-minus-away features, each per team in that
team's own attack frame. Attack direction per period, label-free: home attacks towards the side
where the away team's mean x lies (sign of the median over frames of away minus home mean x).
Frames agreeing with the period sign: 0.97 to 1.00 on BAR-ATH, Metrica and DFL.
- `advance`: mean x towards the opponent goal (the block sits in the defending team's half).
- `forward_speed`: mean player velocity towards the opponent goal (the team on the ball pushes up,
  the other drops).
- `length`, `width`: std of x and y (the team on the ball stretches).
- `in_opp_half`: share of present players in the opponent half.
Swapping the teams negates every feature exactly, reflecting both axes leaves them bit-identical
(centred coordinates, sign applied before any reduction). Rows where a team has fewer than 2
tracked players are 0. Label-free transfer check (`formation_train check`, |feature| quantiles
q5/25/50/75/95): BAR-ATH sits inside the Metrica/DFL `atd` range on every feature, `length` a bit
wider at the top (q95 19.9 m against 13 to 17 m).

Model: logistic, H against A, no intercept, features scaled by std only, IRLS 25 iterations, ridge
1e-3; rows = alive frames of the `atd` variant (BAR-ATH-like slot dropout), every 5th frame, plus
a team-swapped copy of each. Final model: 10 matches, 166,863 rows (333,726 with swaps).
Coefficients (raw units): advance -0.036, forward_speed 0.201, length 0.035, width 0.395,
in_opp_half 1.747.

Channel (`possession/evidence.py`): z = shape . coef (log-odds of H), optional centred moving
average over `formation_smooth_s`, on blackout frames only or on every frame
(`formation_all`); the disfavoured team's column gets -formation_w * |z|, the favoured team and D
get nothing. Viterbi takes the max over alive states, so the best alive score against D is
unchanged: the channel moves only H against A, never alive against D (constraint 2). A
symmetric +-z/2 would raise the favoured team's score and pull blackouts towards alive; adding
log P(team) would lower both and push them towards D.

## 2. Classifier frame accuracy, held out by Metrica game (trained on DFL + the other two)

| held out | n train rows | alive frames (atd) | alive blackout frames (atd+mask) |
|---|---|---|---|
| game 1 | 149,588 | 0.800 (n 85,545) | 0.724 (n 37,541) |
| game 2 | 148,408 | 0.817 (n 91,284) | 0.753 (n 42,999) |
| game 3 | 146,357 | 0.817 (n 101,857) | 0.744 (n 44,352) |

## 3. Grid (`python -m calibration.formation_tune`)
weight {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5} x smoothing {0, 1, 3, 5 s} x {blackout, all}.
Fold choice on J over the two training games (both variants) with the fold classifier.

## 4. Held-out table, run 1 (spellF1; v1 = channel off)

| held out | fold choice | train J v1 -> ch | atd v1 | atd ch | gain | atd+mask v1 | atd+mask ch | gain |
|---|---|---|---|---|---|---|---|---|
| game 1 | w 0.02, no smoothing, all frames | 0.516 -> 0.540 | 0.549 | 0.557 | +0.009 | 0.452 | 0.493 | +0.041 |
| game 2 | w 0.02, no smoothing, all frames | 0.511 -> 0.531 | 0.576 | 0.594 | +0.018 | 0.445 | 0.482 | +0.037 |
| game 3 | w 0.05, no smoothing, all frames | 0.505 -> 0.542 | 0.600 | 0.574 | -0.026 | 0.443 | 0.450 | +0.008 |

Adoption rule: FAIL (game 3 atd -0.026). 5 of 6 cells gain.

All three games, final classifier: v1 J 0.5105 (atd 0.575, atd+mask 0.446); best w 0.05 all
frames J 0.5336 (atd 0.586, atd+mask 0.481); w 0.02 smooth 1 s all 0.5328; w 0.02 all 0.5316.
All frames beat blackout-only in every top-8 row. These all-three numbers are not held-out
evidence (classifier and weight both chosen on the games they are scored on).

## 5. Run 1 verdict: not adopted, formation_w stayed 0 (superseded by run 2, section 6)
The rule was fixed before the run and fails on one cell, so per the brief the weight stays 0 and
no further grid, feature or selection variant was tried (each would be a choice made after seeing
the held-out games). No cost descent was run: it belongs only after adoption.

- Chosen weight / switch / smoothing: 0 (off); the switch and smoothing are unused.
- Re-tuned costs: none (turnover 3.0, stop 4.0, restart 4.0, carry_w 0.2, touch_w 2.0 as v1).
- J before / after: 0.5105 / 0.5105 (macroF1 0.7553).
- DFL mean (clean) before / after: 0.573 / 0.573. For reference only, with the channel on at the
  all-three choice (w 0.05, all frames), DFL clean falls to 0.538/0.908 on all 7 matches, and
  Metrica clean falls 0.666 -> 0.650, although DFL is in the classifier's training set (in-sample).
  So with a good ball the all-frames channel overrides better ball evidence; the gains are on the
  degraded ball (atd 0.575 -> 0.586, atd+mask 0.446 -> 0.481). That is the same direction as the
  game 3 atd failure.
- v2 gates (`python -m possession --no-smoke`, two runs, same sha256
  e09133425dce724a8ddfb62054b6e1e10272343f0308b6a36946aee5457f1f88, identical to v1): D share
  0.421, 3.078 possessions per graded minute, mean possession 11.3 s, home share of alive 0.569.

Observations for the lead (not acted on):
- Each fold chose its weight with a classifier trained on the two games it was chosen on, so the
  classifier is more confident there than on unseen games; that optimism favours larger weights,
  and game 3's fold took 0.05 where the other two took 0.02. A pre-registered follow-up would
  choose the weight with nested classifiers, each trained without the game it is scored on.
- All-frames beat blackout-only everywhere on the proxy, but hurts clean ball. A follow-up could
  pre-register a variant that gates by ball visibility (blackout only) or scales the weight down
  where the ball is seen, and run the same six-cell rule once.
- The masked proxy blackouts are about 0.5 dead against about 0.78 on this match, and the channel
  only acts between the teams, so its real-match effect on blackouts is likely smaller than the
  atd+mask gain suggests.

## 6. Run 2 (specified by the lead after run 1, before running; logged; no tuning beyond it)
Run 1's defect: each fold chose its weight with a classifier trained on the games it was chosen on.
- Merged `submission` first (restart rules; weight-0 sha now 6ddeef44...75d7, 16 tests pass).
- Classifier trained on DFL only (same features, atd, every 5th alive frame, swap-augmented, no
  intercept). It is also the model that ships, so the Metrica games are held out without nesting.
  110,627 rows (221,254 with swaps). Coefficients: advance -0.042, forward_speed 0.210, length
  0.040, width 0.437, in_opp_half 1.883. Deterministic (two runs, same md5).
- Two weights replace `formation_w`/`formation_all` (and the unused smoothing): `formation_w_vis`
  on frames where `Features.visible` is true, `formation_w_blk` elsewhere. Grids w_vis {0, 0.01,
  0.02} x w_blk {0.01, 0.02, 0.05, 0.1}, no smoothing.
- Free gate check: with w_vis = 0, DFL clean codes must be identical to weight 0.
- Same six-cell rule against the current `submission` params. If adopted: weights chosen on all
  three games, one descent pass over turnover/stop/restart costs, carry_w, touch_w, kept only if
  its leave-one-game-out diffs stay within 0.04.

Held-out frame accuracy of the DFL-only model:

| Metrica game | alive frames (atd) | alive blackout frames (atd+mask) |
|---|---|---|
| game 1 | 0.801 (n 85,545) | 0.725 (n 37,541) |
| game 2 | 0.815 (n 91,284) | 0.749 (n 42,999) |
| game 3 | 0.818 (n 101,857) | 0.744 (n 44,352) |

### Run 2 results (`python -m calibration.formation_tune`, calibration/results/formation_loo.json)
Gate check: with w_vis = 0, DFL clean codes are identical to weight 0 on all 7 matches for every
w_blk in the grid (the gating is right).

Held-out table (spellF1; base = current `submission` params, channel off):

| held out | fold choice (on the other two games) | train J base -> ch | atd base | atd ch | gain | atd+mask base | atd+mask ch | gain |
|---|---|---|---|---|---|---|---|---|
| game 1 | w_vis 0.02, w_blk 0.1 | 0.516 -> 0.543 | 0.549 | 0.561 | +0.013 | 0.452 | 0.488 | +0.036 |
| game 2 | w_vis 0.02, w_blk 0.1 | 0.511 -> 0.535 | 0.576 | 0.599 | +0.023 | 0.445 | 0.484 | +0.039 |
| game 3 | w_vis 0.02, w_blk 0.05 | 0.505 -> 0.534 | 0.600 | 0.611 | +0.011 | 0.443 | 0.470 | +0.028 |

Adoption rule: PASS (all six gains positive; atd gains +0.011 to +0.023 are under the 0.04 noise
floor one by one, but all six point the same way).

Grid on all three games: base J 0.5105 (atd 0.575, atd+mask 0.446). J rises with both weights;
w_vis = 0 rows 0.520 to 0.523, w_vis 0.01 rows 0.531 to 0.533, w_vis 0.02 rows 0.535 to 0.537.
Chosen: w_vis 0.02, w_blk 0.1, J 0.5367/0.7665 (atd 0.590, atd+mask 0.483). Both weights sit at
the top of their grids; the grid was fixed in advance and was not extended.

Descent, one pass over stop_cost, restart_cost, carry_w, touch_w, turnover_cost (in that order):
nothing moved (every grid best under the 0.003 MIN_GAIN), so the costs stay as they are and
turnover_cost stays 3.0. Its leave-one-game-out: game 1 diff 0.000, game 2 0.000, game 3 -0.020
(the game-3 fold moved carry_w to 0.1). All within 0.04, so the descent result is kept, which is
the same as the chosen weights with the current costs.

Final J 0.5105 -> 0.5367 (macroF1 0.7553 -> 0.7665). Per cell base -> final: g1 atd 0.549 -> 0.561,
g1 atd+mask 0.452 -> 0.488, g2 atd 0.576 -> 0.599, g2 atd+mask 0.445 -> 0.484, g3 atd 0.600 ->
0.611, g3 atd+mask 0.443 -> 0.478.
DFL clean mean (reference, not optimised; DFL trained the classifier): 0.5735 -> 0.5458. With a
clean ball the visible-frame weight costs more than it brings.

What the channel does to the decoded path (Metrica, labels, aggregates):

| game, variant | true D share | D share base | D share adopted | team right, alive frames, base -> adopted |
|---|---|---|---|---|
| g1 atd | 0.404 | 0.427 | 0.437 | 0.880 -> 0.903 |
| g2 atd | 0.346 | 0.411 | 0.417 | 0.904 -> 0.914 |
| g3 atd | 0.287 | 0.322 | 0.331 | 0.900 -> 0.918 |
| g1 atd+mask | 0.404 | 0.560 | 0.577 | 0.867 -> 0.900 |
| g2 atd+mask | 0.346 | 0.562 | 0.571 | 0.885 -> 0.915 |
| g3 atd+mask | 0.287 | 0.437 | 0.470 | 0.849 -> 0.910 |

Frame by frame the channel never raises D or lowers the favoured team, but the decoded path does
gain D: when the latched team is the disfavoured one, staying costs the penalty and switching
costs a turnover, so a stop and restart can win. Team accuracy is the gain; the extra D moves away
from the true D share. A follow-up could check whether a lower stop/restart ratio offsets this,
but the descent found no gain there.

BAR-ATH, `python -m possession --no-smoke` with the adopted values (two runs, same sha256
1603cfd48e39dfd737633db256e37aa8a1803ccca71bd964d4d6baff1dbde6e4):

| gate | weight 0 (sha 6ddeef44) | adopted |
|---|---|---|
| D share, graded | 0.422 | 0.448 |
| possessions per graded minute | 3.059 | 2.833 |
| mean possession | 11.3 s | 11.7 s |
| home share of alive | 0.572 | 0.596 |
| set-piece windows with D to alive within 3 s | 49 of 59 | 49 of 59 |
| D before the 4 kick-offs after a goal | all ok | 122, 163 (no restart), 131, 221 s, ok |

Graded frames whose label changes: 5.5% of 158,857: H->A 0.9%, A->H 1.7%, H->D 1.0%, A->D 1.7%,
D->H 0.1%, D->A 0.1%.

Test edit: `test_absent_inputs_contribute_zero` now excludes the team-shape channel, because its
fixture has players present and players are that channel's input (PROPERTIES.md updated).
16 tests pass; P4, P5, P8 and the channel property also pass under a second Hypothesis seed.

Commands: `python -m calibration.formation_train [train|check]` (model JSON and accuracies, about
1 min), `python -m calibration.formation_tune` (folds, grid, held-out table, about 5 min; writes
calibration/results/formation_loo.json and rows in notes/calibration-log.tsv).

## 7. Run 3: classifier upgrade (specified by the lead before running; third look at the same games)
Start: `submission` d779de0, v2 frozen (sha 1603cfd4...e6e4, w_vis 0.02, w_blk 0.1).
Features chosen by reasoning before any run, not by trying:
- `space`: mean over a team's players of the distance to the nearest opponent, home minus away.
  Defenders mark goal-side of attackers, so each defender is near some attacker while attackers
  spread and some stay free: the team on the ball has more space.
- `line`: x of the second-deepest player in the team's attack frame (the deepest outfield player
  when the keeper is deepest). The team on the ball pushes its back line up; the mean `advance`
  mixes this with the forwards, who drop less when defending.
- Not added: compactness, which `length` and `width` already measure.
- Temporal context: centred rolling means over 1 s and 3 s of all 7 per-frame features, per period,
  on every frame alike (players only). 21 columns, ordered base 7, then 1 s, then 3 s.
Training as v2: DFL only, atd, every 5th alive frame, team-swap augmented, no intercept, IRLS.
Stage 1 gate: held-out frame accuracy on each Metrica game must rise by at least 0.01 over the v2
classifier on both alive frames (atd) and alive blackout frames (atd+mask), all 3 games. Else stop.
Stage 2: per fold, weights re-chosen on w_vis {0, 0.01, 0.02} x w_blk {0.05, 0.1, 0.2} (J over the
other two games), scored on the held-out game. Adopt only if all six cells gain over v2 and the
mean held-out gain is at least 0.01. No cost descent. Not adopted: nothing committed that changes
the v2 sha.

### Run 3 result: Stage 1 gate FAILS, v2 kept (code not committed)
`python -m calibration.formation_train check` with the 21-column model (DFL only, 110,627 rows),
against the shipped v2 model (5 columns), held-out frame accuracy:

| Metrica game | alive (atd) v2 -> run 3 | gain | alive blackout (atd+mask) v2 -> run 3 | gain |
|---|---|---|---|---|
| game 1 | 0.801 -> 0.812 | +0.011 | 0.725 -> 0.727 | +0.002 |
| game 2 | 0.815 -> 0.826 | +0.011 | 0.749 -> 0.758 | +0.009 |
| game 3 | 0.818 -> 0.820 | +0.002 | 0.744 -> 0.739 | -0.005 |

The gate needs +0.01 on all six; two pass. Stage 2 was not run. Frame counts differ slightly
between models (accuracy counts frames with nonzero log-odds; the rolling means leave fewer exact
zeros), which does not change the verdict. The code is kept as `notes/formation-run3.patch`
(applies on d779de0; with the v2 coefficients it gave the v2 sha unchanged, since the channel uses
only the first len(coef) columns), not committed: it slows `python -m possession` from about 2 s
to about 7 s (the player-to-opponent distance matrix) for no gain.
Blackout frames, the ones the channel is meant for, gained least: rolling means of a shape
that is already slow add little, and space and line overlap with advance and length.

Found on the way (reported to the lead, not changed): frozen v2 fails P8 on one synthetic case,
segments [(A,300),(H,300)], seed 0: spellF1 0.900 with the channel, 1.000 with both weights at 0,
on a clean export of HEAD d779de0. The synthetic players do not change shape with possession, so
the classifier gives a fixed wrong answer that w_vis applies on every visible frame. Earlier test
runs passed because Hypothesis is not derandomized here and had not drawn this case.
