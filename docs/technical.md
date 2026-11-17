# Possession labelling for FC Barcelona vs Athletic Club: technical document

Team: Verynicestyleturbo. Version v2, 2026-11-17. Every version-specific number (file hash, gates, scores) is in section 5, "Final numbers".

## 1. Approach

The submission labels every frame of the tactical video as `H`, `A`, `D` or `X`. The ranking metric is spellF1, which scores whole possessions matched by temporal IoU.
A per-frame classifier tuned for accuracy gives many short, wrong possessions. The method therefore decodes a possession sequence directly.

The model is a three-state sequence model with states home (`H`), away (`A`) and dead ball (`D`).
Each frame gets a score in nats for each state, built from evidence channels (section 2). A Viterbi decoder picks the best path per half.
Switching state has a cost: turnover `H` to `A` 3.0, stop (alive to `D`) 4.0, restart (`D` to alive) 4.0. These did not move in calibration (section 5).
Evidence and costs share one scale, so only their ratios matter. A team takes over when its evidence exceeds the turnover cost. A short intrusion must exceed it twice over.
The only learned part is one small logistic team-shape model (five home-minus-away features, no intercept, trained on the DFL set; section 2). There is no Baum-Welch fitting. Unsupervised fitting of three states finds motion regimes, and nothing ties them to `H`, `A` and `D`.
Run time needs numpy and pandas only.

Design findings that shaped this, measured on the published baselines and public files:

- The ball is missing on 40.7% of graded frames, in 150 long gaps. More than 99% of missing frames lie in gaps longer than 5 s, and the median gap is about 10 s. The vendor has already filled the short ones. So the work is deciding, per gap, whether play is dead and which team has it. Trajectory reconstruction would recover nothing.
- "Ball not tracked" is not "dead ball". Two of the four public stoppages have a fully tracked ball. 44% of untracked public frames are alive.
- The nearest player is the wrong team on 22% of alive frames before smoothing. Possessions are short (six of 17 under 4 s in the public window), so a minimum-duration filter would delete real ones. Evidence has to come from control events.

Guards after decoding are three. (1) A `D` run shorter than 75 frames is merged into its neighbours. (2) `X` is written on every ungraded frame. (3) No `X` is written on a graded frame, because spellF1 forward-fills `X` and macroF1 counts it as a miss.
Hard constraints enter the decoder as data: the pre-kickoff block of the second half is forced `D`, and the first graded frame of the first half is forced alive.
Each half is decoded alone, so no feature reads across halftime.

## 2. Evidence channels

Position data is the Automatic Tracking Data (ATD) of the tactical video, plus the Smart Tagging file. Ball velocity is computed only inside visible runs, never across a blackout. A channel adds zero where its input is missing, so blackouts need no separate code path. Per-event and per-gap weights are counted once and spread over the frames, so one wrong tag cannot outvote the tracking.

Values are the v2 values in `possession/params.py`, which also records the source of each one (`python -m possession.params` prints the table). Four kinds of source: a hand-set first value, the Metrica coordinate descent (section 5), quantile transfer to this match (section 5), and "flat grid", meaning the Metrica games could not tell the values apart and the first value stayed.

| channel | fires when | supports | v2 weight |
|---|---|---|---|
| carry | ball within 3.37 m of a player, nearest opponent at least 0.61 m farther, ball velocity within 7.67 m/s of that player's (all three transferred) | that player's team | 0.2 per frame |
| touch | ball velocity changes by 1.43 m/s or 28.9 degrees across 3 frames with a player within 2.8 m (all transferred) | that team | 2.0 per event |
| restart | first control event within 3 s after a dead-looking stretch of 2 s or more; the evidence sits on the first frame after the stretch | that team | 4.0 per event |
| phase tag | a team-tagged attacking phase covers the frame (8 attacking codes; the 4 DEFENDING codes mirror them and are ignored) | tagged team | 1.0 per phase, spread |
| off pitch | ball outside the lines | `D` | 10 per frame (flat grid) |
| exit gap | blackout starts with ball within 3 m of a line moving outward at 0.5 m/s or more | `D` for the gap | 8 per gap, spread |
| long gap | blackout longer than 30 s | `D` | 4 per gap, spread (flat grid) |
| still ball | ball under 0.5 m/s for 2 s or more | `D` | 0.3 per frame (flat grid) |
| low activity | median player speed under 1.39 m/s (transferred) for 3 s or more | `D` | 0.05 per frame |
| set-piece tag, launch candidate | a `SET PIECES` window. The restart candidate is the first frame with ball speed above 2 m/s after a hidden or still frame inside the window; else the first visible frame after a hidden one; else `D` for the first 6.4 s and nothing after | `D` before the candidate, alive after | 3 per tag |
| goal rule | from a `GOALS` tag's anchor (section 3) to the next kick-off restart | `D` | 1.0 per frame |
| team shape | always, when at least 2 players per team are tracked | penalises the team the shape model disfavours, by w times the absolute log-odds. Never touches `D` or the favoured team | w 0.02 where the ball is visible, 0.1 on blackout frames |
| latch | always | team of the latest carry, touch or restart | 0.001 per frame |

Where a weight shows a bare number it is the first value, and for carry, touch, restart, exit gap and long gap that value was also the best point of the Metrica grid. Phase tag, set-piece tag and latch were not tuned (the Metrica games carry no Smart Tagging). The goal-rule weight is flat from 0.3 to 10 on this match.

The team-shape model reads five features per team in that team's own attack frame, as home minus away: mean x towards the opponent goal, mean velocity towards it, length (std of x), width (std of y) and the share of players in the opponent half. Attack direction per period is label-free (where the away team's mean x lies). A logistic model gives the log-odds of home, with the five coefficients in `possession/formation_model.json`. The channel penalises only the disfavoured team, so it moves `H` against `A` frame by frame and never alive against `D`. It exists to make team calls in blackouts, where the ball gives no evidence.

Two weaknesses of the v0 first values were what calibration targeted. The low-activity threshold of 1.5 m/s sat at the median of the match's median player speed, so it fired on 51% of graded frames and pushed the `D` share to 0.553. The touch threshold of 4 m/s was above the 99th percentile of the ATD's 3-frame velocity change (3.7 m/s), so touches were rare and short possessions got no turnover. In v1 the `D` share is 0.421 and the touch threshold is 1.43 m/s.

Label-free analysis of this match (no labels read) also proposed revised thresholds, per-code tag weights and a set-piece restart rule. The low-activity threshold came from quantile transfer (1.39 m/s), not from the label-free proposal (1.21 m/s, the 20th percentile of this match). Metrica preferred the higher value by 0.015, which is inside the noise. The launch rule for set pieces, which v1 did not use, is adopted in v2 (section 5, step 6). The per-code tag weights are not adopted: they cannot be validated on the proxy, which has no tags, and they compete with the team-shape channel for the same blackout team calls. `phase_w` stays at 1.0 for every code.

## 3. Possession rules in football terms

These are answered from the 5-minute public window as categories for definitions. No number was selected on it.

- **Start of a possession.** The first touch of the new team, not sustained control. The new team's carry run begins at a median of 7 frames after the labelled start. The decoder puts a turnover at the first control frame of the new team.
- **Loose balls and passes in flight.** Possession stays with the last team that controlled the ball. With no player within 3 m, the label equals the latest controlling team on 83.0% of alive frames and the nearest player's team on 64.6%. The latch channel encodes this.
- **Challenges.** A contested ball stays with the last team until another team gains control. A brief intrusion must outweigh twice the turnover cost to flip it.
- **Dead ball starts** when the ball leaves the pitch, when the ball vanishes at a line, at a set-piece tag, after a goal, or when the match goes quiet (still ball, low player activity). Line crossing or tracking loss alone is not enough: of 4 public stoppages the ball crosses a line in 2 and vanishes in 2, and in two cases `D` starts about 3 s before either.
- **Restarts.** Dead ball ends at the strike or throw, before the first control event. The restarting team's carry run starts a median 15 frames after the labelled end (n=4). So v1 puts the restart evidence on the first frame after the dead-looking stretch (the strike), and uses the first control event within 3 s only to name the team.
- **Short possessions.** The two public possessions under 1.5 s show no ball velocity change beside a player (peak 2.1 and 1.3 m/s). The model cannot place them from touch evidence.
- **Goalkeeper holding the ball.** Undetermined from the public window. The still-ball channel treats a held ball as `D`.
- **Kickoffs.** The first-half kickoff frame and the second-half pre-kickoff block are organiser-published constants. After a goal, the decoder must find the stoppage itself.
- **Goals.** The ball is dead from a goal to the kick-off. The `GOALS` tag does not start at the goal: it opens with the scoring move's own set piece and about 20 s of live play. In 3 of the 4 goals the tag start is followed by a 19 to 22 s visible-ball run and a free kick, corner or throw-in window 3 to 10 s in (the fourth tag overlaps a free-kick window for its first 31 s). `D` therefore starts after the last set-piece window the tag overlaps and runs to the next kick-off restart. Starting `D` at the tag start was tried and not adopted: it forced `D` over live play, dropped the set-piece gate to 47 of 59 and removed 8 possessions. The anchor was chosen by looking at the same four stretches the kick-off gate grades, so passing that gate is not independent evidence. What carries it is the structure of the tags. It changes 0.12% of graded frames and nothing in a match without `GOALS` tags.

## 4. Data and assets

**Used.**

- The tactical ATD (tracking of 46 slots and the ball) and the Smart Tagging XML.
- Organiser-published constants: the graded half blocks (`games.half_video_blocks`) and the second-half kickoff frame (`KICKOFF_VIDEO_FRAMES` in `starter/load_smart_tagging.py`, 113742).
- Pitch size from `games.py` (105.3 m by 68.0 m).

**Tried and dropped.**

- Pixel computer vision for the ball in blackouts. About 65,000 graded frames would need inference on a full-pitch 1920x1080 view where the ball is a few pixels wide. The estimate is 9 to 18 hours on the available CPU, with no labels to validate the result. The vendor's detector already ran on those pixels.
- Video and homography. The homography is used only for optional visual debugging. No video is read by the pipeline.
- Broadcast recording. The alignment was found: tactical frame = broadcast frame + 4014 in the first half and + 26920 in the second, each stable within 1 frame (the broadcast cuts halftime), with 85 of 110 one-minute chunks fitting under 2.5 m. Where both see the ball, the median distance is 0.75 m (0.78 m) and 88.5% (89.8%) are within 5 m. The coverage test failed: accepted broadcast ball covers 7.7% (half 1) and 8.3% (half 2) of tactical blackout frames, against a bar of 15% set in advance. Even taking every mapped broadcast ball without acceptance gives only 12.7% and 12.9%. The broadcast shows live play on 30% of blackout frames against 56% of all frames, so it loses the ball in the same moments. The decision is stop, and the broadcast is not read by the pipeline.
- Baum-Welch fitting (reason in section 1). The four DEFENDING Smart Tagging codes (they mirror attacking codes of the other team with identical intervals, so they add no second opinion).

**Training data.** Two external labelled datasets, one of them also training the team-shape classifier, no human labels made by the team, and nothing from the provided game used as a label. The public 5-minute window was used only as a smoke check, scored once per frozen version, and selected nothing.

| dataset | source | who labelled | amount | role |
|---|---|---|---|---|
| Metrica Sports public sample data | github.com/metrica-sports/sample-data. The repository has no licence file. Its README asks users to acknowledge the source, which we do here | Metrica's own event data for the 3 games; our code derived the per-frame labels (`harness/metrica.py`) | 3 anonymised games, about 188 alive minutes | primary calibration set |
| Open DFL Bundesliga tracking, also trains the team-shape classifier | Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12, 195 (2025), CC-BY 4.0. Files from HuggingFace `pysport/idsse-data` (`harness/dfl.py`) | the data provider, as per-frame `BallPossession` and `BallStatus` fields | 7 matches, about 369 alive minutes | team-shape classifier training (110,627 rows from every 5th alive frame, 221,254 with team-swapped copies); otherwise reported, never optimised |

DFL labels: `D` where `BallStatus` is dead, else `H` or `A` by `BallPossession`. A frame with a missing `BallPossession` takes the previous frame's label, because kloppy would otherwise read it as away. The DFL export stores player speed in km/h, and `calibration/variants.py` divides by 3.6 on load.
Metrica labels follow a fixed rule in `harness/metrica.py`. On-ball events (pass, shot, recovery, set piece, carry) give possession to the event's team. Stoppage events (ball out, fault received, offside or hand-ball ball lost, goal) set `D`, which lasts until the next set piece or goal-kick pass. Other events during a stoppage are ignored. Each half starts as `D` until its first on-ball event.
Metrica is the primary calibration set because its possession statistics match the public window and it has real tracking blackouts (32 to 41% of frames missing). Mean possession is 11.8 to 13.2 s on Metrica, 5.5 to 9.2 s on DFL, 14.2 s in the public window. DFL has no blackouts.
The classifier's labels are the DFL provider's possession fields. Metrica is fully held out from it, so every Metrica score of the team-shape channel is held out. Both sets are used for calibration only and are not redistributed. The repository holds none of their data, only the five fitted coefficients.

## 5. Calibration

The calibration objective J is the mean spellF1 over the three Metrica games, each on two variants (six match-variants), with macroF1 as tie-break.

1. **Proxy validation.** Raw Metrica is easier than this match: its ball is glued to the player on the ball and kinks sharply, whereas the ATD ball is smooth and player slots drop out. Two variants (`calibration/variants.py`) make it harder, with three knobs set to match label-free statistics of this match (velocity-change q99, nearest-player distance, tracked players per team). `atd` smooths the ball, offsets it slowly and drops player slots. `atd+mask` adds this match's per-half visibility mask, which raises the missing share to 0.57 to 0.64 against this match's 0.407. The four published baselines, mean spellF1 over the three games:

| variant | raw | smooth25 | smooth50 | smooth125 | v0 first values |
|---|---|---|---|---|---|
| clean | 0.089 | 0.360 | 0.417 | 0.308 | 0.619 |
| atd | 0.079 | 0.287 | 0.348 | 0.283 | 0.498 |
| clean+mask | 0.078 | 0.280 | 0.320 | 0.279 | 0.466 |
| atd+mask | 0.068 | 0.230 | 0.269 | 0.246 | 0.370 |
| this match, published | 0.049 | 0.200 | 0.243 | 0.224 | not applicable |

   smooth50 is first everywhere. Only `atd+mask` also reproduces the published order of the rest (smooth125, smooth25, raw). This match sits between `atd` and `atd+mask`, so J uses both. No knob was fitted to the published row.
2. **Control definition.** Carry only, touch only and both were each tuned from the v0 values (J, six match-variants): v0 0.434, carry only 0.484, touch only 0.454, both 0.502. Both was kept. Its margin over carry only (0.018) is under the 0.04 noise floor, but it has the same sign on both variants and on macroF1.
3. **Coordinate descent.** Two passes over 25 parameters on coarse grids of 3 to 5 values. J rose from 0.502 to 0.521 (`atd` 0.584, `atd+mask` 0.458, macroF1 0.757). Values that moved: carry_radius 2.0 to 3.0, carry_margin 1.0 to 0.5, carry_dv 2 to 8, touch_dv 4 to 1.5, low_w 0.1 to 0.05, the minimum dead run 25 to 75 frames, and the restart anchored at the end of the dead-looking stretch. Every weight and cost with a real slope peaked at its first value (carry, touch, restart and exit weights, stop and restart costs, the 30 s long-gap threshold). Flat grids stayed at the first value: long-gap weight, still-ball weight, off-pitch weight, touch angle. The turnover cost stayed 3.0 as the scale anchor. Five variants of the rules (a 2, 5 or 10 s long-gap threshold, a wider exit rule, a moving-ball alive channel, low_speed 1.2 and 1.0) gained under 0.003 or lost, and were left out.
4. **Quantile transfer.** A threshold at quantile q of a feature on the Metrica `atd` games becomes the value at quantile q of the same feature on this match, both on visible-ball frames.

| threshold | Metrica value | quantile | this match |
|---|---|---|---|
| carry_radius (m) | 3.0 | 0.690 | 3.37 |
| touch_radius (m) | 2.5 | 0.631 | 2.80 |
| carry_margin (m) | 0.5 | 0.105 | 0.61 |
| carry_dv (m/s) | 8.0 | 0.726 | 7.67 |
| touch_dv (m/s) | 1.5 | 0.756 | 1.43 |
| touch_angle (deg) | 35 | 0.977 | 28.93 |
| low_speed (m/s) | 1.5 | 0.324 | 1.39 |
| still_speed (m/s) | 0.5 | 0.023 | 0.18, not applied |

   still_speed stays at 0.5 because the Metrica ball is missing when dead, so its low quantiles say nothing about a dead ball that stays tracked. After transfer J is 0.511 (`atd` 0.575, `atd+mask` 0.446).
5. **Leave one game out.** A full descent on two games, scored on the third (mean of its two variants):

| held out | leave-out choice | all-three choice | difference |
|---|---|---|---|
| game 1 | 0.482 | 0.510 | -0.029 |
| game 2 | 0.502 | 0.530 | -0.028 |
| game 3 | 0.523 | 0.523 | 0.000 |

   All three are within the 0.04 bound. The control definition and the rejected variants were decided on all three games before this check, so it tests the descent, not those choices.

6. **Set-piece launch rule.** The Metrica games have no tags, so a planted-tag simulation was run (`calibration/restarts_proxy.py`): a seeded 59% of each game's true restarts gets a 35 s window starting a resampled offset before the restart (offsets drawn from this match's own launch offsets, median 6.6 s), 5 seeds, both variants. Mean J over weights 1.5, 3, 6, 12: no tags 0.5105; v1 first-touch rule 0.5099, 0.5089, 0.5070, 0.5033; launch rule 0.5113, 0.5114, 0.5105, 0.5098. Launch is at or above the touch rule in every seed at every weight, and level with no tags (inside the 0.003 noise floor). This shows the rule does not hurt, not that it helps: the decoder already finds most restarts on Metrica. On this match the set-piece gate goes from 49 to 50 of 59 with the rule alone, and 0.38% of frames change. `setpiece_w` stays 3.0.
7. **Team-shape channel.** Adoption rule fixed in advance: a gain on all six cells (three games, two variants), held out by game. The first attempt failed: game 3 `atd` lost 0.026, because the weight had been selected with a classifier trained on the games it was scored on. It was not adopted, and one second attempt was specified before it ran: a DFL-only classifier (so Metrica is held out without nesting), separate weights for visible and blackout frames, small fixed grids (visible 0, 0.01, 0.02; blackout 0.01, 0.02, 0.05, 0.1), no smoothing. It passed. Frame accuracy of the classifier on held-out Metrica alive frames, and held-out spellF1 with the weights chosen on the other two games:

| held out | accuracy `atd` | accuracy `atd+mask` blackout frames | weights chosen (visible, blackout) | `atd` without, with | `atd+mask` without, with |
|---|---|---|---|---|---|
| game 1 | 0.801 | 0.725 | 0.02, 0.1 | 0.549, 0.561 (+0.013) | 0.452, 0.488 (+0.036) |
| game 2 | 0.815 | 0.749 | 0.02, 0.1 | 0.576, 0.599 (+0.023) | 0.445, 0.484 (+0.039) |
| game 3 | 0.818 | 0.744 | 0.02, 0.05 | 0.600, 0.611 (+0.011) | 0.443, 0.470 (+0.028) |

   The `atd` gains are each under the 0.04 noise floor, but all six point the same way. On all three games the chosen weights are 0.02 and 0.1, at the top of their grids (the grids were not extended), and J rises from 0.5105 to 0.5367 (macroF1 0.755 to 0.767; `atd` 0.590, `atd+mask` 0.483). A cost descent over turnover, stop, restart, carry and touch costs moved nothing (every gain under 0.003), and its leave-one-game-out differences were 0.000, 0.000 and -0.020, within 0.04. On clean Metrica the channel is neutral (-0.004, 0.000, -0.004). On clean DFL it costs: mean spellF1 falls from 0.573 to 0.546. That is reference only, since DFL trained the classifier. DFL possessions flip 2 to 3 times faster than this match's, and a slow team-shape signal lags quick turnovers, so with a clean ball the visible-frame weight costs more than it brings.

The DFL set was never optimised on. v1 improved on it in all seven matches (mean spellF1 0.512 to 0.573, macroF1 0.900 to 0.915).
The public window is never used to tune. One possession moves spellF1 there by about 6 points.

### Final numbers

This is the one place to update for a later version.

- Version v2, `possession/FROZEN.txt`. File `evaluation/results/Verynicestyleturbo.csv`, 199,319 rows, validator OK, two consecutive runs gave the same hash.
- sha256 `1603cfd48e39dfd737633db256e37aa8a1803ccca71bd964d4d6baff1dbde6e4`. Earlier: v1 `e09133425dce724a8ddfb62054b6e1e10272343f0308b6a36946aee5457f1f88`, v0 `36e19f4c184f3e12629e77b7eb10b23b5492e91fa0ff5596205707bcd2dc0ce3`.
- Harness spellF1 for Metrica games 1 to 3 (v1, then v2). Clean: 0.657, 0.694, 0.646, then 0.653, 0.694, 0.642. `atd`: 0.549, 0.576, 0.600, then 0.561, 0.599, 0.611. `atd+mask`: 0.452, 0.445, 0.443, then 0.488, 0.484, 0.478. J 0.5105 to 0.5367. DFL clean mean over 7 matches: 0.573 (v1), 0.546 (v2). Source: `python -m calibration.report`.
- Public-window smoke, scored once per version and used to select nothing: v2 spellF1 0.487, median latency -13 frames (v1: 0.422 and -12; v0: 0.384 and +11).
- Release gates (`python -m possession --no-smoke`):

| gate | v2 | v1 | range |
|---|---|---|---|
| validator | OK | OK | OK |
| `D` share of graded frames | 0.448 | 0.421 | 0.30 to 0.55 |
| possessions per graded minute | 2.833 | 3.08 | 1.5 to 5 |
| mean possession | 11.7 s | 11.3 s | 6 to 20 s |
| home share of alive frames | 0.596 | 0.569 | 0.45 to 0.75 |
| set-piece windows with `D` to alive within 3 s | 49 of 59 (83%) | 49 of 59 (83%) | 80% or more |
| `D` before the 4 kick-offs after a goal | 122, 163 (no restart), 131, 221 s | 43, 19 (no restart), 131, 221 s | 20 s or more each |

  Only the smoke latency check is out of range: transitions on the public window are early by a median 13 frames (v1: 12) on a handful of matched transitions. The cause was not investigated, because the smoke selects nothing. The kick-off gate passes in v2, but see section 3 on why that is not independent evidence.

## 6. Limitations

- Blackouts. About 40% of graded frames have no ball. In those frames the model relies on player activity, tags, team shape and the edges of the gap, which is a weak signal. The share of alive play inside blackouts is unknown.
- Short possessions. A possession of about one second carries no touch signal in the ATD ball track, which is smooth. The model tends to merge them.
- Transition timing. On the public window v2's matched transitions come a median 13 frames early (v1: 12, v0: 11 late). Too few transitions to say whether the restart anchor causes it.
- Calibration transfer. The DFL possession definition differs from the competition's, and DFL has no blackouts. Metrica's tracking quality differs from this ATD. Quantile transfer and the masked variant reduce this but do not remove it. With three calibration games, over-fitting is the main risk; the leave-one-game-out check bounds it at 0.03.
- Team shape raises the `D` share (0.421 to 0.448). When the latched team is the disfavoured one, a stop plus restart can cost less than a turnover, so `D` is inserted. On the proxy the `D` share moves further above the truth even as spellF1 rises. Team shape also lags quick turnovers, which is why it costs on DFL.
- Smart Tagging windows run past the possession they describe. Agreement with local carry falls from 0.83 in the first third of a tag to 0.56 in the last third. Per-code tag weights are not used.
- Goalkeeper holds are undetermined (section 3).
- The broadcast recording is not used: it covers too few blackout frames (section 4).
- With more time: a better restart anchor (onset of ball motion), a model of alive play inside blackouts, per-code tag weights, and more labelled games with real blackouts. The broadcast could still help as an alive cue: it shows an accepted live view on 28 to 30% of blackout frames, against 56% overall, so a live view in a blackout is informative even when its ball is missing.

## 7. Reproduction

1. Python 3.12 (tested with 3.12.12) with the pinned packages in `requirements-submission.txt`: `numpy==2.4.4`, `pandas==2.2.3`. Tests add `pytest==9.2.0` and `hypothesis==6.170.15` (`requirements-dev.txt`). Byte-identical output depends on the numpy and pandas versions. The calibration harness uses a separate environment (`harness/requirements.txt`, kloppy and newer numpy and pandas) and is not needed to regenerate the file.
2. Place the data bundle as described in `DATA.md` (`./data`), or set `MTAH_DATA_DIR` to the folder that holds it. The data is not in the repository.
3. Run `python -m possession`. On first run it parses the ATD and tagging once into an array cache under `$MTAH_DATA_DIR/cache`, about 11 s. It then writes `evaluation/results/Verynicestyleturbo.csv`, runs `evaluation/validate_submission.py`, prints the sha256 and the release gates. `--no-smoke` skips the public-window gate. `--cache-dir <empty dir>` forces a parse from the raw files.
4. Expected sha256 of the submitted file: see "Final numbers" in section 5. The team-shape coefficients are committed in `possession/formation_model.json`. `python -m calibration.formation_train` regenerates them byte for byte in the submission environment, from the DFL arrays the harness exporters write under `$MTAH_DATA_DIR/cache/ext` (about 1 minute).
5. Tests: `pytest tests/` (property tests listed in `tests/PROPERTIES.md`). Two consecutive runs of the command gave the same sha256.

Every label comes from code. The only constants tied to this match are the organisers' published half blocks and kickoff frame.
