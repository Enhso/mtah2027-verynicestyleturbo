# Possession labelling for FC Barcelona vs Athletic Club: technical document

Team: Verynicestyleturbo. Version v1, 2026-11-16. Every version-specific number (file hash, gates, scores) is in section 5, "Final numbers".

## 1. Approach

The submission labels every frame of the tactical video as `H`, `A`, `D` or `X`. The ranking metric is spellF1, which scores whole possessions matched by temporal IoU.
A per-frame classifier tuned for accuracy gives many short, wrong possessions. The method therefore decodes a possession sequence directly.

The model is a three-state sequence model with states home (`H`), away (`A`) and dead ball (`D`).
Each frame gets a score in nats for each state, built from evidence channels (section 2). A Viterbi decoder picks the best path per half.
Switching state has a cost: turnover `H` to `A` 3.0, stop (alive to `D`) 4.0, restart (`D` to alive) 4.0. These did not move in calibration (section 5).
Evidence and costs share one scale, so only their ratios matter. A team takes over when its evidence exceeds the turnover cost. A short intrusion must exceed it twice over.
There is no learned model and no Baum-Welch fitting. Unsupervised fitting of three states finds motion regimes, and nothing ties them to `H`, `A` and `D`.
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

Values are the v1 values in `possession/params.py`, which also records the source of each one (`python -m possession.params` prints the table). Four kinds of source: a hand-set first value, the Metrica coordinate descent (section 5), quantile transfer to this match (section 5), and "flat grid", meaning the Metrica games could not tell the values apart and the first value stayed.

| channel | fires when | supports | v1 weight |
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
| set-piece tag | a `SET PIECES` window | `D` before the restart candidate, alive after | 3 per tag |
| latch | always | team of the latest carry, touch or restart | 0.001 per frame |

Where a weight shows a bare number it is the first value, and for carry, touch, restart, exit gap and long gap that value was also the best point of the Metrica grid. Phase tag, set-piece tag and latch were not tuned (the Metrica games carry no Smart Tagging).

Two weaknesses of the v0 first values were what calibration targeted. The low-activity threshold of 1.5 m/s sat at the median of the match's median player speed, so it fired on 51% of graded frames and pushed the `D` share to 0.553. The touch threshold of 4 m/s was above the 99th percentile of the ATD's 3-frame velocity change (3.7 m/s), so touches were rare and short possessions got no turnover. In v1 the `D` share is 0.421 and the touch threshold is 1.43 m/s.

Label-free analysis of this match (no labels read) also proposed revised thresholds, per-code tag weights and a set-piece restart rule. Only one proposal touched v1, and only through a different route. The low-activity threshold came from quantile transfer (1.39 m/s), not from the label-free proposal (1.21 m/s, the 20th percentile of this match). Metrica preferred the higher value by 0.015, which is inside the noise. The per-code tag weights and the launch-candidate rule for set pieces were not adopted in v1: `phase_w` stays at 1.0 for every code and the set-piece rule is unchanged. A later version may adopt them, after a check on the DFL set.

## 3. Possession rules in football terms

These are answered from the 5-minute public window as categories for definitions. No number was selected on it.

- **Start of a possession.** The first touch of the new team, not sustained control. The new team's carry run begins at a median of 7 frames after the labelled start. The decoder puts a turnover at the first control frame of the new team.
- **Loose balls and passes in flight.** Possession stays with the last team that controlled the ball. With no player within 3 m, the label equals the latest controlling team on 83.0% of alive frames and the nearest player's team on 64.6%. The latch channel encodes this.
- **Challenges.** A contested ball stays with the last team until another team gains control. A brief intrusion must outweigh twice the turnover cost to flip it.
- **Dead ball starts** when the ball leaves the pitch, when the ball vanishes at a line, at a set-piece tag, or when the match goes quiet (still ball, low player activity). Line crossing or tracking loss alone is not enough: of 4 public stoppages the ball crosses a line in 2 and vanishes in 2, and in two cases `D` starts about 3 s before either.
- **Restarts.** Dead ball ends at the strike or throw, before the first control event. The restarting team's carry run starts a median 15 frames after the labelled end (n=4). So v1 puts the restart evidence on the first frame after the dead-looking stretch (the strike), and uses the first control event within 3 s only to name the team.
- **Short possessions.** The two public possessions under 1.5 s show no ball velocity change beside a player (peak 2.1 and 1.3 m/s). The model cannot place them from touch evidence.
- **Goalkeeper holding the ball.** Undetermined from the public window. The still-ball channel treats a held ball as `D`.
- **Kickoffs.** The first-half kickoff frame and the second-half pre-kickoff block are organiser-published constants. After a goal, the decoder must find the stoppage itself.

## 4. Data and assets

**Used.**

- The tactical ATD (tracking of 46 slots and the ball) and the Smart Tagging XML.
- Organiser-published constants: the graded half blocks (`games.half_video_blocks`) and the second-half kickoff frame (`KICKOFF_VIDEO_FRAMES` in `starter/load_smart_tagging.py`, 113742).
- Pitch size from `games.py` (105.3 m by 68.0 m).

**Tried and dropped.**

- Pixel computer vision for the ball in blackouts. About 65,000 graded frames would need inference on a full-pitch 1920x1080 view where the ball is a few pixels wide. The estimate is 9 to 18 hours on the available CPU, with no labels to validate the result. The vendor's detector already ran on those pixels.
- Video and homography. The homography is used only for optional visual debugging. No video is read by the pipeline.
- Baum-Welch fitting (reason in section 1). The four DEFENDING Smart Tagging codes (they mirror attacking codes of the other team with identical intervals, so they add no second opinion).
- Broadcast recording: alignment found (tactical frame = broadcast frame + 4013 to 4015, stable on 43 of 110 one-minute chunks with a player fit under 2.5 m). Coverage of tactical blackouts was not measured at v1, so the go criterion (accepted broadcast ball on at least 15% of blackout frames) was never tested. Not used in v1.

**Training data.** Two external labelled datasets, no human labels made by the team, and nothing from the provided game used as a label. The public 5-minute window was used only as a smoke check, scored once per frozen version, and selected nothing.

| dataset | source | who labelled | amount | role |
|---|---|---|---|---|
| Metrica Sports public sample data | github.com/metrica-sports/sample-data. The repository has no licence file. Its README asks users to acknowledge the source, which we do here | Metrica's own event data for the 3 games; our code derived the per-frame labels (`harness/metrica.py`) | 3 anonymised games, about 188 alive minutes | primary calibration set |
| Open DFL Bundesliga tracking | Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12, 195 (2025), CC-BY 4.0. Files from HuggingFace `pysport/idsse-data` (`harness/dfl.py`) | the data provider, as per-frame `BallPossession` and `BallStatus` fields | 7 matches, about 369 alive minutes | reported, never optimised |

DFL labels: `D` where `BallStatus` is dead, else `H` or `A` by `BallPossession`. A frame with a missing `BallPossession` takes the previous frame's label, because kloppy would otherwise read it as away. The DFL export stores player speed in km/h, and `calibration/variants.py` divides by 3.6 on load.
Metrica labels follow a fixed rule in `harness/metrica.py`. On-ball events (pass, shot, recovery, set piece, carry) give possession to the event's team. Stoppage events (ball out, fault received, offside or hand-ball ball lost, goal) set `D`, which lasts until the next set piece or goal-kick pass. Other events during a stoppage are ignored. Each half starts as `D` until its first on-ball event.
Metrica is the primary calibration set because its possession statistics match the public window and it has real tracking blackouts (32 to 41% of frames missing). Mean possession is 11.8 to 13.2 s on Metrica, 5.5 to 9.2 s on DFL, 14.2 s in the public window. DFL has no blackouts.
Both are used for calibration only and are not redistributed. The repository holds none of their data.

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

The DFL set was never optimised on and improves on all seven matches: mean spellF1 0.512 (v0) to 0.573 (v1), macroF1 0.900 to 0.915.
The public window is never used to tune. One possession moves spellF1 there by about 6 points.

### Final numbers

This is the one place to update for a later version.

- Version v1, `possession/FROZEN.txt`. File `evaluation/results/Verynicestyleturbo.csv`, 199,319 rows, validator OK, two consecutive runs gave the same hash.
- sha256 `e09133425dce724a8ddfb62054b6e1e10272343f0308b6a36946aee5457f1f88`. For reference, v0 was `36e19f4c184f3e12629e77b7eb10b23b5492e91fa0ff5596205707bcd2dc0ce3`.
- Harness spellF1 (v0, then v1 as shipped), Metrica games 1 to 3. Clean: 0.599, 0.646, 0.611, then 0.657, 0.694, 0.646. `atd`: 0.487, 0.539, 0.469, then 0.549, 0.576, 0.600. `atd+mask`: 0.370, 0.391, 0.349, then 0.452, 0.445, 0.443. DFL mean over 7 matches (spellF1/macroF1): 0.512/0.900, then 0.573/0.915. Source: `python -m calibration.report`.
- Public-window smoke, scored once and used to select nothing: spellF1 0.422 and median latency -12 frames (v0: 0.384 and +11).
- Release gates (`python -m possession --no-smoke`), v1 against v0:

| gate | v1 | v0 | range |
|---|---|---|---|
| validator | OK | OK | OK |
| `D` share of graded frames | 0.421 | 0.553 | 0.30 to 0.55 |
| possessions per graded minute | 3.08 | 1.58 | 1.5 to 5 |
| mean possession | 11.3 s | 17.0 s | 6 to 20 s |
| home share of alive frames | 0.569 | 0.630 | 0.45 to 0.75 |
| set-piece windows with `D` to alive within 3 s | 49 of 59 (83%) | 51 of 59 | 80% or more |
| `D` before the 4 kick-offs after a goal | 43, 19 (no restart), 131, 221 s | 124, 19, 133, 230 s | 20 s or more each |

  Two checks fall outside their range and were not tuned away. The kick-off check fails on one of four (19 s, where no restart is found). The smoke latency flipped sign, so transitions are now early by a median 12 frames on a handful of matched transitions. The restart anchor moved to the strike in this version, which may be the cause. It was not investigated, because the smoke selects nothing.

## 6. Limitations

- Blackouts. About 40% of graded frames have no ball. In those frames the model relies on player activity, tags and the edges of the gap, which is a weak signal. The share of alive play inside blackouts is unknown.
- Short possessions. A possession of about one second carries no touch signal in the ATD ball track, which is smooth. The model tends to merge them.
- Transition timing. On the public window v1's matched transitions come a median 12 frames early (v0: 11 late). Too few transitions to say whether the restart anchor causes it.
- Calibration transfer. The DFL possession definition differs from the competition's, and DFL has no blackouts. Metrica's tracking quality differs from this ATD. Quantile transfer and the masked variant reduce this but do not remove it. With three calibration games, over-fitting is the main risk; the leave-one-game-out check bounds it at 0.03.
- Smart Tagging windows run past the possession they describe. Agreement with local carry falls from 0.83 in the first third of a tag to 0.56 in the last third. v1 does not use this to reweight the tags.
- Goalkeeper holds are undetermined (section 3).
- The broadcast recording is not used. Its alignment to the tactical frames was found, but how many blackout frames it would cover was not measured (section 4).
- With more time: a better restart anchor (onset of ball motion), a model of alive play inside blackouts, and more labelled games with real blackouts.

## 7. Reproduction

1. Python 3.12 (tested with 3.12.12) with the pinned packages in `requirements-submission.txt`: `numpy==2.4.4`, `pandas==2.2.3`. Tests add `pytest==9.2.0` and `hypothesis==6.170.15` (`requirements-dev.txt`). Byte-identical output depends on the numpy and pandas versions. The calibration harness uses a separate environment (`harness/requirements.txt`, kloppy and newer numpy and pandas) and is not needed to regenerate the file.
2. Place the data bundle as described in `DATA.md` (`./data`), or set `MTAH_DATA_DIR` to the folder that holds it. The data is not in the repository.
3. Run `python -m possession`. On first run it parses the ATD and tagging once into an array cache under `$MTAH_DATA_DIR/cache`, about 11 s. It then writes `evaluation/results/Verynicestyleturbo.csv`, runs `evaluation/validate_submission.py`, prints the sha256 and the release gates. `--no-smoke` skips the public-window gate. `--cache-dir <empty dir>` forces a parse from the raw files.
4. Expected sha256 of the submitted file: see "Final numbers" in section 5.
5. Tests: `pytest tests/` (property tests listed in `tests/PROPERTIES.md`). Two consecutive runs of the command gave the same sha256.

Every label comes from code. The only constants tied to this match are the organisers' published half blocks and kickoff frame.
