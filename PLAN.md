# PLAN: frame-level possession, optimised for spellF1

Design plan written on 2026-11-16 before implementation. The working copy's task list and
session handoff sections are omitted here.
Deadline 2026-11-17. Budget about 24 hours, 8 CPU cores, 14 GB RAM, no GPU.

Every factual claim carries a tag. `[doc]` is stated in the repo docs or code. `[measured]` was
computed today from files committed in this repo (the scorer, the public ground truth, the four
baseline CSVs). `[inferred]` is reasoning from those. `[unverified]` must be checked before you
rely on it. Nothing under `data/` or `$MTAH_DATA_DIR` was opened to write this plan.

## 1. Mission and guardrails

Write `evaluation/results/<team>.csv` with 199,319 rows and labels `H`, `A`, `D`, `X`. Maximise
`spellF1` on the private full-match ground truth, with `macroF1` as tie-break. One command
regenerates the file, byte for byte.

Guardrails. Each one changes how you work.

1. **Scripts read the data, agents read aggregates.** The competition terms forbid making the
   videos, the datasets "or any extracts thereof" available to a hosted LLM `[doc: FAQ.md,
   OVERVIEW.md]`. You are a hosted LLM. Scripts open `data/` and `$MTAH_DATA_DIR`. What they print
   to you is counts, rates, quantiles and scores. Rows, coordinates, frames and images stay on
   disk for the team. Put this sentence in every subagent brief.
2. **The public 5 minutes check correctness and nothing else** `[doc: RULES.md]`. Use them to
   answer the categorical questions in section 6 and to run the smoke gate in section 7. Select no
   parameter and no variant on them. They also could not rank anything: one possession moves
   `spellF1` by about 6 points there `[doc]`.
3. **Every label comes from code.** A constant tied to a frame of this match is allowed only when
   the organisers published it. That covers the half blocks and the second-half kickoff. It does
   not cover stoppages, which the pipeline must detect.
4. **External data is about other games**, is public, and is named in the technical document.
5. Leave `evaluation/` and `starter/` unchanged. New code goes in a new top-level package.

## 2. Three findings that set the design

**F1. The missing ball is 150 long blackouts, so trajectory reconstruction recovers nothing.**
`[measured]` In `baseline_raw.csv` a `D` means "ball not tracked on this frame". Over the two
graded halves the ball is missing on 40.7% of frames, and those frames sit in 150 gaps.

| | 1st half | 2nd half |
|---|---|---|
| gaps | 69 | 81 |
| gaps of 5 frames or fewer | 8 | 6 |
| gaps of 6 to 125 frames | 1 | 3 |
| gaps longer than 5 s | 60 | 72 |
| median gap | 289 frames (11.6 s) | 245 frames (9.8 s) |
| 90th percentile | 1,083 frames | 639 frames |
| longest | 3,466 frames (2.3 min) | 5,537 frames (3.7 min) |
| gaps longer than 30 s | 11, holding 19.5% of the half | 8, holding 16.2% |
| median visible run | 379 frames | 299 frames |

More than 99% of missing frames lie in gaps longer than 5 s. The vendor has already filled the
short occlusions `[inferred]`. Interpolate the 14 gaps of 5 frames or fewer and stop there. The
work that pays is deciding, for each blackout, whether play is dead, and which team carries
through it. Published baseline scores suggest about two thirds of blackout frames are dead ball
`[inferred]`. `starter/make_dummy_submission.py` states that dead ball is the most common label
`[doc]`. The public window is unrepresentative here. Its ball is visible on 88.2% of frames
against 59.3% for the graded match `[measured]`.

**F2. The ground truth flips on events, and the nearest player is a poor witness.** `[measured]`
The 17 public possessions last 0.8, 1.3, 2.8, 3.2, 3.5, 3.6, 6.6, 7.4, 7.6, 8.1, 11.8, 16.0,
20.0, 28.6, 35.7, 41.6 and 42.4 s. Six are under 4 s. In 5 of the 17 the nearest tracked player
belongs to the other team on more than half of the frames. The 4 public stoppages last 5.5, 14.5,
18.3 and 20.4 s. Two of them have a fully tracked ball and two are mostly untracked, so "no ball" and
"dead ball" are different things in both directions. Consequences: evidence must come from
control events, a minimum-duration filter on possessions would delete real ones, and stoppages
need their own detector.

**F3. No legal labelled signal exists on this match, so calibration needs another one.** Without
an external labelled set every weight in the decoder is a guess that nothing can test. Section 4
names the set.

## 3. Design

One pipeline. The possession state machine is the transition structure, the hidden Markov model is
the scoring, Viterbi is the decoder. Three states, `H`, `A`, `D`, decoded separately in each
graded half. Run time needs numpy and pandas only.

| stage | does | output |
|---|---|---|
| load | parse the ATD text and the Smart Tagging XML once, cache arrays outside git | ball (N,2) in metres, players (N,46,2), speed (N,46), team (46), tag intervals |
| features | per-frame geometry and kinematics | ball velocity inside visible runs only, nearest home and away distance, per-slot player velocity with a jump guard, median player speed, distance to the nearest line |
| evidence | turn features into a score per state | (N,3) float64 in nats |
| decode | Viterbi per half, then guards | (N,) labels |
| write | CSV, validator, sha256 | the submission |

Evidence channels. A channel contributes zero wherever its input is missing, so blackouts need no
separate code path. "First value" is a prior that lets v0 run. No number in that column was
fitted to anything.

| channel | fires when | supports | counted | first value | calibrate on |
|---|---|---|---|---|---|
| carry | ball within 2.0 m of a player, the nearest opponent at least 1.0 m farther away, ball velocity within 2 m/s of the player's | that team | per frame | 0.2 | DFL |
| touch | ball velocity changes by 4 m/s or 35 degrees across 3 frames, nearest player within 2.5 m | that team | per event | 2.0 | DFL |
| restart | first touch after a `D` stretch | that team | per event | 4.0 | DFL |
| phase tag | a team-tagged attacking phase covers the frame | tagged team | once per phase, spread over its frames | log-odds of its measured agreement | this match, label-free |
| off pitch | ball position outside the lines | `D` | per frame | 10 | law of the game |
| exit gap | a blackout starts with the ball within 3 m of a line and moving outward | `D` for the gap | once per gap, spread | 8 | DFL with this match's mask |
| long gap | a blackout longer than 30 s | `D` | once per gap, spread | 4 | this match, label-free |
| still ball | ball speed under 0.5 m/s for 2 s or longer | `D` | per frame | 0.3 | DFL |
| low activity | median player speed under 1.5 m/s for 3 s or longer | `D` | per frame | 0.1 | DFL |
| set-piece tag | a `SET PIECES` window | `D` before the restart candidate inside it, alive after | once per tag | 3 | this match, label-free |
| pre-kickoff block | frames 110808 to 113741 | `D` | hard constraint | infinite | `[doc]` |
| latch | always | the team of the latest carry, touch or restart | per frame | 0.001 | fixed |

Decoder. Transition costs are three numbers: turnover (`H` to `A` or back) 3.0, stop (alive to
`D`) 4.0, restart (`D` to alive) 4.0. Frame 11185 is alive. Costs and weights share one scale, so
only their ratios matter. Calibrate them together.

How the user's ideas map onto this:

- Dwell thresholds become accumulated evidence. A team takes over when its evidence exceeds the
  turnover cost, and a brief intrusion must exceed twice that cost. This matches the two proxies
  in CHALLENGE.md, holding the ball and completing consecutive passes.
- The latch places a turnover on the first control frame of the new team, which is where the
  ground-truth definition puts it. A loose or contested ball stays with the last team `[doc]`.
- Count a tag or a blackout once, not once per frame. Frames inside one interval are one
  observation, and summing them lets a single wrong tag overrule the tracking.
- The second-half pre-kickoff block is hard-coded from the published constant. Import
  `KICKOFF_VIDEO_FRAMES` from `starter/load_smart_tagging.py`.
- No Baum-Welch. Unsupervised fitting of three states finds motion regimes, and nothing ties them
  to `H`, `A` and `D`.

Guards after decoding, kept to three. Merge a `D` run shorter than 25 frames into its neighbours
and print how many were merged. Write `X` on every ungraded frame. Write no `X` on a graded
frame, because the decoder always has an answer and `macroF1` charges an `X` as a miss.

One rule decides every doubtful case. Adding a predicted possession raises F1 when its chance of
matching exceeds half the current F1 `[inferred from the formula]`. At F1 0.5 a one-in-four guess
pays. So in an alive blackout, commit to a coarse team call from tags and the edges of the gap.

## 4. Calibration, external data and computer vision

Each parameter takes its value from the first source in this list that can supply it.

1. Laws of the game and pitch geometry. Fixed.
2. The DFL harness. All control and dead-ball weights, the three costs, the thresholds.
3. Label-free statistics of this match. Tag reliability, set-piece timing, threshold transfer.
4. First values from section 3, when a source above is unavailable.

**External data: yes, the open DFL set, with a 3-hour box.** Seven Bundesliga matches, TRACAB
tracking at 25 Hz, CC-BY 4.0, with a per-frame `BallPossession` and `BallStatus` on the ball
`[verified today from the kloppy docs and source]`. `kloppy.sportec.load_open_tracking_data(
match_id=...)` downloads and parses a match. The ids are J03WMX, J03WN1, J03WPY, J03WOH, J03WQQ,
J03WOY and J03WR9. These labels have the same shape as the competition's: dead where the status
says so, otherwise the owning team. A missing `BallPossession` attribute reads as away in kloppy,
so check the label statistics before trusting a match. The docs read today describe kloppy
3.19.1. Whether the pinned 3.18.0 has this loader is `[unverified]`, so the harness may need its
own environment.

What to calibrate on it. Not the transition matrix alone. In a decoder with hand-scaled evidence
the transition costs and the evidence weights trade off against each other, and the transition
frequencies of another league have low leverage. Use the harness to choose between control
definitions and to set the cost-to-weight ratios against `spellF1` itself. Read the transition
counts and duration distributions from the DFL labels as starting costs.

Three limits, each with its countermeasure.

- The DFL definition of possession may differ from Metrica's. Print, per DFL match, possessions
  per alive minute, the share of possessions under 2 s and under 4 s, and the median stoppage.
  Compare with the published public figures (17 possessions in 5 minutes, mean 14.2 s, six under
  4 s). A large mismatch means you keep the DFL-tuned thresholds and return the turnover cost to
  its first value.
- DFL tracking is clean and the ATD is automatic. Transfer each threshold by quantile matching.
  A threshold at quantile q of a feature on DFL becomes the value at quantile q of the same
  feature on this match, both computed on visible-ball frames.
- DFL has no blackouts. Apply this match's ball-visibility mask to the DFL ball and re-tune only
  the blackout channels. The transplanted gaps are unrelated to the DFL stoppages, so use this
  run for the alive-blackout policy and not for the exit-gap weight.

The Metrica open sample data (3 games, same vendor and coordinates) is the alternative. It has no
per-frame possession field, so labels would have to be derived from its events. Use it only if
the DFL export fails.

**Computer vision on pixels: no.** The gap to close is the ball in long blackouts. Metrica's
production detector already ran on those pixels. On a full-pitch 1920x1080 view the ball is a few
pixels wide `[inferred]`. About 65,000 graded frames need it, which is 9 to 18 hours of tiled
small-object inference on this CPU `[inferred]`, with no labels to validate the result.

**Broadcast tracking: an optional 90-minute probe, last in line.** The broadcast ATD already
carries a ball track (50.5% of its frames) `[doc]`. It needs time alignment, not vision. A
broadcast recording runs in real time, so expect one constant offset per continuous stretch
`[unverified]`. Estimate the offset per one-minute chunk by minimising the mean nearest-neighbour
distance between broadcast and tactical players of the same team. Accept a broadcast frame only
when that distance is under 2.5 m, which rejects replays. Go on if the accepted broadcast ball
covers at least 15% of tactical blackout frames and the offset varies by 2 frames or less within
a stretch. Stop otherwise.

## 5. Steps

Times are hours from the start of the implementation session. Each phase ends on a check.

**Phase 0, setup. T+0 to T+1.**
Create the environment and pin it. Write `load` and the cache.
Done when a script reproduces these `[doc]` figures: 199,319 records with frame ids 1 to 199,319;
ball present on 113,488 frames overall and 94,214 of 158,857 graded frames; the first team in the
metadata is Barcelona; `python starter/load_smart_tagging.py` prints its OK line.

**Phase 1, core v0. T+1 to T+4.**
Write `features`, `evidence` with first values, `decode`, `write`, and the property tests of
section 7. Answer the semantic questions of section 6 and adjust channel definitions to the
answers.
Done when the validator passes, two runs give the same sha256, the property tests pass, and the
release gates are printed. Keep this file. It is the insurance submission.

**Phase 2, DFL harness. T+1 to T+4, one builder in its own worktree.**
Export the seven matches to the array schema of `load` plus a `frame,label` file. Wrap
`evaluation/possession_metrics.evaluate`.
Done when each label file scores `spellF1` 1.0 against itself, the pipeline runs end to end on
all seven, and the label statistics above are printed.
If this is not done at T+5, stop it and continue with first values and label-free calibration.

**Phase 3, calibrate. T+4 to T+9.**
Compare three control definitions on DFL: carry only, touch only, both. Tune by coordinate
descent on coarse grids of 3 to 5 values, two passes, objective mean `spellF1` over matches.
Then the masked run for blackout channels. Then quantile transfer to this match.
Done when parameters chosen on six matches score within 0.04 of the all-seven choice on the
seventh, for every held-out match, and `params.py` records a source for each value. Freeze v1.

**Phase 4, dead ball and blackouts on this match. T+9 to T+12.**
Run the label-free analysis of section 6. Set the phase-tag and set-piece channels from it.
Done when every question has a printed answer and a recorded decision, and v2 passes the release
gates or each failed gate has a written explanation.

**Phase 5, broadcast probe. T+12 to T+15, optional.** Section 4 gives the go and stop numbers.

**Phase 6, freeze. T+15 to T+19.**
Regenerate from a clean clone. Write the technical document from `params.py` and the decision
trail: approach, assets used and dropped, the DFL data and its licence, possession rules,
limitations, reproduction command.
Done when the clean-clone sha256 equals the frozen one and the validator passes on that exact
file.

T+19 to T+24 is slack for the team's submission.

When late, cut in this order: Phase 5, the masked re-tune, the phase-tag channel, the low-activity
channel. Keep the pre-kickoff block, the validator and the determinism check to the end.

If one piece goes to the strongest model, send the control-definition comparison of Phase 3.

## 6. Analysis guidelines

Every script prints aggregates. Each question names the decision it feeds.

Semantic questions for the public window. Write each answer as one category before looking at a
second question. These decide definitions, never numbers.

1. Does a possession start on the first touch of the new team or after sustained control?
2. Does the owner stay unchanged through a pass in flight and through a loose ball?
3. Does `D` start when the ball crosses the line or stops being tracked, and by how many frames
   do the two differ?
4. Does `D` end on the frame the ball is struck or thrown?
5. In the two possessions under 1.5 s, did the ball change velocity next to a player of the
   owning team?
6. Is the keeper holding the ball alive?

Label-free questions on the full match.

| question | print | decides |
|---|---|---|
| Do positions leave the 0 to 1 range? | min, max, share outside, for ball and players | form of the off-pitch channel |
| What does each blackout look like? | per gap class: count, duration quantiles, share starting within 3 m of a line with outward velocity, share ending near a line or a restart spot | coverage of the exit-gap channel |
| Is player activity lower in blackouts? | median player speed quantiles in visible frames, in blackouts by length class, in the pre-kickoff block | long-gap weight, low-activity threshold |
| Which slots are not players? | per slot: presence share, share of frames within 1 m of a touchline or outside the pitch; pairs of same-team slots under 0.5 m apart | slot filter |
| Which way does each team attack? | mean x of each team in the first minute of each half | restart-spot geometry |
| How much does the tagging cover? | share of graded frames under a team-tagged attacking phase; whether each `DEFENDING` phase mirrors an attacking phase of the other team | whether the tag channel is worth keeping |
| How reliable is each tag code? | per code: agreement with the decoder on frames where carry evidence is strong | per-code tag weight |
| Where is the restart inside a set-piece window? | quantiles of restart candidate minus window start, share of the 59 windows with a candidate | set-piece channel timing |
| Is the ball speed usable? | speed quantiles, count and duration of still runs, speed on the frame before and after each gap | kink and still-ball thresholds |

## 7. Checks

Properties for `tests/PROPERTIES.md`, one test each. Ask the team to confirm every line.

1. The file has 199,319 rows, frames 0 to 199,318 in order, labels in `H A D X`.
2. `X` appears on every ungraded frame and on no graded frame.
3. Frames 110808 to 113741 are `D`.
4. Swapping the two teams in the input swaps `H` and `A` in the output and changes nothing else.
5. Reflecting both pitch axes leaves the output unchanged.
6. On random score matrices of up to 8 frames the Viterbi path scores the same as the best of all
   paths found by enumeration.
7. No ball velocity is computed across a blackout.
8. A synthetic match built from a known label sequence is recovered with `spellF1` of 0.95 or more.
9. Two runs on the same input write the same bytes.

Release gates, printed for every frozen version. A gate outside its range is a reason to
investigate. It is never a target to tune toward.

| gate | range |
|---|---|
| `evaluation/validate_submission.py` | OK |
| share of `D` on graded frames | 0.30 to 0.55 |
| possessions per graded minute | 1.5 to 5 |
| mean possession | 6 to 20 s |
| home share of alive frames | 0.45 to 0.75 |
| set-piece windows with a `D` to alive transition within 3 s | 80% or more of 59 |
| `D` before each of the 4 kick-offs after a goal | 20 s or more |
| public-window smoke, scored once per frozen version | `spellF1` above 0.42, median transition latency within 8 frames |
| DFL harness `spellF1`, clean and masked | recorded |

## 8. Reference

### Frame alignment `[doc]`

| source | index | submission row |
|---|---|---|
| submission, tactical video, homography | 0 to 199,318 | itself |
| ATD | `frame_id` 1 to 199,319 | `frame_id - 1` |
| Smart Tagging | seconds from the first tactical frame | `round(seconds * 25)` |

| block | frames | graded |
|---|---|---|
| before the match | 0 to 11,184 | no |
| 1st half | 11,185 to 90,651 (79,467) | yes, alive from the first frame |
| halftime | 90,652 to 110,807 | no |
| 2nd half | 110,808 to 190,197 (79,390) | yes, all `D` until the kickoff at 113,742 |
| after the match | 190,198 to 199,318 | no |

`games.half_video_blocks("BAR-ATH")` returns the two graded blocks. The per-half offsets in
`games.py` belong to the counter the ground truth was derived from. You do not need them.

### ATD `[doc]`

- Raw line: `<frame_id>:<slot>;<slot>;...:<ball>`. Each group is `x,y,speed`. A missing
  observation is `NaN`. `starter/baseline.py` is the reference parser.
- 46 slots on the tactical recording, in the order of the `Player` entries of the metadata XML.
  `teamId` gives the team. The first `Team` is home, Barcelona, `H`. Athletic is `A`.
- A slot is a tracklet. It appears, vanishes and changes person. Mean 20.5 tracked objects on
  graded frames, sometimes more than 22.
- Speed is in m/s. Whether the ball group has a third field is `[unverified]`.
- kloppy 3.18.0 reads the pair with `metrica.load_tracking_epts`. A full load costs minutes and
  gigabytes, so parse the text directly.

### Coordinates

- Positions are normalised to 0 to 1 on both axes `[doc]`. Metres are x times 105.3 and y times
  68.0, the pitch in `games.py`. The metadata says 105 by 68.
- Origin and axis direction are not stated in this repo. Metrica's convention is origin top
  left, (1,1) bottom right, (0.5,0.5) the kickoff point `[Metrica sample-data README]`. Distances
  do not depend on it.
- Which goal each team attacks is undocumented. Derive it per half.
- The homography maps normalised pitch coordinates to pixels of the same recording. Columns:
  `frame,h00..h22,reprojection_error,valid`. Valid on every tactical frame and on 65.8% of
  broadcast frames. This plan uses it only for the team's visual debugging.

### Smart Tagging `[doc]`

- SportsCode XML, 638 `instance` elements with `start`, `end`, `code` and `label` pairs of
  `group` and `text`. `starter/load_smart_tagging.py` returns half-open frame intervals.
- Codes are upper case, tag values title case. Match case-insensitively.
- `Team` is `Barcelona` (314), `Atletic Club` (288, spelled this way) or `N/A` (36, including
  every throw-in).
- Attacking codes: `BUILD UP` 22, `PROGRESSION` 66, `CREATING CHANCES` 30, `ATTACKING TRANSITION`
  24, `LONG BALL` 29, `BALL IN FINAL THIRD` 93, `BALL IN THE BOX` 72, `PLAYERS IN THE BOX` 97.
- Defending codes: `DEFENDING IN ATTACKING THIRD` 22, `IN MIDDLE THIRD` 66, `IN DEFENSIVE THIRD`
  30, `DEFENSIVE TRANSITION` 24. The counts equal those of four attacking codes, so they are
  probably mirrors and carry no second opinion `[inferred]`.
- `SET PIECES` 59: throw-in 26, corner 12, goal kick 10, free kick 5, kick-off after goal 4,
  kick-off start 2. `GOALS` 4. A real match has more restarts than this, so the tags find some
  stoppages and the tracking must find the rest `[inferred]`.
- An interval is a window. The two kick-off tags open about 3.5 s before the real kickoff.

### Broadcast recording `[doc]`

Own timeline, 164,488 frames, 51 slots, ball on 50.5% of frames, 122 stretches without a
homography. No mapping to the tactical frames is given.

### Metric mechanics `[doc: evaluation/possession_metrics.py]`

- A possession is a maximal run of `H` or of `A` inside one graded half. `D` runs are stoppages.
- Before matching, each `X` takes the last non-`X` label of its half.
- Pairs of the same team are matched one to one, greedily by temporal IoU. F1 is averaged over
  IoU 0.3, 0.4, 0.5, 0.6 and 0.7.
- Cost of a uniform delay: 1 frame 1.000, 3 frames 0.982, 8 frames 0.893, 1 s 0.654.
- A missed stoppage merges two possessions. A false one splits a possession. `D` boundaries
  weigh as much as turnovers.
- Missing a short true possession costs it and one of its neighbours. Inventing one costs two
  false possessions. The two errors are about equal, so flip near even odds.
- Differences under 0.04 on a full match are noise.
- `pm.evaluate(gt_codes, pred_codes)` returns `spell_f1_mean`, `macro_f1`, `n_pred_spells`,
  `mean_spell_duration_s`, `per_class` and `transitions`. The scorer also writes
  `reports/<name>_transitions.csv` with `status`, `gt_frame`, `pred_frame`, `latency`.

### Baseline weaknesses

The baseline gives the ball to the team of the nearest tracked player, writes `D` where the ball
is untracked, then takes a centred majority vote that ignores `D` unless the whole window is `D`.

| | raw | smooth25 | smooth50 | smooth125 |
|---|---|---|---|---|
| full match `spellF1` `[doc]` | 0.0486 | 0.1996 | 0.2433 | 0.2242 |
| full match `macroF1` `[doc]` | 0.6532 | 0.6752 | 0.6724 | 0.6356 |
| public `spellF1` `[measured]` | 0.0378 | 0.2868 | 0.3256 | 0.4242 |
| predicted possessions, full match | 1,949 | 746 | 594 | 398 |
| mean predicted possession | 1.9 s | 5.4 s | 7.2 s | 12.4 s |
| public `D` recall | 0.34 | 0.28 | 0.21 | 0.08 |
| public team accuracy on alive frames | 0.78 | 0.83 | 0.86 | 0.89 |
| public transitions matched within 1 s, of 20 | 11, with 125 spurious | 6 | 2 | 3 |

1. `D` means "ball untracked". Two of the four public stoppages have a fully tracked ball and
   are labelled as possessions. 44% of untracked public frames are alive.
2. The nearest player is the wrong team on 22% of alive frames before smoothing.
3. The vote has no event. It cannot hold a 0.8 s possession or put a boundary on a touch.
4. Boundaries are misplaced. At best 6 of 20 transitions land within 1 s once smoothed.
5. It labels possessions through the second-half pre-kickoff block, where the ball is tracked on
   80.8% of frames.
6. It ignores the tagging, the restarts and the difference between the two halves of the label.

