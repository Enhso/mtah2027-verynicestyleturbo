# labelfree-answers (PLAN section 6, label-free questions on BAR-ATH)

Source: `analysis/labelfree.py` (one command: `.venv/bin/python analysis/labelfree.py`), printed
output in `notes/labelfree-output.txt`. No ground truth used. Proposals only: nothing in
`possession/` was edited. All "proposed" values are label-free and still need the DFL check.

## One row per question

| # | question | answer (numbers from the output) | decision / proposed value |
|---|---|---|---|
| 1 | Positions outside 0 to 1? | Ball: 4.2% of tracked graded frames outside (y 3.6%, x 0.6%), range x -0.07..1.06, y -0.11..1.34. Of those frames 85% are more than 0.5 m and 64% more than 1 m beyond the line (median 1.9 m). Off-pitch runs: 81, median 0.8 s, 25% under 0.3 s. Players: 0.9% of graded slot-frames outside (y 0.8%). | Keep the per-frame off-pitch channel at margin 0.0 (law of the game; only 15% of outside frames are within 0.5 m, so noise is not the issue). Short runs (25% under 0.3 s) rely on the 25-frame dead-merge guard. Do not clip positions. |
| 2 | Blackout anatomy | 134 interior gaps (3 more touch a period edge) cover 40.3% of graded frames. None lasts 1 to 5 s and only 3 are under 1 s: 112 gaps are 5-30 s (median 10.7 s), 19 are over 30 s (median 45.6 s). Start within 3 m of a line: 35% (5-30 s) and 32% (over 30 s); with outward speed >= 0.5 m/s: 24% and 21%. Ball speed on the last visible frame: median 1.8 m/s (25% under 0.84); first frame after: median 11.5 m/s (5-30 s), 15.4 (over 30 s). Gap ends within 3 m of a line 33%/37%, of a restart spot 8%/26%. | Exit channel as written fires on 24% of 5-30 s gaps and 21% of over-30 s gaps (the latter are covered by long-gap). Proposed: `exit_out_speed` 0.0 (30% of gaps instead of 24%) and `exit_dist` 5 m (40%); add an "entry" side: start or end within 3 m of a line or restart spot covers 53% of gaps (70% at 5 m). Because the ball is slow when it vanishes (median 1.8 m/s), the 0.5 m/s outward test is the binding limit. Precision of these needs the DFL mask. |
| 3 | Activity lower in blackouts? | Median player speed (m/s), q10/q50/q90: visible graded 1.01/1.72/3.52; pre-kickoff dead block 0.69/1.17/1.54; blackout 5-30 s 0.61/1.16/1.97; blackout over 30 s 0.24/0.80/1.42. Yes: blackouts look like the dead block (5-30 s) or quieter (over 30 s). | Long-gap weight is supported. Because 5-30 s gaps match the dead block, propose `long_gap_s` 5 and tiering: weight 2.0 for 5-30 s gaps, 4.0 for over 30 s (the tier split is a judgement; the activity evidence supports both). Low-activity threshold: see proposal 4. |
| 4 | Which slots are not players? | 46 slots, 24 home and 22 away. Slots 21, 22 and 43 are never present. Slots 23 and 45 are present 93% of frames with mean speed ~1.0 m/s: the two keepers (their mean x is at the goal ends, see Q5). No present slot has more than 4.7% of its frames near a touchline or beyond. Of 423 same-team pairs, none is within 0.5 m on more than 1% of the frames both are present (maximum 0.8%). Present slots per graded frame: q25 21, q50 22, q95 22. | No slot filter needed. Drop the three never-present slots only if a speed-up is wanted. |
| 5 | Attack direction | Keeper mean x (normalised): 1st half home 0.10, away 0.85; 2nd half home 0.89, away 0.15. First-minute outfield mean x: home 0.31 vs away 0.37 (1st half), home 0.69 vs away 0.61 (2nd). | 1st half: home defends low x, attacks high x; away the reverse. 2nd half swapped. Restart-spot geometry: a team's goal-kick spot is near its own keeper's end. |
| 6 | Tag coverage and mirrors | 68.1% of graded frames lie under a team-tagged attacking phase (home 46.1%, away 31.2%; both teams tagged at once on 9.2%). DEFENDING tags: 142, of which 140 have a team; every one has an attacking tag of the other team with identical interval (IoU 1.0 for 100%) and the paired code; same-team overlap 1.4%. | Keep the phase channel. DEFENDING codes mirror exactly and add no second opinion; keep ignoring them. The Team field on a DEFENDING tag names the defender (agreement with carry team 9% for the attacking third, 8% middle, 17% transition, 35% defensive third, i.e. inverted). |
| 7 | Per-code tag reliability | Agreement with local strong carry (>= 10 carry frames of one team, none of the other, within +-1 s; 24.9% of graded frames; the no-tag decoder agrees with this carry team on 96.9% of alive strong frames). Tag-level majority agreement, all attack codes pooled: 73.4% (Wilson 69-78%), log-odds +1.01. By code (shrunk, see proposal 1): progression 0.92, build up 0.91, attacking transition 0.83, ball in final third 0.74, long ball 0.74, ball in the box 0.68, creating chances 0.65, players in the box 0.57. Agreement by third of the tag: first 0.83, middle 0.77, last 0.56. | Per-code weights: proposal 1. Tag windows run past the possession (last third near chance level): spread the weight over the first two thirds of each tag window. |
| 8 | Restart inside set-piece windows | 59 windows, 58 fully inside graded periods; length is 35.0 s for at least 75% of them. Candidate offsets from window start (s), share of windows with a candidate: touch event 69%, median 14.6 (q25 9.3, q75 22.0); first carry frame 93%, median 7.2 (goal kicks 0.6, the ball is carried before the kick); ball reappears 78%, median 6.4 (IQR 4.5-9.9); ball speed > 2 m/s after a hidden or still frame 80%, median 6.4 (IQR 4.6-8.7). By type, launch candidate median: throw-in 6.8, corner 6.8, free kick 9.0, goal kick 7.3, kick-off after goal 3.4, kick-off start 3.7. Decoder D share before / after the candidate: touch 0.59/0.29, carry 0.87/0.33, reappears 0.91/0.39, launch 0.94/0.40. | Replace the touch-event candidate (late, missing in 31% of windows) with the launch candidate. Rule in proposal 2. |
| 9 | Ball speed usable? | Centred speed (m/s) on tracked graded frames: q5 0.50, q25 2.4, q50 5.1, q75 9.5, q95 17.3, q99 24.6, q99.9 54.6; 0.34% above 35 m/s and 0.17% above 45 m/s (tracking glitches). Only 0.6% of tracked frames have no velocity because of a blackout. 3-frame velocity change: q50 0.68, q90 2.2, q99 4.6; 1.4% of tracked frames meet the 4 m/s kink before clustering. Still runs (speed < 0.5 m/s for >= 2 s): 22, median 3.6 s, 1.6% of graded frames (at 1.0 m/s 46 runs, 3.1%; at 1.5 m/s 71 runs, 4.8%). The ATD ball group carries a third field on every record sampled. | Ball speed is usable. Keep still-ball at 0.5 m/s, 2 s (rare, so it adds little but costs nothing). Keep the 4 m/s kink (it sits near the 98.6th percentile). Treat speeds above 45 m/s as glitches (not a feature of the channels; mention only). |

## Proposals

### 1. phase_w and per-code tag weights (log-odds of measured agreement)

Measured as tag-level majority agreement with local strong carry, shrunk toward the pooled rate
with 10 pseudo-tags, floor 0. The decoder used as a cross-check had phase and set-piece weights set
to 0 so the tag does not agree with itself.

| code | tags with evidence | raw agreement | proposed weight (nats per phase) |
|---|---|---|---|
| pooled (all 8) | 380 | 0.734 | 1.0 (`phase_w`) |
| progression | 64 | 0.922 | 2.2 |
| build up | 22 | 0.909 | 1.8 |
| attacking transition | 24 | 0.833 | 1.4 |
| ball in final third | 84 | 0.738 | 1.0 |
| long ball | 19 | 0.737 | 1.0 |
| ball in the box | 62 | 0.677 | 0.8 |
| creating chances | 26 | 0.654 | 0.7 |
| players in the box | 79 | 0.570 | 0.4 |

Caveat: the reference is local carry, itself an imperfect truth (the 25% of graded frames with
strong carry are the easy ones). Agreement on harder frames is probably lower, so treat these as
upper bounds. Spread each phase over the first two thirds of its window (agreement 0.83, 0.77,
0.56 by third).

### 2. Set-piece channel timing

Restart candidate `c` = first frame inside the window at which centred ball speed exceeds 2 m/s
after at least one hidden or still frame inside the window. Present in 80% of windows (47 of 59),
offset median 6.4 s, IQR 4.6-8.7 s, 5% quantile 2.9 s. Rule: `D` from window start to `c`, alive
(either team, tag weight 3 as in v0) from `c` to window end. When no candidate exists (20% of
windows) fall back to the frame the ball reappears; when neither exists, `D` for the first 6.4 s
(the median) and nothing after. Drop the touch-event candidate. Windows are 35 s long, so most of
the window is alive: only about 5 to 9 s of each is `D`.

### 3. Do long-gap and exit-gap cover the blackouts?

| gap class | gaps | exit fires | long-gap fires | either | blackout frames |
|---|---|---|---|---|---|
| under 5 s | 3 | 0% | 0% | 0% | negligible |
| 5-30 s | 112 | 24.1% | 0% | 24.1% | 35,254 (22% of graded) |
| over 30 s | 19 | 21.1% | 100% | 100% | 28,223 (18% of graded) |

Frame view: of all blackout frames in gaps of 5 s or longer, 59.2% sit in a gap where either channel
fires. The gap in coverage is the 5-30 s class (76% of those gaps unreached). Evidence for
extending: their median player speed (1.16 m/s) matches the dead block (1.17), and 53% of them
start or end within 3 m of a line or restart spot (70% within 5 m).

### 4. Low-activity threshold as a quantile

Quantiles of median player speed (m/s):

| group | q5 | q10 | q25 | q50 | q75 | q90 | q95 |
|---|---|---|---|---|---|---|---|
| visible graded frames | 0.82 | 1.01 | 1.29 | 1.72 | 2.69 | 3.52 | 4.05 |
| blackout 5-30 s | 0.50 | 0.61 | 0.85 | 1.16 | 1.43 | 1.97 | 2.90 |
| blackout over 30 s | 0.18 | 0.24 | 0.42 | 0.80 | 1.22 | 1.42 | 1.53 |
| pre-kickoff block | 0.55 | 0.69 | 0.96 | 1.17 | 1.34 | 1.54 | 1.79 |

Proposal: threshold = the 20th percentile of the graded visible-frame distribution (1.21 m/s in
this match), run length 3 s as in v0. It fires on 12.4% of visible frames, 35% of the pre-kickoff
block (likelihood ratio 2.9, log-odds 1.35 nats) and 57% of blackout frames. The v0 value 1.5 m/s
is the 38th percentile and fires on about a quarter of visible frames, which is why `D` share hit
0.55. The 25th percentile (1.29 m/s) has the highest ratio, 3.0, but fires on 16% of visible
frames; take 15% to 25% as the range for DFL to settle. The per-frame weight stays small (0.1
nats): these frames are strongly autocorrelated, so the per-frame ratio must not be summed
at face value.

## Snags
- `analysis/labelfree.py` needs the cache; it exists under `$MTAH_DATA_DIR`, no rebuild was needed.
- The Q3 blackout rows include the 3 period-edge gaps that Q2 skips (they are the small 6-25 f and
  26-125 f classes with 24 and 122 frames).
