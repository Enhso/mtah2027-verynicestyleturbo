# Broadcast probe progress (Phase 5)

Difficulty: medium-hard. Parsing is routine, but the offset search has an unknown range, unknown
orientation, and the broadcast camera's partial view gives a weak, ambiguous signal.

## Log
- Started 2026-11-16 (90 minute box).
- Parsed broadcast ATD (164488 frames, 51 slots, ball 50.5%, same 0..1 normalisation). Identity orientation (no flip) fits.
- Probe chunks (broadcast frames 5k..70k) all peak at offset about +4010 (tactical = broadcast + offset), score 0.7 to 3 m vs median 8 to 20 m at wrong offsets.
- Full per-minute search (broadcast/align.py, 110 chunks, window 0..12000): 43 chunks fit under 2.5 m; their offsets span 4013..4015 (spread 2 frames).
- Checkpoint status (cut for usage limit): offset stability criterion looks met. NOT measured: per-frame acceptance under 2.5 m, coverage of tactical blackout frames in graded halves (go needs >= 15%), ball agreement, output .npy. No verdict yet; the coverage number decides go/stop.

## Run 2 (2026-11-16, 2 hour box)
- Difficulty: medium-high (stretch detection on a cut-and-replayed timeline is the unknown).
- Step 0 done: core.py / parse.py / align.py now write to possession.match.CACHE_DIR (bc.npz, chunks.npz, bc_ball.npz). parse.py rerun: 164488 frames, ball 50.5%.
- align.py window 0..12000 reproduced: 43/110 chunks under 2.5 m, offsets 4013..4015, covering broadcast frames ~3000..85000 (first half; tactical first half = 11185..90652).
- After broadcast frame ~85500 the 0..12000 window finds nothing (chunk scores 5-18 m). Wide test (4 flips, 3 chunks at 100k/120k/140k, offsets 12000..end): identity orientation, offset ~26920 (score 0.6 to 1.7 m; the three other flips 6 to 11 m). So the second half sits at a different offset (~+26920), as expected from a cut halftime.
- align.py rewritten to search the whole range (offset 0..end of tactical track, coarse step 8, 7 processes) so the stretch offsets are found rather than assumed.
- Whole-range align.py (110 chunks): 85 fit under 2.5 m. Two offset clusters: 4013..4015 (broadcast frames ~3000..85000) and 26917..26923 (broadcast ~87000..164488, mostly 26919/26920). One boundary (halftime gap, ~85000..87000). Wide align reproduces 4014 exactly for the first half.
- Pre-registered trust bar for ball agreement (set before computing it): median distance <= 2 m and >= 90% of jointly visible frames within 5 m. Why: the possession call turns on which player is nearest the ball, and 5 m is about the typical nearest-opponent spacing; a 1 frame offset error alone moves a fast ball about 0.5 to 1 m.

## Result (fill.py, 2026-11-17)
Command chain (about 25 min cold; bc_ball.npz hash identical after deleting the fit cache and rerunning):
    export MTAH_DATA_DIR=$MTAH_DATA_DIR
    PY=.venv/bin/python
    $PY broadcast/parse.py && $PY -W ignore broadcast/align.py && $PY -W ignore broadcast/fill.py

Stretches (window = 250 frames, offset cluster per window, boundary at the midpoint of the gap between clusters):
| stretch | broadcast frames | offset (tactical = broadcast + o) | tactical frames | per-minute offset |
|---|---|---|---|---|
| 1 | 0..85875 | 4014 | 4014..89889 | 4010..4020; 86% within +-1, 96% within +-2 (49 minutes) |
| 2 | 85875..164488 | 26920 | 112795..191408 | 26913..26925; 83% within +-1, 90% within +-2 (48 minutes) |
Pooled by thirds of each stretch the best offset is 4014/4014/4013 and 26920/26920/26919, so within a stretch the offset moves by 1 frame; the per-minute range of 10 to 12 is single-minute noise (MAD sd 1.5 frames), no drift. The broadcast cuts halftime: graded block 1 (11185..90652) is stretch 1, graded block 2 (110808..190198) is stretch 2 from tactical 112795 on; the first 2000 frames of block 2 and the last ~750 of block 1 have no broadcast.
Null control: acceptance at a wrong offset (+500 frames) is 0.0000, at the stretch offsets 0.556 of all broadcast frames.

| per graded half | half 1 | half 2 |
|---|---|---|
| graded frames with an accepted broadcast frame | 0.587 | 0.553 |
| graded frames with an accepted broadcast ball (tactical ball 0.593) | 0.481 | 0.434 |
| blackout frames (tactical ball NaN) | 0.407 of half | 0.407 of half |
| blackout frames covered by accepted broadcast ball, all | 0.077 | 0.083 |
| covered, blackouts 5-30 s (n=49 / 64; any coverage 30 / 35) | 0.114 | 0.111 |
| covered, blackouts over 30 s (n=11 / 8; any coverage 8 / 4) | 0.035 | 0.041 |
| blackout frames with an accepted broadcast frame (any ball) | 0.278 | 0.297 |
| blackout frames where the broadcast ball exists at all, acceptance ignored | 0.127 | 0.129 |
| ball agreement (both see): n | 35746 | 31759 |
| median distance | 0.75 m | 0.78 m |
| within 2 m / within 5 m | 0.761 / 0.885 | 0.731 / 0.898 |

Reading: STOP. Accepted broadcast ball covers 7.7% and 8.3% of blackout frames against a go bar of 15%. Acceptance is not the limiter: even taking every mapped broadcast ball regardless of fit, only 12.7% and 12.9% of blackout frames have one. The blackouts are where the broadcast also lacks the ball (30% of blackout frames show live play at all against 56% of frames overall, so replays or close-ups, and the broadcast ball tracker loses the ball in the same moments). Over-30 s blackouts, the main open loss, get 3.5% and 4.1%. Offset stability passes (1 frame within a stretch). Ball agreement is marginal: median 0.75 m is fine, but within 5 m is 88.5% and 89.8% against the pre-registered 90%. Not run: any use of bc_ball.npz in the pipeline, since the coverage gate fails.
