# Broadcast probe progress (Phase 5)

Difficulty: medium-hard. Parsing is routine, but the offset search has an unknown range, unknown
orientation, and the broadcast camera's partial view gives a weak, ambiguous signal.

## Log
- Started 2026-11-16 (90 minute box).
- Parsed broadcast ATD (164488 frames, 51 slots, ball 50.5%, same 0..1 normalisation). Identity orientation (no flip) fits.
- Probe chunks (broadcast frames 5k..70k) all peak at offset about +4010 (tactical = broadcast + offset), score 0.7 to 3 m vs median 8 to 20 m at wrong offsets.
- Full per-minute search (broadcast/align.py, 110 chunks, window 0..12000): 43 chunks fit under 2.5 m; their offsets span 4013..4015 (spread 2 frames).
- Checkpoint status (cut for usage limit): offset stability criterion looks met. NOT measured: per-frame acceptance under 2.5 m, coverage of tactical blackout frames in graded halves (go needs >= 15%), ball agreement, output .npy. No verdict yet; the coverage number decides go/stop.
