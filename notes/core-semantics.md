# Semantic questions of PLAN section 6, answered on the public window

Produced by `python -m possession.semantics` (frames 11185 to 18684; 17 possessions, 20
transitions, 4 stoppages, label shares H 0.461, A 0.344, D 0.196). Aggregates only. Offsets are in
frames: an offset of +15 means the event comes 15 frames after the GT transition. The sample is tiny (see each n), so each answer is a category for a definition, not a
measurement of a number. No parameter was selected on these figures.

| # | question | category | evidence |
|---|---|---|---|
| 1 | possession starts on first touch or after sustained control | first touch | New team's nearest carry run starts at median +7 frames from the GT start (n=13 of 16 transitions; 3 have no carry run within 125 frames). A sustained-control rule would put the GT start at least 12 frames after the carry onset, and the median says the opposite. The touch detector fires near only 6 of the 16 (median -7). |
| 2 | owner unchanged through a pass in flight or a loose ball | latch (unchanged) | On alive frames with no player within 3 m (1666 frames) the GT equals the latest controlling team on 0.830 and the nearest player's team on 0.646. With no player within 6 m (483 frames): 0.826 against 0.727. |
| 3 | D starts at the line crossing or when tracking is lost | neither alone | Of 4 stoppages, the ball crosses a line in 2 and a blackout starts in 2. Within 25 frames of the GT start: off-pitch 1, blackout 1. In two cases D starts about 3 s before the ball leaves or vanishes. Two stoppages have a fully tracked ball that never leaves the pitch. |
| 4 | D ends on the frame the ball is struck or thrown | before the first carry | Restarting team's carry run starts a median 15 frames after the GT D end (n=4). The touch detector finds an event near only one restart. So D ends at the strike and control is detected about 0.5 s later, which is the expected shape of a strike. The ball is not at rest on GT D frames (median 10-frame speed 2.6 m/s, 5th percentile 0.77), so rest does not locate the end. |
| 5 | in the two possessions under 1.5 s, did the ball change velocity next to a player of the owning team | no | 0 of 2. Peak velocity change inside the spells is 2.1 m/s (0.8 s spell) and 1.3 m/s (1.3 s spell), both under the 4 m/s touch threshold. The ATD ball track is smooth: the 99th percentile of the 3-frame velocity change over a whole half is 3.7 m/s, so the touch channel fires 134 times in the first half. |
| 6 | the keeper holding the ball is alive | undetermined | No run of still ball with a player within 1.5 m exists in the window, inside a penalty area or outside. Match-wide the still-ball rule (under 0.5 m/s for 2 s) fires on 2540 frames. |

## Decisions that follow

- Q1: the pipeline places a turnover at the first control frame of the new team (latch, restart and
  carry evidence), which the Viterbi does by construction. No change.
- Q2: keep the latch channel and the "contested or loose ball stays with the last team" reading.
  No change.
- Q3: keep the off-pitch, exit-gap and long-gap channels as separate sources of D. The two
  stoppages whose ball stays tracked on the pitch get D only from the still-ball, low-activity,
  set-piece and restart channels. No change.
- Q4: the restart evidence arrives at the first control event, about 15 frames after the GT D end.
  PLAN section 8 prices a uniform 8-frame delay at 0.893 of spellF1, so a 15-frame lag on restarts
  is a real cost. No shift is built in on 4 cases; Phase 3 should test an earlier restart anchor
  (for example the onset of ball motion) on DFL.
- Q5: a short possession carries no touch signal, so the pipeline cannot place it from the touch
  channel. Whether the carry and touch first values can be re-scaled to resolve it is Phase 3's
  question (quantile transfer from DFL).
- Q6: undetermined. The still-ball channel treats a held ball as `D` and nothing was changed.
