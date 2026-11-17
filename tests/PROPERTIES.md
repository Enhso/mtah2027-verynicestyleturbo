# Properties encoded by the tests

One line per property of PLAN section 7. Each names the test in `tests/test_properties.py` that
encodes it. The team confirmed that each line holds.

1. The written file has one row per frame, `frame` runs 0 to N-1 in order, and every label is one of `H A D X`. `test_p1_file_shape`
2. `X` appears on every frame outside the graded periods and on no frame inside them. `test_p2_x_only_on_ungraded_frames`
3. Every frame the per-match config marks as forced dead is `D`, and for BAR-ATH that block is frames 110808 to 113741. `test_p3_forced_dead_block_is_dead`, `test_p3_bar_ath_block_bounds`
4. Swapping the two teams in the input (slot teams and the Smart Tagging `Team` values) swaps `H` and `A` in the output and changes nothing else. This holds except where an exact H/A tie has no data to break it, for example a forced-alive frame with no team evidence: ties keep the previous state, then the lower state index, so the BAR-ATH first frame (forced alive) could go to H in the swapped input too. The test therefore does not force a frame alive. `test_p4_team_swap`
5. Reflecting both pitch axes leaves the output unchanged. `test_p5_reflection`
6. On random score matrices of up to 8 frames, with random costs and random hard constraints, the Viterbi path scores the same as the best of all paths found by enumeration, and respects the constraints. `test_p6_viterbi_matches_enumeration`
7. No ball velocity is computed across a blackout: wherever a velocity is finite, every frame in its window is a visible frame of one run, and gaps of more than 5 frames stay NaN. `test_p7_no_velocity_across_blackout`
8. A synthetic match built from a random known label sequence is recovered with `spellF1` of 0.95 or more. The synthetic world is built to match the channels (a holder dribbles and passes, opponents stay far, stoppages put the ball off the touchline, players slow down), so this tests the machinery and says nothing about realism. `test_p8_synthetic_recovery`
9. Two runs on the same input write the same bytes. `test_p9_same_bytes`
10. The goal rule puts weight only on `D`, and only between a GOALS tag start and the end of the next kick-off-after-goal window; the set-piece channel puts weight only inside set-piece windows, for both the touch and the launch rule. `test_p10_restart_channels_stay_inside_their_tags`
11. The team-shape channel at any visible-ball and blackout weights: swapping the teams negates the shape features exactly and swaps its H and A columns, reflecting both pitch axes leaves features and channel bit-identical, and it never adds to the favoured team or to D (it only penalises the disfavoured team). `test_formation_channel_symmetry`

Supporting checks that are not PLAN properties: the parameter table has a source for every field
(`test_params_have_sources`), the guard that merges short `D` runs respects hard constraints
(`test_guard_merges_short_dead_runs`), a match with no ball, no speed and no tags gets zero evidence in every channel that reads them, that is every channel except the team-shape one, which reads only players (`test_absent_inputs_contribute_zero`), and the real-data file checks run only when the cache and
the submission exist (`test_real_*`).
