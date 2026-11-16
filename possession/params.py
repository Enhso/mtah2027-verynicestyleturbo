"""Every number the pipeline uses, each with where it came from.

Sources: `law of the game` (fixed by the rules), `[doc]` (stated in the repo docs), `first value`
(PLAN section 3 prior that lets v0 run; fitted to nothing), `first value (builder)` (a number the
PLAN does not give, chosen by the v0 builder), `DFL` (checked on the DFL harness).
Phase 3 sources: `Metrica sample games` (chosen by coordinate descent on the three Metrica sample
games, variants atd and atd+mask, see notes/calibration-progress.md), `quantile transfer` (the
Metrica-tuned value moved to the same quantile of the same feature on this match, visible-ball
frames, calibration/transfer.py), `first value` (kept; where the Metrica grid was flat it says so).
Costs and weights share one scale (nats), so only their ratios matter.
"""

from dataclasses import dataclass, fields

FIRST = "first value"
BUILDER = "first value (builder)"


@dataclass(frozen=True)
class Params:
    # --- features -------------------------------------------------------------------------------
    interp_max_gap: int = 5          # frames; ball gaps this short are interpolated, longer stay NaN
    vel_half: int = 2                # frames; centred velocity spans 2*vel_half frames
    still_half: int = 5              # frames; ball speed for the still-ball channel spans 10 frames
    touch_span: int = 3              # frames; velocity before/after a touch is taken over this span
    player_vmax: float = 14.0        # m/s; faster apparent slot motion is an identity jump, not speed
    # --- carry ----------------------------------------------------------------------------------
    carry_radius: float = 3.37       # m
    carry_margin: float = 0.61       # m; nearest opponent at least this much farther
    carry_dv: float = 7.67           # m/s; ball velocity within this of the player's
    carry_w: float = 0.2             # nats per frame
    # --- touch ----------------------------------------------------------------------------------
    touch_dv: float = 1.43           # m/s change across touch_span frames
    touch_angle: float = 28.93       # degrees
    touch_min_speed: float = 1.0     # m/s; the angle test needs both speeds above this
    touch_radius: float = 2.8        # m; nearest player must be this close
    touch_cluster: int = 5           # frames; candidate frames this close form one event
    touch_w: float = 2.0             # nats per event
    # --- restart --------------------------------------------------------------------------------
    restart_min_s: float = 2.0       # s; a pre-decode dead-looking stretch this long precedes a restart
    restart_window_s: float = 3.0    # s; the first control event must follow the stretch within this
    restart_w: float = 4.0           # nats per event
    # --- phase tag ------------------------------------------------------------------------------
    phase_w: float = 1.0             # nats per phase, spread; placeholder until Phase 4 measures agreement
    # --- off pitch ------------------------------------------------------------------------------
    offpitch_margin: float = 0.0     # m beyond the line
    offpitch_w: float = 10.0         # nats per frame
    # --- exit gap -------------------------------------------------------------------------------
    exit_dist: float = 3.0           # m; ball this close to a line when it disappears
    exit_out_speed: float = 0.5      # m/s; outward speed at the last visible frame
    exit_w: float = 8.0              # nats per gap, spread
    # --- long gap -------------------------------------------------------------------------------
    long_gap_s: float = 30.0         # s
    long_gap_w: float = 4.0          # nats per gap, spread
    # --- still ball -----------------------------------------------------------------------------
    still_speed: float = 0.5         # m/s
    still_s: float = 2.0             # s
    still_w: float = 0.3             # nats per frame
    # --- low activity ---------------------------------------------------------------------------
    low_speed: float = 1.39          # m/s median player speed
    low_s: float = 3.0               # s
    low_w: float = 0.05              # nats per frame
    # --- set-piece tag --------------------------------------------------------------------------
    setpiece_w: float = 3.0          # nats per tag, each side of the restart candidate
    # --- latch ----------------------------------------------------------------------------------
    latch_w: float = 0.001           # nats per frame
    # --- decoder --------------------------------------------------------------------------------
    turnover_cost: float = 3.0       # H<->A
    stop_cost: float = 4.0           # alive -> D
    restart_cost: float = 4.0        # D -> alive
    # --- guards ---------------------------------------------------------------------------------
    min_dead_frames: int = 75        # D runs shorter than this are merged into their neighbours


METRICA = "Metrica sample games"
KEPT = METRICA + " (first value kept: best of its grid; DFL improves with the set)"
FLAT = FIRST + " (Metrica sample games cannot identify it: flat grid)"


def _qt(tuned, q, feature):
    return "quantile transfer ({} {} at quantile {} of {})".format(METRICA, tuned, q, feature)


SOURCES = {
    "interp_max_gap": FIRST + " (PLAN F1: interpolate gaps of 5 frames or fewer and stop there)",
    "vel_half": BUILDER,
    "still_half": BUILDER,
    "touch_span": FIRST + " (PLAN section 3: across 3 frames)",
    "player_vmax": BUILDER,
    "carry_radius": _qt(3.0, 0.690, "ball to nearest player distance"),
    "carry_margin": _qt(0.5, 0.105, "nearest opponent margin"),
    "carry_dv": _qt(8.0, 0.726, "ball minus player velocity"),
    "carry_w": KEPT,
    "touch_dv": _qt(1.5, 0.756, "3-frame ball velocity change"),
    "touch_angle": _qt(35.0, 0.977, "ball turning angle") + "; the Metrica grid over it was flat",
    "touch_min_speed": BUILDER,
    "touch_radius": _qt(2.5, 0.631, "ball to nearest player distance"),
    "touch_cluster": BUILDER, "touch_w": KEPT,
    "restart_min_s": BUILDER, "restart_window_s": BUILDER, "restart_w": KEPT,
    "phase_w": BUILDER + " (placeholder; Phase 4 measures agreement label-free)",
    "offpitch_margin": "law of the game (lines at 0 and the pitch size)",
    "offpitch_w": FLAT,
    "exit_dist": FIRST, "exit_out_speed": BUILDER, "exit_w": KEPT,
    "long_gap_s": KEPT, "long_gap_w": FLAT,
    "still_speed": FIRST + " (not transferred: the proxy ball is missing when dead)",
    "still_s": FIRST, "still_w": FLAT,
    "low_speed": _qt(1.5, 0.324, "median player speed"),
    "low_s": KEPT, "low_w": METRICA,
    "setpiece_w": FIRST, "latch_w": FIRST + " (PLAN: fixed)",
    "turnover_cost": FIRST + " (the scale anchor: every other weight was tuned relative to it)",
    "stop_cost": KEPT, "restart_cost": KEPT,
    "min_dead_frames": METRICA,
}

assert set(SOURCES) == {f.name for f in fields(Params)}, "every Params field needs a source"


def describe():
    """One line per parameter: name, value, source."""
    p = Params()
    return ["{:<16} {:>8}  {}".format(f.name, str(getattr(p, f.name)), SOURCES[f.name])
            for f in fields(Params)]


if __name__ == "__main__":
    print("\n".join(describe()))
