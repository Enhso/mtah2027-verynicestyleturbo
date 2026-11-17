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

import json
import os
from dataclasses import dataclass, fields

FIRST = "first value"
BUILDER = "first value (builder)"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formation_model.json")

with open(MODEL_PATH) as _fh:
    _FORMATION_COEF = tuple(json.load(_fh)["coef"])


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
    setpiece_launch: bool = True     # restart candidate = launch frame (True) or first touch event (False)
    setpiece_launch_speed: float = 2.0   # m/s; centred ball speed that marks the launch
    setpiece_fallback_s: float = 6.4     # s; D from the window start when no candidate exists
    goal_dead_w: float = 1.0         # nats per frame, D from a goal to the next kick-off restart; 0 = off
    goal_after_setpiece: bool = True     # the goal starts where the last overlapped set-piece window ends, not at the GOALS tag start
    # --- latch ----------------------------------------------------------------------------------
    latch_w: float = 0.001           # nats per frame
    # --- team shape -----------------------------------------------------------------------------
    formation_w_vis: float = 0.02    # nats per frame per unit of log-odds where the ball is seen
    formation_w_blk: float = 0.1     # the same on blackout frames (no ball after interpolation)
    formation_coef: tuple = _FORMATION_COEF   # log-odds of H per unit of features.SHAPE_NAMES
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
    "setpiece_launch": "label-free analysis of this match (notes/labelfree-answers.md proposal 2); no loss on the planted-tag proxy, notes/restarts-progress.md",
    "setpiece_launch_speed": "label-free analysis of this match (notes/labelfree-answers.md proposal 2)",
    "setpiece_fallback_s": "label-free analysis of this match (median launch offset, proposal 2)",
    "goal_after_setpiece": "label-free structure of the GOALS tags of this match (the tag opens with a set piece; notes/restarts-progress.md)",
    "goal_dead_w": "law of the game (the ball is dead from a goal to the kick-off); evidence weight, the goal start is a tag heuristic; flat 0.3 to 10 on BAR-ATH, notes/restarts-progress.md",
    "setpiece_w": FIRST, "latch_w": FIRST + " (PLAN: fixed)",
    "formation_w_vis": METRICA + " (held out by game: gain on all 3 games on atd and atd+mask; top of its grid; calibration/formation_tune.py, notes/formation-progress.md)",
    "formation_w_blk": METRICA + " (held out by game: gain on all 3 games on atd and atd+mask; top of its grid; calibration/formation_tune.py, notes/formation-progress.md)",
    "formation_coef": "DFL matches, atd variant (calibration/formation_train.py, "
                      "possession/formation_model.json); Metrica is kept out so it scores held out",
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
