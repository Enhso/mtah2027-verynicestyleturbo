"""Per-match facts the pipeline is told, never infers: tag-to-side mapping and hard constraints.

The channels and the decoder are generic over Match. Anything that belongs to one match lives here,
so an external match with no entry in CONFIGS runs with no constraints and no tags.
"""

from dataclasses import dataclass

import numpy as np

from . import ingest  # noqa: F401  (puts the repo root and starter/ on sys.path)

HOME, AWAY = 0, 1


@dataclass(frozen=True)
class MatchConfig:
    # Smart Tagging `Team` text (lower case) -> HOME or AWAY. Tags of other teams are ignored.
    tag_team: tuple = ()
    # [start, stop) frame ranges that must be D.
    forced_dead: tuple = ()
    # frames that must not be D (alive from the first frame).
    forced_alive: tuple = ()

    def tag_side(self, text):
        return dict(self.tag_team).get((text or "").strip().lower())

    def allowed(self, start, stop):
        """(stop-start, 3) bool mask of the states each frame of [start, stop) may take."""
        mask = np.ones((stop - start, 3), dtype=bool)
        for a, b in self.forced_dead:
            a, b = max(a, start), min(b, stop)
            if a < b:
                mask[a - start:b - start, :2] = False
        for f in self.forced_alive:
            if start <= f < stop:
                mask[f - start, 2] = False
        return mask


EMPTY = MatchConfig()


def bar_ath():
    """Barcelona (home) vs Athletic. Constants are the published ones: the half blocks of
    games.py and the second-half kickoff of starter/load_smart_tagging.py."""
    import games
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES
    blocks = games.half_video_blocks("BAR-ATH")
    second_start = blocks[1][1]
    return MatchConfig(
        tag_team=(("barcelona", HOME), ("atletic club", AWAY)),
        forced_dead=((second_start, KICKOFF_VIDEO_FRAMES["2nd Half"]),),
        forced_alive=(KICKOFF_VIDEO_FRAMES["1st Half"],))


def for_match(name):
    return bar_ath() if name == "BAR-ATH" else EMPTY
