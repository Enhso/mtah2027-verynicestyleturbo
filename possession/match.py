"""The one array schema every source is converted to: this match, DFL, Metrica sample games.

Coordinates are metres with the origin at a pitch corner, x along the length in [0, L] and y along
the width in [0, W]. Which corner does not matter: the pipeline only uses distances and lines.
Frame i of every array is frame i of the source's video timeline (for BAR-ATH, submission row i).
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

HOME, AWAY = 0, 1
# Same code order as evaluation/possession_metrics.LABELS.
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])

CACHE_DIR = os.path.join(
    os.environ.get("MTAH_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
    "cache")


@dataclass(frozen=True)
class Match:
    name: str
    fps: int
    pitch: tuple            # (length, width) in metres
    ball: np.ndarray        # (N, 2) float32, NaN where the ball is not tracked
    players: np.ndarray     # (N, K, 2) float32, NaN where the slot is empty
    speed: np.ndarray       # (N, K) float32 m/s, NaN where the slot is empty
    team: np.ndarray        # (K,) int8, HOME or AWAY
    periods: np.ndarray     # (P, 2) int64, half-open [start, stop) frame ranges decoded separately
    tags: list = field(default_factory=list)  # Smart Tagging dicts; empty for other sources
    labels: np.ndarray | None = None  # (N,) int8 H/A/D/X; external matches only, X outside periods

    @property
    def n_frames(self):
        return len(self.ball)


def path_for(name, cache_dir=None):
    return os.path.join(cache_dir or CACHE_DIR, name + ".npz")


def save(match, cache_dir=None):
    path = path_for(match.name, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = dict(
        name=np.array(match.name), fps=np.array(match.fps), pitch=np.asarray(match.pitch, float),
        ball=match.ball, players=match.players, speed=match.speed, team=match.team,
        periods=match.periods, tags=np.array(json.dumps(match.tags)))
    if match.labels is not None:
        arrays["labels"] = match.labels
    np.savez(path, **arrays)
    return path


def load(name, cache_dir=None):
    with np.load(path_for(name, cache_dir)) as z:
        return Match(
            name=str(z["name"]), fps=int(z["fps"]), pitch=tuple(z["pitch"].tolist()),
            ball=z["ball"], players=z["players"], speed=z["speed"], team=z["team"],
            periods=z["periods"], tags=json.loads(str(z["tags"])),
            labels=z["labels"] if "labels" in z.files else None)
