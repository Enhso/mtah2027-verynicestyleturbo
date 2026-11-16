"""The three post-decode guards of PLAN section 3."""

import numpy as np

H, A, D, X = 0, 1, 2, 3


def merge_short_dead(path, min_len, allowed):
    """Guard 1: merge each D run shorter than min_len frames into its neighbours.

    The run takes the label of its neighbour when only one side exists or both agree; between two
    different teams it is split at its midpoint. A run is left alone when a hard constraint forbids
    the replacement. Returns (path, number of runs merged).
    """
    path = path.copy()
    merged = 0
    d = path == D
    edges = np.diff(np.concatenate(([0], d.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= min_len:
            continue
        left = path[a - 1] if a > 0 else None
        right = path[b] if b < len(path) else None
        if left is None and right is None:
            continue
        fill = np.empty(b - a, dtype=np.int8)
        mid = (b - a + 1) // 2   # an odd run gives its middle frame to the left neighbour
        fill[:mid] = left if left is not None else right
        fill[mid:] = right if right is not None else left
        if allowed[np.arange(a, b), fill].all():
            path[a:b] = fill
            merged += 1
    return path, merged


def assemble(n, periods, paths):
    """Guards 2 and 3: X on every ungraded frame, and no X on any graded frame."""
    codes = np.full(n, X, dtype=np.int8)
    for (a, b), path in zip(periods, paths):
        if (path == X).any():
            raise ValueError("a decoded period contains X")
        codes[a:b] = path
    return codes
