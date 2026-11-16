"""Viterbi over three states (H, A, D) with transition costs and hard constraints given as data."""

import numpy as np

H, A, D = 0, 1, 2
NEG = float("-inf")


def transition_costs(params):
    """cost[p][s] of moving from state p to state s; zero on the diagonal."""
    t, stop, restart = params.turnover_cost, params.stop_cost, params.restart_cost
    return [[0.0, t, stop],
            [t, 0.0, stop],
            [restart, restart, 0.0]]


def viterbi(score, cost, allowed=None):
    """Best path of states 0..2 for score (T,3) in nats, maximising sum(score) - sum(costs).

    allowed is an optional (T,3) bool mask: a False entry forbids that state at that frame. Ties
    keep the previous state (a self transition), then prefer the lower state index, so a run of
    zero evidence never invents a switch. Raises ValueError when a frame allows no state.
    """
    score = np.asarray(score, dtype=np.float64)
    n = len(score)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if allowed is None:
        allowed = np.ones((n, 3), dtype=bool)
    if not allowed.any(axis=1).all():
        raise ValueError("constraints leave a frame with no allowed state")

    rows = score.tolist()
    ok = allowed.tolist()
    back = np.zeros((n, 3), dtype=np.int8)
    delta = [rows[0][s] if ok[0][s] else NEG for s in range(3)]
    for t in range(1, n):
        row, okt = rows[t], ok[t]
        new = [NEG, NEG, NEG]
        bt = [0, 1, 2]
        for s in range(3):
            if not okt[s]:
                continue
            best, arg = delta[s], s
            for p in range(3):
                if p != s:
                    value = delta[p] - cost[p][s]
                    if value > best:
                        best, arg = value, p
            new[s] = best + row[s]
            bt[s] = arg
        back[t] = bt
        delta = new

    state = max(range(3), key=lambda s: (delta[s], -s))
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path
