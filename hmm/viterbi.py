"""C2: Viterbi decoder skeleton.

Finds the highest-scoring sequence of candidate arcs for one trip, working
entirely in log space so long trips cannot underflow to zero.

Generic on purpose: candidate counts may differ from one observation to the
next (up to max_candidates, 20 by default per the brief), and impossible
transitions are written as -inf. Later steps plug real scores into this same
interface:

- C3 builds the transition matrices from road connectivity and travel time;
- the playbook decode() contract (4.5) groups A's candidate table by
  obs_idx, calls viterbi_decode, and maps the returned local candidate
  indices back to arc_ids, one per GPS observation.

If every candidate at some observation is unreachable, the function raises
NoFeasiblePathError naming the step. How to recover from such gaps (missing
candidates, large GPS gaps) is a C4 decision, intentionally not made here.

Memory:
- core DP working storage: one O(K) score column plus one (K_{t-1}, K_t)
  transition matrix at a time;
- backpointer table: T int32 rows, at most T x max_candidates
  (19,313 x 20 ≈ 1.5 MB for the largest training trip);
- inputs: the emissions list costs O(T x K); pass transitions as an
  iterator (as the tests do) and they are never all materialized - a
  materialized list would cost O(T x K^2).
"""

import numpy as np

NEG_INF = -np.inf  # marks a forbidden candidate or an impossible transition


class NoFeasiblePathError(ValueError):
    """Raised when no candidate sequence survives the transition constraints."""


def _check_scores(arr, name):
    if np.isnan(arr).any() or np.isposinf(arr).any():
        raise ValueError(f"{name} must be finite or -inf, got NaN or +inf")


def viterbi_decode(log_emissions, log_transitions, log_start=None,
                   max_candidates=20):
    """Run Viterbi decoding over per-observation candidate lists.

    Parameters
    ----------
    log_emissions : list of (K_t,) float arrays
        Log score of each candidate at each observation (higher is better).
        -inf forbids a candidate; NaN and +inf are rejected. Lengths K_t
        may vary up to max_candidates; one array per GPS observation, in
        time order.
    log_transitions : list or iterator of (K_{t-1}, K_t) float arrays
        Log score of moving from candidate i at t-1 to candidate j at t.
        -inf marks an impossible move. Must yield exactly
        len(log_emissions) - 1 matrices; an iterator avoids materializing
        all of them at once.
    log_start : (K_0,) float array, optional
        Log prior over first-observation candidates. Defaults to uniform
        (all zeros), meaning no preference.
    max_candidates : int
        Maximum candidates allowed per observation (20 per the brief).
        Over-limit observations are rejected, never silently pruned:
        choosing which candidates to drop is a modelling decision and
        belongs to the candidate-scoring step.

    Returns
    -------
    path : (T,) int32 array
        Local index of the chosen candidate at each observation.
    best_score : float
        Total log score of that path, always finite for a returned path.

    Raises
    ------
    NoFeasiblePathError
        If every candidate at some observation is unreachable (-inf).
    ValueError
        On malformed input: wrong counts or shapes, NaN/+inf scores, or
        more than max_candidates candidates at an observation.
    """
    emissions = [np.asarray(e, dtype=float) for e in log_emissions]
    if len(emissions) == 0:
        raise ValueError("need at least one observation")
    for t, e in enumerate(emissions):
        if e.ndim != 1 or e.size == 0:
            raise ValueError(f"observation {t} has no candidates")
        if e.size > max_candidates:
            raise ValueError(
                f"observation {t} has {e.size} candidates, over the limit "
                f"of {max_candidates}; prune before decoding, choosing "
                "which to drop is a modelling decision")
        _check_scores(e, f"emissions at observation {t}")

    T = len(emissions)
    start = (np.zeros(emissions[0].size) if log_start is None
             else np.asarray(log_start, dtype=float))
    if start.shape != emissions[0].shape:
        raise ValueError(
            f"log_start has shape {start.shape}, expected {emissions[0].shape}")
    _check_scores(start, "log_start")

    # Forward pass. Keep only the previous score column (O(K) memory) and
    # store one backpointer row per step for the traceback.
    delta = start + emissions[0]
    if np.all(np.isneginf(delta)):
        raise NoFeasiblePathError(
            "every candidate at observation 0 is forbidden (-inf)")

    backpointers = [np.zeros(emissions[0].size, dtype=np.int32)]
    transitions = iter(log_transitions)

    for t in range(1, T):
        try:
            trans = np.asarray(next(transitions), dtype=float)
        except StopIteration:
            raise ValueError(
                f"expected {T - 1} transition matrices, got {t - 1}") from None
        if trans.shape != (emissions[t - 1].size, emissions[t].size):
            raise ValueError(
                f"transition {t - 1}->{t} has shape {trans.shape}, expected "
                f"({emissions[t - 1].size}, {emissions[t].size})")
        _check_scores(trans, f"transition {t - 1}->{t}")

        # Total score of arriving at each candidate j via each predecessor i.
        totals = delta[:, None] + trans                    # (K_{t-1}, K_t)
        best_prev = np.argmax(totals, axis=0)              # (K_t,)
        delta = totals[best_prev, np.arange(emissions[t].size)] + emissions[t]

        if np.all(np.isneginf(delta)):
            raise NoFeasiblePathError(
                f"no reachable candidate at observation {t}: every "
                "path into it has score -inf")
        backpointers.append(best_prev.astype(np.int32))

    sentinel = object()
    if next(transitions, sentinel) is not sentinel:
        raise ValueError(f"expected {T - 1} transition matrices, got more")

    best_last = int(np.argmax(delta))
    best_score = float(delta[best_last])  # finite: every step checked above

    # Traceback: follow the backpointers from the best final candidate.
    path = np.empty(T, dtype=np.int32)
    path[T - 1] = best_last
    for t in range(T - 1, 0, -1):
        path[t - 1] = backpointers[t][path[t]]

    return path, best_score
