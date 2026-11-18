"""Tests for hmm/viterbi.py (C2).

Run with:  python -m pytest hmm/test_viterbi.py -v
or plain:  python hmm/test_viterbi.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from viterbi import NEG_INF, NoFeasiblePathError, viterbi_decode  # noqa: E402

LOG = np.log


def toy_example():
    """Tiny hand-made road network with a known correct answer.

    5 nodes (x, y metres):  0:(0,0)  1:(100,0)  2:(200,0)  3:(300,0)  4:(100,60)
    4 arcs (from -> to):    0: 0->1   1: 1->2    2: 2->3     3: 1->4 (dead end)

    Three GPS observations travel along the main route 0->1->2. The middle
    point sits slightly closer to the side branch (arc 3), so picking the
    best candidate per point would wrongly choose arc 3. Viterbi must still
    return [arc0, arc1, arc2], because arc 3 is a dead end: no transition
    leads from it to arc 2, so any path through it scores -inf.
    """
    log_emissions = [
        np.log([0.9]),              # obs 0: candidates [arc0]
        np.log([0.4, 0.6]),         # obs 1: candidates [arc1, arc3] - arc3 looks closer
        np.log([0.9]),              # obs 2: candidates [arc2]
    ]
    # Allowed when prev.to_node == cur.from_node, score log(1) = 0; else -inf.
    log_transitions = [
        np.array([[0.0, 0.0]]),     # arc0 -> arc1 and arc0 -> arc3 both leave node 1
        np.array([[0.0],            # arc1 -> arc2 connects
                  [NEG_INF]]),      # arc3 is a dead end, cannot reach arc2
    ]
    expected_score = LOG(0.9) + LOG(0.4) + LOG(0.9)
    return log_emissions, log_transitions, [0, 0, 0], expected_score


def test_toy_known_answer():
    em, tr, expected_path, expected_score = toy_example()
    path, score = viterbi_decode(em, tr)
    assert path.tolist() == expected_path, f"got {path.tolist()}"
    assert np.isclose(score, expected_score)


def test_matches_brute_force_on_random_instance():
    """Viterbi must agree with enumerating every possible path."""
    rng = np.random.default_rng(7)
    T = 6
    counts = rng.integers(1, 5, size=T)          # variable candidates per step
    em = [rng.normal(size=k) for k in counts]
    tr = []
    for t in range(1, T):
        m = rng.normal(size=(counts[t - 1], counts[t]))
        m[rng.random(m.shape) < 0.3] = NEG_INF   # some impossible moves
        tr.append(m)

    path, score = viterbi_decode(em, tr)

    best_score, best_path = NEG_INF, None
    for candidate_path in itertools.product(*(range(k) for k in counts)):
        total = em[0][candidate_path[0]]
        for t in range(1, T):
            total += tr[t - 1][candidate_path[t - 1], candidate_path[t]]
            total += em[t][candidate_path[t]]
        if total > best_score:
            best_score, best_path = total, candidate_path

    assert np.isclose(score, best_score)
    assert path.tolist() == list(best_path)


def test_numerical_stability_long_chain():
    """Log space must survive scores that underflow to 0 in linear space.

    50,000 steps with per-step probabilities around 1e-300: their linear
    product underflows immediately, but the log-space total stays finite.
    Transitions are passed as an iterator, the form real trips will use.
    """
    T = 50_000
    p_good, p_bad, p_stay = 1e-300, 1e-301, 0.99
    em = [np.log([p_good, p_bad])] * T
    trans_matrix = np.log([[p_stay, 1 - p_stay],
                           [1 - p_stay, p_stay]])
    tr = itertools.repeat(trans_matrix, T - 1)   # iterator, not a list

    path, score = viterbi_decode(em, tr)

    expected = T * LOG(p_good) + (T - 1) * LOG(p_stay)   # stay on candidate 0
    assert np.isfinite(score)
    assert np.isclose(score, expected, rtol=1e-9)
    assert (path == 0).all()


def test_memory_bounded_large_trip():
    """Largest training trip scale: 100k observations x 16 candidates.

    Transitions stream from a generator, so the only growing storage is the
    (T, K) int32 backpointer table (~6.4 MB here). Nothing network-sized is
    built and the full input is never materialized.
    """
    rng = np.random.default_rng(1)
    T, K = 100_000, 16
    emission_bank = [rng.normal(size=K) for _ in range(8)]
    trans_bank = [rng.normal(size=(K, K)) for _ in range(4)]
    em = [emission_bank[i % 8] for i in range(T)]          # refs, not copies
    tr = (trans_bank[i % 4] for i in range(T - 1))         # lazy generator

    path, score = viterbi_decode(em, tr)

    assert path.shape == (T,)
    assert path.dtype == np.int32
    assert np.isfinite(score)


def test_max_candidates_enforced():
    """Over-limit observations are rejected, never silently pruned."""
    em, tr, _, _ = toy_example()
    path, _ = viterbi_decode(em, tr, max_candidates=2)     # within limit: fine
    assert path.tolist() == [0, 0, 0]

    over_limit = [np.zeros(21)]
    try:
        viterbi_decode(over_limit, [], max_candidates=20)
    except ValueError as exc:
        assert "21" in str(exc) and "20" in str(exc)
    else:
        raise AssertionError("expected ValueError for 21 > 20 candidates")


def test_rejects_nan_and_posinf():
    em, tr, _, _ = toy_example()
    cases = [
        lambda: viterbi_decode([np.array([np.nan])], []),
        lambda: viterbi_decode([np.array([np.inf])], []),
        lambda: viterbi_decode(em, [np.array([[np.nan, 0.0]]), tr[1]]),
        lambda: viterbi_decode(em, [np.array([[np.inf, 0.0]]), tr[1]]),
        lambda: viterbi_decode(em, tr, log_start=np.array([np.nan])),
    ]
    for bad_call in cases:
        try:
            bad_call()
        except ValueError as exc:
            assert "NaN or +inf" in str(exc)
        else:
            raise AssertionError("expected ValueError for NaN/+inf")


def test_unreachable_step_raises_with_index():
    em = [np.log([0.5]), np.log([0.5]), np.log([0.5])]
    tr = [np.full((1, 1), NEG_INF),   # nothing reaches observation 1
          np.zeros((1, 1))]
    try:
        viterbi_decode(em, tr)
    except NoFeasiblePathError as exc:
        assert "observation 1" in str(exc)
    else:
        raise AssertionError("expected NoFeasiblePathError")


def test_all_forbidden_candidates_raise():
    # All candidates forbidden at a middle observation.
    try:
        viterbi_decode([np.log([0.5]), np.array([NEG_INF])],
                       [np.zeros((1, 1))])
    except NoFeasiblePathError:
        pass
    else:
        raise AssertionError("expected NoFeasiblePathError")

    # All candidates forbidden at the first observation.
    try:
        viterbi_decode([np.array([NEG_INF, NEG_INF])], [])
    except NoFeasiblePathError as exc:
        assert "observation 0" in str(exc)
    else:
        raise AssertionError("expected NoFeasiblePathError")


def test_input_validation():
    em, tr, _, _ = toy_example()
    for bad_call in (
        lambda: viterbi_decode([], []),
        lambda: viterbi_decode(em, tr[:-1]),                    # too few
        lambda: viterbi_decode(em, tr + [np.zeros((1, 1))]),    # too many
        lambda: viterbi_decode(em, [tr[0], np.zeros((3, 3))]),  # wrong shape
        lambda: viterbi_decode([np.array([])], []),             # no candidates
        lambda: viterbi_decode(em, tr, log_start=np.zeros(5)),  # start shape
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} tests passed")
