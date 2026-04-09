import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_local_spr import make_internal_a_tree


def test_compute_log_likelihood_invalid_branch_order_returns_negative_infinity():
    tree = make_internal_a_tree()
    pi = np.array([0.5, 0.5])

    tree.time[5] = -0.1
    assert tree.compute_log_likelihood(1.0, pi) == -math.inf
    assert tree.compute_site_log_likelihood(0, 1.0, pi) == -math.inf


def test_propose_local_time_root_overflow_self_rejects(monkeypatch):
    tree = make_internal_a_tree()
    root = tree.root
    tree.time[root] = 1.7
    old_time = float(tree.time[root])

    monkeypatch.setattr(tree, "sample_internal_node", lambda: root)
    monkeypatch.setattr(np.random, "randn", lambda: 1e308)

    node, returned_old_time, log_hastings = tree.propose_local_time(step_size=1.0)

    assert node == root
    assert returned_old_time == old_time
    assert log_hastings == -math.inf
    assert tree.time[root] == old_time


def test_resample_mutation_rate_overflow_self_rejects(monkeypatch):
    tree = make_internal_a_tree()
    tree.mutation_rate = 1.0
    old_rate = float(tree.mutation_rate)

    monkeypatch.setattr(np.random, "randn", lambda: 1e308)

    old_returned, log_forward, log_reverse = tree.resample_mutation_rate(step_size=1.0)

    assert old_returned == old_rate
    assert log_forward == 0.0
    assert log_reverse == -math.inf
    assert tree.mutation_rate == old_rate


def test_root_time_move_records_debug_metadata(monkeypatch):
    tree = make_internal_a_tree()
    root = tree.root

    monkeypatch.setattr(tree, "sample_internal_node", lambda: root)
    monkeypatch.setattr(np.random, "randn", lambda: 0.0)

    node, old_time, log_hastings = tree.propose_local_time(step_size=1.0)

    assert node == root
    assert tree.last_time_move_debug["is_root"] is True
    assert tree.last_time_move_debug["node"] == root
    assert tree.last_time_move_debug["old_time"] == old_time
    assert tree.last_time_move_debug["new_time"] == tree.time[root]
    assert math.isfinite(log_hastings)


def test_repeated_root_time_moves_do_not_immediately_run_away_with_prior_correction(monkeypatch):
    tree = make_internal_a_tree()
    root = tree.root

    monkeypatch.setattr(tree, "sample_internal_node", lambda: root)
    np.random.seed(123)
    random.seed(123)

    max_root_time = float(tree.time[root])
    for _ in range(2000):
        old_time = float(tree.time[root])
        old_prior = tree.log_likelihood()
        _, _, log_hastings = tree.propose_local_time(step_size=1.0)
        new_prior = tree.log_likelihood()
        alpha = (new_prior - old_prior) + log_hastings
        if math.log(random.random()) < alpha:
            max_root_time = max(max_root_time, float(tree.time[root]))
        else:
            tree.time[root] = old_time

    assert math.isfinite(tree.time[root])
    assert max_root_time < 1e6
