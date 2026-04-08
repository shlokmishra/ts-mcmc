import math
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
