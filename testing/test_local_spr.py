import random

import numpy as np
import pytest

from mcmc import kingman_mcmc
from recorder import Recorder
from tree import Tree


def make_tree_with_structure(left_child, right_child, parent, time, root, n_states=2):
    sample_size = (len(parent) + 1) // 2
    tree = Tree(sample_size, n_states)
    tree.left_child = np.array(left_child, dtype=int)
    tree.right_child = np.array(right_child, dtype=int)
    tree.parent = np.array(parent, dtype=int)
    tree.time = np.array(time, dtype=float)
    tree.root = root
    tree.sequences = [[0, 0, 0, 0] for _ in range(sample_size)]
    return tree


def make_internal_a_tree():
    # (((0,1),2),(3,4))
    return make_tree_with_structure(
        left_child=[-1, -1, -1, -1, -1, 0, 5, 3, 6],
        right_child=[-1, -1, -1, -1, -1, 1, 2, 4, 7],
        parent=[5, 5, 6, 7, 7, 6, 8, 8, -1],
        time=[0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.9, 1.7],
        root=8,
    )


def make_leaf_a_tree():
    # ((0,1),(2,3))
    return make_tree_with_structure(
        left_child=[-1, -1, -1, -1, 0, 2, 4],
        right_child=[-1, -1, -1, -1, 1, 3, 5],
        parent=[4, 4, 5, 5, 6, 6, -1],
        time=[0.0, 0.0, 0.0, 0.0, 0.7, 0.9, 1.8],
        root=6,
    )


def make_root_adjacent_tree():
    # ((0,1),2)
    return make_tree_with_structure(
        left_child=[-1, -1, -1, 0, 3],
        right_child=[-1, -1, -1, 1, 2],
        parent=[3, 3, 4, 4, -1],
        time=[0.0, 0.0, 0.0, 0.5, 1.3],
        root=4,
    )


def assert_valid_tree(tree):
    roots = [node for node, parent in enumerate(tree.parent) if parent == -1]
    assert roots == [tree.root]

    for leaf in range(tree.sample_size):
        assert tree.left_child[leaf] == -1
        assert tree.right_child[leaf] == -1

    for node in range(tree.sample_size, len(tree.parent)):
        left = tree.left_child[node]
        right = tree.right_child[node]
        assert left != -1
        assert right != -1
        assert tree.parent[left] == node
        assert tree.parent[right] == node
        assert tree.time[node] >= tree.time[left]
        assert tree.time[node] >= tree.time[right]

    seen = set()
    stack = [tree.root]
    while stack:
        node = stack.pop()
        assert node not in seen
        seen.add(node)
        if node >= tree.sample_size:
            stack.append(tree.left_child[node])
            stack.append(tree.right_child[node])

    assert seen == set(range(len(tree.parent)))


def test_local_spr_candidates_for_internal_a_node():
    tree = make_internal_a_tree()
    candidates = tree.get_local_spr_candidates(child=2)
    by_node = {candidate["branch_node"]: candidate for candidate in candidates}

    assert set(by_node) == {0, 1, 5, 7, 8}
    assert by_node[5]["sources"] == ["above_a"]
    assert by_node[0]["sources"] == ["below_a_left"]
    assert by_node[1]["sources"] == ["below_a_right"]
    assert by_node[7]["sources"] == ["above_sibling_of_a"]
    assert by_node[8]["sources"] == ["above_parent_of_a"]

    proposal = tree.propose_local_spr(child=2, branch_node=7, debug=True)
    assert proposal["debug"]["a_node"] == 5
    assert proposal["debug"]["chosen_candidate"]["branch_node"] == 7
    assert proposal["debug"]["sampled_time"] >= proposal["debug"]["chosen_candidate"]["interval_lower"]


def test_local_spr_forward_reverse_bookkeeping_internal_case():
    tree = make_internal_a_tree()
    metadata = tree.build_local_spr_proposal_metadata(child=2, new_sib=7, new_time=1.1)

    assert [candidate["branch_node"] for candidate in metadata["forward_candidates"]] == [0, 1, 5, 7, 8]
    assert [candidate["branch_node"] for candidate in metadata["reverse_candidates"]] == [3, 4, 5, 7, 8]
    assert metadata["forward_candidate_count"] == 5
    assert metadata["reverse_candidate_count"] == 5
    assert metadata["reverse_chosen_candidate"]["branch_node"] == 5
    assert metadata["log_q_forward"] == pytest.approx(-2.995732273553991)
    assert metadata["log_q_reverse"] == pytest.approx(-3.4812400893356914)
    assert metadata["log_hastings"] == pytest.approx(-0.48550781578170055)


def test_local_spr_candidates_for_leaf_a_node():
    tree = make_leaf_a_tree()
    candidates = tree.get_local_spr_candidates(child=0)
    branch_nodes = {candidate["branch_node"] for candidate in candidates}

    assert branch_nodes == {1, 5, 6}


def test_local_spr_forward_reverse_candidate_count_can_differ():
    tree = make_internal_a_tree()
    metadata = tree.build_local_spr_proposal_metadata(child=0, new_sib=6, new_time=1.1)

    assert metadata["forward_candidate_count"] == 3
    assert metadata["reverse_candidate_count"] == 5
    assert metadata["log_hastings"] == pytest.approx(-0.3930425881096071)


def test_local_spr_root_adjacent_case_handles_missing_parent_neighborhood():
    tree = make_root_adjacent_tree()
    candidates = tree.get_local_spr_candidates(child=2)
    by_node = {candidate["branch_node"]: candidate for candidate in candidates}

    assert set(by_node) == {0, 1, 3}
    assert "above_parent_of_a" not in sum((candidate["sources"] for candidate in candidates), [])

    proposal = tree.propose_local_spr(child=2, branch_node=3)
    old_sibling = proposal["old_sibling"]
    old_time = tree.time[proposal["old_parent"]]
    tree.detach_reattach(proposal["child"], proposal["new_sibling"], proposal["new_time"])
    assert_valid_tree(tree)
    tree.detach_reattach(proposal["child"], old_sibling, old_time)
    assert_valid_tree(tree)

    metadata = tree.build_local_spr_proposal_metadata(child=2, new_sib=3, new_time=0.7)
    assert [candidate["branch_node"] for candidate in metadata["forward_candidates"]] == [0, 1, 3]
    assert [candidate["branch_node"] for candidate in metadata["reverse_candidates"]] == [0, 1, 3]
    assert metadata["reverse_candidate_count"] == 3
    assert metadata["log_hastings"] == pytest.approx(-0.6)


def test_local_spr_candidate_nodes_are_unique():
    tree = make_internal_a_tree()
    candidates = tree.get_local_spr_candidates(child=2)
    branch_nodes = [candidate["branch_node"] for candidate in candidates]
    assert len(branch_nodes) == len(set(branch_nodes))


def test_tree_validity_after_repeated_local_spr_proposals():
    np.random.seed(7)
    random.seed(7)
    tree = make_internal_a_tree()

    for _ in range(25):
        proposal = tree.propose_local_spr(debug=True)
        assert np.isfinite(proposal["log_q_forward"])
        assert np.isfinite(proposal["log_q_reverse"])
        assert np.isfinite(proposal["log_hastings"])
        tree.detach_reattach(proposal["child"], proposal["new_sibling"], proposal["new_time"])
        assert_valid_tree(tree)


def test_local_spr_acceptance_uses_hastings_correction(monkeypatch):
    np.random.seed(19)
    random.seed(19)
    tree = make_internal_a_tree()
    original_parent = tree.parent.copy()
    original_left = tree.left_child.copy()
    original_right = tree.right_child.copy()
    original_time = tree.time.copy()
    original_root = tree.root

    proposal = tree.propose_local_spr(child=0, branch_node=6, debug=True)
    assert proposal["log_hastings"] < 0

    monkeypatch.setattr(tree, "propose_local_spr", lambda debug=False: proposal)
    monkeypatch.setattr(tree, "compute_log_likelihood", lambda mutation_rate, pi: 0.0)
    monkeypatch.setattr(tree, "propose_local_time", lambda step_size=1.0: (tree.root, tree.time[tree.root], 0.0))
    monkeypatch.setattr(tree, "resample_mutation_rate", lambda step_size=0.1: (tree._mutation_rate, 0.0, 0.0))

    draws = iter([0.8, 0.1, 0.1])
    monkeypatch.setattr(random, "random", lambda: next(draws))

    kingman_mcmc(
        tree,
        Recorder(sample_size=5, seq_length=4),
        np.array([0.5, 0.5]),
        steps=1,
        record=False,
        compute_gradients=False,
        print_every=None,
        spr_proposal="local_spr",
        time_move="local",
    )

    np.testing.assert_array_equal(tree.parent, original_parent)
    np.testing.assert_array_equal(tree.left_child, original_left)
    np.testing.assert_array_equal(tree.right_child, original_right)
    np.testing.assert_array_equal(tree.time, original_time)
    assert tree.root == original_root


def test_local_spr_is_selectable_in_mcmc():
    np.random.seed(11)
    random.seed(11)
    tree = make_internal_a_tree()
    recorder = Recorder(sample_size=5, seq_length=4)
    pi = np.array([0.5, 0.5])

    acceptances = kingman_mcmc(
        tree,
        recorder,
        pi,
        steps=4,
        record=True,
        compute_gradients=False,
        print_every=None,
        spr_proposal="local_spr",
    )

    assert len(acceptances) == 3
    assert recorder.tree_sequence().num_samples == 5
