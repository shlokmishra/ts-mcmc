import random

import numpy as np
import pytest

from mcmc import kingman_mcmc
from recorder import Recorder
from stein_thinning_trees.stein import compute_stein_kernel_matrix
from tree import Tree, coalescence_tree_with_sequences


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


def make_base_tree():
    # ((0,1),(2,3))
    return make_tree_with_structure(
        left_child=[-1, -1, -1, -1, 0, 2, 4],
        right_child=[-1, -1, -1, -1, 1, 3, 5],
        parent=[4, 4, 5, 5, 6, 6, -1],
        time=[0.0, 0.0, 0.0, 0.0, 1.0, 1.2, 2.0],
        root=6,
    )


def make_topology_changed_tree():
    # ((0,2),(1,3)) with the same internal node times.
    return make_tree_with_structure(
        left_child=[-1, -1, -1, -1, 0, 1, 4],
        right_child=[-1, -1, -1, -1, 2, 3, 5],
        parent=[4, 5, 4, 5, 6, 6, -1],
        time=[0.0, 0.0, 0.0, 0.0, 1.0, 1.2, 2.0],
        root=6,
    )


def make_time_changed_tree():
    tree = make_base_tree()
    tree.time[4] = 1.4
    return tree


def source_tree_signature(tree):
    signatures = []

    def descendants(node):
        if node < tree.sample_size:
            return frozenset([node])
        left = descendants(tree.left_child[node])
        right = descendants(tree.right_child[node])
        return left | right

    for node in range(tree.sample_size, len(tree.parent)):
        signatures.append((descendants(node), float(tree.time[node])))
    return sorted(signatures, key=lambda item: (tuple(sorted(item[0])), item[1]))


def recorded_tree_signature(ts_tree):
    signatures = []
    ts = ts_tree.tree_sequence
    for node in ts_tree.nodes():
        if ts_tree.num_children(node) > 0:
            signatures.append((frozenset(ts_tree.samples(node)), float(ts.node(node).time)))
    return sorted(signatures, key=lambda item: (tuple(sorted(item[0])), item[1]))


def assert_tree_matches_recorded(source_tree, ts_tree):
    assert source_tree_signature(source_tree) == recorded_tree_signature(ts_tree)


def test_identical_trees_extend_intervals():
    recorder = Recorder(sample_size=4, seq_length=10)
    tree = make_base_tree()

    recorder.append_tree(tree, mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))
    recorder.append_tree(tree, mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))

    assert recorder.node_table.num_rows == 7
    # Live edge rows are appended per recorded state and compressed on finalize.
    assert recorder.edge_table.num_rows == 12

    ts = recorder.tree_sequence()
    assert ts.sequence_length == 2.0
    assert ts.num_trees == 1
    assert ts.num_edges == 6
    first_tree = ts.first(sample_lists=True)
    assert first_tree.interval.left == 0.0
    assert first_tree.interval.right == 2.0
    assert_tree_matches_recorded(tree, first_tree)

    # Sample-aligned access should still recover one tree per recorded state.
    recorded = recorder.recorded_trees()
    assert len(recorded) == 2
    assert_tree_matches_recorded(tree, recorded[0])
    assert_tree_matches_recorded(tree, recorded[1])

    simplified = ts.simplify()
    assert simplified.num_samples == 4


def test_local_time_change_creates_only_needed_rows():
    recorder = Recorder(sample_size=4, seq_length=10)

    recorder.append_tree(make_base_tree(), mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))
    recorder.append_tree(make_time_changed_tree(), mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))

    # Only the changed internal node is duplicated.
    assert recorder.node_table.num_rows == 8

    ts = recorder.tree_sequence()
    assert ts.num_trees == 2
    # Only the incident edges change after compression: 6 original + 3 new.
    assert ts.num_edges == 9


def test_local_topology_change_reuses_nodes_and_only_adds_changed_edges():
    recorder = Recorder(sample_size=4, seq_length=10)

    recorder.append_tree(make_base_tree(), mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))
    recorder.append_tree(make_topology_changed_tree(), mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))

    # Times are unchanged, so no new node rows are needed.
    assert recorder.node_table.num_rows == 7

    ts = recorder.tree_sequence()
    assert ts.num_trees == 2
    # After squashing, only the changed relationships require extra rows.
    assert ts.num_edges == 8


def test_reconstructed_trees_match_source_sequence():
    recorder = Recorder(sample_size=4, seq_length=10)
    source_trees = [
        make_base_tree(),
        make_base_tree(),
        make_time_changed_tree(),
        make_topology_changed_tree(),
    ]

    for tree in source_trees:
        recorder.append_tree(tree, mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))

    recorded_trees = recorder.recorded_trees()
    assert len(recorded_trees) == len(source_trees)

    for source_tree, ts_tree in zip(source_trees, recorded_trees):
        assert_tree_matches_recorded(source_tree, ts_tree)


def test_downstream_recorder_consumer_still_aligns_samples():
    recorder = Recorder(sample_size=4, seq_length=10)
    tree = make_base_tree()
    recorder.append_tree(tree, mutation_rate=1.0, log_likelihood=-1.0, gradient=np.zeros(7))
    recorder.append_tree(tree, mutation_rate=1.1, log_likelihood=-1.2, gradient=np.ones(7))

    K = compute_stein_kernel_matrix(recorder)
    assert K.shape == (2, 2)
    np.testing.assert_array_almost_equal(K, K.T)


def test_short_mcmc_end_to_end_recorder_path():
    np.random.seed(123)
    random.seed(123)

    tree, sequences = coalescence_tree_with_sequences(4, 2, 8, 1.0)
    tree.sequences = sequences
    recorder = Recorder(sample_size=4, seq_length=8)
    pi = np.array([0.5, 0.5])

    kingman_mcmc(
        tree,
        recorder,
        pi,
        steps=5,
        step_size=0.2,
        record=True,
        compute_gradients=True,
        print_every=None,
    )

    ts = recorder.tree_sequence()
    recorded_trees = recorder.recorded_trees()

    assert len(recorder.mutation_rates) == 5
    assert len(recorded_trees) == 5
    assert ts.sequence_length == 5.0
    assert ts.num_samples == 4

    K = compute_stein_kernel_matrix(recorder)
    assert K.shape == (5, 5)
    np.testing.assert_array_almost_equal(K, K.T)
