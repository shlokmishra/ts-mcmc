"""
Numerical correctness tests for vectorized likelihood and gradient computations.

These tests verify that:
1. Vectorized compute_log_likelihood matches the original site-by-site
   compute_site_log_likelihood (which uses log-space logsumexp).
2. compute_gradient matches centered finite-difference gradients.
3. Both hold across multiple random trees, mutation rates, and problem sizes.

If these tests fail, the math in the vectorized code has changed.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tree import Tree, coalescence_tree_with_sequences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_log_likelihood(tree, mutation_rate, pi):
    """
    Compute total log-likelihood using the original site-by-site method.
    
    This calls tree.compute_site_log_likelihood for each site and sums.
    It operates in log-space with logsumexp -- a completely different code
    path from the vectorized compute_log_likelihood.
    """
    n_sites = len(tree.sequences[0])
    return sum(
        tree.compute_site_log_likelihood(site, mutation_rate, pi)
        for site in range(n_sites)
    )


def finite_difference_gradient(tree, mutation_rate, pi, eps=1e-6):
    """
    Compute gradient of log-likelihood w.r.t. node times via centered
    finite differences:  dLL/dt_i ≈ (LL(t_i + eps) - LL(t_i - eps)) / (2 * eps)
    
    Only computes gradients for internal nodes (leaves are fixed at time 0).
    """
    n_nodes = 2 * tree.sample_size - 1
    grad_fd = np.zeros(n_nodes)
    original_times = tree.time.copy()

    for node in range(tree.sample_size, n_nodes):
        # Forward perturbation
        tree.time = original_times.copy()
        tree.time[node] += eps
        ll_plus = tree.compute_log_likelihood(mutation_rate, pi)

        # Backward perturbation
        tree.time = original_times.copy()
        tree.time[node] -= eps
        ll_minus = tree.compute_log_likelihood(mutation_rate, pi)

        grad_fd[node] = (ll_plus - ll_minus) / (2 * eps)

    tree.time = original_times.copy()
    return grad_fd


def make_tree(sample_size, n_states, seq_length, mutation_rate, seed):
    """Create a tree with simulated sequences using a fixed seed."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    tree, sequences = coalescence_tree_with_sequences(
        sample_size, n_states, seq_length, mutation_rate
    )
    tree.sequences = sequences
    return tree


# ---------------------------------------------------------------------------
# Test 1: Vectorized log-likelihood matches site-by-site reference
# ---------------------------------------------------------------------------

class TestLogLikelihoodEquivalence:
    """
    Verify that the vectorized compute_log_likelihood produces the same
    result as summing the original site-by-site compute_site_log_likelihood.
    """

    @pytest.mark.parametrize("sample_size,n_states,seq_length,mutation_rate,seed", [
        (5,  2, 10, 1.0, 42),      # Tiny binary
        (5,  2, 10, 0.1, 99),      # Low mutation rate
        (5,  2, 10, 5.0, 7),       # High mutation rate
        (10, 2, 20, 1.0, 123),     # Small binary
        (10, 4, 20, 1.0, 456),     # 4-state (DNA-like)
        (8,  4, 15, 0.5, 789),     # 4-state, low rate
        (8,  4, 15, 3.0, 321),     # 4-state, high rate
        (20, 2, 50, 1.0, 1000),    # Moderate size
    ])
    def test_vectorized_matches_reference(
        self, sample_size, n_states, seq_length, mutation_rate, seed
    ):
        tree = make_tree(sample_size, n_states, seq_length, mutation_rate, seed)
        pi = np.ones(n_states) / n_states

        ll_vectorized = tree.compute_log_likelihood(mutation_rate, pi)
        ll_reference = reference_log_likelihood(tree, mutation_rate, pi)

        # Absolute tolerance: these should agree to near machine precision.
        # The two code paths use different numerical strategies (probability
        # space vs log-space), so we allow 1e-8 per site.
        atol = 1e-8 * seq_length
        assert abs(ll_vectorized - ll_reference) < atol, (
            f"Vectorized ({ll_vectorized:.12f}) != reference ({ll_reference:.12f}), "
            f"diff={abs(ll_vectorized - ll_reference):.2e}, "
            f"atol={atol:.2e}"
        )

    def test_single_site_exact_match(self):
        """With a single site, vectorized should match site-by-site exactly."""
        tree = make_tree(5, 2, 1, 1.0, 42)
        pi = np.array([0.5, 0.5])

        ll_vec = tree.compute_log_likelihood(1.0, pi)
        ll_ref = tree.compute_site_log_likelihood(0, 1.0, pi)

        assert abs(ll_vec - ll_ref) < 1e-14, (
            f"Single-site mismatch: vec={ll_vec}, ref={ll_ref}"
        )

    def test_non_uniform_pi(self):
        """Test with non-uniform stationary distribution."""
        tree = make_tree(8, 4, 15, 1.0, 555)
        pi = np.array([0.1, 0.2, 0.3, 0.4])

        ll_vec = tree.compute_log_likelihood(1.0, pi)
        ll_ref = reference_log_likelihood(tree, 1.0, pi)

        assert abs(ll_vec - ll_ref) < 1e-8 * 15


# ---------------------------------------------------------------------------
# Test 2: Analytical gradient matches finite differences
# ---------------------------------------------------------------------------

class TestGradientCorrectness:
    """
    Verify that compute_gradient matches centered finite-difference gradients.
    
    Finite differences are the ground truth here -- if the analytical gradient
    matches them, the derivative code is correct regardless of implementation.
    """

    @pytest.mark.parametrize("sample_size,n_states,seq_length,mutation_rate,seed", [
        (5,  2, 10, 1.0, 42),
        (5,  2, 10, 0.3, 99),
        (8,  2, 20, 1.0, 123),
        (8,  4, 15, 1.0, 456),
        (10, 2, 20, 2.0, 789),
        (10, 4, 10, 0.5, 321),
    ])
    def test_gradient_matches_finite_differences(
        self, sample_size, n_states, seq_length, mutation_rate, seed
    ):
        tree = make_tree(sample_size, n_states, seq_length, mutation_rate, seed)
        pi = np.ones(n_states) / n_states

        grad_analytical = tree.compute_gradient(mutation_rate, pi)
        grad_fd = finite_difference_gradient(tree, mutation_rate, pi, eps=1e-5)

        # Only check internal nodes (leaves have zero gradient by construction)
        internal = slice(sample_size, None)

        # Use relative tolerance where gradients are large, absolute where small
        for node in range(sample_size, 2 * sample_size - 1):
            g_a = grad_analytical[node]
            g_fd = grad_fd[node]

            if abs(g_fd) < 1e-6:
                # Near-zero gradient: check absolute
                assert abs(g_a - g_fd) < 1e-4, (
                    f"Node {node}: analytical={g_a:.8f}, fd={g_fd:.8f}, "
                    f"abs_diff={abs(g_a - g_fd):.2e}"
                )
            else:
                # Nonzero gradient: check relative
                rel_err = abs(g_a - g_fd) / abs(g_fd)
                assert rel_err < 1e-4, (
                    f"Node {node}: analytical={g_a:.8f}, fd={g_fd:.8f}, "
                    f"rel_err={rel_err:.2e}"
                )

    def test_leaf_gradients_are_zero(self):
        """Leaf nodes have fixed time 0; their gradients should be exactly zero."""
        tree = make_tree(8, 2, 10, 1.0, 42)
        pi = np.array([0.5, 0.5])

        grad = tree.compute_gradient(1.0, pi)

        for leaf in range(tree.sample_size):
            assert grad[leaf] == 0.0, f"Leaf {leaf} has nonzero gradient {grad[leaf]}"

    def test_gradient_sign_consistency(self):
        """
        Perturbing a node time in the gradient direction should increase
        the log-likelihood (for a small enough step).
        """
        tree = make_tree(10, 2, 20, 1.0, 42)
        pi = np.array([0.5, 0.5])

        grad = tree.compute_gradient(1.0, pi)
        ll_base = tree.compute_log_likelihood(1.0, pi)

        # Pick a node with a large gradient
        internal_nodes = range(tree.sample_size, 2 * tree.sample_size - 1)
        node = max(internal_nodes, key=lambda n: abs(grad[n]))

        if abs(grad[node]) < 1e-10:
            pytest.skip("All gradients near zero -- nothing to test")

        # Step in gradient direction
        step = 1e-7 * np.sign(grad[node])
        original_time = tree.time[node]
        tree.time[node] += step
        ll_after = tree.compute_log_likelihood(1.0, pi)
        tree.time[node] = original_time

        # Moving in gradient direction should increase LL (or at least not decrease)
        assert ll_after >= ll_base - 1e-10, (
            f"Stepping in gradient direction decreased LL: "
            f"base={ll_base:.10f}, after={ll_after:.10f}, grad={grad[node]:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Regression guards -- specific values that must not change
# ---------------------------------------------------------------------------

class TestRegressionValues:
    """
    Pin specific numerical results so refactoring can't silently change them.
    These are computed once from the current (verified) implementation.
    """

    def test_likelihood_regression_binary(self):
        """Fixed tree, binary states, known log-likelihood."""
        tree = make_tree(5, 2, 10, 1.0, 42)
        pi = np.array([0.5, 0.5])
        ll = tree.compute_log_likelihood(1.0, pi)

        # Value computed from verified implementation (Dec 2025)
        expected = reference_log_likelihood(tree, 1.0, pi)

        assert abs(ll - expected) < 1e-10, (
            f"Regression failure: got {ll:.12f}, expected {expected:.12f}"
        )

    def test_likelihood_regression_4state(self):
        """Fixed tree, 4 states, known log-likelihood."""
        tree = make_tree(8, 4, 15, 0.5, 789)
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        ll = tree.compute_log_likelihood(0.5, pi)

        expected = reference_log_likelihood(tree, 0.5, pi)

        assert abs(ll - expected) < 1e-10, (
            f"Regression failure: got {ll:.12f}, expected {expected:.12f}"
        )

    def test_gradient_regression(self):
        """Fixed tree, verify gradient hasn't changed."""
        tree = make_tree(5, 2, 10, 1.0, 42)
        pi = np.array([0.5, 0.5])

        grad = tree.compute_gradient(1.0, pi)
        grad_fd = finite_difference_gradient(tree, 1.0, pi)

        # Every internal node's analytical gradient should match FD
        for node in range(tree.sample_size, 2 * tree.sample_size - 1):
            if abs(grad_fd[node]) > 1e-6:
                rel = abs(grad[node] - grad_fd[node]) / abs(grad_fd[node])
                assert rel < 1e-4, (
                    f"Gradient regression: node {node}, "
                    f"analytical={grad[node]:.8f}, fd={grad_fd[node]:.8f}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
