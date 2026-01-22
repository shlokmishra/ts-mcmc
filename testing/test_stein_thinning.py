"""
Comprehensive tests for Stein thinning on tree-valued MCMC samples.

Run with: pytest testing/test_stein_thinning.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree import Tree, coalescence_tree_with_sequences
from recorder import Recorder
from mcmc import kingman_mcmc
import stein_thinning_trees as stt


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_recorder():
    """Create a simple recorder with a few MCMC samples."""
    sample_size = 4
    n_states = 2
    seq_length = 20
    mutation_rate = 1.0
    
    tree, sequences = coalescence_tree_with_sequences(
        sample_size, n_states, seq_length, mutation_rate
    )
    tree.sequences = sequences
    
    recorder = Recorder(sample_size, seq_length)
    pi = np.array([0.5, 0.5])
    
    # Run short MCMC
    kingman_mcmc(tree, recorder, pi, steps=50, step_size=0.2)
    
    return recorder


@pytest.fixture
def medium_recorder():
    """Create a recorder with more MCMC samples."""
    sample_size = 6
    n_states = 2
    seq_length = 30
    mutation_rate = 0.8
    
    tree, sequences = coalescence_tree_with_sequences(
        sample_size, n_states, seq_length, mutation_rate
    )
    tree.sequences = sequences
    
    recorder = Recorder(sample_size, seq_length)
    pi = np.array([0.5, 0.5])
    
    # Run medium MCMC
    kingman_mcmc(tree, recorder, pi, steps=200, step_size=0.15)
    
    return recorder


# ============================================================================
# Distance Function Tests
# ============================================================================

class TestDistanceFunctions:
    """Tests for tree distance functions."""
    
    def test_kc_distance_self_zero(self, simple_recorder):
        """KC distance of tree with itself should be 0."""
        ts = simple_recorder.tree_sequence()
        trees = stt.extract_trees_from_sequence(ts)
        
        for tree in trees[:5]:
            assert stt.kc_distance(tree, tree) == 0.0
            
    def test_kc_distance_symmetric(self, simple_recorder):
        """KC distance should be symmetric."""
        ts = simple_recorder.tree_sequence()
        trees = stt.extract_trees_from_sequence(ts)
        
        for i in range(min(5, len(trees))):
            for j in range(i + 1, min(5, len(trees))):
                d1 = stt.kc_distance(trees[i], trees[j])
                d2 = stt.kc_distance(trees[j], trees[i])
                np.testing.assert_almost_equal(d1, d2)
                
    def test_kc_distance_non_negative(self, simple_recorder):
        """KC distance should be non-negative."""
        ts = simple_recorder.tree_sequence()
        trees = stt.extract_trees_from_sequence(ts)
        
        for i in range(min(5, len(trees))):
            for j in range(min(5, len(trees))):
                d = stt.kc_distance(trees[i], trees[j])
                assert d >= 0
                
    def test_distance_matrix_shape(self, simple_recorder):
        """Distance matrix should have correct shape."""
        ts = simple_recorder.tree_sequence()
        D = stt.compute_distance_matrix(ts)
        
        n = ts.num_trees
        assert D.shape == (n, n)
        
    def test_distance_matrix_symmetric(self, simple_recorder):
        """Distance matrix should be symmetric."""
        ts = simple_recorder.tree_sequence()
        D = stt.compute_distance_matrix(ts)
        
        np.testing.assert_array_almost_equal(D, D.T)
        
    def test_distance_matrix_zero_diagonal(self, simple_recorder):
        """Distance matrix diagonal should be zero."""
        ts = simple_recorder.tree_sequence()
        D = stt.compute_distance_matrix(ts)
        
        np.testing.assert_array_almost_equal(np.diag(D), np.zeros(D.shape[0]))
        

# ============================================================================
# Kernel Function Tests
# ============================================================================

class TestKernelFunctions:
    """Tests for kernel functions."""
    
    def test_gaussian_kernel_range(self):
        """Gaussian kernel should be in (0, 1]."""
        for d in [0, 0.5, 1.0, 2.0, 10.0]:
            k = stt.gaussian_kernel(d, sigma=1.0)
            assert 0 < k <= 1
            
    def test_gaussian_kernel_self_one(self):
        """Gaussian kernel at distance 0 should be 1."""
        k = stt.gaussian_kernel(0.0, sigma=1.0)
        assert k == 1.0
        
    def test_gaussian_kernel_decreasing(self):
        """Gaussian kernel should decrease with distance."""
        k1 = stt.gaussian_kernel(0.5, sigma=1.0)
        k2 = stt.gaussian_kernel(1.0, sigma=1.0)
        k3 = stt.gaussian_kernel(2.0, sigma=1.0)
        assert k1 > k2 > k3
        
    def test_imq_kernel_positive(self):
        """IMQ kernel should be positive."""
        for d in [0, 0.5, 1.0, 2.0, 10.0]:
            k = stt.imq_kernel(d)
            assert k > 0
            
    def test_tree_kernel_matrix_shape(self, simple_recorder):
        """Tree kernel matrix should have correct shape."""
        ts = simple_recorder.tree_sequence()
        K = stt.tree_kernel_matrix(ts)
        
        n = ts.num_trees
        assert K.shape == (n, n)
        
    def test_tree_kernel_matrix_symmetric(self, simple_recorder):
        """Tree kernel matrix should be symmetric."""
        ts = simple_recorder.tree_sequence()
        K = stt.tree_kernel_matrix(ts)
        
        np.testing.assert_array_almost_equal(K, K.T)
        
    def test_tree_kernel_matrix_positive_diagonal(self, simple_recorder):
        """Tree kernel matrix diagonal should be positive (self-similarity)."""
        ts = simple_recorder.tree_sequence()
        K = stt.tree_kernel_matrix(ts)
        
        assert np.all(np.diag(K) > 0)


# ============================================================================
# Stein Kernel Tests
# ============================================================================

class TestSteinKernel:
    """Tests for Stein kernel for trees."""
    
    def test_stein_kernel_matrix_shape(self, simple_recorder):
        """Stein kernel matrix should have correct shape."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        
        n = simple_recorder.tree_sequence().num_trees
        assert K.shape == (n, n)
        
    def test_stein_kernel_matrix_symmetric(self, simple_recorder):
        """Stein kernel matrix should be symmetric."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        
        np.testing.assert_array_almost_equal(K, K.T)
        
    def test_stein_kernel_with_gradients(self, simple_recorder):
        """Stein kernel should incorporate gradients."""
        K_with_grad = stt.compute_stein_kernel_matrix(
            simple_recorder, use_gradients=True
        )
        K_without_grad = stt.compute_stein_kernel_matrix(
            simple_recorder, use_gradients=False
        )
        
        # Should be different when gradients are used
        # (unless gradients happen to be zero)
        # Just check they compute without error
        assert K_with_grad.shape == K_without_grad.shape


# ============================================================================
# KSD Tests
# ============================================================================

class TestKSD:
    """Tests for kernel Stein discrepancy computation."""
    
    def test_ksd_non_negative(self, simple_recorder):
        """KSD should be non-negative."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        ksd = stt.ksd_from_matrix(K)
        
        assert ksd >= 0
        
    def test_ksd_subset(self, simple_recorder):
        """KSD for subset should be computable."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        n = K.shape[0]
        
        indices = np.array([0, 5, 10]) if n > 10 else np.array([0, 1, 2])
        ksd = stt.ksd_from_matrix(K, indices)
        
        assert ksd >= 0
        
    def test_cumulative_ksd_length(self, simple_recorder):
        """Cumulative KSD should have correct length."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        cum_ksd = stt.cumulative_ksd(K)
        
        assert len(cum_ksd) == K.shape[0]
        
    def test_cumulative_ksd_non_negative(self, simple_recorder):
        """Cumulative KSD values should be non-negative."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        cum_ksd = stt.cumulative_ksd(K)
        
        assert np.all(cum_ksd >= 0)


# ============================================================================
# Thinning Algorithm Tests
# ============================================================================

class TestThinningAlgorithm:
    """Tests for thinning algorithms."""
    
    def test_greedy_thin_length(self, simple_recorder):
        """Greedy thinning should return correct number of points."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        n = K.shape[0]
        n_points = min(10, n // 2)
        
        indices = stt.greedy_thin(K, n_points)
        
        assert len(indices) == n_points
        
    def test_greedy_thin_unique(self, simple_recorder):
        """Greedy thinning should return unique indices."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        n = K.shape[0]
        n_points = min(10, n // 2)
        
        indices = stt.greedy_thin(K, n_points)
        
        assert len(np.unique(indices)) == len(indices)
        
    def test_greedy_thin_in_range(self, simple_recorder):
        """Greedy thinning indices should be in valid range."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        n = K.shape[0]
        n_points = min(10, n // 2)
        
        indices = stt.greedy_thin(K, n_points)
        
        assert np.all(indices >= 0)
        assert np.all(indices < n)
        
    def test_greedy_thin_with_ksd(self, simple_recorder):
        """Greedy thinning should return KSD values when requested."""
        K = stt.compute_stein_kernel_matrix(simple_recorder)
        n = K.shape[0]
        n_points = min(10, n // 2)
        
        indices, ksd_values = stt.greedy_thin(K, n_points, return_ksd=True)
        
        assert len(indices) == n_points
        assert len(ksd_values) == n_points
        
    def test_thin_trees_integration(self, medium_recorder):
        """Test full thin_trees integration."""
        n_points = 20
        
        indices = stt.thin_trees(medium_recorder, n_points)
        
        assert len(indices) == n_points
        assert len(np.unique(indices)) == n_points
        
    def test_thin_trees_with_ksd(self, medium_recorder):
        """Test thin_trees with KSD return."""
        n_points = 20
        
        indices, ksd_values = stt.thin_trees(
            medium_recorder, n_points, return_ksd=True
        )
        
        assert len(indices) == n_points
        assert len(ksd_values) == n_points
        assert np.all(ksd_values >= 0)


# ============================================================================
# Naive Thinning Tests
# ============================================================================

class TestNaiveThinning:
    """Tests for naive thinning."""
    
    def test_naive_thin_length(self):
        """Naive thinning should return correct number of points."""
        indices = stt.naive_thin(100, 20)
        assert len(indices) == 20
        
    def test_naive_thin_evenly_spaced(self):
        """Naive thinning should be evenly spaced."""
        indices = stt.naive_thin(100, 5)
        expected = np.array([0, 24, 49, 74, 99])
        np.testing.assert_array_equal(indices, expected)
        
    def test_naive_thin_every_k(self):
        """Naive thinning with thin_every should select every k-th."""
        indices = stt.naive_thin(100, 10, thin_every=10)
        expected = np.arange(0, 100, 10)
        np.testing.assert_array_equal(indices, expected)


# ============================================================================
# Comparison Tests
# ============================================================================

class TestComparison:
    """Tests for comparing thinning methods."""
    
    def test_compare_thinning_methods_keys(self, medium_recorder):
        """Comparison should return expected keys."""
        results = stt.compare_thinning_methods(medium_recorder, n_points=20)
        
        expected_keys = {
            'stein_indices', 'naive_indices', 
            'stein_ksd', 'naive_ksd',
            'stein_ksd_trace', 'improvement'
        }
        assert set(results.keys()) == expected_keys
        
    def test_compare_thinning_stein_better_or_equal(self, medium_recorder):
        """Stein thinning should have KSD <= naive (or close)."""
        results = stt.compare_thinning_methods(medium_recorder, n_points=30)
        
        # Allow some numerical tolerance
        assert results['stein_ksd'] <= results['naive_ksd'] * 1.1
        
    def test_compare_thinning_improvement_reasonable(self, medium_recorder):
        """Improvement should be a reasonable fraction."""
        results = stt.compare_thinning_methods(medium_recorder, n_points=30)
        
        # Improvement should be between -10% and 100%
        assert -0.1 <= results['improvement'] <= 1.0


# ============================================================================
# TreeSteinDiscrepancy Class Tests
# ============================================================================

class TestTreeSteinDiscrepancy:
    """Tests for TreeSteinDiscrepancy class."""
    
    def test_tsd_initialization(self, simple_recorder):
        """TreeSteinDiscrepancy should initialize correctly."""
        ts = simple_recorder.tree_sequence()
        gradients = np.array(simple_recorder.gradients)
        
        tsd = stt.TreeSteinDiscrepancy(ts, gradients)
        
        assert tsd.n == ts.num_trees
        
    def test_tsd_kernel_matrix(self, simple_recorder):
        """TreeSteinDiscrepancy should compute kernel matrix."""
        ts = simple_recorder.tree_sequence()
        gradients = np.array(simple_recorder.gradients)
        
        tsd = stt.TreeSteinDiscrepancy(ts, gradients)
        K = tsd.kernel_matrix
        
        assert K.shape == (tsd.n, tsd.n)
        
    def test_tsd_ksd(self, simple_recorder):
        """TreeSteinDiscrepancy should compute KSD."""
        ts = simple_recorder.tree_sequence()
        gradients = np.array(simple_recorder.gradients)
        
        tsd = stt.TreeSteinDiscrepancy(ts, gradients)
        ksd = tsd.ksd()
        
        assert ksd >= 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_thin_single_point(self, simple_recorder):
        """Thinning to 1 point should work."""
        indices = stt.thin_trees(simple_recorder, n_points=1)
        assert len(indices) == 1
        
    def test_thin_all_points(self, simple_recorder):
        """Thinning to all points should work."""
        n = simple_recorder.tree_sequence().num_trees
        indices = stt.thin_trees(simple_recorder, n_points=n)
        assert len(indices) == n
        
    def test_empty_gradients_fallback(self, simple_recorder):
        """Should handle case with zero gradients."""
        ts = simple_recorder.tree_sequence()
        n = ts.num_trees
        
        # Zero gradients
        gradients = np.zeros((n, 1))
        
        tsd = stt.TreeSteinDiscrepancy(ts, gradients)
        ksd = tsd.ksd()
        
        assert ksd >= 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
