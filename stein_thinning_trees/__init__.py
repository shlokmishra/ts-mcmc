"""
Stein Thinning for Tree-Valued MCMC Samples

This package implements Stein thinning algorithms adapted for phylogenetic
trees from MCMC sampling, based on the work of Riabiz et al. (2022).

Main Functions
--------------
thin_trees : Main thinning function for Recorder objects
thin_tree_sequence : Thin directly from a TreeSequence
compare_thinning_methods : Compare Stein vs naive thinning

Classes
-------
TreeSteinDiscrepancy : Compute KSD for tree samples
SteinKernelTree : Stein kernel adapted for trees

Example
-------
>>> from recorder import Recorder
>>> from stein_thinning_trees import thin_trees, compare_thinning_methods
>>> 
>>> # After running MCMC and collecting samples in recorder...
>>> indices = thin_trees(recorder, n_points=100)
>>> 
>>> # Compare with naive thinning
>>> results = compare_thinning_methods(recorder, n_points=100)
>>> print(f"Stein KSD: {results['stein_ksd']:.4f}")
>>> print(f"Naive KSD: {results['naive_ksd']:.4f}")
>>> print(f"Improvement: {results['improvement']:.1%}")
"""

from .distance import (
    kc_distance,
    rf_distance,
    node_time_distance,
    compute_distance_matrix,
    extract_trees_from_sequence,
    median_distance,
)

from .kernel import (
    gaussian_kernel,
    imq_kernel,
    tree_kernel_matrix,
    combined_kernel_matrix,
    SteinKernelTree,
)

from .stein import (
    ksd_from_matrix,
    cumulative_ksd,
    TreeSteinDiscrepancy,
    compute_stein_kernel_matrix,
)

from .thinning import (
    greedy_thin,
    thin_trees,
    thin_tree_sequence,
    naive_thin,
    compare_thinning_methods,
    stein_thinning,  # Backwards compatibility
)

__version__ = "0.1.0"

__all__ = [
    # Distance functions
    'kc_distance',
    'rf_distance',
    'node_time_distance',
    'compute_distance_matrix',
    'extract_trees_from_sequence',
    'median_distance',
    
    # Kernel functions
    'gaussian_kernel',
    'imq_kernel',
    'tree_kernel_matrix',
    'combined_kernel_matrix',
    'SteinKernelTree',
    
    # Stein discrepancy
    'ksd_from_matrix',
    'cumulative_ksd',
    'TreeSteinDiscrepancy',
    'compute_stein_kernel_matrix',
    
    # Thinning
    'greedy_thin',
    'thin_trees',
    'thin_tree_sequence',
    'naive_thin',
    'compare_thinning_methods',
    'stein_thinning',
]
