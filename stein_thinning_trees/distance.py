"""
Tree distance functions for Stein thinning.

This module provides wrappers around tskit's tree distance metrics
and additional utilities for working with tree distances.
"""

import numpy as np
import tskit
from typing import List, Union, Optional, Callable


def node_time_distance(tree1: tskit.Tree, tree2: tskit.Tree) -> float:
    """
    Compute a simple distance between trees based on internal node times.
    
    This is a fallback distance metric that works even when trees have
    multiple roots (which KC distance doesn't support).
    
    The distance is the L2 norm of the difference in sorted node times.
    
    Parameters
    ----------
    tree1 : tskit.Tree
        First tree
    tree2 : tskit.Tree
        Second tree
        
    Returns
    -------
    float
        Distance between trees based on node times
    """
    # Get node times for non-sample nodes
    ts1 = tree1.tree_sequence
    ts2 = tree2.tree_sequence
    
    # Get times of all nodes that have children in each tree
    times1 = []
    times2 = []
    
    for node in tree1.nodes():
        if tree1.num_children(node) > 0:  # Internal node
            times1.append(ts1.node(node).time)
            
    for node in tree2.nodes():
        if tree2.num_children(node) > 0:  # Internal node
            times2.append(ts2.node(node).time)
    
    # Handle edge cases
    if len(times1) == 0 and len(times2) == 0:
        return 0.0
    if len(times1) == 0 or len(times2) == 0:
        # One tree has no internal nodes - return sum of other's times
        return float(np.sum(times1) + np.sum(times2))
    
    # Sort times for comparison
    times1 = np.sort(times1)
    times2 = np.sort(times2)
    
    # Pad shorter array if needed
    max_len = max(len(times1), len(times2))
    if len(times1) < max_len:
        times1 = np.pad(times1, (0, max_len - len(times1)), constant_values=0)
    if len(times2) < max_len:
        times2 = np.pad(times2, (0, max_len - len(times2)), constant_values=0)
    
    # L2 distance
    return float(np.linalg.norm(times1 - times2))


def kc_distance(tree1: tskit.Tree, tree2: tskit.Tree, lambda_: float = 0.0) -> float:
    """
    Compute the Kendall-Colijn distance between two trees.
    
    The KC distance combines topological and branch length differences:
    - lambda_ = 0: topology only
    - lambda_ = 1: branch lengths only
    - Intermediate values mix both
    
    Parameters
    ----------
    tree1 : tskit.Tree
        First tree
    tree2 : tskit.Tree
        Second tree (must have same samples as tree1)
    lambda_ : float
        Weight parameter in [0, 1]. Default 0.0 (topology only).
        
    Returns
    -------
    float
        KC distance between the trees
        
    Raises
    ------
    ValueError
        If either tree has multiple roots (KC distance requires single-rooted trees)
    """
    # Check for multiple roots
    if tree1.num_roots != 1:
        raise ValueError(f"tree1 has {tree1.num_roots} roots, KC distance requires single-rooted trees")
    if tree2.num_roots != 1:
        raise ValueError(f"tree2 has {tree2.num_roots} roots, KC distance requires single-rooted trees")
    
    return tree1.kc_distance(tree2, lambda_=lambda_)


def rf_distance(tree1: tskit.Tree, tree2: tskit.Tree) -> int:
    """
    Compute the Robinson-Foulds distance between two trees.
    
    RF distance counts the number of bipartitions (clades) present in one
    tree but not the other. This is a purely topological metric.
    
    Parameters
    ----------
    tree1 : tskit.Tree
        First tree (must be rooted)
    tree2 : tskit.Tree
        Second tree (must be rooted, same samples as tree1)
        
    Returns
    -------
    int
        RF distance (number of differing splits)
    """
    return tree1.rf_distance(tree2)


def extract_trees_from_sequence(
    ts: tskit.TreeSequence,
    sample_lists: bool = True
) -> List[tskit.Tree]:
    """
    Extract all trees from a TreeSequence as a list.
    
    Note: tskit's trees() iterator reuses the same Tree object for efficiency.
    We must call .copy() on each tree to get independent objects.
    
    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to extract trees from
    sample_lists : bool
        Whether to enable sample lists (required for distance computation)
        
    Returns
    -------
    List[tskit.Tree]
        List of Tree objects (independent copies)
    """
    # IMPORTANT: ts.trees() reuses the same Tree object during iteration.
    # Using list(ts.trees()) gives you multiple refs to the same exhausted tree!
    # We must call .copy() on each tree to get independent objects.
    return [tree.copy() for tree in ts.trees(sample_lists=sample_lists)]


def compute_distance_matrix(
    trees: Union[tskit.TreeSequence, List[tskit.Tree]],
    distance_func: Callable[[tskit.Tree, tskit.Tree], float] = None,
    lambda_: float = 0.0,
    metric: str = 'kc'
) -> np.ndarray:
    """
    Compute pairwise distance matrix for a collection of trees.
    
    Parameters
    ----------
    trees : TreeSequence or List[Tree]
        Collection of trees
    distance_func : Callable, optional
        Custom distance function taking two trees. If None, uses metric parameter.
    lambda_ : float
        Lambda parameter for KC distance (ignored if distance_func provided)
    metric : str
        Distance metric to use if distance_func is None: 'kc', 'rf', or 'time'
        
    Returns
    -------
    np.ndarray
        n x n symmetric distance matrix
    """
    # Convert TreeSequence to list if needed
    if isinstance(trees, tskit.TreeSequence):
        tree_list = extract_trees_from_sequence(trees)
    else:
        tree_list = trees
        
    n = len(tree_list)
    D = np.zeros((n, n))
    
    # Check if trees have multiple roots (KC/RF won't work)
    has_multiple_roots = any(t.num_roots != 1 for t in tree_list)
    
    # Set up distance function
    if distance_func is None:
        if metric == 'time':
            distance_func = node_time_distance
        elif metric == 'kc':
            if has_multiple_roots:
                # Fall back to time-based distance
                import warnings
                warnings.warn(
                    "Trees have multiple roots, falling back to node_time_distance "
                    "instead of kc_distance. This is common when using TreeSequence "
                    "to store MCMC samples.",
                    RuntimeWarning
                )
                distance_func = node_time_distance
            else:
                distance_func = lambda t1, t2: kc_distance(t1, t2, lambda_=lambda_)
        elif metric == 'rf':
            if has_multiple_roots:
                import warnings
                warnings.warn(
                    "Trees have multiple roots, falling back to node_time_distance "
                    "instead of rf_distance.",
                    RuntimeWarning
                )
                distance_func = node_time_distance
            else:
                distance_func = rf_distance
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'kc', 'rf', or 'time'.")
    
    # Compute upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(tree_list[i], tree_list[j])
            D[i, j] = d
            D[j, i] = d
            
    return D


def median_distance(D: np.ndarray) -> float:
    """
    Compute the median of non-zero pairwise distances.
    
    Useful for setting kernel bandwidth parameters.
    
    Parameters
    ----------
    D : np.ndarray
        Symmetric distance matrix
        
    Returns
    -------
    float
        Median distance (or 1.0 if all distances are zero)
    """
    # Get upper triangle (excluding diagonal)
    n = D.shape[0]
    upper_tri = D[np.triu_indices(n, k=1)]
    
    if len(upper_tri) == 0 or np.all(upper_tri == 0):
        return 1.0
        
    return float(np.median(upper_tri[upper_tri > 0]))
