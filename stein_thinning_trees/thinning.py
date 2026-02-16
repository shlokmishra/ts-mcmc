"""
Stein thinning algorithms for tree-valued MCMC samples.

This module implements the core thinning algorithms adapted for
phylogenetic trees, following the approach of Riabiz et al. (2022).
"""

import logging
import numpy as np
import tskit
from typing import Union, List, Optional, Tuple

from .kernel import SteinKernelTree, tree_kernel_matrix
from .stein import TreeSteinDiscrepancy, compute_stein_kernel_matrix

logger = logging.getLogger(__name__)


def greedy_thin(
    K: np.ndarray,
    n_points: int,
    return_ksd: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Greedy point selection to minimize kernel Stein discrepancy.
    
    This implements the core greedy algorithm from Riabiz et al. (2022),
    selecting points one at a time to minimize the running KSD.
    
    Parameters
    ----------
    K : np.ndarray
        n x n Stein kernel matrix
    n_points : int
        Number of points to select
    return_ksd : bool
        If True, also return KSD values at each step
        
    Returns
    -------
    np.ndarray
        Indices of selected points (length n_points)
    np.ndarray (optional)
        KSD values at each step (if return_ksd=True)
    """
    n = K.shape[0]
    assert n_points <= n, f"Cannot select {n_points} points from {n} samples"
    
    # Pre-allocate arrays
    selected = np.empty(n_points, dtype=np.int32)
    
    # Running sum: for each point j, stores sum of K[j, selected_points]
    # This allows O(1) update when adding a new point
    running_sums = K.diagonal().copy().astype(np.float64)  # Start with K[j,j] for each j
    
    if return_ksd:
        ksd_values = np.zeros(n_points)
    
    # Select first point: minimize K[j,j]
    selected[0] = np.argmin(running_sums)
    logger.debug(f'Selected point 1/{n_points}: index {selected[0]}')
    
    # Mark selected point to prevent re-selection
    running_sums[selected[0]] = np.inf
    
    if return_ksd:
        ksd_values[0] = np.sqrt(max(0, K[selected[0], selected[0]]))
    
    # Greedy selection of remaining points
    for i in range(1, n_points):
        # Update running sums with contribution from last selected point
        last_idx = selected[i - 1]
        running_sums += 2 * K[:, last_idx]
        
        # Select point with minimum running sum (already selected have inf)
        selected[i] = np.argmin(running_sums)
        logger.debug(f'Selected point {i+1}/{n_points}: index {selected[i]}')
        
        # Mark selected point to prevent re-selection
        running_sums[selected[i]] = np.inf
        
        if return_ksd:
            # Compute KSD for selected subset
            K_sub = K[np.ix_(selected[:i+1], selected[:i+1])]
            ksd_values[i] = np.sqrt(max(0, np.sum(K_sub))) / (i + 1)
    
    if return_ksd:
        return selected, ksd_values
    return selected


def thin_trees(
    recorder,
    n_points: int,
    sigma: float = None,
    lambda_: float = 0.5,
    use_gradients: bool = True,
    return_ksd: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform Stein thinning on MCMC tree samples from a Recorder.
    
    This is the main entry point for thinning tree-valued MCMC chains.
    
    Parameters
    ----------
    recorder : Recorder
        Recorder object containing MCMC samples
    n_points : int
        Number of representative trees to select
    sigma : float, optional
        Kernel bandwidth. If None, uses median heuristic.
    lambda_ : float
        Lambda for KC distance (0=topology, 1=branch lengths)
    use_gradients : bool
        Whether to use gradient information in the Stein kernel
    return_ksd : bool
        If True, also return KSD values at each selection step
        
    Returns
    -------
    np.ndarray
        Indices of selected trees
    np.ndarray (optional)
        KSD values at each step (if return_ksd=True)
        
    Examples
    --------
    >>> from recorder import Recorder
    >>> from tree import Tree
    >>> # After running MCMC...
    >>> indices = thin_trees(recorder, n_points=100)
    >>> # Get thinned mutation rates
    >>> thinned_rates = [recorder.mutation_rates[i] for i in indices]
    """
    # Compute Stein kernel matrix
    K = compute_stein_kernel_matrix(
        recorder,
        sigma=sigma,
        lambda_=lambda_,
        use_gradients=use_gradients
    )
    
    # Run greedy selection
    return greedy_thin(K, n_points, return_ksd=return_ksd)


def thin_tree_sequence(
    ts: tskit.TreeSequence,
    gradients: np.ndarray,
    n_points: int,
    mutation_rates: np.ndarray = None,
    sigma: float = None,
    lambda_: float = 0.5,
    return_ksd: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform Stein thinning directly on a TreeSequence.
    
    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence containing MCMC samples
    gradients : np.ndarray
        Gradient information for each tree
    n_points : int
        Number of trees to select
    mutation_rates : np.ndarray, optional
        Mutation rates (not currently used in kernel, for future extension)
    sigma : float, optional
        Kernel bandwidth
    lambda_ : float
        Lambda for KC distance
    return_ksd : bool
        If True, also return KSD values
        
    Returns
    -------
    np.ndarray
        Indices of selected trees
    """
    # IMPORTANT: ts.trees() reuses the same Tree object - must copy!
    trees = [t.copy() for t in ts.trees(sample_lists=True)]
    
    # Ensure gradients is proper shape
    gradients = np.atleast_2d(gradients)
    if gradients.shape[0] != len(trees):
        if gradients.shape[1] == len(trees):
            gradients = gradients.T
            
    # Compute Stein kernel matrix
    stein_kernel = SteinKernelTree(sigma=sigma, lambda_=lambda_)
    K = stein_kernel.compute_matrix(trees, gradients, mutation_rates)
    
    return greedy_thin(K, n_points, return_ksd=return_ksd)


def naive_thin(n_samples: int, n_points: int, thin_every: int = None) -> np.ndarray:
    """
    Naive (uniform) thinning for comparison.
    
    Selects every k-th point, or evenly spaced points.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_points : int
        Number of points to select
    thin_every : int, optional
        Select every k-th point. If None, uses evenly spaced indices.
        
    Returns
    -------
    np.ndarray
        Indices of selected points
    """
    if thin_every is not None:
        # Select every k-th point
        indices = np.arange(0, n_samples, thin_every)[:n_points]
    else:
        # Evenly spaced
        indices = np.linspace(0, n_samples - 1, n_points, dtype=int)
        
    return indices


def compare_thinning_methods(
    recorder,
    n_points: int,
    sigma: float = None,
    lambda_: float = 0.5
) -> dict:
    """
    Compare Stein thinning with naive thinning.
    
    Parameters
    ----------
    recorder : Recorder
        Recorder object with MCMC samples
    n_points : int
        Number of points to select
    sigma : float, optional
        Kernel bandwidth
    lambda_ : float
        Lambda for KC distance
        
    Returns
    -------
    dict
        Dictionary with:
        - 'stein_indices': Indices from Stein thinning
        - 'naive_indices': Indices from naive thinning
        - 'stein_ksd': Final KSD for Stein thinning
        - 'naive_ksd': Final KSD for naive thinning
    """
    # Compute Stein kernel matrix once
    K = compute_stein_kernel_matrix(
        recorder,
        sigma=sigma,
        lambda_=lambda_,
        use_gradients=True
    )
    
    n_samples = K.shape[0]
    
    # Stein thinning
    stein_indices, stein_ksd_values = greedy_thin(K, n_points, return_ksd=True)
    
    # Naive thinning
    naive_indices = naive_thin(n_samples, n_points)
    
    # Compute KSD for naive thinning
    K_naive = K[np.ix_(naive_indices, naive_indices)]
    naive_ksd = np.sqrt(max(0, np.sum(K_naive))) / n_points
    
    return {
        'stein_indices': stein_indices,
        'naive_indices': naive_indices,
        'stein_ksd': stein_ksd_values[-1],
        'naive_ksd': naive_ksd,
        'stein_ksd_trace': stein_ksd_values,
        'improvement': (naive_ksd - stein_ksd_values[-1]) / naive_ksd if naive_ksd > 0 else 0
    }


def thin_trees_scalable(
    recorder,
    n_points: int,
    max_kernel_size: int = 2000,
    burnin_fraction: float = 0.1,
    sigma: float = None,
    lambda_: float = 0.5,
    use_gradients: bool = True,
    return_ksd: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Scalable Stein thinning for large MCMC chains.
    
    For chains larger than max_kernel_size, this first subsamples
    (after discarding burn-in) and then applies Stein thinning to
    the subsample. Indices are mapped back to the original chain.
    
    Parameters
    ----------
    recorder : Recorder
        Recorder object containing MCMC samples
    n_points : int
        Number of representative trees to select
    max_kernel_size : int
        Maximum number of samples for kernel matrix computation.
        Larger values give better results but use O(n^2) memory.
    burnin_fraction : float
        Fraction of initial samples to discard as burn-in (0 to 1).
    sigma : float, optional
        Kernel bandwidth. If None, uses median heuristic.
    lambda_ : float
        Lambda for KC distance
    use_gradients : bool
        Whether to use gradient information in the Stein kernel
    return_ksd : bool
        If True, also return KSD values at each selection step
        
    Returns
    -------
    np.ndarray
        Indices into the original recorder (after burn-in applied)
    np.ndarray (optional)
        KSD values at each step (if return_ksd=True)
    """
    n_total = len(recorder.mutation_rates)
    burnin = int(n_total * burnin_fraction)
    post_burnin = n_total - burnin
    
    logger.info(f"Scalable thinning: {n_total} total, {burnin} burn-in, "
                f"{post_burnin} post-burn-in, selecting {n_points}")
    
    # Create a post-burn-in index map
    all_indices = np.arange(burnin, n_total)
    
    if post_burnin <= max_kernel_size:
        # Small enough to do full thinning
        # Build a sub-recorder or pass indices
        subsample_indices = all_indices
    else:
        # Subsample to max_kernel_size
        rng = np.random.default_rng(42)
        subsample_indices = np.sort(
            rng.choice(all_indices, size=max_kernel_size, replace=False)
        )
        logger.info(f"Subsampled {post_burnin} -> {max_kernel_size} for kernel computation")
    
    # Build a temporary recorder-like object for the subsample
    class _SubRecorder:
        def __init__(self, parent_recorder, indices):
            self.mutation_rates = [parent_recorder.mutation_rates[i] for i in indices]
            self.log_likelihoods = [parent_recorder.log_likelihoods[i] for i in indices]
            self.gradients = [parent_recorder.gradients[i] for i in indices]
            self._parent = parent_recorder
            self._indices = indices
        
        def tree_sequence(self):
            """Build tree sequence from the selected subset of trees."""
            import tskit
            full_ts = self._parent.tree_sequence()
            # Extract only the trees at specified indices
            tables = tskit.TableCollection(sequence_length=len(self._indices))
            
            # Add sample nodes
            sample_size = len(list(full_ts.first().samples()))
            sample_ids = []
            for _ in range(sample_size):
                nid = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
                sample_ids.append(nid)
            
            pos = 0.0
            all_trees = [t.copy() for t in full_ts.trees()]
            for idx in self._indices:
                tree = all_trees[idx]
                node_map = {}
                for s in tree.samples():
                    node_map[s] = sample_ids[s]
                
                for n in tree.nodes(order="timeasc"):
                    if tree.is_sample(n):
                        continue
                    nid = tables.nodes.add_row(time=tree.time(n))
                    node_map[n] = nid
                
                for n in tree.nodes():
                    if tree.is_sample(n) or tree.is_root(n):
                        continue
                    tables.edges.add_row(
                        left=pos, right=pos + 1.0,
                        parent=node_map[tree.parent(n)],
                        child=node_map[n]
                    )
                    # Also add root's children edges
                for n in tree.nodes():
                    if tree.parent(n) == tree.root and not tree.is_root(n):
                        pass  # Already handled above
                
                pos += 1.0
            
            tables.sort()
            return tables.tree_sequence()
    
    # For the subsample, use the standard kernel computation
    sub_rec = _SubRecorder(recorder, subsample_indices)
    
    # Compute Stein kernel matrix on subsample
    K = compute_stein_kernel_matrix(
        sub_rec,
        sigma=sigma,
        lambda_=lambda_,
        use_gradients=use_gradients
    )
    
    # Greedy select from subsample
    result = greedy_thin(K, n_points, return_ksd=return_ksd)
    
    if return_ksd:
        local_indices, ksd_values = result
        original_indices = subsample_indices[local_indices]
        return original_indices, ksd_values
    else:
        local_indices = result
        original_indices = subsample_indices[local_indices]
        return original_indices


# Backwards compatibility - keep old function signature
def stein_thinning(recorder, subset_size):
    """
    Perform Stein thinning on recorded trees.
    
    DEPRECATED: Use thin_trees() instead.
    
    Parameters
    ----------
    recorder : Recorder
        Recorder object with recorded trees and data
    subset_size : int
        Number of trees to select
        
    Returns
    -------
    list
        Indices of selected trees
    """
    import warnings
    warnings.warn(
        "stein_thinning() is deprecated, use thin_trees() instead",
        DeprecationWarning
    )
    return list(thin_trees(recorder, subset_size))
