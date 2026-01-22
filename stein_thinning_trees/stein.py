"""
Stein discrepancy computation for tree-valued samples.

This module provides functions for computing kernel Stein discrepancy (KSD)
and related quantities for tree-valued MCMC samples.
"""

import numpy as np
import tskit
from typing import Union, List, Callable, Optional
from .kernel import SteinKernelTree, tree_kernel_matrix, combined_kernel_matrix


def ksd_from_matrix(K: np.ndarray, indices: np.ndarray = None) -> float:
    """
    Compute kernel Stein discrepancy from a kernel matrix.
    
    KSD^2 = (1/n^2) * sum_{i,j} K[i,j]
    
    Parameters
    ----------
    K : np.ndarray
        n x n Stein kernel matrix
    indices : np.ndarray, optional
        Indices of points to include. If None, uses all points.
        
    Returns
    -------
    float
        KSD value (square root of mean kernel value)
    """
    if indices is not None:
        K_sub = K[np.ix_(indices, indices)]
    else:
        K_sub = K
        
    n = K_sub.shape[0]
    ksd_squared = np.sum(K_sub) / (n * n)
    
    return np.sqrt(max(0, ksd_squared))


def cumulative_ksd(K: np.ndarray) -> np.ndarray:
    """
    Compute cumulative KSD values for increasing subsets.
    
    Returns KSD for first 1, 2, ..., n points.
    
    Parameters
    ----------
    K : np.ndarray
        n x n Stein kernel matrix
        
    Returns
    -------
    np.ndarray
        Array of length n with cumulative KSD values
    """
    n = K.shape[0]
    ksd_values = np.zeros(n)
    
    cum_sum = 0.0
    for i in range(n):
        # Add contribution from new point
        # K[i,i] + 2 * sum_{j<i} K[i,j]
        cum_sum += K[i, i] + 2 * np.sum(K[i, :i])
        ksd_values[i] = np.sqrt(max(0, cum_sum)) / (i + 1)
        
    return ksd_values


class TreeSteinDiscrepancy:
    """
    Compute Stein discrepancy for tree-valued MCMC samples.
    
    This class provides methods for computing KSD and related
    quantities using the adapted Stein kernel for trees.
    """
    
    def __init__(
        self,
        trees: Union[tskit.TreeSequence, List[tskit.Tree]],
        gradients: np.ndarray,
        mutation_rates: np.ndarray = None,
        log_likelihoods: np.ndarray = None,
        sigma: float = None,
        lambda_: float = 0.5
    ):
        """
        Initialize with MCMC samples.
        
        Parameters
        ----------
        trees : TreeSequence or List[Tree]
            Collection of trees from MCMC
        gradients : np.ndarray
            Gradients of log-likelihood w.r.t. parameters
        mutation_rates : np.ndarray, optional
            Mutation rates at each MCMC step
        log_likelihoods : np.ndarray, optional
            Log-likelihoods at each MCMC step
        sigma : float, optional
            Kernel bandwidth
        lambda_ : float
            Lambda for KC distance
        """
        # Store trees
        # IMPORTANT: ts.trees() reuses the same Tree object - must copy!
        if isinstance(trees, tskit.TreeSequence):
            self.trees = [t.copy() for t in trees.trees(sample_lists=True)]
        else:
            self.trees = trees
            
        self.n = len(self.trees)
        self.gradients = np.atleast_2d(gradients)
        if self.gradients.shape[0] == 1:
            self.gradients = self.gradients.T
        
        self.mutation_rates = mutation_rates
        self.log_likelihoods = log_likelihoods
        
        # Initialize Stein kernel
        self.stein_kernel = SteinKernelTree(sigma=sigma, lambda_=lambda_)
        
        # Lazily computed kernel matrix
        self._K = None
        
    @property
    def kernel_matrix(self) -> np.ndarray:
        """Get or compute the Stein kernel matrix."""
        if self._K is None:
            self._K = self.stein_kernel.compute_matrix(
                self.trees, 
                self.gradients,
                self.mutation_rates
            )
        return self._K
    
    def ksd(self, indices: np.ndarray = None) -> float:
        """
        Compute KSD for given indices.
        
        Parameters
        ----------
        indices : np.ndarray, optional
            Indices to include. If None, uses all points.
            
        Returns
        -------
        float
            KSD value
        """
        return ksd_from_matrix(self.kernel_matrix, indices)
    
    def cumulative_ksd(self) -> np.ndarray:
        """
        Compute cumulative KSD for increasing chain lengths.
        
        Returns
        -------
        np.ndarray
            Array of length n with cumulative KSD values
        """
        return cumulative_ksd(self.kernel_matrix)
    
    def ksd_for_subset(self, indices: np.ndarray) -> float:
        """
        Compute KSD for a specific subset of points.
        
        Parameters
        ----------
        indices : np.ndarray
            Indices of points in the subset
            
        Returns
        -------
        float
            KSD for the subset
        """
        K_sub = self.kernel_matrix[np.ix_(indices, indices)]
        n = len(indices)
        return np.sqrt(max(0, np.sum(K_sub))) / n


def compute_stein_kernel_matrix(
    recorder,
    sigma: float = None,
    lambda_: float = 0.5,
    use_gradients: bool = True
) -> np.ndarray:
    """
    Convenience function to compute Stein kernel matrix from a Recorder.
    
    Parameters
    ----------
    recorder : Recorder
        Recorder object containing MCMC samples
    sigma : float, optional
        Kernel bandwidth
    lambda_ : float
        Lambda for KC distance
    use_gradients : bool
        Whether to incorporate gradient information
        
    Returns
    -------
    np.ndarray
        Stein kernel matrix
    """
    ts = recorder.tree_sequence()
    # IMPORTANT: ts.trees() reuses the same Tree object - must copy!
    trees = [t.copy() for t in ts.trees(sample_lists=True)]
    
    if use_gradients and recorder.gradients:
        gradients = np.array(recorder.gradients)
    else:
        # Use zeros if no gradients (falls back to pure kernel)
        gradients = np.zeros((len(trees), 1))
    
    stein_kernel = SteinKernelTree(sigma=sigma, lambda_=lambda_)
    return stein_kernel.compute_matrix(trees, gradients)
