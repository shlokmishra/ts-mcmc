"""
Kernel functions for tree-valued Stein thinning.

This module provides kernel functions defined on tree space and
combined tree-parameter space for use in Stein thinning.

The key insight is that while traditional Stein kernels are defined in R^d,
we can adapt them for trees by:
1. Using tree distances (KC, RF) as the base metric
2. Combining with continuous parameters (mutation rate, branch times)
3. Incorporating gradient information where available
"""

import numpy as np
import tskit
from typing import Union, List, Callable, Optional
from .distance import kc_distance, rf_distance, compute_distance_matrix, median_distance


def gaussian_kernel(d: float, sigma: float = 1.0) -> float:
    """
    Gaussian (RBF) kernel: k(d) = exp(-d^2 / (2 * sigma^2))
    
    Parameters
    ----------
    d : float
        Distance between points
    sigma : float
        Bandwidth parameter
        
    Returns
    -------
    float
        Kernel value in [0, 1]
    """
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


def imq_kernel(d: float, c: float = 1.0, beta: float = -0.5) -> float:
    """
    Inverse Multiquadric (IMQ) kernel: k(d) = (c + d^2)^beta
    
    The IMQ kernel is commonly used in Stein thinning due to its
    theoretical properties.
    
    Parameters
    ----------
    d : float
        Distance between points
    c : float
        Positive constant. Default 1.0.
    beta : float
        Exponent, typically negative. Default -0.5.
        
    Returns
    -------
    float
        Kernel value
    """
    return (c + d ** 2) ** beta


def tree_kernel_matrix(
    trees: Union[tskit.TreeSequence, List[tskit.Tree]],
    kernel_func: Callable[[float], float] = None,
    sigma: float = None,
    lambda_: float = 0.5,
    metric: str = 'kc'
) -> np.ndarray:
    """
    Compute kernel matrix for a collection of trees.
    
    Parameters
    ----------
    trees : TreeSequence or List[Tree]
        Collection of trees
    kernel_func : Callable, optional
        Function mapping distance to kernel value. If None, uses Gaussian.
    sigma : float, optional
        Bandwidth for Gaussian kernel. If None, uses median heuristic.
    lambda_ : float
        Lambda parameter for KC distance
    metric : str
        Distance metric: 'kc' or 'rf'
        
    Returns
    -------
    np.ndarray
        n x n kernel matrix
    """
    # Compute distance matrix
    D = compute_distance_matrix(trees, lambda_=lambda_, metric=metric)
    n = D.shape[0]
    
    # Set bandwidth using median heuristic if not provided
    if sigma is None:
        sigma = median_distance(D)
        if sigma <= 0:
            sigma = 1.0
    
    # Set default kernel function
    if kernel_func is None:
        kernel_func = lambda d: gaussian_kernel(d, sigma)
    
    # Compute kernel matrix
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = kernel_func(0.0)  # Self-similarity
        for j in range(i + 1, n):
            k_val = kernel_func(D[i, j])
            K[i, j] = k_val
            K[j, i] = k_val
            
    return K


def combined_kernel_matrix(
    trees: Union[tskit.TreeSequence, List[tskit.Tree]],
    params: np.ndarray,
    sigma_tree: float = None,
    sigma_param: float = None,
    lambda_: float = 0.5,
    tree_weight: float = 0.5
) -> np.ndarray:
    """
    Compute kernel matrix combining tree structure and continuous parameters.
    
    This creates a product kernel:
    K(i,j) = K_tree(T_i, T_j) * K_param(θ_i, θ_j)
    
    Or a weighted sum:
    K(i,j) = w * K_tree(T_i, T_j) + (1-w) * K_param(θ_i, θ_j)
    
    Parameters
    ----------
    trees : TreeSequence or List[Tree]
        Collection of trees
    params : np.ndarray
        n x d array of continuous parameters (e.g., mutation rates)
        If 1D, will be reshaped to n x 1.
    sigma_tree : float, optional
        Bandwidth for tree kernel. Uses median heuristic if None.
    sigma_param : float, optional
        Bandwidth for parameter kernel. Uses median heuristic if None.
    lambda_ : float
        Lambda parameter for KC distance
    tree_weight : float
        Weight for tree kernel in [0, 1]. Default 0.5 for equal weighting.
        
    Returns
    -------
    np.ndarray
        n x n combined kernel matrix
    """
    # Ensure params is 2D
    params = np.atleast_2d(params)
    if params.shape[0] == 1:
        params = params.T  # Convert to column vector
    n = params.shape[0]
    
    # Compute tree kernel
    K_tree = tree_kernel_matrix(trees, sigma=sigma_tree, lambda_=lambda_)
    
    # Compute parameter distance matrix
    from scipy.spatial.distance import cdist
    D_param = cdist(params, params, metric='euclidean')
    
    # Set parameter bandwidth using median heuristic
    if sigma_param is None:
        upper_tri = D_param[np.triu_indices(n, k=1)]
        sigma_param = np.median(upper_tri[upper_tri > 0]) if np.any(upper_tri > 0) else 1.0
        if sigma_param <= 0:
            sigma_param = 1.0
    
    # Compute parameter kernel
    K_param = np.exp(-(D_param ** 2) / (2 * sigma_param ** 2))
    
    # Combine kernels (weighted sum)
    K = tree_weight * K_tree + (1 - tree_weight) * K_param
    
    return K


class SteinKernelTree:
    """
    Stein kernel for tree-valued MCMC samples.
    
    This implements a Stein kernel adapted for phylogenetic trees,
    incorporating both tree structure and gradient information.
    
    The Stein kernel has the form:
    κ(x,y) = ∇_x · ∇_y k(x,y) + ∇_x k(x,y) · ∇ log p(y) 
             + ∇_y k(x,y) · ∇ log p(x) + k(x,y) · ∇ log p(x) · ∇ log p(y)
             
    For trees, we approximate this using:
    - Base kernel k based on tree distance
    - Gradients w.r.t. node times (from MCMC)
    """
    
    def __init__(
        self,
        sigma: float = None,
        lambda_: float = 0.5,
        c: float = 1.0,
        beta: float = -0.5,
        kernel_type: str = 'imq'
    ):
        """
        Initialize Stein kernel for trees.
        
        Parameters
        ----------
        sigma : float, optional
            Bandwidth parameter. If None, uses median heuristic.
        lambda_ : float
            Lambda for KC distance (0=topology, 1=branch lengths)
        c : float
            IMQ kernel parameter c
        beta : float
            IMQ kernel exponent
        kernel_type : str
            'imq' for inverse multiquadric, 'gaussian' for RBF
        """
        self.sigma = sigma
        self.lambda_ = lambda_
        self.c = c
        self.beta = beta
        self.kernel_type = kernel_type
        
    def _base_kernel(self, d: float, sigma: float) -> float:
        """Compute base kernel value from distance."""
        if self.kernel_type == 'imq':
            # IMQ: (c + d^2/sigma^2)^beta
            return (self.c + (d / sigma) ** 2) ** self.beta
        else:
            # Gaussian
            return gaussian_kernel(d, sigma)
    
    def compute_matrix(
        self,
        trees: Union[tskit.TreeSequence, List[tskit.Tree]],
        gradients: np.ndarray,
        mutation_rates: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute Stein kernel matrix.
        
        Parameters
        ----------
        trees : TreeSequence or List[Tree]
            Collection of n trees from MCMC
        gradients : np.ndarray
            n x d array of gradients (d = number of internal nodes, or scalar)
            These are gradients of log-likelihood w.r.t. node times.
        mutation_rates : np.ndarray, optional
            n-length array of mutation rates
            
        Returns
        -------
        np.ndarray
            n x n Stein kernel matrix
        """
        # Extract trees if TreeSequence
        # IMPORTANT: ts.trees() reuses the same Tree object - must copy!
        if isinstance(trees, tskit.TreeSequence):
            tree_list = [t.copy() for t in trees.trees(sample_lists=True)]
        else:
            tree_list = trees
            
        n = len(tree_list)
        
        # Ensure gradients is proper shape
        gradients = np.atleast_2d(gradients)
        if gradients.shape[0] == 1:
            gradients = gradients.T
        if gradients.shape[0] != n:
            # If gradients are stored differently, try to reshape
            if gradients.shape[1] == n:
                gradients = gradients.T
                
        # Compute distance matrix
        D = compute_distance_matrix(tree_list, lambda_=self.lambda_, metric='kc')
        
        # Set bandwidth using median heuristic
        sigma = self.sigma
        if sigma is None:
            sigma = median_distance(D)
            if sigma <= 0:
                sigma = 1.0
        
        # Compute Stein kernel matrix
        # Simplified form for tree space:
        # κ(i,j) = k(d_ij) * (1 + grad_i · grad_j)
        # This captures the essence of Stein kernel while being applicable to trees
        
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                d_ij = D[i, j]
                k_ij = self._base_kernel(d_ij, sigma)
                
                # Gradient term
                grad_i = gradients[i].flatten()
                grad_j = gradients[j].flatten()
                
                # Ensure same length (take min if different)
                min_len = min(len(grad_i), len(grad_j))
                grad_term = np.dot(grad_i[:min_len], grad_j[:min_len])
                
                # Stein kernel value
                kappa_ij = k_ij * (1 + grad_term)
                
                K[i, j] = kappa_ij
                K[j, i] = kappa_ij
                
        return K
