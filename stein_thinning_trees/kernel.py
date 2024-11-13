"""Kernel definitions for phylogenetic trees"""

import numpy as np
from scipy.spatial.distance import pdist
import tskit

def kc_distance_wrapper(tree1, tree2, lambda_=0.0):
    """
    Wrapper function to compute the Kendall-Colijn (KC) distance between two trees.
    
    Parameters
    ----------
    tree1: tskit.TreeSequence
        The first tree sequence.
    tree2: tskit.TreeSequence
        The second tree sequence.
    lambda_: float
        The KC metric lambda parameter determining the relative weight of topology and branch length.

    Returns
    -------
    float
        The KC distance between tree1 and tree2.
    """
    return tree1.kc_distance(tree2, lambda_)

def vfk0_imq_trees(
        tree_x,
        tree_y,
        grad_x,
        grad_y,
        c: float = 1.0,
        beta: float = -0.5,
        lambda_: float = 0.0,
    ) -> float:
    """
    Evaluate Stein kernel based on inverse multiquadratic kernel for phylogenetic trees.

    Parameters
    ----------
    tree_x: tskit.TreeSequence
        The first tree.
    tree_y: tskit.TreeSequence
        The second tree.
    grad_x: float
        Gradient of the log-likelihood at tree_x.
    grad_y: float
        Gradient of the log-likelihood at tree_y.
    c: float
        Parameter of the inverse multiquadratic kernel. Default: 1.0.
    beta: float
        Exponent of the inverse multiquadratic kernel. Default: -0.5.
    lambda_: float
        The KC metric lambda parameter.

    Returns
    -------
    float
        The value of the kernel evaluated for the pair of trees.
    """
    # Compute the distance between trees
    dist = kc_distance_wrapper(tree_x, tree_y, lambda_)
    qf = c + dist ** 2

    # Compute the kernel value
    kernel_value = (qf) ** beta

    # Compute the Stein kernel components
    t1 = -2 * beta * (grad_x - grad_y) * dist / (qf ** (1 - beta))
    t2 = grad_x * grad_y / (qf ** (-beta))

    return t1 + t2 + kernel_value

def make_imq_trees(sample_trees, gradients, c: float = 1.0, beta: float = -0.5, lambda_: float = 0.0):
    """
    Creates a kernel function for phylogenetic trees using the inverse multiquadratic kernel.

    Parameters
    ----------
    sample_trees: list of tskit.TreeSequence
        List of sampled trees.
    gradients: list of floats
        Gradients of the log-likelihood at each sampled tree.
    c: float
        Parameter of the inverse multiquadratic kernel. Default: 1.0.
    beta: float
        Exponent of the inverse multiquadratic kernel. Default: -0.5.
    lambda_: float
        The KC metric lambda parameter.

    Returns
    -------
    function
        A function that computes the Stein kernel between pairs of trees.
    """
    def kernel(tree_i, tree_j, grad_i, grad_j):
        return vfk0_imq_trees(tree_i, tree_j, grad_i, grad_j, c, beta, lambda_)
    return kernel
