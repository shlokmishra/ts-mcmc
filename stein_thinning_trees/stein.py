"""Kernel matrix functions for phylogenetic trees"""

from typing import Callable, List, Union

import numpy as np
import tskit

def kmat(
    integrand: Callable[[int, int], float],
    n: int,
) -> np.ndarray:
    """Compute a Stein kernel matrix for phylogenetic trees.

    The matrix is obtained by evaluating the provided Stein kernel
    on all pairs of trees in the sample.

    Parameters
    ----------
    integrand: Callable[[int, int], float]
        Function returning the value of the kernel between two trees,
        given their indices.
    n: int
        Size of the matrix to return (number of trees).

    Returns
    -------
    np.ndarray
        n x n array containing the Stein kernel matrix.
    """
    k0 = np.zeros((n, n))
    ind1, ind2 = np.triu_indices(n)
    # Compute the kernel values for the upper triangle indices
    for i, j in zip(ind1, ind2):
        k_value = integrand(i, j)
        k0[i, j] = k_value
        k0[j, i] = k_value  # Since the kernel matrix is symmetric
    return k0

def ksd(
    integrand: Callable[[int, Union[int, slice]], Union[float, np.ndarray]],
    n: int,
) -> np.ndarray:
    """Compute a cumulative sequence of KSD values for phylogenetic trees.

    KSD values are calculated from sums of elements in each i x i square in the top-left
    corner of the kernel Stein matrix.

    Parameters
    ----------
    integrand: Callable[[int, Union[int, slice]], Union[float, np.ndarray]]
        Function returning the kernel values between a tree at index i
        and trees at indices specified by the second argument.
    n: int
        Number of terms to calculate.

    Returns
    -------
    np.ndarray
        Array shaped (n,) containing the sequence of KSD values.
    """
    assert n > 0
    cum_sum = np.zeros(n)
    cum_sum[0] = integrand(0, 0)
    for i in range(1, n):
        # Compute kernel values between tree i and all previous trees (0 to i)
        vals = integrand(i, slice(0, i + 1))
        cum_sum[i] = cum_sum[i - 1] + 2 * np.sum(vals) - vals[-1]
    return np.sqrt(cum_sum) / np.arange(1, n + 1)

def create_integrand(
    trees: List[tskit.TreeSequence],
    gradients: List[float],
    kernel_func: Callable[[tskit.TreeSequence, tskit.TreeSequence, float, float], float],
) -> Callable[[int, Union[int, slice]], Union[float, np.ndarray]]:
    """Create an integrand function for use in kmat and ksd functions.

    Parameters
    ----------
    trees: List[tskit.TreeSequence]
        List of tree sequences.
    gradients: List[float]
        Gradients of the log-likelihood at each tree.
    kernel_func: Callable[[tskit.TreeSequence, tskit.TreeSequence, float, float], float]
        Function that computes the kernel value between two trees and their gradients.

    Returns
    -------
    Callable[[int, Union[int, slice]], Union[float, np.ndarray]]
        Integrand function compatible with kmat and ksd functions.
    """
    def integrand(i: int, j: Union[int, slice, List[int]]) -> Union[float, np.ndarray]:
        if isinstance(j, int):
            # Compute the kernel value between trees i and j
            return kernel_func(trees[i], trees[j], gradients[i], gradients[j])
        elif isinstance(j, slice):
            # Compute kernel values between tree i and trees in the slice
            indices = list(range(*j.indices(len(trees))))
            values = []
            for idx in indices:
                value = kernel_func(trees[i], trees[idx], gradients[i], gradients[idx])
                values.append(value)
            return np.array(values)
        elif isinstance(j, list):
            # If j is a list of indices
            values = []
            for idx in j:
                value = kernel_func(trees[i], trees[idx], gradients[i], gradients[idx])
                values.append(value)
            return np.array(values)
        else:
            raise TypeError("Index j must be an int, slice, or list of ints.")
    return integrand
