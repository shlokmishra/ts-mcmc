"""Implementation of Stein thinning for phylogenetic trees"""

import logging
from typing import Callable, List, Union
import numpy as np
import tskit

logger = logging.getLogger(__name__)


def _validate_trees_and_gradients(
    trees: List[tskit.TreeSequence],
    gradients: List[float],
):
    """
    Validate the trees and gradients.

    Parameters
    ----------
    trees: List[tskit.TreeSequence]
        List of tree sequences.
    gradients: List[float]
        List of gradients corresponding to each tree.

    Returns
    -------
    Tuple[List[tskit.TreeSequence], List[float]]
        Validated trees and gradients.
    """
    n = len(trees)
    assert n > 0, 'No trees provided.'
    assert len(gradients) == n, 'Number of gradients does not match number of trees.'

    for i, grad in enumerate(gradients):
        assert not np.isnan(grad), f'Gradient at index {i} contains NaNs.'
        assert not np.isinf(grad), f'Gradient at index {i} contains infinities.'

    return trees, gradients


def _make_stein_integrand_trees(
    trees: List[tskit.TreeSequence],
    gradients: List[float],
    kernel_func: Callable[[tskit.TreeSequence, tskit.TreeSequence, float, float], float],
):
    """
    Create an integrand function for use in thinning algorithms.

    Parameters
    ----------
    trees: List[tskit.TreeSequence]
        List of tree sequences.
    gradients: List[float]
        List of gradients corresponding to each tree.
    kernel_func: Callable[[tskit.TreeSequence, tskit.TreeSequence, float, float], float]
        Kernel function that computes the kernel value between two trees and their gradients.

    Returns
    -------
    Callable[[Union[int, slice, List[int]], Union[int, slice, List[int]]], np.ndarray]
        Integrand function compatible with the thinning algorithm.
    """
    trees, gradients = _validate_trees_and_gradients(trees, gradients)
    n = len(trees)

    def integrand(ind1, ind2):
        # Convert indices to lists
        if isinstance(ind1, slice):
            ind1 = list(range(*ind1.indices(n)))
        elif isinstance(ind1, int):
            ind1 = [ind1]
        else:
            ind1 = list(ind1)

        if isinstance(ind2, slice):
            ind2 = list(range(*ind2.indices(n)))
        elif isinstance(ind2, int):
            ind2 = [ind2]
        else:
            ind2 = list(ind2)

        values = np.zeros((len(ind1), len(ind2)))
        for i, idx1 in enumerate(ind1):
            for j, idx2 in enumerate(ind2):
                values[i, j] = kernel_func(
                    trees[idx1], trees[idx2], gradients[idx1], gradients[idx2]
                )
        return values

    return integrand


def _greedy_search(
    n_points: int,
    integrand: Callable[[Union[int, slice, List[int]], Union[int, slice, List[int]]], np.ndarray],
) -> np.ndarray:
    """
    Select points minimizing total kernel Stein distance.

    Parameters
    ----------
    n_points: int
        Number of points to select.
    integrand: Callable[[IndexerT, IndexerT], np.ndarray]
        Function returning values of the integrand in the KSD integral
        for points identified by two indices (row and column).

    Returns
    -------
    np.ndarray
        Indices of selected points.
    """
    idx = np.empty(n_points, dtype=np.uint32)
    k_matrix = integrand(slice(None), slice(None))
    k0 = np.diag(k_matrix)
    idx[0] = np.argmin(k0)
    logger.debug('THIN: %d of %d', 1, n_points)

    for i in range(1, n_points):
        k_vector = integrand(slice(None), [idx[i - 1]]).flatten()
        k0 += 2 * k_vector
        idx[i] = np.argmin(k0)
        logger.debug('THIN: %d of %d', i + 1, n_points)

    return idx


def thin_trees(
    trees: List[tskit.TreeSequence],
    gradients: List[float],
    n_points: int,
    kernel_func: Callable[[tskit.TreeSequence, tskit.TreeSequence, float, float], float],
) -> np.ndarray:
    """
    Optimally select m points from n > m sampled trees.

    Parameters
    ----------
    trees: List[tskit.TreeSequence]
        List of tree sequences.
    gradients: List[float]
        List of gradients corresponding to each tree.
    n_points: int
        Number of points to select.
    kernel_func: Callable[[tskit.TreeSequence, tskit.TreeSequence, float, float], float]
        Kernel function that computes the kernel value between two trees and their gradients.

    Returns
    -------
    np.ndarray
        Array containing the indices of the selected trees.
    """
    integrand = _make_stein_integrand_trees(trees, gradients, kernel_func)
    selected_indices = _greedy_search(n_points, integrand)
    return selected_indices
