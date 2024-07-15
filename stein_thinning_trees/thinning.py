import numpy as np
import tskit
from stein_thinning.kernel import make_imq

def thin(trees, gradients, m, pre='id', verb=False):
    """
    Optimally select m points from n > m samples generated from a target
    distribution of d dimensions.

    Args:
    trees      - List of tskit.Tree objects.
    gradients  - List of corresponding gradients.
    m          - Integer specifying the desired number of points.
    pre        - Optional string, either 'id' (default), 'med', 'sclmed', or
                 'smpcov', specifying the preconditioner to be used. Alternatively,
                 a numeric string can be passed as the single length-scale parameter
                 of an isotropic kernel.
    verb       - Optional logical, either 'True' or 'False' (default), indicating
                 whether or not to be verbose about the thinning progress.

    Returns:
    array shaped (m,) containing the indices in trees (and gradients) of the
    selected points.
    """
    # Argument checks
    if not isinstance(trees, list) or not isinstance(gradients, list):
        raise Exception('trees or gradients is not a list.')
    n = len(trees)
    if n == 0 or len(gradients) != n:
        raise Exception('trees and gradients must have the same non-zero length.')
    if any(tree is None for tree in trees) or any(grad is None for grad in gradients):
        raise Exception('trees or gradients contains None elements.')

    # Vectorised Stein kernel function
    vfk0 = make_imq(trees, gradients, pre)

    # Pre-allocate arrays
    k0 = np.empty((n, m))
    idx = np.empty(m, dtype=np.uint32)

    # Populate columns of k0 as new points are selected
    k0[:, 0] = vfk0(trees, trees, gradients, gradients)
    idx[0] = np.argmin(k0[:, 0])
    if verb:
        print(f'THIN: {1} of {m}')
    for i in range(1, m):
        tree_last = [trees[idx[i - 1]] for _ in range(n)]
        grad_last = [gradients[idx[i - 1]] for _ in range(n)]
        k0[:, i] = vfk0(trees, tree_last, gradients, grad_last)
        idx[i] = np.argmin(k0[:, 0] + 2 * np.sum(k0[:, 1:(i + 1)], axis=1))
        if verb:
            print(f'THIN: {i + 1} of {m}')
    return idx
