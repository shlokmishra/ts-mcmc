import tskit
import numpy as np

def kc_distance_wrapper(tree1, tree2, lambda_=0.0):
    """
    Wrapper for tskit.Tree.kc_distance to compute Kendall-Colijn distance.
    Both inputs should be tskit.Tree objects (from the same TreeSequence or with matching samples).
    """
    if not isinstance(tree1, tskit.Tree) or not isinstance(tree2, tskit.Tree):
        raise TypeError("kc_distance_wrapper expects tskit.Tree instances")
    # Compute Kendall-Colijn distance using tskit (lambda_ controls branch length weight)
    return tree1.kc_distance(tree2, lambda_)

def rf_distance_wrapper(tree1, tree2):
    """
    Optional: Wrapper for Robinson-Foulds distance (topological distance) using tskit.
    """
    if not isinstance(tree1, tskit.Tree) or not isinstance(tree2, tskit.Tree):
        raise TypeError("rf_distance_wrapper expects tskit.Tree instances")
    return tree1.rf_distance(tree2)

# Example kernel function using kc_distance (could be passed into SteinThinner):
def gaussian_kernel_tree(tree1, tree2, sigma=1.0):
    """
    Gaussian kernel on tree space using Kendall-Colijn distance.
    k(T1, T2) = exp(-d(T1,T2)^2 / (2*sigma^2))
    """
    d = kc_distance_wrapper(tree1, tree2)
    return float(np.exp(-(d**2) / (2 * (sigma**2))))
