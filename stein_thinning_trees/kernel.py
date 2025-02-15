import tskit
import numpy as np

def kc_distance_wrapper(tree1, tree2, **kwargs):
    """
    Compute the Kendall-Colijn distance between two tree structures (tskit.Tree or TreeSequence).
    Ensures that if a TreeSequence is provided, it is converted to a Tree with sample lists enabled.
    Additional kwargs (like lambda_) are passed to tskit.Tree.kc_distance().
    """
    # If tree1 is a TreeSequence, extract the first tree with sample lists enabled
    if isinstance(tree1, tskit.TreeSequence):
        # Ensure we get a Tree with sample_lists=True to avoid NoSampleListsError
        tree1 = tree1.first(sample_lists=True)
    # If tree2 is a TreeSequence, do the same
    if isinstance(tree2, tskit.TreeSequence):
        tree2 = tree2.first(sample_lists=True)
    # Now tree1 and tree2 are tskit.Tree objects with sample lists (if originally TreeSequences)
    return tree1.kc_distance(tree2, **kwargs)

def rf_distance_wrapper(tree1, tree2):
    """
    Another metric for distance: Wrapper for Robinson-Foulds distance (topological distance) using tskit.
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
