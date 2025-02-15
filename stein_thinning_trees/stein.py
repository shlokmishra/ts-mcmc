import numpy as np
import tskit
from . import kernel  # kernel module provides kc_distance_wrapper and kernel functions

class SteinThinner:
    def __init__(self, kernel_func=None, kernel_param=None):
        """
        Stein Thinner for tree-space samples.
        :param kernel_func: A kernel function accepting two tskit.Tree objects.
        :param kernel_param: Additional parameters for the kernel (e.g., bandwidth).
        """
        # Use a default Gaussian kernel on Kendall-Colijn distance if none provided
        if kernel_func is None:
            self.kernel_func = lambda t1, t2: np.exp(- kernel.kc_distance_wrapper(t1, t2)**2 / 2.0)
        else:
            self.kernel_func = kernel_func
        self.kernel_param = kernel_param

    def thin(self, trees, m):
        """
        Select m representative trees from the given collection (trees can be a TreeSequence or list of Tree).
        Returns indices of selected trees in the original collection.
        """
        # Ensure we have a list of tskit.Tree objects
        if isinstance(trees, tskit.TreeSequence):
            # Extract all trees from the TreeSequence
            tree_list = [tree for tree in trees.trees(sample_lists=True)]
        elif isinstance(trees, list) and all(isinstance(t, tskit.Tree) for t in trees):
            tree_list = trees
        else:
            raise TypeError("Input must be a tskit.TreeSequence or list of tskit.Tree objects")
        n = len(tree_list)
        if m >= n:
            return list(range(n))  # no thinning needed (return all indices)

        # Pre-compute kernel matrix for efficiency
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = self.kernel_func(tree_list[i], tree_list[i])
            for j in range(i+1, n):
                K_val = self.kernel_func(tree_list[i], tree_list[j])
                K[i, j] = K_val
                K[j, i] = K_val

        # Greedily select m samples to minimize kernel Stein discrepancy
        selected_idx = []
        remaining_idx = list(range(n))
        # (Optional) initialization: pick first point (e.g., at random or maximize diversity)
        # Here, pick the first tree as the initial representative
        selected_idx.append(remaining_idx.pop(0))

        # Iteratively select remaining points
        for _ in range(1, m):
            best_idx = None
            best_obj_val = None
            # Evaluate Stein discrepancy objective for each candidate if added
            for j in remaining_idx:
                candidate_idx = selected_idx + [j]
                # Compute objective (KSD) for candidate set
                obj_val = self._stein_discrepancy(K, candidate_idx)
                if best_obj_val is None or obj_val < best_obj_val:
                    best_obj_val = obj_val
                    best_idx = j
            # Select the index that gives minimum discrepancy
            selected_idx.append(best_idx)
            remaining_idx.remove(best_idx)
        return selected_idx

    def _stein_discrepancy(self, K, idx_set):
        """
        Compute (approximate) kernel Stein discrepancy for a set of indices.
        Here we use a simplified measure: sum of pairwise kernel values within the set.
        Lower is better for diversity (assuming kernel is similarity measure).
        """
        # Sum of K over all pairs in idx_set (including self-pairs)
        subK = K[np.ix_(idx_set, idx_set)]
        return np.sum(subK)
