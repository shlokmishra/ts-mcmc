import math
import numpy as np
import tskit

def stein_thinning(recorder, subset_size):
    """
    Perform Stein thinning on the recorded trees to select a subset of given size.
    :param recorder: Recorder object with recorded trees and data.
    :param subset_size: Number of trees to select.
    :return: List of indices of selected trees.
    """
    ts = recorder.tree_sequence()  # Get the finalized tree sequence
    n = ts.num_trees
    if subset_size >= n:
        # If requested subset is larger or equal to total, return all indices
        return list(range(n))
    # Ensure data arrays are numpy arrays
    rates = recorder.mutation_rates  # numpy array of shape (n,)
    grads = recorder.gradients      # numpy array of shape (n,)
    # (We don't use log_likelihoods directly in this algorithm.)

    # Precompute Kendall-Colijn distances between all pairs of trees
    # Using lambda=1.0 to include branch length differences. 
    # (Can adjust lambda for different emphasis on topology vs branch lengths.)
    tree_list = [tree for tree in ts.trees()]  # list of Tree objects for each index
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = tree_list[i].kc_distance(tree_list[j], lambda_=1.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Define kernel parameters (bandwidths) based on data dispersion
    # For tree distances, use median distance as a scale (or 1.0 if only one tree).
    if n > 1:
        d_vals = dist_matrix[np.triu_indices(n, k=1)]  # upper triangle distances
        sigma_t = np.median(d_vals) if len(d_vals) > 0 else 1.0
        if sigma_t <= 0: 
            sigma_t = 1.0
    else:
        sigma_t = 1.0
    # For mutation rates, use standard deviation as scale
    sigma_r = np.std(rates) if np.std(rates) > 0 else 1.0

    # Precompute diagonal of Stein kernel (kappa(i,i)) for each i.
    # Stein kernel \kappa(i,i) includes grad_i^2 term (and a constant term that is same for all, which we omit for selection).
    diag_kappa = grads**2  # (We ignore constant terms that are equal for all points.)

    selected = []
    remaining = list(range(n))

    # Greedy selection
    # Step 1: pick the first point (minimize kappa(i,i) which ~ grad^2)
    first_index = int(np.argmin(diag_kappa))
    selected.append(first_index)
    remaining.remove(first_index)

    # Initialize sum of kernel values for each candidate (cost_sum[j] = sum_{i in S} kappa(i,j))
    cost_sum = np.zeros(n)
    # Compute contributions from the first selected point
    i = first_index
    for j in remaining:
        # Base kernel k0 = exp[-(d_tree^2/(sigma_t^2)) - ((r_i - r_j)^2/(sigma_r^2))]
        dist_term = dist_matrix[i, j] / sigma_t if sigma_t > 0 else 0.0
        rate_term = (rates[i] - rates[j]) / sigma_r if sigma_r > 0 else 0.0
        base_k = math.exp(- (dist_term**2) - (rate_term**2))
        # Stein kernel: kappa(i,j) = base_k * (grad_i * grad_j - (4/(sigma_r^4)) * (rates[i]-rates[j])^2)
        # (Constant 2/sigma_r^2 term omitted as it adds equal offset for all pairs.)
        kappa_ij = base_k * (grads[i] * grads[j] - (4.0 * (rates[i] - rates[j])**2) / (sigma_r**4 if sigma_r != 0 else 1.0))
        cost_sum[j] = kappa_ij

    # Iteratively select the remaining points
    while len(selected) < subset_size:
        best_val = None
        best_idx = None
        # Find the remaining index with minimum (cost_sum[j] + 0.5 * kappa(j,j))
        for j in remaining:
            # kappa(j,j) ~ grads[j]^2  (again ignoring constant term 2/sigma_r^2)
            val = cost_sum[j] + 0.5 * (grads[j]**2)
            if best_val is None or val < best_val:
                best_val = val
                best_idx = j
        # Select the best candidate
        selected.append(best_idx)
        remaining.remove(best_idx)
        # Update cost_sum for the new selection
        i = best_idx
        for j in remaining:
            dist_term = dist_matrix[i, j] / sigma_t if sigma_t > 0 else 0.0
            rate_term = (rates[i] - rates[j]) / sigma_r if sigma_r > 0 else 0.0
            base_k = math.exp(- (dist_term**2) - (rate_term**2))
            kappa_ij = base_k * (grads[i] * grads[j] - (4.0 * (rates[i] - rates[j])**2) / (sigma_r**4 if sigma_r != 0 else 1.0))
            cost_sum[j] += kappa_ij
        # (The selected point is removed from 'remaining', so we won't use cost_sum for it again.)

    return selected
