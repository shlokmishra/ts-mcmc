import math
import numpy
import random
from scipy.stats import gamma

from recorder import Recorder
from tree import Tree


def kingman_mcmc(tree, recorder, pi, steps=None, step_size=0.1):
    """
    Performs Markov Chain Monte Carlo (MCMC) sampling on a phylogenetic tree using Kingman's coalescent.
    
    Parameters:
    - tree: Instance of the Tree class representing the current phylogenetic tree.
    - recorder: Instance of the Recorder class for storing sampled trees and related information.
    - pi: List or array representing the base frequency distribution.
    - steps: Optional integer specifying the number of MCMC iterations.
    - step_size: float
        The standard deviation for the Gaussian proposal in mutation rate resampling.
    
    Returns:
    - List containing acceptance probabilities for SPR moves, resampling times, and mutation rate resampling.
    """
    # Initialize acceptance counters
    acceptance_count_spr = 0
    acceptance_count_times = 0
    acceptance_count_mutations = 0

    # Initialize log-likelihood and prior (use only compute_log_likelihood)
    log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)

    # Define the prior over the mutation rate
    def log_prior_mutation_rate(rate):
        # Gamma prior with shape a and scale b
        a = 2.0  # Shape parameter
        b = 1.0  # Scale parameter
        if rate <= 0:
            return -math.inf  # Log-probability of zero for invalid rates
        return gamma.logpdf(rate, a=a, scale=b)

    current_log_prior = log_prior_mutation_rate(tree._mutation_rate)

    # Determine the number of MCMC steps
    if steps is None:
        steps = int(recorder.tables.sequence_length)

    for i in range(steps):
        # -------------------
        # 1. Subtree Pruning and Regrafting (SPR) Move
        # -------------------
        # Sample a leaf node to detach
        child = tree.sample_leaf()
        parent = tree.parent[child]
        sib = tree.sibling(child)
        new_sib = sib

        # Select a new sibling node ensuring it's not the child, parent, or current sibling
        if tree.sample_size > 2:
            attempts = 0
            max_attempts = 100
            while new_sib in [child, parent, sib] and attempts < max_attempts:
                new_sib = tree.sample_node()
                attempts += 1
            if attempts == max_attempts:
                raise RuntimeError("Failed to sample a valid new sibling node.")

        # Propose a new attachment time for the subtree
        new_time = tree.sample_reattach_time(child, new_sib)

        # Compute the acceptance ratio for the SPR move
        alpha = -tree.log_reattach_density(child, new_sib, new_time)  # log(q(new | old))
        old_time = tree.time[parent]
        tree.detach_reattach(child, new_sib, new_time)  # Apply the SPR move
        alpha += tree.log_reattach_density(child, sib, old_time)  # log(q(old | new))
        proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)
        alpha += proposal_log_likelihood - log_likelihood  # log(p(new)) - log(p(old))

        # Metropolis-Hastings acceptance step for SPR move
        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            acceptance_count_spr += 1
        else:
            tree.detach_reattach(child, sib, old_time)  # Revert SPR move

        # -------------------
        # 2. Resample Node Times
        # -------------------
        # Note: resample_times() proposes new times from the Kingman coalescent prior.
        # Since the proposal distribution equals the prior, the MH acceptance ratio
        # simplifies to just the likelihood ratio (prior terms cancel with proposal terms).
        old_times = tree.time.copy()
        tree.resample_times()  # Propose new times from prior
        proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)

        # Compute acceptance ratio: just likelihood ratio since proposal = prior
        alpha = proposal_log_likelihood - log_likelihood

        # Metropolis-Hastings acceptance step for resampling times
        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            acceptance_count_times += 1
        else:
            tree.time = old_times.copy()

        # -------------------
        # 3. Mutation Rate Resampling (Log-Normal Proposal)
        # -------------------
        old_mutation_rate, log_proposal_density, log_reverse_proposal_density = tree.resample_mutation_rate(step_size=step_size)

        # Compute the log prior for the new and old mutation rates
        proposal_log_prior = log_prior_mutation_rate(tree._mutation_rate)
        reverse_log_prior = log_prior_mutation_rate(old_mutation_rate)

        # Compute the proposal log-likelihood
        proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)

        # Compute the acceptance ratio with correct proposal densities
        alpha = (proposal_log_likelihood + proposal_log_prior + log_reverse_proposal_density) - (log_likelihood + current_log_prior + log_proposal_density)

        # Metropolis-Hastings acceptance step for mutation rate resampling
        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            current_log_prior = proposal_log_prior
            acceptance_count_mutations += 1
        else:
            tree._mutation_rate = old_mutation_rate  # Revert to old mutation rate

        # -------------------
        # 4. Record the Current State
        # -------------------
        # Compute gradient after all updates
        gradient = tree.compute_gradient(tree._mutation_rate, pi)
        recorder.append_tree(tree, tree._mutation_rate, log_likelihood, gradient)

        # -------------------
        # 5. Optional: Progress Monitoring
        # -------------------
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Step {i+1}/{steps}: Mutation Rate = {tree._mutation_rate:.4f}, "
                  f"Log Likelihood = {log_likelihood:.4f}, "
                  f"Acceptance Rates - SPR: {acceptance_count_spr/(i+1):.2f}, "
                  f"Times: {acceptance_count_times/(i+1):.2f}, "
                  f"Mutations: {acceptance_count_mutations/(i+1):.2f}")

    # Compute final acceptance probabilities
    acceptance_prob_spr = acceptance_count_spr / steps
    acceptance_prob_times = acceptance_count_times / steps
    acceptance_prob_mutations = acceptance_count_mutations / steps

    print(f"Final Acceptance Probabilities: SPR = {acceptance_prob_spr:.2f}, "
          f"Times = {acceptance_prob_times:.2f}, "
          f"Mutations = {acceptance_prob_mutations:.2f}")

    return [acceptance_prob_spr, acceptance_prob_times, acceptance_prob_mutations]
