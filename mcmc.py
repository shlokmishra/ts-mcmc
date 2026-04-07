import math
import numpy
import random
from scipy.stats import gamma

from recorder import Recorder
from tree import Tree


def kingman_mcmc(tree, recorder, pi, steps=None, step_size=0.1,
                 record=True, compute_gradients=True, print_every=10,
                 mutation_step_size=None, time_move="global", time_step_size=1.0,
                 spr_local_k=None, spr_moves_per_step=1):
    """
    Performs Markov Chain Monte Carlo (MCMC) sampling on a phylogenetic tree
    using Kingman's coalescent.
    
    Parameters:
    - tree: Instance of the Tree class representing the current phylogenetic tree.
    - recorder: Instance of the Recorder class for storing sampled trees and related information.
    - pi: List or array representing the base frequency distribution.
    - steps: Optional integer specifying the number of MCMC iterations.
    - step_size: float
        Backwards-compatible alias for mutation-rate proposal scale.
    - record: bool
        Whether to record trees into the recorder (set False for benchmarking).
    - compute_gradients: bool
        Whether to compute gradients (needed for Stein thinning, skip for speed).
    - print_every: int or None
        Print progress every N steps. Set to None to suppress output.
    - mutation_step_size: float or None
        Log-normal mutation-rate proposal scale. Defaults to `step_size`.
    - time_move: {"global", "local"}
        Global resampling of all coalescent times or a local one-node move.
    - time_step_size: float
        Proposal scale for the local time move.
    - spr_local_k: int or None
        If set, choose the reattachment target uniformly from the `k` eligible
        nodes closest in time to the current parent.
    - spr_moves_per_step: int
        Number of SPR proposals attempted per outer MCMC iteration.
    
    Returns:
    - List containing acceptance probabilities for SPR moves, resampling times,
      and mutation rate resampling.
    """
    if mutation_step_size is None:
        mutation_step_size = step_size

    # Initialize acceptance counters
    acceptance_count_spr = 0
    acceptance_count_times = 0
    acceptance_count_mutations = 0

    # Initialize log-likelihood and prior
    log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)

    # Define the prior over the mutation rate
    def log_prior_mutation_rate(rate):
        a = 2.0  # Shape parameter
        b = 1.0  # Scale parameter
        if rate <= 0:
            return -math.inf
        return gamma.logpdf(rate, a=a, scale=b)

    current_log_prior = log_prior_mutation_rate(tree._mutation_rate)

    # Determine the number of MCMC steps
    if steps is None:
        steps = int(recorder.tables.sequence_length)

    for i in range(steps):
        # -------------------
        # 1. SPR Move(s)
        # -------------------
        for _ in range(spr_moves_per_step):
            child = tree.sample_leaf()
            parent = tree.parent[child]
            sib = tree.sibling(child)
            new_sib = sib

            if tree.sample_size > 2:
                attempts = 0
                max_attempts = 100
                while new_sib in [child, parent, sib] and attempts < max_attempts:
                    if spr_local_k is None:
                        new_sib = tree.sample_node()
                    else:
                        new_sib = tree.sample_reattach_target(child, local_k=spr_local_k)
                    attempts += 1
                if attempts == max_attempts:
                    raise RuntimeError("Failed to sample a valid new sibling node.")

            new_time = tree.sample_reattach_time(child, new_sib)

            alpha = -tree.log_reattach_density(child, new_sib, new_time)
            old_time = tree.time[parent]
            tree.detach_reattach(child, new_sib, new_time)
            alpha += tree.log_reattach_density(child, sib, old_time)
            proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)
            alpha += proposal_log_likelihood - log_likelihood

            if math.log(random.random()) < alpha:
                log_likelihood = proposal_log_likelihood
                acceptance_count_spr += 1
            else:
                tree.detach_reattach(child, sib, old_time)

        # -------------------
        # 2. Resample Node Times
        # -------------------
        old_times = tree.time.copy()
        log_time_hastings = 0.0
        if time_move == "global":
            tree.resample_times()
        elif time_move == "local":
            _, _, log_time_hastings = tree.propose_local_time(step_size=time_step_size)
        else:
            raise ValueError(f"Unknown time_move: {time_move}")
        proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)

        alpha = proposal_log_likelihood - log_likelihood + log_time_hastings

        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            acceptance_count_times += 1
        else:
            tree.time = old_times.copy()

        # -------------------
        # 3. Mutation Rate Resampling
        # -------------------
        old_mutation_rate, log_proposal_density, log_reverse_proposal_density = tree.resample_mutation_rate(step_size=mutation_step_size)

        proposal_log_prior = log_prior_mutation_rate(tree._mutation_rate)

        proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)

        alpha = (proposal_log_likelihood + proposal_log_prior + log_reverse_proposal_density) - (log_likelihood + current_log_prior + log_proposal_density)

        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            current_log_prior = proposal_log_prior
            acceptance_count_mutations += 1
        else:
            tree._mutation_rate = old_mutation_rate

        # -------------------
        # 4. Record the Current State
        # -------------------
        if record:
            gradient = tree.compute_gradient(tree._mutation_rate, pi) if compute_gradients else numpy.zeros(1)
            recorder.append_tree(tree, tree._mutation_rate, log_likelihood, gradient)

        # -------------------
        # 5. Progress Monitoring
        # -------------------
        if print_every and ((i + 1) % print_every == 0 or i == 0):
            print(f"Step {i+1}/{steps}: Mutation Rate = {tree._mutation_rate:.4f}, "
                  f"Log Likelihood = {log_likelihood:.4f}, "
                  f"Acceptance Rates - SPR: {acceptance_count_spr/((i+1) * spr_moves_per_step):.2f}, "
                  f"Times: {acceptance_count_times/(i+1):.2f}, "
                  f"Mutations: {acceptance_count_mutations/(i+1):.2f}")

    # Compute final acceptance probabilities
    acceptance_prob_spr = acceptance_count_spr / (steps * spr_moves_per_step)
    acceptance_prob_times = acceptance_count_times / steps
    acceptance_prob_mutations = acceptance_count_mutations / steps

    if print_every:
        print(f"Final Acceptance Probabilities: SPR = {acceptance_prob_spr:.2f}, "
              f"Times = {acceptance_prob_times:.2f}, "
              f"Mutations = {acceptance_prob_mutations:.2f}")

    return [acceptance_prob_spr, acceptance_prob_times, acceptance_prob_mutations]
