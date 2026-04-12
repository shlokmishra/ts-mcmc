import math
import numpy
import random
import time
from collections import deque
from scipy.stats import gamma

from recorder import Recorder
from tree import Tree


def kingman_mcmc(tree, recorder, pi, steps=None, step_size=0.1,
                 record=True, compute_gradients=True, print_every=10,
                 mutation_step_size=None, time_move="global", time_step_size=1.0,
                 spr_local_k=None, spr_moves_per_step=1,
                 spr_proposal="local_spr", spr_debug=False,
                 iteration_logger=None, acceptance_window=100):
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
    - spr_proposal: {"spr", "local_spr"}
        Select the legacy global SPR proposal or the canonical local SPR
        neighborhood proposal. `local_spr` is now the default experiment path.
    - spr_debug: bool
        If True, store and print structured SPR proposal metadata.
    - iteration_logger: callable or None
        Optional callback invoked once per outer iteration with a dictionary of
        diagnostic fields for the topology proposal and current chain state.
    - acceptance_window: int
        Window size for rolling topology acceptance diagnostics.
    
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
    current_log_tree_prior = tree.log_likelihood()
    run_start = time.perf_counter()
    rolling_spr_accepts = deque(maxlen=max(1, acceptance_window))

    # Determine the number of MCMC steps
    if steps is None:
        steps = int(recorder.tables.sequence_length)

    for i in range(steps):
        iteration_topology = None
        # -------------------
        # 1. SPR Move(s)
        # -------------------
        for spr_attempt in range(spr_moves_per_step):
            if spr_proposal == "spr":
                proposal = tree.propose_global_spr(local_k=spr_local_k, debug=spr_debug)
                child = proposal["child"]
                parent = proposal["old_parent"]
                sib = proposal["old_sibling"]
                new_sib = proposal["new_sibling"]
                new_time = proposal["new_time"]

                alpha = -tree.log_reattach_density(child, new_sib, new_time)
                old_time = tree.time[parent]
                tree.detach_reattach(child, new_sib, new_time)
                alpha += tree.log_reattach_density(child, sib, old_time)
            elif spr_proposal == "local_spr":
                proposal = tree.propose_local_spr(debug=spr_debug)
                child = proposal["child"]
                parent = proposal["old_parent"]
                sib = proposal["old_sibling"]
                new_sib = proposal["new_sibling"]
                new_time = proposal["new_time"]
                old_time = tree.time[parent]
                tree.detach_reattach(child, new_sib, new_time)
                alpha = proposal["log_hastings"]
            else:
                raise ValueError(f"Unknown spr_proposal: {spr_proposal}")

            proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)
            alpha += proposal_log_likelihood - log_likelihood
            accepted_spr = math.log(random.random()) < alpha

            if accepted_spr:
                log_likelihood = proposal_log_likelihood
                current_log_tree_prior = tree.log_likelihood()
                acceptance_count_spr += 1
            else:
                tree.detach_reattach(child, sib, old_time)

            rolling_spr_accepts.append(1 if accepted_spr else 0)
            iteration_topology = {
                "proposal_type": spr_proposal,
                "spr_attempt": spr_attempt,
                "accepted": bool(accepted_spr),
                "log_alpha": float(alpha),
                "detached_leaf": int(child),
                "old_parent": int(parent),
                "old_sibling": int(sib),
                "chosen_branch": int(new_sib),
                "log_hastings": proposal.get("log_hastings"),
                "log_q_forward": proposal.get("log_q_forward"),
                "log_q_reverse": proposal.get("log_q_reverse"),
                "forward_candidate_count": (
                    None if proposal["debug"].get("forward_candidate_count") is None
                    else int(proposal["debug"]["forward_candidate_count"])
                ),
                "reverse_candidate_count": (
                    None if proposal["debug"].get("reverse_candidate_count") is None
                    else int(proposal["debug"]["reverse_candidate_count"])
                ),
                "forward_candidates": proposal["debug"].get("forward_candidates"),
                "reverse_candidates": proposal["debug"].get("reverse_candidates"),
                "sampled_time": float(new_time),
            }

            if spr_debug:
                print(f"SPR debug: {proposal['debug']}")

        # -------------------
        # 2. Resample Node Times
        # -------------------
        old_times = tree.time.copy()
        log_time_hastings = 0.0
        log_time_prior_delta = 0.0
        accepted_time_move = False
        time_move_debug = None
        if time_move == "global":
            tree.resample_times()
        elif time_move == "local":
            _, _, log_time_hastings = tree.propose_local_time(step_size=time_step_size)
            time_move_debug = dict(tree.last_time_move_debug) if tree.last_time_move_debug is not None else None
        else:
            raise ValueError(f"Unknown time_move: {time_move}")
        proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)
        proposal_log_tree_prior = tree.log_likelihood()
        log_time_prior_delta = proposal_log_tree_prior - current_log_tree_prior

        alpha = proposal_log_likelihood - log_likelihood + log_time_hastings + log_time_prior_delta

        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            current_log_tree_prior = proposal_log_tree_prior
            acceptance_count_times += 1
            accepted_time_move = True
        else:
            tree.time = old_times.copy()

        # -------------------
        # 3. Mutation Rate Resampling
        # -------------------
        old_mutation_rate, log_proposal_density, log_reverse_proposal_density = tree.resample_mutation_rate(step_size=mutation_step_size)
        accepted_mutation_move = False

        proposal_log_prior = log_prior_mutation_rate(tree._mutation_rate)

        proposal_log_likelihood = tree.compute_log_likelihood(tree._mutation_rate, pi)

        alpha = (proposal_log_likelihood + proposal_log_prior + log_reverse_proposal_density) - (log_likelihood + current_log_prior + log_proposal_density)

        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            current_log_prior = proposal_log_prior
            acceptance_count_mutations += 1
            accepted_mutation_move = True
        else:
            tree._mutation_rate = old_mutation_rate

        current_log_target = log_likelihood + current_log_prior + current_log_tree_prior
        if iteration_logger is not None and iteration_topology is not None:
            rolling_acceptance = sum(rolling_spr_accepts) / len(rolling_spr_accepts)
            cumulative_acceptance = acceptance_count_spr / ((i + 1) * spr_moves_per_step)
            iteration_logger(
                {
                    "iteration": int(i),
                    "proposal_type": iteration_topology["proposal_type"],
                    "accepted": iteration_topology["accepted"],
                    "log_likelihood": float(log_likelihood),
                    "log_target": float(current_log_target),
                    "log_alpha": iteration_topology["log_alpha"],
                    "log_hastings": iteration_topology["log_hastings"],
                    "log_q_forward": iteration_topology["log_q_forward"],
                    "log_q_reverse": iteration_topology["log_q_reverse"],
                    "mutation_rate": float(tree._mutation_rate),
                    "root_time": float(tree.time[tree.root]),
                    "cumulative_acceptance_rate": float(cumulative_acceptance),
                    "rolling_acceptance_rate": float(rolling_acceptance),
                    "elapsed_s": float(time.perf_counter() - run_start),
                    "detached_leaf": iteration_topology["detached_leaf"],
                    "chosen_branch": iteration_topology["chosen_branch"],
                    "forward_candidate_count": iteration_topology["forward_candidate_count"],
                    "reverse_candidate_count": iteration_topology["reverse_candidate_count"],
                    "sampled_time": iteration_topology["sampled_time"],
                    "forward_candidates_json": iteration_topology["forward_candidates"],
                    "reverse_candidates_json": iteration_topology["reverse_candidates"],
                    "time_move_accepted": bool(accepted_time_move),
                    "time_move_node": None if time_move_debug is None else int(time_move_debug["node"]),
                    "time_move_was_root": None if time_move_debug is None else bool(time_move_debug["is_root"]),
                    "time_move_old_time": None if time_move_debug is None else float(time_move_debug["old_time"]),
                    "time_move_new_time": None if time_move_debug is None else float(time_move_debug["new_time"]),
                    "time_move_delta": None if time_move_debug is None else float(time_move_debug["delta"]),
                    "time_move_log_hastings": None if time_move_debug is None else float(time_move_debug["log_hastings"]),
                    "time_move_log_prior_delta": float(log_time_prior_delta),
                    "time_move_log_alpha": float(alpha),
                    "mutation_move_accepted": bool(accepted_mutation_move),
                }
            )

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
