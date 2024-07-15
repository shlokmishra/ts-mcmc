import math
import numpy
import random

from recorder import Recorder
from tree import Tree


def kingman_mcmc(tree, recorder):
    acceptance_count_spr = 0
    acceptance_count_times = 0
    acceptance_count_mutations = 0
    log_likelihood = float(tree.log_likelihood())
    steps = int(recorder.tables.sequence_length) - 1

    for i in range(steps):
        child = tree.sample_leaf()
        parent = tree.parent[child]
        sib = tree.sibling(child)
        new_sib = sib
        if tree.sample_size > 2:
            while new_sib in [child, parent, sib]:
                new_sib = tree.sample_node()
        new_time = tree.sample_reattach_time(child, new_sib)
        alpha = float(-tree.log_reattach_density(child, new_sib, new_time))
        old_time = tree.time[parent]
        tree.detach_reattach(child, new_sib, new_time)
        alpha += float(tree.log_reattach_density(child, sib, old_time))
        proposal_log_likelihood = float(tree.log_likelihood())
        alpha += proposal_log_likelihood - log_likelihood
        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            acceptance_count_spr += 1
        else:
            tree.detach_reattach(child, sib, old_time)

        old_times = tree.time.copy()
        alpha = float(-tree.resample_times())
        proposal_log_likelihood = float(tree.log_likelihood())
        alpha += float(tree.log_resample_times_density(old_times))
        alpha += proposal_log_likelihood - log_likelihood
        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            acceptance_count_times += 1
        else:
            tree.time = old_times.copy()

        # Mutation rate resampling step
        old_mutation_rates = tree.resample_mutation_rates()
        alpha = -tree.log_resample_mutation_rates_density(old_mutation_rates)
        proposal_log_likelihood = float(tree.compute_log_likelihood())
        alpha += proposal_log_likelihood - log_likelihood
        if math.log(random.random()) < alpha:
            log_likelihood = proposal_log_likelihood
            acceptance_count_mutations += 1
        else:
            tree._mutation_rates = old_mutation_rates.copy()

        recorder.append_tree(tree)

    acceptance_prob_spr = acceptance_count_spr / steps
    acceptance_prob_times = acceptance_count_times / steps
    acceptance_prob_mutations = acceptance_count_mutations / steps

    return [acceptance_prob_spr, acceptance_prob_times, acceptance_prob_mutations]
