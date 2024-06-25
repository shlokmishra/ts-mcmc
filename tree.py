import math
import numpy
import random
import scipy.special


class Tree:
    def __init__(self, sample_size, n_states, sequences=None):
        self.left_child = numpy.full(2 * sample_size - 1, -1)
        self.right_child = numpy.full(2 * sample_size - 1, -1)
        self.parent = numpy.full(2 * sample_size - 1, -1)
        self.time = numpy.zeros(2 * sample_size - 1)
        self.sample_size = sample_size
        self.n_states = n_states
        self.sequences = sequences if sequences is not None else []

        # Initialize the tree structure and times
        t = 0
        active_lineages = list(range(sample_size))
        n = len(active_lineages)
        next_parent = sample_size
        while n > 1:
            t += random.expovariate(lambd=scipy.special.binom(n, 2))
            [l_child, r_child] = numpy.random.choice(active_lineages, size=2, replace=False)
            self.parent[l_child] = next_parent
            self.parent[r_child] = next_parent
            self.left_child[next_parent] = l_child
            self.right_child[next_parent] = r_child
            self.time[next_parent] = t
            active_lineages.remove(l_child)
            active_lineages.remove(r_child)
            active_lineages.append(next_parent)
            next_parent += 1
            n -= 1


    def sample_reattach_time(self, child, new_sib):
        sib = self.sibling(child)
        new_sib_parent = self.parent[new_sib]
        new_sib_grandparent = self.grandparent(new_sib)
        if sib == new_sib:
            if new_sib_grandparent == -1:
                ret = self.time[new_sib] + random.expovariate(lambd=1)
            else:
                ret = numpy.random.uniform(
                    self.time[new_sib], self.time[new_sib_grandparent]
                )
        else:
            if new_sib_parent == -1:
                ret = self.time[new_sib] + random.expovariate(lambd=1)
            else:
                ret = numpy.random.uniform(
                    self.time[new_sib], self.time[new_sib_parent]
                )
        return ret

    def log_reattach_density(self, child, new_sib, new_time):
        sib = self.sibling(child)
        new_sib_parent = self.parent[new_sib]
        new_sib_grandparent = self.grandparent(new_sib)
        if sib == new_sib:
            if new_sib_grandparent == -1:
                ret = self.time[new_sib] - new_time
            else:
                ret = -math.log(self.time[new_sib_grandparent] - self.time[new_sib])
        else:
            if new_sib_parent == -1:
                ret = self.time[new_sib] - new_time
            else:
                ret = -math.log(self.time[new_sib_parent] - self.time[new_sib])
        return ret

    def resample_times(self):
        ret = 0
        lb = 0
        [sorted_times, ind] = numpy.unique(self.time, return_index=True)
        for i in range(1, len(sorted_times)):
            rate = scipy.special.binom(self.sample_size - i + 1, 2)
            index = ind[i]
            self.time[index] = lb + random.expovariate(lambd=rate)
            ret -= rate * (self.time[index] - lb)
            lb = self.time[index]
        return ret

    def log_resample_times_density(self, t):
        ret = 0
        lb = 0
        [sorted_times, ind] = numpy.unique(t, return_index=True)
        for i in range(1, len(sorted_times)):
            rate = scipy.special.binom(self.sample_size - i + 1, 2)
            index = ind[i]
            ret -= rate * (t[index] - lb)
            lb = t[index]
        return ret

    def sample_leaf(self):
        node = numpy.random.choice(numpy.arange(self.sample_size))
        return node

    def sample_node(self):
        node = numpy.random.choice(len(self.parent))
        return node

    def replace_child(self, node, child, new_child):
        if self.left_child[node] == child:
            self.left_child[node] = new_child
        else:
            self.right_child[node] = new_child

    def detach_reattach(self, child, new_sib, new_time):
        sib = self.sibling(child)
        child_parent = self.parent[child]
        sib_grandparent = self.grandparent(child)
        new_sib_parent = self.parent[new_sib]
        self.time[child_parent] = new_time
        if sib != new_sib:
            self.parent[new_sib] = child_parent
            self.parent[sib] = sib_grandparent
            self.parent[child_parent] = new_sib_parent
            self.replace_child(child_parent, sib, new_sib)
            if sib_grandparent != -1:
                self.replace_child(sib_grandparent, child_parent, sib)
            if new_sib_parent != -1:
                self.replace_child(new_sib_parent, new_sib, child_parent)

    def sibling(self, node):
        ret = -1
        p = self.parent[node]
        if p != -1:
            ret = self.left_child[p]
            if ret == node:
                ret = self.right_child[p]
        return ret

    def grandparent(self, node):
        ret = -1
        if self.parent[node] != -1:
            ret = self.parent[self.parent[node]]
        return ret

    def log_likelihood(self):
        sorted_times = numpy.unique(self.time)
        ret = 0
        for i in range(self.sample_size - 1):
            ret -= scipy.special.binom(self.sample_size - i, 2) * (
                sorted_times[i + 1] - sorted_times[i]
            )
        return ret


    def transition_probability(self, t, mutation_rate, pi):
        exp_term = numpy.exp(-mutation_rate * t)
        p_matrix = exp_term * numpy.eye(self.n_states) + (1 - exp_term) * pi
        return p_matrix

    def compute_site_log_likelihood(self, site, mutation_rate, pi):
        n_states = self.n_states
        log_likelihoods = numpy.full((len(self.parent), n_states), -numpy.inf)

        # Initialize leaf log-likelihoods based on observed data
        for leaf in range(self.sample_size):
            observed_state = self.sequences[leaf][site]
            log_likelihoods[leaf][observed_state] = 0.0  # log(1) = 0

        # Order nodes by their times
        nodes_order = numpy.argsort(self.time)

        # Traverse the tree in the order of increasing time
        for node in nodes_order:
            if node < self.sample_size:  # Skip leaf nodes
                continue

            l_child = self.left_child[node]
            r_child = self.right_child[node]
            t_l = self.time[node] - self.time[l_child]
            t_r = self.time[node] - self.time[r_child]

            p_l = self.transition_probability(t_l, mutation_rate, pi)
            p_r = self.transition_probability(t_r, mutation_rate, pi)

            for s in range(n_states):
                log_likelihood_l_child = numpy.logaddexp.reduce([log_likelihoods[l_child][sl] + numpy.log(p_l[s][sl]) for sl in range(n_states)])
                log_likelihood_r_child = numpy.logaddexp.reduce([log_likelihoods[r_child][sr] + numpy.log(p_r[s][sr]) for sr in range(n_states)])
                log_likelihoods[node][s] = log_likelihood_l_child + log_likelihood_r_child

        root = nodes_order[-1]
        # Sum over all possible states at the root
        root_log_likelihood = numpy.logaddexp.reduce([numpy.log(pi[s]) + log_likelihoods[root][s] for s in range(n_states)])
        return root_log_likelihood

    def compute_log_likelihood(self, mutation_rate, pi):
        n_sites = len(self.sequences[0])
        log_likelihood = 0.0

        for site in range(n_sites):
            site_log_likelihood = self.compute_site_log_likelihood(site, mutation_rate, pi)
            if site_log_likelihood == -numpy.inf:
                return -numpy.inf  # If any site log-likelihood is -inf, the overall log-likelihood is -inf
            log_likelihood += site_log_likelihood

        return log_likelihood


def generate_root_sequence(seq_length, num_states):
        return numpy.random.choice(range(num_states), size=seq_length)

def mutate_sequence(sequence, time, mutation_rate, num_states):
    seq_length = len(sequence)
    mutated_sequence = sequence.copy()
    num_mutations = numpy.random.poisson(mutation_rate * time * seq_length)
    mutation_sites = numpy.random.choice(seq_length, num_mutations, replace=False)
    for site in mutation_sites:
        mutated_sequence[site] = numpy.random.choice([state for state in range(num_states) if state != mutated_sequence[site]])
    return mutated_sequence

def simulate_sequences(tree, root_sequence, mutation_rate, num_states):
    sequences = numpy.full((2 * tree.sample_size - 1, len(root_sequence)), -1)
    sequences[-1] = root_sequence

    for node in range(2 * tree.sample_size - 2, -1, -1):
        if tree.left_child[node] != -1 and tree.right_child[node] != -1:
            parent_seq = sequences[node]
            left_child = tree.left_child[node]
            right_child = tree.right_child[node]
            t_left = max(0, tree.time[left_child] - tree.time[node])
            t_right = max(0, tree.time[right_child] - tree.time[node])
            left_seq = mutate_sequence(parent_seq, t_left, mutation_rate, num_states)
            right_seq = mutate_sequence(parent_seq, t_right, mutation_rate, num_states)
            sequences[left_child] = left_seq
            sequences[right_child] = right_seq

    return sequences[:tree.sample_size]


def coalescence_tree_with_sequences(sample_size, num_states, seq_length, mutation_rate, root_sequence=None):
    if root_sequence is None:
        root_sequence = generate_root_sequence(seq_length, num_states)
    
    tree = Tree(sample_size, num_states)
    sequences = simulate_sequences(tree, root_sequence, mutation_rate, num_states)
    return tree, sequences
