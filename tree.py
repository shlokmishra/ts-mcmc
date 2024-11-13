import math
import numpy
import random
import scipy.special


class Tree:
    def __init__(self, sample_size, n_states,sequences=None):
        self.left_child = numpy.full(2 * sample_size - 1, -1)
        self.right_child = numpy.full(2 * sample_size - 1, -1)
        self.parent = numpy.full(2 * sample_size - 1, -1)
        self.time = numpy.zeros(2 * sample_size - 1)
        self.sample_size = sample_size
        self.n_states = n_states
        self.sequences = sequences if sequences is not None else []
        self._mutation_rate = 1.0  # Initialized to 1
        # self.likelihood = numpy.random.rand(max_nodes, n_states)  


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
        sorted_times, ind = numpy.unique(self.time, return_index=True)
        for i in range(1, len(sorted_times)):
            rate = scipy.special.binom(self.sample_size - i + 1, 2)
            index = ind[i]
            new_time = lb + random.expovariate(lambd=rate)
            self.time[index] = new_time
            ret += numpy.log(rate) - rate * (new_time - lb)  # Include log(rate)
            lb = new_time
        return ret


    def log_resample_times_density(self, t):
        ret = 0
        lb = 0
        sorted_times, ind = numpy.unique(t, return_index=True)
        for i in range(1, len(sorted_times)):
            rate = scipy.special.binom(self.sample_size - i + 1, 2)
            index = ind[i]
            ret += numpy.log(rate) - rate * (t[index] - lb)  # Include log(rate)
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

    def transition_probability(self, t, mutation_rate):
        """
        Compute the transition probability matrix P(t) for all state pairs
        at time t using the Jukes-Cantor model generalized for n_states.
        """
        n_states = self.n_states
        alpha = mutation_rate
        
        exp_factor = numpy.exp(-n_states * alpha * t)
        
        P_same = 1.0 / n_states + (1.0 - 1.0 / n_states) * exp_factor
        P_diff = 1.0 / n_states - (1.0 / n_states) * exp_factor
        
        P = numpy.full((n_states, n_states), P_diff)
        numpy.fill_diagonal(P, P_same)
        return P

    def transition_probability_derivative(self, t, mutation_rate):
        """
        Compute the derivative of the transition probability matrix P'(t)
        with respect to time t for all state pairs using the Jukes-Cantor model generalized for n_states.
        """
        n_states = self.n_states
        alpha = mutation_rate
        
        exp_factor = numpy.exp(-n_states * alpha * t)
        factor = -n_states * alpha * exp_factor
        
        dP_same = factor * (1.0 - 1.0 / n_states)
        dP_diff = factor * (-1.0 / n_states)
        
        dP_dt = numpy.full((n_states, n_states), dP_diff)
        numpy.fill_diagonal(dP_dt, dP_same)
        return dP_dt

    def log_transition_probability(self, t, mutation_rate, s1, s2):
        n_states = self.n_states
        alpha = mutation_rate
        exp_factor = numpy.exp(-n_states * alpha * t)

        if s1 == s2:
            prob = 1.0 / n_states + (1.0 - 1.0 / n_states) * exp_factor
        else:
            prob = 1.0 / n_states - (1.0 / n_states) * exp_factor

        # Ensure probability is positive to avoid log of zero
        prob = max(prob, 1e-300)

        return numpy.log(prob)


    def log_sum_exp(self, log_values):
        """
        Compute log-sum-exp in a numerically stable way.
        """
        max_log = numpy.max(log_values)
        return max_log + numpy.log(numpy.sum(numpy.exp(log_values - max_log)))

    def compute_site_log_likelihood(self, site, mutation_rate, pi):
        """
        Compute the log-likelihood for a single site (position) in the sequences.
        """
        L = numpy.full((2 * self.sample_size - 1, self.n_states), -numpy.inf)  # Log-likelihood matrix initialized to -inf

        # Initialize the log-likelihoods at the leaves
        for i in range(self.sample_size):
            observed_state = self.sequences[i][site]
            for s in range(self.n_states):
                L[i][s] = 0.0 if s == observed_state else -numpy.inf  # Set 0 for the observed state (log(1)), -inf otherwise

        # Debug statement
        # print(f"Initialized log-likelihoods at leaves for site {site}: {L[:self.sample_size]}")

        # Propagate log-likelihoods up the tree
        for p in range(self.sample_size, 2 * self.sample_size - 1):  # Internal nodes
            left = self.left_child[p]
            right = self.right_child[p]

            # Initialize L[p][s] to 0.0 (log(1)) for all states s
            for s in range(self.n_states):
                L[p][s] = 0.0

            # Process left child
            if left != -1:
                t_left = self.time[p] - self.time[left]
                for s in range(self.n_states):
                    log_probs_left = [
                        self.log_transition_probability(t_left, mutation_rate, s, sl) + L[left][sl]
                        for sl in range(self.n_states)
                    ]
                    L[p][s] += self.log_sum_exp(log_probs_left)
            
                # Debug statement
                # print(f"Log-likelihood at node {p}, after processing left child {left}: {L[p]}")

            # Process right child
            if right != -1:
                t_right = self.time[p] - self.time[right]
                for s in range(self.n_states):
                    log_probs_right = [
                        self.log_transition_probability(t_right, mutation_rate, s, sr) + L[right][sr]
                        for sr in range(self.n_states)
                    ]
                    L[p][s] += self.log_sum_exp(log_probs_right)

                # Debug statement
                # print(f"Log-likelihood at node {p}, after processing right child {right}: {L[p]}")

        # Likelihood at the root
        root = 2 * self.sample_size - 2  # Root node index
        root_log_probs = [
            numpy.log(pi[s]) + L[root][s] for s in range(self.n_states)
        ]  # Sum over all root states with log(pi[s])

        root_log_likelihood = self.log_sum_exp(root_log_probs)

        # Debug statement
        # print(f"Root log-likelihood: {root_log_likelihood}")

        return root_log_likelihood

    def compute_log_likelihood(self, mutation_rate, pi):
        """
        Compute the total log-likelihood across all sites.
        """
        total_log_likelihood = 0.0
        for site in range(len(self.sequences[0])):  # Iterate over each site in the sequences
            site_log_likelihood = self.compute_site_log_likelihood(site, mutation_rate, pi)
            total_log_likelihood += site_log_likelihood
            
            # Debug statement
            # print(f"Log-likelihood for site {site}: {site_log_likelihood}")

        # Debug statement
        # print(f"Total log-likelihood across all sites: {total_log_likelihood}")
        
        return total_log_likelihood

    def compute_gradient(self, mutation_rate, pi):
        n_sites = len(self.sequences[0])
        n_states = self.n_states
        gradients = numpy.zeros(2 * self.sample_size - 1)

        for site in range(n_sites):
            # Initialize likelihoods and conditionals
            L = numpy.ones((len(self.parent), n_states))  # Likelihoods at each node
            C = numpy.zeros((len(self.parent), n_states))  # Conditionals at each node

            # Initialize leaf likelihoods
            for leaf in range(self.sample_size):
                observed_state = self.sequences[leaf][site]
                for s in range(n_states):
                    L[leaf][s] = 1.0 if s == observed_state else 0.0

            # Post-order traversal to compute likelihoods
            nodes_order = numpy.argsort(self.time)
            for node in nodes_order:
                if node < self.sample_size:
                    continue  # Skip leaves
                l_child = self.left_child[node]
                r_child = self.right_child[node]
                t_l = self.time[node] - self.time[l_child]
                t_r = self.time[node] - self.time[r_child]

                P_l = self.transition_probability(t_l, mutation_rate)
                P_r = self.transition_probability(t_r, mutation_rate)

                for s in range(n_states):
                    L_l = numpy.dot(P_l[s, :], L[l_child, :])
                    L_r = numpy.dot(P_r[s, :], L[r_child, :])
                    L[node][s] = L_l * L_r

            # Compute root likelihood
            root = nodes_order[-1]
            L_root = numpy.dot(pi, L[root, :])

            # Correct initialization of conditionals at root
            C[root, :] = (pi * L[root, :]) / L_root

            reversed_order = nodes_order[::-1]  # From root to leaves
            for node in reversed_order:
                if node < self.sample_size:
                    continue  # Skip leaves
                l_child = self.left_child[node]
                r_child = self.right_child[node]
                t_l = self.time[node] - self.time[l_child]
                t_r = self.time[node] - self.time[r_child]

                P_l = self.transition_probability(t_l, mutation_rate)
                P_r = self.transition_probability(t_r, mutation_rate)

                # Update conditionals for left child
                C_l_child = numpy.dot(C[node, :], P_l)
                C[l_child, :] += C_l_child

                # Update conditionals for right child
                C_r_child = numpy.dot(C[node, :], P_r)
                C[r_child, :] += C_r_child

            # Compute gradients
            for node in range(len(self.parent)):
                if node < self.sample_size:
                    continue  # Skip leaves
                parent = self.parent[node]
                if parent == -1:
                    continue  # Root node
                t = self.time[parent] - self.time[node]
                P = self.transition_probability(t, mutation_rate)
                dP_dt = self.transition_probability_derivative(t, mutation_rate)

                grad = 0.0
                for s_i in range(n_states):
                    for s_j in range(n_states):
                        grad += C[parent, s_i] * dP_dt[s_i, s_j] * L[node, s_j]

                gradients[node] += grad / L_root

        return gradients




    @property
    def mutation_rate(self):
        return self._mutation_rate

    def resample_mutation_rate(self, step_size=0.1):
        """
        Resample the mutation rate using a reflective Gaussian proposal.
        
        Parameters:
        - step_size: float
            The standard deviation of the Gaussian proposal.
        
        Returns:
        - old_mutation_rate: float
            The previous mutation rate before resampling.
        """
        old_mutation_rate = self._mutation_rate
        new_rate = old_mutation_rate + step_size * numpy.random.randn()
        
        # Reflective Gaussian: mirror the negative rates back into positive domain
        if new_rate < 0:
            new_rate = -new_rate
        
        self._mutation_rate = new_rate
        return old_mutation_rate
    

    

    def log_resample_mutation_rate_density(self, old_mutation_rate):
        """
        Computes the log proposal density ratio for mutation rate resampling.
        Since the proposal is symmetric, this ratio is zero.
        
        Parameters:
        - old_mutation_rate: float
            The previous mutation rate before resampling.
        
        Returns:
        - log_density_ratio: float
            The log of the proposal density ratio. Zero for symmetric proposals.
        """
        return 0.0  # Symmetric proposal, so log(q(theta | theta') / q(theta' | theta)) = 0


    @mutation_rate.setter
    def mutation_rate(self, value):
        self._mutation_rate = value





def generate_root_sequence(seq_length, num_states):
        return numpy.random.choice(range(num_states), size=seq_length)

def mutate_sequence(sequence, time, mutation_rate, num_states):
    seq_length = len(sequence)
    mutated_sequence = sequence.copy()
    num_mutations = numpy.random.poisson(mutation_rate * time * seq_length)
    mutation_sites = numpy.random.choice(seq_length, num_mutations, replace=True)
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
