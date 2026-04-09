import math
import numpy
import random
import scipy.special

MAX_LOG_FLOAT = math.log(numpy.finfo(float).max)


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
            
        self.root = next_parent - 1
        self.last_spr_debug = None
        self.last_time_move_debug = None
        

    def is_internal(self, node):
        """Checks if a node is an internal node (not a leaf)."""
        return self.left_child[node] != -1

    def reattach_interval(self, child, new_sib):
        sib = self.sibling(child)
        new_sib_parent = self.parent[new_sib]
        new_sib_grandparent = self.grandparent(new_sib)
        if sib == new_sib:
            if new_sib_grandparent == -1:
                return float(self.time[new_sib]), math.inf
            return float(self.time[new_sib]), float(self.time[new_sib_grandparent])
        if new_sib_parent == -1:
            return float(self.time[new_sib]), math.inf
        return float(self.time[new_sib]), float(self.time[new_sib_parent])

    def sample_reattach_time(self, child, new_sib):
        lower, upper = self.reattach_interval(child, new_sib)
        if math.isinf(upper):
            return lower + random.expovariate(lambd=1)
        return float(numpy.random.uniform(lower, upper))

    def log_reattach_density(self, child, new_sib, new_time):
        lower, upper = self.reattach_interval(child, new_sib)
        if math.isinf(upper):
            return lower - new_time
        return -math.log(upper - lower)

    def resample_times(self):
        """
        Resample coalescent times using the correct Kingman coalescent process.
        Returns the log density of the proposal.
        """
        # Get unique times and their indices, sorted
        sorted_times, ind = numpy.unique(self.time, return_index=True)
        
        lb = 0.0  # Lower bound for current interval
        
        # Resample each coalescent event time
        for i in range(1, len(sorted_times)):
            # Number of lineages at this point
            n_lineages = self.sample_size - i + 1
            # Coalescent rate
            rate = scipy.special.binom(n_lineages, 2)
            
            # Sample new time from exponential distribution
            new_time = lb + random.expovariate(lambd=rate)
            
            # Update the time for this coalescent event
            index = ind[i]
            self.time[index] = new_time
            
            # Update lower bound for next interval
            lb = new_time
        
        # Calculate and return the log density of the new times
        return self.log_resample_times_density(self.time)

    def log_resample_times_density(self, t):
        """
        Calculate the log density of the times resampling proposal for given times t.
        This should match the density used in resample_times().
        """
        # Get unique times and their indices, sorted
        sorted_times, ind = numpy.unique(t, return_index=True)
        
        log_density = 0.0
        lb = 0.0  # Lower bound for current interval
        
        # Calculate density for each coalescent event time
        for i in range(1, len(sorted_times)):
            # Number of lineages at this point
            n_lineages = self.sample_size - i + 1
            # Coalescent rate
            rate = scipy.special.binom(n_lineages, 2)
            
            # Get the time for this coalescent event
            index = ind[i]
            event_time = t[index]
            
            # Add log density: log(rate * exp(-rate * (event_time - lb)))
            # = log(rate) - rate * (event_time - lb)
            log_density += numpy.log(rate) - rate * (event_time - lb)
            
            # Update lower bound for next interval
            lb = event_time
            
        return log_density

    def sample_internal_node(self):
        """Sample a single internal node uniformly."""
        return numpy.random.randint(self.sample_size, len(self.parent))

    def propose_local_time(self, step_size=1.0):
        """
        Propose a local update to one internal node time.

        Non-root nodes use a symmetric random walk in logit space within the
        feasible interval `(max(child_times), parent_time)`. The root uses a
        log-normal random walk on its gap above the oldest child.

        Returns
        -------
        node : int
            Updated node index.
        old_time : float
            Previous node time.
        log_reverse_minus_forward : float
            Hastings correction term `log q(old|new) - log q(new|old)`.
        """
        node = self.sample_internal_node()
        old_time = float(self.time[node])
        lower = max(self.time[self.left_child[node]], self.time[self.right_child[node]])
        parent = self.parent[node]

        if parent == -1:
            old_gap = max(old_time - lower, 1e-12)
            log_old_gap = numpy.log(old_gap)
            log_new_gap = log_old_gap + step_size * numpy.random.randn()
            if not numpy.isfinite(log_new_gap) or log_new_gap > MAX_LOG_FLOAT:
                self.last_time_move_debug = {
                    "node": int(node),
                    "old_time": old_time,
                    "new_time": old_time,
                    "delta": 0.0,
                    "is_root": True,
                    "accepted_geometry": False,
                    "proposal_status": "overflow_log_gap",
                    "log_hastings": -math.inf,
                }
                self.time[node] = old_time
                return node, old_time, -math.inf
            new_gap = float(numpy.exp(log_new_gap))
            if not numpy.isfinite(new_gap):
                self.last_time_move_debug = {
                    "node": int(node),
                    "old_time": old_time,
                    "new_time": old_time,
                    "delta": 0.0,
                    "is_root": True,
                    "accepted_geometry": False,
                    "proposal_status": "overflow_gap",
                    "log_hastings": -math.inf,
                }
                self.time[node] = old_time
                return node, old_time, -math.inf
            proposed_time = lower + new_gap
            if not numpy.isfinite(proposed_time):
                self.last_time_move_debug = {
                    "node": int(node),
                    "old_time": old_time,
                    "new_time": proposed_time,
                    "delta": proposed_time - old_time,
                    "is_root": True,
                    "accepted_geometry": False,
                    "proposal_status": "overflow_root_time",
                    "log_hastings": -math.inf,
                }
                self.time[node] = old_time
                return node, old_time, -math.inf
            self.time[node] = proposed_time
            log_hastings = log_new_gap - log_old_gap
            self.last_time_move_debug = {
                "node": int(node),
                "old_time": old_time,
                "new_time": proposed_time,
                "delta": proposed_time - old_time,
                "is_root": True,
                "accepted_geometry": True,
                "proposal_status": "ok",
                "log_hastings": float(log_hastings),
                "lower_bound": float(lower),
                "old_gap": float(old_gap),
                "new_gap": float(new_gap),
            }
            return node, old_time, log_hastings

        upper = float(self.time[parent])
        span = max(upper - lower, 1e-12)
        old_frac = min(max((old_time - lower) / span, 1e-12), 1 - 1e-12)
        logit_old = numpy.log(old_frac / (1.0 - old_frac))
        logit_new = logit_old + step_size * numpy.random.randn()
        new_frac = 1.0 / (1.0 + numpy.exp(-logit_new))
        new_frac = min(max(new_frac, 1e-12), 1 - 1e-12)
        self.time[node] = lower + span * new_frac

        log_hastings = (
            numpy.log(new_frac * (1.0 - new_frac)) - numpy.log(old_frac * (1.0 - old_frac))
        )
        self.last_time_move_debug = {
            "node": int(node),
            "old_time": old_time,
            "new_time": float(self.time[node]),
            "delta": float(self.time[node] - old_time),
            "is_root": False,
            "accepted_geometry": True,
            "proposal_status": "ok",
            "log_hastings": float(log_hastings),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "span": float(span),
        }
        return (
            node,
            old_time,
            log_hastings,
        )


    def sample_leaf(self):
        node = numpy.random.choice(numpy.arange(self.sample_size))
        return node

    def sample_node(self):
        node = numpy.random.choice(len(self.parent))
        return node

    def _validate_leaf_spr_child(self, child):
        if child < 0 or child >= self.sample_size:
            raise ValueError("SPR moves currently detach leaves only.")
        parent = self.parent[child]
        if parent == -1:
            raise ValueError("Cannot detach the root as an SPR child.")
        return int(parent)

    def sample_reattach_target(self, child, local_k=None):
        """
        Sample a reattachment target for a leaf SPR move.

        When `local_k` is provided, choose uniformly among the `k` eligible
        nodes whose times are closest to the current parent time. This makes
        SPR moves less disruptive in larger trees while preserving the original
        global mode as a fallback.
        """
        parent = self.parent[child]
        sib = self.sibling(child)
        excluded = {child, parent, sib}
        candidates = [node for node in range(len(self.parent)) if node not in excluded]
        if not candidates:
            raise RuntimeError("No valid reattachment target available.")

        if local_k is None or local_k <= 0 or local_k >= len(candidates):
            return int(numpy.random.choice(candidates))

        parent_time = self.time[parent]
        candidates.sort(key=lambda node: (abs(self.time[node] - parent_time), numpy.random.random()))
        return int(numpy.random.choice(candidates[:local_k]))

    def _detached_parent_after_leaf_spr(self, child, node):
        """
        Parent of `node` in the detached topology where `child` has been
        removed and its original parent suppressed.
        """
        self._validate_leaf_spr_child(child)
        surviving_sibling = self.sibling(child)
        if node == surviving_sibling:
            return self.grandparent(child)
        return int(self.parent[node])

    def _detached_sibling_after_leaf_spr(self, child, node):
        old_parent = self._validate_leaf_spr_child(child)
        surviving_sibling = self.sibling(child)
        if node == surviving_sibling:
            grandparent = self.grandparent(child)
            if grandparent == -1:
                return -1
            return int(self.sibling(old_parent))

        parent = self._detached_parent_after_leaf_spr(child, node)
        if parent == -1:
            return -1
        sibling = self.left_child[parent]
        if sibling == node:
            sibling = self.right_child[parent]
        return int(sibling)

    def _add_local_spr_candidate(self, candidates_by_node, child, branch_node, source, description):
        if branch_node in (-1, child):
            return
        candidate = candidates_by_node.setdefault(
            int(branch_node),
            {
                "branch_node": int(branch_node),
                "sources": [],
                "description": [],
            },
        )
        candidate["sources"].append(source)
        candidate["description"].append(description)

    def get_local_spr_candidates(self, child):
        """
        Enumerate the local SPR neighborhood around node `A`.

        `A` is defined here as the surviving sibling of the detached leaf
        after suppressing the detached leaf's old parent. Candidates are
        branches in that detached topology.
        """
        self._validate_leaf_spr_child(child)
        a_node = self.sibling(child)
        candidates_by_node = {}

        # Original reattachment branch above A.
        self._add_local_spr_candidate(
            candidates_by_node,
            child,
            a_node,
            "above_a",
            "branch above A, the surviving sibling after detach",
        )

        # Two branches below A when A is internal.
        if self.is_internal(a_node):
            self._add_local_spr_candidate(
                candidates_by_node,
                child,
                self.left_child[a_node],
                "below_a_left",
                "left branch below A",
            )
            self._add_local_spr_candidate(
                candidates_by_node,
                child,
                self.right_child[a_node],
                "below_a_right",
                "right branch below A",
            )

        # Branch above the detached-tree sibling of A.
        self._add_local_spr_candidate(
            candidates_by_node,
            child,
            self._detached_sibling_after_leaf_spr(child, a_node),
            "above_sibling_of_a",
            "branch above the detached-tree sibling of A",
        )

        # Branch above the detached-tree parent of A.
        self._add_local_spr_candidate(
            candidates_by_node,
            child,
            self._detached_parent_after_leaf_spr(child, a_node),
            "above_parent_of_a",
            "branch above the detached-tree parent of A",
        )

        candidates = []
        for branch_node, candidate in sorted(candidates_by_node.items()):
            lower, upper = self.reattach_interval(child, branch_node)
            candidates.append(
                {
                    "branch_node": int(branch_node),
                    "sources": candidate["sources"],
                    "description": candidate["description"],
                    "interval_lower": lower,
                    "interval_upper": upper,
                    "parent_after_detach": self._detached_parent_after_leaf_spr(child, branch_node),
                }
            )
        return candidates

    def propose_global_spr(self, local_k=None, child=None, debug=False):
        if child is None:
            child = int(self.sample_leaf())
        parent = self._validate_leaf_spr_child(child)
        a_node = self.sibling(child)
        new_sib = a_node

        if self.sample_size > 2:
            attempts = 0
            max_attempts = 100
            while new_sib in [child, parent, a_node] and attempts < max_attempts:
                if local_k is None:
                    new_sib = int(self.sample_node())
                else:
                    new_sib = int(self.sample_reattach_target(child, local_k=local_k))
                attempts += 1
            if attempts == max_attempts:
                raise RuntimeError("Failed to sample a valid new sibling node.")

        interval_lower, interval_upper = self.reattach_interval(child, new_sib)
        new_time = self.sample_reattach_time(child, new_sib)
        debug_info = {
            "proposal": "spr",
            "child": int(child),
            "old_parent": int(parent),
            "a_node": int(a_node),
            "chosen_branch_node": int(new_sib),
            "interval_lower": interval_lower,
            "interval_upper": interval_upper,
            "sampled_time": float(new_time),
        }
        if debug:
            self.last_spr_debug = debug_info
        return {
            "child": int(child),
            "old_parent": int(parent),
            "old_sibling": int(a_node),
            "new_sibling": int(new_sib),
            "new_time": float(new_time),
            "log_q_forward": None,
            "log_q_reverse": None,
            "log_hastings": None,
            "debug": debug_info,
        }

    def _snapshot_tree_state(self):
        return {
            "left_child": self.left_child.copy(),
            "right_child": self.right_child.copy(),
            "parent": self.parent.copy(),
            "time": self.time.copy(),
            "root": int(self.root),
        }

    def _restore_tree_state(self, state):
        self.left_child = state["left_child"].copy()
        self.right_child = state["right_child"].copy()
        self.parent = state["parent"].copy()
        self.time = state["time"].copy()
        self.root = int(state["root"])

    def _local_spr_log_q(self, child, candidates, branch_node, attach_time):
        if not candidates:
            return -math.inf
        candidate_map = {candidate["branch_node"]: candidate for candidate in candidates}
        chosen = candidate_map.get(int(branch_node))
        if chosen is None:
            return -math.inf
        return (
            -math.log(self.sample_size)
            -math.log(len(candidates))
            + self.log_reattach_density(child, int(branch_node), attach_time)
        )

    def build_local_spr_proposal_metadata(self, child, new_sib, new_time):
        parent = self._validate_leaf_spr_child(child)
        old_sibling = self.sibling(child)
        old_time = float(self.time[parent])
        a_node = old_sibling
        forward_candidates = self.get_local_spr_candidates(child)
        forward_log_q = self._local_spr_log_q(child, forward_candidates, new_sib, new_time)

        state = self._snapshot_tree_state()
        try:
            self.detach_reattach(child, new_sib, new_time)
            reverse_candidates = self.get_local_spr_candidates(child)
            reverse_log_q = self._local_spr_log_q(child, reverse_candidates, old_sibling, old_time)
        finally:
            self._restore_tree_state(state)

        chosen_forward = next(
            candidate for candidate in forward_candidates if candidate["branch_node"] == int(new_sib)
        )
        reverse_chosen = None
        for candidate in reverse_candidates:
            if candidate["branch_node"] == int(old_sibling):
                reverse_chosen = candidate
                break

        return {
            "proposal": "local_spr",
            "child": int(child),
            "old_parent": int(parent),
            "old_sibling": int(old_sibling),
            "old_parent_time": old_time,
            "a_node": int(a_node),
            "forward_candidates": forward_candidates,
            "forward_candidate_count": len(forward_candidates),
            "chosen_candidate": chosen_forward,
            "chosen_branch_node": int(new_sib),
            "forward_interval_lower": chosen_forward["interval_lower"],
            "forward_interval_upper": chosen_forward["interval_upper"],
            "sampled_time": float(new_time),
            "reverse_candidates": reverse_candidates,
            "reverse_candidate_count": len(reverse_candidates),
            "reverse_chosen_candidate": reverse_chosen,
            "reverse_branch_node": int(old_sibling),
            "reverse_interval_lower": None if reverse_chosen is None else reverse_chosen["interval_lower"],
            "reverse_interval_upper": None if reverse_chosen is None else reverse_chosen["interval_upper"],
            "log_q_forward": float(forward_log_q),
            "log_q_reverse": float(reverse_log_q),
            "log_hastings": float(reverse_log_q - forward_log_q),
        }

    def propose_local_spr(self, child=None, branch_node=None, debug=False):
        if child is None:
            child = int(self.sample_leaf())
        parent = self._validate_leaf_spr_child(child)
        a_node = self.sibling(child)
        candidates = self.get_local_spr_candidates(child)
        if not candidates:
            raise RuntimeError("No valid local SPR candidates available.")

        candidate_map = {candidate["branch_node"]: candidate for candidate in candidates}
        if branch_node is None:
            chosen = random.choice(candidates)
        else:
            branch_node = int(branch_node)
            if branch_node not in candidate_map:
                raise ValueError(f"Branch node {branch_node} is not a valid local SPR candidate.")
            chosen = candidate_map[branch_node]

        new_sib = chosen["branch_node"]
        new_time = self.sample_reattach_time(child, new_sib)
        debug_info = self.build_local_spr_proposal_metadata(child, new_sib, new_time)
        if debug:
            self.last_spr_debug = debug_info
        return {
            "child": int(child),
            "old_parent": int(parent),
            "old_sibling": int(a_node),
            "new_sibling": int(new_sib),
            "new_time": float(new_time),
            "log_q_forward": debug_info["log_q_forward"],
            "log_q_reverse": debug_info["log_q_reverse"],
            "log_hastings": debug_info["log_hastings"],
            "debug": debug_info,
        }

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
            new_root = self.root
            if sib_grandparent == -1:
                new_root = sib
            self.parent[new_sib] = child_parent
            self.parent[sib] = sib_grandparent
            self.parent[child_parent] = new_sib_parent
            self.replace_child(child_parent, sib, new_sib)
            if sib_grandparent != -1:
                self.replace_child(sib_grandparent, child_parent, sib)
            if new_sib_parent != -1:
                self.replace_child(new_sib_parent, new_sib, child_parent)
            else:
                new_root = child_parent
            self.root = new_root

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

    def has_valid_times(self, tol=1e-12):
        if not numpy.all(numpy.isfinite(self.time)):
            return False
        if not numpy.isfinite(self._mutation_rate) or self._mutation_rate <= 0:
            return False
        for node in range(self.sample_size, len(self.parent)):
            left = self.left_child[node]
            right = self.right_child[node]
            if left == -1 or right == -1:
                return False
            if self.time[node] + tol < self.time[left]:
                return False
            if self.time[node] + tol < self.time[right]:
                return False
        return True

    def transition_probability(self, t, mutation_rate):
        """
        Compute the transition probability matrix P(t) for all state pairs
        at time t using the Jukes-Cantor model generalized for n_states.
        """
        n_states = self.n_states
        alpha = mutation_rate
        if t < 0 or not numpy.isfinite(t):
            raise ValueError(f"Invalid branch length {t}.")
        if not numpy.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid mutation rate {alpha}.")
        
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
        if t < 0 or not numpy.isfinite(t):
            raise ValueError(f"Invalid branch length {t}.")
        if not numpy.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid mutation rate {alpha}.")
        
        exp_factor = numpy.exp(-n_states * alpha * t)
        factor = -n_states * alpha * exp_factor
        
        dP_same = factor * (1.0 - 1.0 / n_states)
        dP_diff = factor * (-1.0 / n_states)
        
        dP_dt = numpy.full((n_states, n_states), dP_diff)
        numpy.fill_diagonal(dP_dt, dP_same)
        return dP_dt

    def compute_site_log_likelihood(self, site, mutation_rate, pi):
        """
        Compute the log-likelihood for a single site (position) in the sequences.
        Uses matrix-based transition probabilities for efficiency.
        """
        if not self.has_valid_times() or not numpy.isfinite(mutation_rate) or mutation_rate <= 0:
            return -math.inf
        n_states = self.n_states
        L = numpy.full((2 * self.sample_size - 1, n_states), -numpy.inf)

        # Initialize the log-likelihoods at the leaves
        for i in range(self.sample_size):
            observed_state = self.sequences[i][site]
            L[i, observed_state] = 0.0  # log(1) for observed state, -inf otherwise

        # Propagate log-likelihoods up the tree
        for p in range(self.sample_size, 2 * self.sample_size - 1):
            left = self.left_child[p]
            right = self.right_child[p]

            # Initialize L[p] to 0.0 (log(1)) for all states
            L[p, :] = 0.0

            # Process left child using matrix operations
            if left != -1:
                t_left = self.time[p] - self.time[left]
                P_left = self.transition_probability(t_left, mutation_rate)
                # Ensure no zero probabilities for log
                log_P_left = numpy.log(numpy.maximum(P_left, 1e-300))
                # For each parent state s, compute logsumexp over child states
                # log_P_left[s, :] + L[left, :] gives log(P(s->child_state) * L_child)
                for s in range(n_states):
                    L[p, s] += scipy.special.logsumexp(log_P_left[s, :] + L[left, :])

            # Process right child using matrix operations
            if right != -1:
                t_right = self.time[p] - self.time[right]
                P_right = self.transition_probability(t_right, mutation_rate)
                log_P_right = numpy.log(numpy.maximum(P_right, 1e-300))
                for s in range(n_states):
                    L[p, s] += scipy.special.logsumexp(log_P_right[s, :] + L[right, :])

        # Likelihood at the root
        log_pi = numpy.log(pi)
        root_log_likelihood = scipy.special.logsumexp(log_pi + L[self.root, :])

        return root_log_likelihood

    def compute_log_likelihood(self, mutation_rate, pi):
        """
        Compute the total log-likelihood across all sites (vectorized).
        
        Processes all sites simultaneously using matrix operations instead of
        looping site-by-site. Uses probability space with Felsenstein's pruning.
        """
        if not self.has_valid_times() or not numpy.isfinite(mutation_rate) or mutation_rate <= 0:
            return -math.inf
        n_sites = len(self.sequences[0])
        n_states = self.n_states
        n_nodes = 2 * self.sample_size - 1
        pi_arr = numpy.asarray(pi)

        # Partial likelihoods: L[node, site, state]
        L = numpy.zeros((n_nodes, n_sites, n_states))

        # Initialize leaves: L[leaf, site, observed_state] = 1.0
        seq_array = numpy.asarray(self.sequences)
        leaf_idx = numpy.arange(self.sample_size)[:, None]
        site_idx = numpy.arange(n_sites)[None, :]
        L[leaf_idx, site_idx, seq_array] = 1.0

        # Post-order traversal (internal nodes are numbered in time order)
        for p in range(self.sample_size, n_nodes):
            left = self.left_child[p]
            right = self.right_child[p]

            P_left = self.transition_probability(
                self.time[p] - self.time[left], mutation_rate
            )
            P_right = self.transition_probability(
                self.time[p] - self.time[right], mutation_rate
            )

            # (n_sites, n_states) @ (n_states, n_states) for each child
            L[p] = (L[left] @ P_left.T) * (L[right] @ P_right.T)

        # Root likelihood per site, then sum log-likelihoods
        root_L = L[self.root] @ pi_arr
        return numpy.sum(numpy.log(numpy.maximum(root_L, 1e-300)))

    def compute_gradient(self, mutation_rate, pi):
        """
        Compute gradients of log-likelihood w.r.t. node times (vectorized).
        
        Uses three passes:
        1. Post-order: partial likelihoods L[node, site, state] and per-child
           contributions R_left, R_right at each internal node.
        2. Pre-order: from-above weights F[node, site, state], accounting for
           the sibling subtree contribution at each split.
        3. Per-edge derivatives aggregated to per-node time gradients via the
           chain rule: d(LL)/d(time[node]) sums contributions from all edges
           incident to node, with signs reflecting how time[node] affects each
           branch length.
        """
        n_sites = len(self.sequences[0])
        n_states = self.n_states
        n_nodes = 2 * self.sample_size - 1
        pi_arr = numpy.asarray(pi)

        # --- Post-order: partial likelihoods ---
        L = numpy.zeros((n_nodes, n_sites, n_states))
        seq_array = numpy.asarray(self.sequences)
        leaf_idx = numpy.arange(self.sample_size)[:, None]
        site_idx = numpy.arange(n_sites)[None, :]
        L[leaf_idx, site_idx, seq_array] = 1.0

        # Per-child contributions at each internal node
        R_left = numpy.zeros((n_nodes, n_sites, n_states))
        R_right = numpy.zeros((n_nodes, n_sites, n_states))

        for p in range(self.sample_size, n_nodes):
            left = self.left_child[p]
            right = self.right_child[p]
            P_l = self.transition_probability(
                self.time[p] - self.time[left], mutation_rate
            )
            P_r = self.transition_probability(
                self.time[p] - self.time[right], mutation_rate
            )
            R_left[p] = L[left] @ P_l.T
            R_right[p] = L[right] @ P_r.T
            L[p] = R_left[p] * R_right[p]

        L_root = L[self.root] @ pi_arr
        L_root = numpy.maximum(L_root, 1e-300)

        # --- Pre-order: from-above weights F[node, site, state] ---
        # F satisfies: L_total(site) = sum_s F[node,site,s] * L[node,site,s]
        F = numpy.zeros((n_nodes, n_sites, n_states))
        F[self.root] = pi_arr

        for p in range(n_nodes - 1, self.sample_size - 1, -1):
            left = self.left_child[p]
            right = self.right_child[p]
            P_l = self.transition_probability(
                self.time[p] - self.time[left], mutation_rate
            )
            P_r = self.transition_probability(
                self.time[p] - self.time[right], mutation_rate
            )
            F[left] += (F[p] * R_right[p]) @ P_l
            F[right] += (F[p] * R_left[p]) @ P_r

        # --- Per-edge gradients, indexed by child node ---
        # edge_grad[c] = d(LL)/d(t_branch) for the edge from parent(c) to c
        edge_grad = numpy.zeros(n_nodes)

        for p in range(self.sample_size, n_nodes):
            left = self.left_child[p]
            right = self.right_child[p]

            # Edge (p -> left): weight at p excluding left subtree
            dP_l = self.transition_probability_derivative(
                self.time[p] - self.time[left], mutation_rate
            )
            w_l = F[p] * R_right[p]
            edge_grad[left] = numpy.sum(
                numpy.sum(w_l * (L[left] @ dP_l.T), axis=1) / L_root
            )

            # Edge (p -> right): weight at p excluding right subtree
            dP_r = self.transition_probability_derivative(
                self.time[p] - self.time[right], mutation_rate
            )
            w_r = F[p] * R_left[p]
            edge_grad[right] = numpy.sum(
                numpy.sum(w_r * (L[right] @ dP_r.T), axis=1) / L_root
            )

        # --- Aggregate to node-time gradients via chain rule ---
        # d(LL)/d(time[p]) = +edge_grad[left_child] + edge_grad[right_child]
        #                     - edge_grad[p]   (if p is not root)
        # Signs: time[p] increases branch lengths to children (+1 each)
        #        time[p] decreases branch length to parent (-1)
        gradients = numpy.zeros(n_nodes)
        for node in range(self.sample_size, n_nodes):
            left = self.left_child[node]
            right = self.right_child[node]
            gradients[node] = edge_grad[left] + edge_grad[right]
            if self.parent[node] != -1:
                gradients[node] -= edge_grad[node]

        return gradients




    @property
    def mutation_rate(self):
        return self._mutation_rate

    def resample_mutation_rate(self, step_size=0.1):
        """
        Resample the mutation rate using a log-normal proposal (better for positive parameters).
        
        Parameters:
        - step_size: float
            The standard deviation of the log-normal proposal.
        
        Returns:
        - old_mutation_rate: float
            The previous mutation rate before resampling.
        - log_proposal_density: float
            Log of the proposal density q(new|old)
        - log_reverse_proposal_density: float
            Log of the reverse proposal density q(old|new)
        """
        old_mutation_rate = self._mutation_rate
        
        # Log-normal proposal: log(new_rate) ~ N(log(old_rate), step_size^2)
        log_old_rate = numpy.log(old_mutation_rate)
        log_new_rate = log_old_rate + step_size * numpy.random.randn()
        if not numpy.isfinite(log_new_rate) or log_new_rate > MAX_LOG_FLOAT:
            return old_mutation_rate, 0.0, -math.inf
        new_rate = numpy.exp(log_new_rate)
        if not numpy.isfinite(new_rate):
            return old_mutation_rate, 0.0, -math.inf
        
        # Calculate proposal densities
        # q(new|old) = 1/(new_rate * step_size * sqrt(2*pi)) * exp(-0.5 * ((log(new_rate) - log(old_rate))/step_size)^2)
        log_proposal_density = -numpy.log(new_rate * step_size * numpy.sqrt(2 * numpy.pi)) - 0.5 * ((log_new_rate - log_old_rate) / step_size)**2
        
        # q(old|new) = 1/(old_rate * step_size * sqrt(2*pi)) * exp(-0.5 * ((log(old_rate) - log(new_rate))/step_size)^2)
        log_reverse_proposal_density = -numpy.log(old_mutation_rate * step_size * numpy.sqrt(2 * numpy.pi)) - 0.5 * ((log_old_rate - log_new_rate) / step_size)**2
        
        self._mutation_rate = new_rate
        return old_mutation_rate, log_proposal_density, log_reverse_proposal_density


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
    # Initialize the sequences array
    num_nodes = 2 * tree.sample_size - 1
    sequences = numpy.full((num_nodes, len(root_sequence)), -1, dtype=int)

    # Use the new attribute to set the root sequence
    sequences[tree.root] = root_sequence

    # Iterate downwards from the root
    for node in range(num_nodes - 1, -1, -1):
        # Check if the node's sequence has been set and it's an internal node
        if tree.is_internal(node) and numpy.all(sequences[node] != -1):
            parent_seq = sequences[node]
            left_child = tree.left_child[node]
            right_child = tree.right_child[node]
            
            # Correctly calculate branch length
            t_left = tree.time[node] - tree.time[left_child]
            t_right = tree.time[node] - tree.time[right_child]
            
            # Mutate and assign sequences to children
            sequences[left_child] = mutate_sequence(parent_seq, t_left, mutation_rate, num_states)
            sequences[right_child] = mutate_sequence(parent_seq, t_right, mutation_rate, num_states)

    return sequences[:tree.sample_size]

def coalescence_tree_with_sequences(sample_size, num_states, seq_length, mutation_rate, root_sequence=None):
    if root_sequence is None:
        root_sequence = generate_root_sequence(seq_length, num_states)
    
    tree = Tree(sample_size, num_states)
    sequences = simulate_sequences(tree, root_sequence, mutation_rate, num_states)
    return tree, sequences
