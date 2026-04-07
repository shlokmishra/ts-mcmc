import tskit


class Recorder:
    def __init__(self, sample_size, seq_length):
        """
        Initialize the recorder with persistent sample nodes and append-only tables.

        Parameters
        ----------
        sample_size : int
            Number of leaves in each recorded tree.
        seq_length : float
            Initial sequence length passed to the TableCollection. This is
            overwritten with the number of recorded states when materializing a
            TreeSequence.
        """
        self.sample_size = sample_size
        self.tables = tskit.TableCollection(sequence_length=seq_length)
        self.node_table = self.tables.nodes
        self.edge_table = self.tables.edges

        self.mutation_rates = []
        self.log_likelihoods = []
        self.gradients = []
        self.current_position = 0.0

        # Leaves are stable across the whole chain and can always be reused.
        self.sample_node_ids = []
        for _ in range(sample_size):
            node_id = self.node_table.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
            self.sample_node_ids.append(node_id)

        # Map tree node index -> currently active table node id.
        self.tree_to_table_map = list(self.sample_node_ids) + [tskit.NULL] * (sample_size - 1)

    def _ensure_capacity(self, tree):
        num_nodes = len(tree.parent)
        if len(self.tree_to_table_map) != num_nodes:
            raise ValueError(
                f"Recorder initialized for {len(self.tree_to_table_map)} nodes, "
                f"but tree has {num_nodes} nodes."
            )

    def append_tree(self, tree, mutation_rate, log_likelihood, gradient):
        """
        Append the current tree state while reusing nodes when possible.

        Internal nodes are reused across adjacent samples if their time is
        unchanged. Edge interval sharing is finalized in tree_sequence() via
        tskit edge squashing, which keeps per-step append work proportional to
        current tree size instead of total recorded history.
        """
        self._ensure_capacity(tree)
        num_nodes = len(tree.parent)

        # Leaves always reuse the persistent sample nodes.
        for i in range(tree.sample_size):
            self.tree_to_table_map[i] = self.sample_node_ids[i]

        # Internal nodes can be reused if their time is unchanged.
        for i in range(tree.sample_size, num_nodes):
            table_node_id = self.tree_to_table_map[i]
            if table_node_id == tskit.NULL:
                self.tree_to_table_map[i] = self.node_table.add_row(time=tree.time[i])
                continue

            current_time = self.node_table.time[table_node_id]
            if current_time != tree.time[i]:
                self.tree_to_table_map[i] = self.node_table.add_row(time=tree.time[i])

        next_right = self.current_position + 1.0

        # Record one incoming edge per non-root child for this step. We defer
        # interval merging to tree_sequence(), where tskit can squash adjacent
        # identical rows after sorting.
        for child in range(num_nodes):
            parent = tree.parent[child]
            if parent == -1:
                continue
            self.edge_table.add_row(
                left=self.current_position,
                right=next_right,
                parent=self.tree_to_table_map[parent],
                child=self.tree_to_table_map[child],
            )

        self.current_position = next_right

        self.mutation_rates.append(mutation_rate)
        self.log_likelihoods.append(log_likelihood)
        self.gradients.append(gradient)

    def tree_sequence(self):
        """Return the compressed TreeSequence over recorded states."""
        tables = self.tables.copy()
        tables.sequence_length = self.current_position
        tables.sort()
        tables.edges.squash()
        tables.sort()
        return tables.tree_sequence()

    def recorded_trees(self, sample_lists=True):
        """
        Return one copied tskit tree per recorded MCMC state.

        The recorder may compress adjacent identical states into a single
        interval in the TreeSequence. This accessor preserves sample alignment by
        querying the tree covering the midpoint of each recorded unit interval.
        """
        ts = self.tree_sequence()
        trees = []
        for step in range(len(self.mutation_rates)):
            position = step + 0.5
            trees.append(ts.at(position, sample_lists=sample_lists).copy())
        return trees

    def write_additional_info(self, filename):
        """Write mutation rates, log-likelihoods, and gradients to a text file."""
        with open(filename, "w") as f:
            for i in range(len(self.mutation_rates)):
                f.write(f"Tree {i}:\n")
                f.write(f"Mutation Rate: {self.mutation_rates[i]}\n")
                f.write(f"Log Likelihood: {self.log_likelihoods[i]}\n")
                f.write(f"Gradient: {self.gradients[i]}\n\n")
