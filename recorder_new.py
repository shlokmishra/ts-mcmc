import tskit
import math
import numpy as np

class Recorder:
    """Records a sequence of phylogenetic trees and associated data, 
    building a tskit TreeSequence for Stein thinning and analysis."""
    def __init__(self, samples):
        """
        Initialize the recorder with sample nodes.
        :param samples: Either an int (number of samples, labeled 0..N-1) 
                        or a list of sample names/identifiers.
        """
        # Create a new TableCollection with initial sequence_length 0.
        self.tables = tskit.TableCollection(sequence_length=0.0)
        self.sample_id_map = {}
        self.num_samples = 0

        # Initialize sample nodes in the node table (flags=1 for samples, time=0).
        if isinstance(samples, int):
            self.num_samples = samples
            # If integer, sample IDs will be 0..num_samples-1
            for i in range(samples):
                node_id = self.tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
                # node_id should equal i here
                self.sample_id_map[i] = node_id
        else:
            # If a list of names is provided
            self.num_samples = len(samples)
            for idx, name in enumerate(samples):
                node_id = self.tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
                self.sample_id_map[name] = node_id

        # Prepare storage for additional data per tree
        self.mutation_rates = []    # list of mutation rate values
        self.log_likelihoods = []   # list of log-likelihoods
        self.gradients = []         # list of gradient values (w.rt some parameter)

        # Track the current end of sequence for assigning new tree intervals
        self._current_left = 0.0

    def append_tree(self, tree, mutation_rate, log_likelihood, gradient):
        """
        Add a new tree to the sequence along with its mutation rate, log-likelihood, and gradient.
        The tree can be provided as a Bio.Phylo Clade, a custom tree node, or any structure 
        with 'children' (or 'clades') and branch length attributes.
        """
        # Start a new interval for this tree
        left = self._current_left
        right = left + 1.0
        self._current_left = right
        # Update the tables' sequence_length to cover this new interval
        if right > self.tables.sequence_length:
            self.tables.sequence_length = right

        # Recursive function to process nodes
        def add_node_recursive(node):
            """
            Recursively add nodes and edges for the subtree rooted at 'node'.
            Returns the node ID in the tables corresponding to this 'node'.
            """
            # Determine if this is a leaf
            is_leaf = False
            # Use Bio.Phylo Clade interface if available
            if hasattr(node, "is_terminal") and hasattr(node, "clades"):
                is_leaf = node.is_terminal()
            elif hasattr(node, "children"):
                # Custom node with 'children' list
                is_leaf = (len(node.children) == 0)
            else:
                # If node has no 'children' attribute, assume it's a leaf identifier (e.g., int or str)
                is_leaf = True

            if is_leaf:
                # Leaf node: get its sample ID
                if hasattr(node, "name") or hasattr(node, "id"):
                    # Use name or id attribute as key for sample map
                    label = getattr(node, "name", None)
                    if label is None:
                        label = getattr(node, "id", None)
                else:
                    # If node is given directly as a label (int/str)
                    label = node
                if label not in self.sample_id_map:
                    raise ValueError(f"Leaf label '{label}' not found in sample map.")
                return self.sample_id_map[label], 0.0  # (node_id, time)
            else:
                # Internal node: process all children to add them first
                if hasattr(node, "clades"):  # Bio.Phylo Clade
                    children = node.clades
                else:
                    children = node.children
                child_ids = []
                child_times = []
                for child in children:
                    child_id, child_time = add_node_recursive(child)
                    child_ids.append(child_id)
                    # Determine branch length to this child
                    if hasattr(child, "branch_length"):
                        bl = child.branch_length if child.branch_length is not None else 0.0
                    elif hasattr(child, "length"):
                        bl = child.length if child.length is not None else 0.0
                    else:
                        bl = 0.0
                    child_times.append(child_time + bl)
                # Determine this internal node's time as the max of child_times
                node_time = 0.0 if len(child_times) == 0 else max(child_times)
                # Ensure strict parent>child time (add a tiny epsilon if equal)
                if len(child_times) > 0 and node_time in child_times:
                    node_time += 1e-9

                # Add this internal node to the tables
                parent_id = self.tables.nodes.add_row(time=node_time)
                # Add edges from this parent to each child
                for child_id in child_ids:
                    self.tables.edges.add_row(left=left, right=right, parent=parent_id, child=child_id)
                return parent_id, node_time

        # Process the root of the tree
        root_id, root_time = add_node_recursive(tree)
        # (Optionally, we could handle the case of multiple roots, but assume one root per tree.)

        # Record the additional data for this tree
        self.mutation_rates.append(mutation_rate)
        self.log_likelihoods.append(log_likelihood)
        self.gradients.append(gradient)
        return root_id  # returning root_id is optional, mainly for internal use

    def tree_sequence(self):
        """Finalize and return the tskit.TreeSequence of all recorded trees."""
        # Sort tables to satisfy tskit requirements (edges by position, nodes by time)&#8203;:contentReference[oaicite:28]{index=28}
        self.tables.sort()
        ts = self.tables.tree_sequence()
        # Convert stored lists to numpy arrays for efficient use
        self.mutation_rates = np.array(self.mutation_rates)
        self.log_likelihoods = np.array(self.log_likelihoods)
        self.gradients = np.array(self.gradients)
        return ts
