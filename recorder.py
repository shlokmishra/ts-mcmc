import math
import numpy
import random
import tskit

from tree import Tree

class Recorder:
    def __init__(self, sample_size, seq_length):
        """
        Initialize the Recorder with sample nodes and empty tables.

        Parameters:
        - sample_size: int
            Number of sample nodes (leaves) in each tree.
        - seq_length: float
            Total length of the genomic sequence.
        """
        self.tables = tskit.TableCollection(sequence_length=seq_length)
        self.node_table = self.tables.nodes
        self.edge_table = self.tables.edges
        self.mutation_rates = []
        self.log_likelihoods = []
        self.gradients = []
        self.current_position = 0.0  # Current genomic position

        # Create sample nodes once and reuse them for each tree
        self.sample_node_ids = []
        for _ in range(sample_size):
            node_id = self.node_table.add_row(
                flags=tskit.NODE_IS_SAMPLE,
                time=0.0  # Assuming all samples are at time 0
            )
            self.sample_node_ids.append(node_id)

    def append_tree(self, tree, mutation_rate, log_likelihood, gradient):
        """
        Append a new tree to the TreeSequence.

        Parameters:
        - tree: Instance of the Tree class representing the current phylogenetic tree.
        - mutation_rate: float
            Mutation rate for the current tree.
        - log_likelihood: float
            Log-likelihood of the current tree.
        - gradient: float
            Gradient of the log-likelihood with respect to parameters.
        """
        num_nodes = len(tree.parent)
        tree_to_table_map = [0] * num_nodes

        # Map sample nodes to existing node IDs
        for i in range(tree.sample_size):
            tree_to_table_map[i] = self.sample_node_ids[i]

        # Add internal nodes and map them
        for i in range(tree.sample_size, num_nodes):
            node_id = self.node_table.add_row(time=tree.time[i])
            tree_to_table_map[i] = node_id

        # Add edges for the current tree
        for i in range(tree.sample_size, num_nodes):
            parent = tree_to_table_map[i]
            left_child = tree_to_table_map[tree.left_child[i]]
            right_child = tree_to_table_map[tree.right_child[i]]

            # Add edge for left child
            self.edge_table.add_row(
                left=self.current_position,
                right=self.current_position + 1.0,  # Each tree occupies [r, r+1)
                parent=parent,
                child=left_child,
            )

            # Add edge for right child
            self.edge_table.add_row(
                left=self.current_position,
                right=self.current_position + 1.0,
                parent=parent,
                child=right_child,
            )

        # Increment genomic position for the next tree
        self.current_position += 1.0

        # Store additional information
        self.mutation_rates.append(mutation_rate)
        self.log_likelihoods.append(log_likelihood)
        self.gradients.append(gradient)

    def tree_sequence(self):
        """
        Generate and return the TreeSequence.

        Returns:
        - ts: tskit.TreeSequence
            The complete tree sequence containing all appended trees.
        """
        self.tables.sequence_length = self.current_position  # Set the total sequence length
        self.tables.sort()  # Sort tables by genomic position
        ts = self.tables.tree_sequence()
        return ts

    def write_additional_info(self, filename):
        """
        Write mutation rates, log-likelihoods, and gradients to a file.

        Parameters:
        - filename: str
            Path to the output text file.
        """
        with open(filename, 'w') as f:
            for i in range(len(self.mutation_rates)):
                f.write(f"Tree {i}:\n")
                f.write(f"Mutation Rate: {self.mutation_rates[i]}\n")
                f.write(f"Log Likelihood: {self.log_likelihoods[i]}\n")
                f.write(f"Gradient: {self.gradients[i]}\n\n")
