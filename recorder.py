import math
import numpy
import random
import tskit

from tree import Tree


class Recorder:
    def __init__(self, tree, sample_size, seq_length):
        self.tables = tskit.TableCollection(sequence_length=seq_length)
        self.node_table = self.tables.nodes
        self.edge_table = self.tables.edges
        num_nodes = len(tree.parent)
        self.tree_to_table_map = list(range(num_nodes))
        self.left_edge_below = []
        self.right_edge_below = []
        for i in self.tree_to_table_map:
            if i < sample_size:
                node = self.node_table.add_row(time=tree.time[i], flags=1)
                self.left_edge_below.append(-1)
                self.right_edge_below.append(-1)
            else:
                node = self.node_table.add_row(time=tree.time[i])
                edge = self.edge_table.add_row(
                    left=0, right=1, child=tree.left_child[i], parent=i
                )
                self.left_edge_below.append(edge)
                self.edge_table.add_row(
                    left=0, right=1, child=tree.right_child[i], parent=i
                )
                self.right_edge_below.append(edge)

    def append_tree(self, tree):
        r = self.edge_table[self.left_edge_below[-1]].right
        for i in range(tree.sample_size, len(self.tree_to_table_map)):
            if tree.time[i] != self.node_table[self.tree_to_table_map[i]].time:
                new_node = self.node_table.add_row(time=tree.time[i])
                self.tree_to_table_map[i] = new_node

        for i in range(tree.sample_size, len(self.left_edge_below)):
            tree_parent = self.tree_to_table_map[i]
            tree_child = self.tree_to_table_map[tree.left_child[i]]
            edge = self.left_edge_below[i]
            edge_child = self.edge_table[edge].child
            edge_parent = self.edge_table[edge].parent
            if tree_child != edge_child or tree_parent != edge_parent:
                new_edge = self.edge_table.add_row(
                    left=r,
                    right=r + 1,
                    child=tree_child,
                    parent=tree_parent,
                )
                self.left_edge_below[i] = new_edge
            else:
                self.edge_table[edge] = self.edge_table[edge].replace(right=r + 1)
            edge = self.right_edge_below[i]
            edge_child = self.edge_table[edge].child
            edge_parent = self.edge_table[edge].parent
            tree_child = self.tree_to_table_map[tree.right_child[i]]
            if tree_child != edge_child or tree_parent != edge_parent:
                new_edge = self.edge_table.add_row(
                    left=r,
                    right=r + 1,
                    child=tree_child,
                    parent=tree_parent,
                )
                self.right_edge_below[i] = new_edge
            else:
                self.edge_table[edge] = self.edge_table[edge].replace(right=r + 1)

    def tree_sequence(self):
        self.tables.sort()
        ts = self.tables.tree_sequence()
        return ts

