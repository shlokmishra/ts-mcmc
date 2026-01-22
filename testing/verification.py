import numpy
import math
import matplotlib.pyplot as plt
import tskit

from mcmc import kingman_mcmc
from recorder import Recorder
from tree import Tree


def verify_kingman(samples, steps):
    len_samples = len(samples)
    tmrca = numpy.zeros(len_samples)
    branch_length = numpy.zeros(len_samples)
    first_clade_prob = numpy.zeros(len_samples)
    acceptance_probs = []
    ind = 0
    for i in samples:
        tree = Tree(i)
        recorder = Recorder(tree, i, steps)
        acceptance_prob = kingman_mcmc(tree, recorder)
        ts = recorder.tree_sequence()
        node_table = recorder.node_table
        for tree in ts.trees():
            tmrca[ind] += (
                node_table[tree.root].time
                * (tree.interval.right - tree.interval.left)
                / steps
            )
            branch_length[ind] += (
                tree.total_branch_length
                * (tree.interval.right - tree.interval.left)
                / steps
            )
            nodes = tree.timeasc()
            node = nodes[0]
            node_index = 0
            while tree.is_sample(node) == True:
                node_index += 1
                node = nodes[node_index]
            if 0 in tree.children(node) and 1 in tree.children(node):
                first_clade_prob[ind] += (
                    tree.interval.right - tree.interval.left
                ) / steps
        acceptance_probs.append(acceptance_prob)
        ind += 1
    mean_tmrca = numpy.array([2 * (1 - 1 / n) for n in samples])
    mean_branch_length = numpy.cumsum([2 / (n - 1) for n in samples])
    mean_first_clade_prob = numpy.array([math.log(2 / (n * (n - 1))) for n in samples])
    first_clade_prob = [math.log(x) for x in first_clade_prob]

    plt.plot(samples, tmrca, label="Observed mean TMRCA")
    plt.plot(
        samples,
        mean_tmrca,
        label="Exact mean TMRCA",
    )
    plt.plot(
        samples,
        branch_length,
        label="Observed mean branch length",
    )
    plt.plot(
        samples,
        mean_branch_length,
        label="Exact mean branch length",
    )
    plt.plot(
        samples,
        first_clade_prob,
        label="Observed log(P(0 & 1 merge first))",
    )
    plt.plot(
        samples,
        mean_first_clade_prob,
        label="Exact log(P(0 & 1 merge first))",
    )
    plt.legend()
    plt.savefig("verification.png")
    plt.close("all")

    print(acceptance_probs)


min_sample = 2
max_sample = 10
samples = range(min_sample, max_sample + 1)
steps = int(1e5)

verify_kingman(samples, steps)
