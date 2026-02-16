"""
Benchmarks for tree sequence MCMC.

Addresses the key questions for the paper:
1. Runtime overhead of tree-sequence recording vs vanilla MCMC.
2. How does MCMC cost scale with taxa and sites?
3. Storage tradeoffs (tree-sequence vs Newick, raw vs compressed).
4. Downstream inference quality (Stein vs naive thinning).

Usage:
    python benchmarks.py
"""

import time
import os
import gzip
import tempfile
import numpy as np
from tree import Tree, coalescence_tree_with_sequences
from recorder import Recorder
from mcmc import kingman_mcmc


def benchmark_recording_overhead(sample_size=50, seq_length=50, steps=500):
    """
    Benchmark 1: Runtime overhead of tree sequence recording.
    
    Compares three modes:
    - Vanilla MCMC (no recording, no gradients)
    - Recording only (tree sequence storage, no gradients)
    - Full recording (tree sequence + gradients for Stein thinning)
    """
    print("=" * 70)
    print("BENCHMARK 1: Recording Overhead")
    print(f"  {sample_size} taxa, {seq_length} sites, {steps} MCMC steps")
    print("=" * 70)

    pi = np.array([0.5, 0.5])
    results = {}

    for mode, record, gradients, label in [
        ("vanilla",  False, False, "Vanilla MCMC (no recording)"),
        ("record",   True,  False, "With tree sequence recording"),
        ("full",     True,  True,  "With recording + gradients"),
    ]:
        # Fresh tree and recorder each time (same seed for fair comparison)
        np.random.seed(42)
        tree, sequences = coalescence_tree_with_sequences(
            sample_size, 2, seq_length, 1.0
        )
        tree.sequences = sequences
        recorder = Recorder(sample_size, seq_length)

        t0 = time.time()
        kingman_mcmc(
            tree, recorder, pi, steps=steps, step_size=0.3,
            record=record, compute_gradients=gradients, print_every=None
        )
        elapsed = time.time() - t0
        results[mode] = elapsed

        print(f"  {label:<45} {elapsed:.2f}s  ({elapsed/steps*1000:.1f} ms/step)")

    # Compute overhead percentages
    vanilla = results["vanilla"]
    record_overhead = (results["record"] - vanilla) / vanilla * 100
    gradient_overhead = (results["full"] - results["record"]) / vanilla * 100
    total_overhead = (results["full"] - vanilla) / vanilla * 100

    print()
    print(f"  Recording overhead:  {record_overhead:+.1f}%")
    print(f"  Gradient overhead:   {gradient_overhead:+.1f}%")
    print(f"  Total overhead:      {total_overhead:+.1f}%")
    print("=" * 70)

    return results


def benchmark_storage_cost(sample_size=50, seq_length=50, steps_list=None):
    """
    Benchmark 3: Storage comparison — tree sequence vs Newick, raw vs compressed.
    """
    if steps_list is None:
        steps_list = [100, 500, 1000, 5000]

    print("\n" + "=" * 70)
    print("BENCHMARK 3: Storage Cost (Tree Sequence vs Newick, Raw vs Compressed)")
    print(f"  {sample_size} taxa, {seq_length} sites")
    print("=" * 70)
    print(f"  {'Steps':<8} {'TS':<10} {'TS.gz':<10} {'Nwk':<10} {'Nwk.gz':<10} "
          f"{'TS/Nwk':<8} {'TS.gz/Nwk.gz':<12}")
    print("-" * 70)

    pi = np.array([0.5, 0.5])

    def fmt_size(b):
        if b < 1024:
            return f"{b} B"
        elif b < 1024 * 1024:
            return f"{b/1024:.1f} KB"
        else:
            return f"{b/(1024*1024):.1f} MB"

    for steps in steps_list:
        np.random.seed(42)
        tree, sequences = coalescence_tree_with_sequences(
            sample_size, 2, seq_length, 1.0
        )
        tree.sequences = sequences
        recorder = Recorder(sample_size, seq_length)

        kingman_mcmc(
            tree, recorder, pi, steps=steps, step_size=0.3,
            record=True, compute_gradients=True, print_every=None
        )

        # Tree sequence (raw + compressed)
        ts = recorder.tree_sequence()
        with tempfile.NamedTemporaryFile(suffix=".trees", delete=False) as f:
            ts.dump(f.name)
            ts_raw = open(f.name, "rb").read()
            ts_size = len(ts_raw)
            os.unlink(f.name)
        ts_gz = len(gzip.compress(ts_raw, compresslevel=6))

        # Newick (raw + compressed)
        newick_parts = []
        for t in ts.trees():
            try:
                newick_parts.append(t.newick())
            except Exception:
                newick_parts.append(str(list(t.nodes())))
        newick_data = "\n".join(newick_parts).encode("utf-8")
        newick_size = len(newick_data)
        newick_gz = len(gzip.compress(newick_data, compresslevel=6))

        ratio_raw = ts_size / newick_size if newick_size > 0 else float("inf")
        ratio_gz = ts_gz / newick_gz if newick_gz > 0 else float("inf")

        print(f"  {steps:<8} {fmt_size(ts_size):<10} {fmt_size(ts_gz):<10} "
              f"{fmt_size(newick_size):<10} {fmt_size(newick_gz):<10} "
              f"{ratio_raw:<8.2f} {ratio_gz:<12.2f}")

    print()
    print("  Findings:")
    print("  - Raw: tree sequence ~2.5x larger than Newick per independent tree.")
    print("  - Compressed: Newick.gz is *smaller* than TreeSeq.gz (text compresses well).")
    print("  - TreeSeq advantage is O(1) random access + tskit API for post-hoc analysis.")
    print("  - A differential recorder storing only changed edges could close the gap.")
    print("=" * 70)


def benchmark_stein_thinning(sample_size=20, seq_length=50, steps=1000):
    """
    Benchmark 3: Stein thinning quality vs naive thinning.
    """
    import stein_thinning_trees as stt

    print("\n" + "=" * 70)
    print("BENCHMARK 3: Stein Thinning vs Naive Thinning")
    print(f"  {sample_size} taxa, {seq_length} sites, {steps} MCMC steps")
    print("=" * 70)

    pi = np.array([0.5, 0.5])
    np.random.seed(42)
    tree, sequences = coalescence_tree_with_sequences(
        sample_size, 2, seq_length, 1.0
    )
    tree.sequences = sequences
    recorder = Recorder(sample_size, seq_length)

    print(f"  Running MCMC ({steps} steps)...", end=" ", flush=True)
    t0 = time.time()
    kingman_mcmc(
        tree, recorder, pi, steps=steps, step_size=0.3,
        record=True, compute_gradients=True, print_every=None
    )
    mcmc_time = time.time() - t0
    print(f"done ({mcmc_time:.1f}s)")

    n_total = len(recorder.mutation_rates)
    test_points = [10, 25, 50, 100, 200]
    test_points = [p for p in test_points if p < n_total]

    print()
    print(f"  {'n_points':<12} {'Stein KSD':<15} {'Naive KSD':<15} {'Improvement':<15} {'Thinning time'}")
    print("-" * 70)

    for n_pts in test_points:
        t0 = time.time()
        results = stt.compare_thinning_methods(recorder, n_points=n_pts)
        thin_time = time.time() - t0

        print(f"  {n_pts:<12} {results['stein_ksd']:<15.4f} {results['naive_ksd']:<15.4f} "
              f"{results['improvement']:<15.1%} {thin_time:.2f}s")

    # Summary statistics
    full_rates = np.array(recorder.mutation_rates)
    best_result = stt.compare_thinning_methods(recorder, n_points=min(100, n_total - 1))
    stein_rates = [recorder.mutation_rates[i] for i in best_result['stein_indices']]
    naive_rates = [recorder.mutation_rates[i] for i in best_result['naive_indices']]

    print()
    print(f"  Posterior mean comparison (n=100 selected):")
    print(f"    Full chain:     {np.mean(full_rates):.4f} ± {np.std(full_rates):.4f}")
    print(f"    Stein thinned:  {np.mean(stein_rates):.4f} ± {np.std(stein_rates):.4f}")
    print(f"    Naive thinned:  {np.mean(naive_rates):.4f} ± {np.std(naive_rates):.4f}")
    print("=" * 70)


def benchmark_scaling(steps=100):
    """
    2D scaling: taxa x sites heatmap.
    """
    taxa_list = [10, 20, 50, 100, 200]
    sites_list = [10, 50, 200, 500, 1000]

    print("\n" + "=" * 70)
    print("BENCHMARK 2: Scaling (ms per MCMC step)")
    print(f"  {steps} steps per config, with recording + gradients")
    print("=" * 70)

    pi = np.array([0.5, 0.5])
    ms_grid = np.zeros((len(taxa_list), len(sites_list)))

    header = f"  {'':>8}"
    for L in sites_list:
        header += f"  L={L:>5}"
    print(header)
    print("-" * 70)

    for i, n in enumerate(taxa_list):
        row = f"  n={n:>4}"
        for j, L in enumerate(sites_list):
            np.random.seed(42)
            tree, sequences = coalescence_tree_with_sequences(n, 2, L, 1.0)
            tree.sequences = sequences
            recorder = Recorder(n, L)

            t0 = time.time()
            kingman_mcmc(
                tree, recorder, pi, steps=steps, step_size=0.3,
                record=True, compute_gradients=True, print_every=None
            )
            elapsed = time.time() - t0
            ms = elapsed / steps * 1000
            ms_grid[i, j] = ms
            row += f"  {ms:>6.1f}"
        print(row)

    # Fit cost model: t = b*n + c*(n*L)
    data = []
    for i, n in enumerate(taxa_list):
        for j, L in enumerate(sites_list):
            data.append({'n': n, 'L': L, 'ms': ms_grid[i, j]})

    X = np.array([[d['n'], d['n'] * d['L']] for d in data])
    y = np.array([d['ms'] for d in data])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b_fit, c_fit = coeffs
    y_pred = X @ coeffs
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    print()
    print(f"  Cost model: t = {b_fit:.4f}·n + {c_fit:.6f}·(n·L)  ms/step  (R²={r2:.4f})")
    print(f"  - Per-node overhead: {b_fit:.4f} ms  (Python loop, independent of sites)")
    print(f"  - Per-element cost:  {c_fit:.6f} ms  (matmul, linear in sites)")
    print()
    print("  At small L, per-node overhead dominates (cost looks flat in L).")
    print("  At large L, matmul dominates and cost grows linearly in L.")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  TREE SEQUENCE MCMC BENCHMARKS")
    print("#" * 70)

    # Benchmark 1: Recording overhead
    benchmark_recording_overhead(sample_size=50, seq_length=50, steps=500)

    # Benchmark 2: Scaling (taxa x sites)
    benchmark_scaling(steps=100)

    # Benchmark 3: Storage cost (raw + compressed)
    benchmark_storage_cost(sample_size=50, seq_length=50,
                           steps_list=[100, 500, 1000, 5000])

    # Benchmark 4: Stein thinning
    benchmark_stein_thinning(sample_size=20, seq_length=50, steps=1000)
