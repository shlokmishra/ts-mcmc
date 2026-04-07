"""
Authoritative recorder benchmark driver.

This script focuses on the paper-critical storage/runtime/query story around
the recorder and tree-sequence encoding. It avoids notebook-only logic and
uses moderate, reproducible settings by default.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tskit

from mcmc import kingman_mcmc
from recorder import Recorder
from tree import coalescence_tree_with_sequences


FLOAT_RE = re.compile(r":([0-9eE.+-]+)")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def fmt_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024**2:.2f} MB"


def median_timing_ms(fn, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(timings))


def dump_ts_bytes(ts: tskit.TreeSequence) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".trees", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ts.dump(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def build_fresh_tree_sequence_from_recorded_trees(recorded_trees: list[tskit.Tree]) -> tskit.TreeSequence:
    sample_size = len(list(recorded_trees[0].samples()))
    tables = tskit.TableCollection(sequence_length=float(len(recorded_trees)))

    sample_ids = []
    for _ in range(sample_size):
        sample_ids.append(tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0))

    for pos, tree in enumerate(recorded_trees):
        node_map = {}
        for sample in tree.samples():
            node_map[sample] = sample_ids[sample]

        for node in tree.nodes(order="timeasc"):
            if tree.is_sample(node):
                continue
            node_map[node] = tables.nodes.add_row(time=tree.time(node))

        for node in tree.nodes():
            parent = tree.parent(node)
            if parent == tskit.NULL:
                continue
            tables.edges.add_row(
                left=float(pos),
                right=float(pos + 1),
                parent=node_map[parent],
                child=node_map[node],
            )

    tables.sort()
    return tables.tree_sequence()


def newick_text_from_recorded_trees(recorded_trees: list[tskit.Tree]) -> str:
    return "\n".join(tree.newick() for tree in recorded_trees)


def sum_branch_lengths_from_newick_line(line: str) -> float:
    return float(sum(float(x) for x in FLOAT_RE.findall(line)))


def make_recorder(
    sample_size: int,
    seq_length: int,
    steps: int,
    record: bool,
    gradients: bool,
    seed: int,
    mutation_step_size: float = 0.3,
    time_move: str = "global",
    time_step_size: float = 1.0,
    spr_moves_per_step: int = 1,
):
    seed_all(seed)
    tree, sequences = coalescence_tree_with_sequences(sample_size, 2, seq_length, 1.0)
    tree.sequences = sequences
    recorder = Recorder(sample_size, seq_length)
    pi = np.array([0.5, 0.5])
    acceptance = kingman_mcmc(
        tree,
        recorder,
        pi,
        steps=steps,
        mutation_step_size=mutation_step_size,
        record=record,
        compute_gradients=gradients,
        print_every=None,
        time_move=time_move,
        time_step_size=time_step_size,
        spr_moves_per_step=spr_moves_per_step,
    )
    return recorder, acceptance


def validate_recording_outputs(recorder: Recorder) -> None:
    ts = recorder.tree_sequence()
    recorded = recorder.recorded_trees()
    n_states = len(recorder.mutation_rates)

    assert len(recorded) == n_states
    assert ts.sequence_length == float(n_states)
    assert ts.num_samples == recorder.sample_size

    if n_states > 0:
        sample_positions = sorted({0, n_states // 2, n_states - 1})
        for idx in sample_positions:
            ts_tree = ts.at(idx + 0.5)
            rec_tree = recorded[idx]
            assert abs(ts_tree.total_branch_length - rec_tree.total_branch_length) < 1e-9


@dataclass
class RuntimeResult:
    sample_size: int
    seq_length: int
    steps: int
    repeats: int
    vanilla_ms_per_step: float
    record_ms_per_step: float
    full_ms_per_step: float
    record_overhead_pct: float
    full_overhead_pct: float


def benchmark_runtime(
    sample_size: int,
    seq_length: int,
    steps: int,
    repeats: int,
    seed: int,
    mutation_step_size: float = 0.3,
    time_move: str = "global",
    time_step_size: float = 1.0,
    spr_moves_per_step: int = 1,
) -> RuntimeResult:
    def run_mode(record: bool, gradients: bool):
        return median_timing_ms(
            lambda: make_recorder(
                sample_size,
                seq_length,
                steps,
                record,
                gradients,
                seed,
                mutation_step_size=mutation_step_size,
                time_move=time_move,
                time_step_size=time_step_size,
                spr_moves_per_step=spr_moves_per_step,
            ),
            repeats,
        ) / steps

    vanilla = run_mode(False, False)
    record = run_mode(True, False)
    full = run_mode(True, True)

    return RuntimeResult(
        sample_size=sample_size,
        seq_length=seq_length,
        steps=steps,
        repeats=repeats,
        vanilla_ms_per_step=vanilla,
        record_ms_per_step=record,
        full_ms_per_step=full,
        record_overhead_pct=(record - vanilla) / vanilla * 100.0,
        full_overhead_pct=(full - vanilla) / vanilla * 100.0,
    )


@dataclass
class StorageResult:
    steps: int
    recorded_states: int
    compressed_intervals: int
    compressed_nodes: int
    compressed_edges: int
    fresh_intervals: int
    fresh_nodes: int
    fresh_edges: int
    trees_raw_bytes: int
    trees_gz_bytes: int
    fresh_trees_raw_bytes: int
    fresh_trees_gz_bytes: int
    newick_raw_bytes: int
    newick_gz_bytes: int


def benchmark_storage(
    sample_size: int,
    seq_length: int,
    steps_list: list[int],
    seed: int,
    mutation_step_size: float = 0.3,
    time_move: str = "global",
    time_step_size: float = 1.0,
    spr_moves_per_step: int = 1,
) -> list[StorageResult]:
    results = []
    for steps in steps_list:
        recorder, _ = make_recorder(
            sample_size,
            seq_length,
            steps,
            True,
            False,
            seed,
            mutation_step_size=mutation_step_size,
            time_move=time_move,
            time_step_size=time_step_size,
            spr_moves_per_step=spr_moves_per_step,
        )
        validate_recording_outputs(recorder)

        ts = recorder.tree_sequence()
        recorded = recorder.recorded_trees()
        fresh_ts = build_fresh_tree_sequence_from_recorded_trees(recorded)

        ts_bytes = dump_ts_bytes(ts)
        fresh_ts_bytes = dump_ts_bytes(fresh_ts)
        newick_text = newick_text_from_recorded_trees(recorded)
        newick_bytes = newick_text.encode("utf-8")

        results.append(
            StorageResult(
                steps=steps,
                recorded_states=len(recorded),
                compressed_intervals=ts.num_trees,
                compressed_nodes=ts.num_nodes,
                compressed_edges=ts.num_edges,
                fresh_intervals=fresh_ts.num_trees,
                fresh_nodes=fresh_ts.num_nodes,
                fresh_edges=fresh_ts.num_edges,
                trees_raw_bytes=len(ts_bytes),
                trees_gz_bytes=len(gzip.compress(ts_bytes, compresslevel=6)),
                fresh_trees_raw_bytes=len(fresh_ts_bytes),
                fresh_trees_gz_bytes=len(gzip.compress(fresh_ts_bytes, compresslevel=6)),
                newick_raw_bytes=len(newick_bytes),
                newick_gz_bytes=len(gzip.compress(newick_bytes, compresslevel=6)),
            )
        )
    return results


@dataclass
class QueryResult:
    sample_size: int
    seq_length: int
    steps: int
    repeats: int
    recorded_states: int
    compressed_intervals: int
    trees_file_bytes: int
    newick_file_bytes: int
    newick_gz_file_bytes: int
    load_trees_and_access_ms: float
    load_newick_and_access_ms: float
    load_newick_gz_and_access_ms: float
    compressed_interval_scan_ms: float
    compressed_interval_mrca_ms: float
    per_sample_ts_at_branch_length_ms: float
    per_sample_recorded_trees_branch_length_ms: float
    per_sample_recorded_trees_mrca_ms: float
    per_sample_newick_branch_length_ms: float
    random_access_ts_at_ms: float
    random_access_newick_ms: float


def benchmark_queries(
    sample_size: int,
    seq_length: int,
    steps: int,
    repeats: int,
    seed: int,
    mutation_step_size: float = 0.3,
    time_move: str = "global",
    time_step_size: float = 1.0,
    spr_moves_per_step: int = 1,
) -> QueryResult:
    recorder, _ = make_recorder(
        sample_size,
        seq_length,
        steps,
        True,
        False,
        seed,
        mutation_step_size=mutation_step_size,
        time_move=time_move,
        time_step_size=time_step_size,
        spr_moves_per_step=spr_moves_per_step,
    )
    validate_recording_outputs(recorder)
    ts = recorder.tree_sequence()
    recorded = recorder.recorded_trees()
    newick_lines = [tree.newick() for tree in recorded]
    random_indices = np.random.default_rng(seed).choice(len(recorded), size=min(100, len(recorded)), replace=False)

    with tempfile.NamedTemporaryFile(suffix=".trees", delete=False) as tmp_trees:
        ts_path = tmp_trees.name
    with tempfile.NamedTemporaryFile(suffix=".nwk", delete=False, mode="w") as tmp_nwk:
        nwk_path = tmp_nwk.name
    with tempfile.NamedTemporaryFile(suffix=".nwk.gz", delete=False) as tmp_nwk_gz:
        nwk_gz_path = tmp_nwk_gz.name

    try:
        ts.dump(ts_path)
        with open(nwk_path, "w") as f:
            f.write("\n".join(newick_lines))
        with gzip.open(nwk_gz_path, "wt") as f:
            f.write("\n".join(newick_lines))

        load_trees_and_access = median_timing_ms(
            lambda: tskit.load(ts_path).at(len(recorded) / 2).total_branch_length,
            repeats,
        )
        load_newick_and_access = median_timing_ms(
            lambda: sum_branch_lengths_from_newick_line(open(nwk_path).read().splitlines()[len(recorded) // 2]),
            repeats,
        )
        load_newick_gz_and_access = median_timing_ms(
            lambda: sum_branch_lengths_from_newick_line(gzip.open(nwk_gz_path, "rt").read().splitlines()[len(recorded) // 2]),
            repeats,
        )
    finally:
        trees_file_bytes = os.path.getsize(ts_path)
        newick_file_bytes = os.path.getsize(nwk_path)
        newick_gz_file_bytes = os.path.getsize(nwk_gz_path)
        os.unlink(ts_path)
        os.unlink(nwk_path)
        os.unlink(nwk_gz_path)

    compressed_interval_scan_ms = median_timing_ms(
        lambda: [tree.total_branch_length for tree in ts.trees()],
        repeats,
    )
    compressed_interval_mrca_ms = median_timing_ms(
        lambda: [tree.mrca(0, 1) for tree in ts.trees()],
        repeats,
    )
    per_sample_ts_at_branch_length_ms = median_timing_ms(
        lambda: [ts.at(i + 0.5).total_branch_length for i in range(len(recorded))],
        repeats,
    )
    per_sample_recorded_trees_branch_length_ms = median_timing_ms(
        lambda: [tree.total_branch_length for tree in recorder.recorded_trees()],
        repeats,
    )
    per_sample_recorded_trees_mrca_ms = median_timing_ms(
        lambda: [tree.mrca(0, 1) for tree in recorder.recorded_trees()],
        repeats,
    )
    per_sample_newick_branch_length_ms = median_timing_ms(
        lambda: [sum_branch_lengths_from_newick_line(line) for line in newick_lines],
        repeats,
    )
    random_access_ts_at_ms = median_timing_ms(
        lambda: [ts.at(int(i) + 0.5).total_branch_length for i in random_indices],
        repeats,
    )
    random_access_newick_ms = median_timing_ms(
        lambda: [sum_branch_lengths_from_newick_line(newick_lines[int(i)]) for i in random_indices],
        repeats,
    )

    return QueryResult(
        sample_size=sample_size,
        seq_length=seq_length,
        steps=steps,
        repeats=repeats,
        recorded_states=len(recorded),
        compressed_intervals=ts.num_trees,
        trees_file_bytes=trees_file_bytes,
        newick_file_bytes=newick_file_bytes,
        newick_gz_file_bytes=newick_gz_file_bytes,
        load_trees_and_access_ms=load_trees_and_access,
        load_newick_and_access_ms=load_newick_and_access,
        load_newick_gz_and_access_ms=load_newick_gz_and_access,
        compressed_interval_scan_ms=compressed_interval_scan_ms,
        compressed_interval_mrca_ms=compressed_interval_mrca_ms,
        per_sample_ts_at_branch_length_ms=per_sample_ts_at_branch_length_ms,
        per_sample_recorded_trees_branch_length_ms=per_sample_recorded_trees_branch_length_ms,
        per_sample_recorded_trees_mrca_ms=per_sample_recorded_trees_mrca_ms,
        per_sample_newick_branch_length_ms=per_sample_newick_branch_length_ms,
        random_access_ts_at_ms=random_access_ts_at_ms,
        random_access_newick_ms=random_access_newick_ms,
    )


@dataclass
class MatrixRow:
    regime: str
    sample_size: int
    seq_length: int
    steps: int
    seed: int
    repeats: int
    acceptance_spr: float
    acceptance_times: float
    acceptance_mutation: float
    recorded_states: int
    compressed_intervals: int
    compressed_interval_fraction: float
    trees_raw_bytes: int
    trees_gz_bytes: int
    newick_raw_bytes: int
    newick_gz_bytes: int
    fresh_trees_raw_bytes: int
    fresh_trees_gz_bytes: int
    vanilla_ms_per_step: float
    record_ms_per_step: float
    full_ms_per_step: float
    record_overhead_pct: float
    full_overhead_pct: float
    load_trees_and_access_ms: float
    compressed_interval_scan_ms: float
    compressed_interval_mrca_ms: float
    per_sample_ts_at_branch_length_ms: float
    per_sample_recorded_trees_branch_length_ms: float
    per_sample_recorded_trees_mrca_ms: float
    per_sample_newick_branch_length_ms: float
    random_access_ts_at_ms: float
    random_access_newick_ms: float


def matrix_configs(preset: str):
    if preset == "paper":
        seeds = [42, 123, 999]
        configs = [
            {"regime": "small", "sample_size": 10, "seq_length": 20, "steps": 100},
            {"regime": "small", "sample_size": 10, "seq_length": 20, "steps": 300},
            {"regime": "moderate", "sample_size": 20, "seq_length": 50, "steps": 100},
            {"regime": "moderate", "sample_size": 20, "seq_length": 50, "steps": 300},
            {"regime": "moderate", "sample_size": 20, "seq_length": 50, "steps": 500},
        ]
        return seeds, configs

    if preset == "paper_bridge":
        seeds = [42, 123, 999, 2024, 31415]
        configs = [
            {"regime": "small", "sample_size": 10, "seq_length": 20, "steps": 100},
            {"regime": "small", "sample_size": 10, "seq_length": 20, "steps": 300},
            {"regime": "moderate", "sample_size": 20, "seq_length": 50, "steps": 100},
            {"regime": "moderate", "sample_size": 20, "seq_length": 50, "steps": 300},
            {"regime": "moderate", "sample_size": 20, "seq_length": 50, "steps": 500},
            # Conservative bridge regime: larger than the paper matrix in both
            # taxa and sequence length, but still tractable for multi-seed runs.
            {"regime": "bridge", "sample_size": 32, "seq_length": 100, "steps": 500},
        ]
        return seeds, configs

    raise ValueError(f"Unknown matrix preset: {preset}")


def run_matrix(preset: str, repeats: int) -> list[MatrixRow]:
    seeds, configs = matrix_configs(preset)
    rows = []
    for config in configs:
        for seed in seeds:
            sample_size = config["sample_size"]
            seq_length = config["seq_length"]
            steps = config["steps"]

            recorder, acceptance = make_recorder(sample_size, seq_length, steps, True, False, seed)
            validate_recording_outputs(recorder)

            runtime = benchmark_runtime(sample_size, seq_length, steps, repeats, seed)
            storage = benchmark_storage(sample_size, seq_length, [steps], seed)[0]
            queries = benchmark_queries(sample_size, seq_length, steps, repeats, seed)

            rows.append(
                MatrixRow(
                    regime=config["regime"],
                    sample_size=sample_size,
                    seq_length=seq_length,
                    steps=steps,
                    seed=seed,
                    repeats=repeats,
                    acceptance_spr=acceptance[0],
                    acceptance_times=acceptance[1],
                    acceptance_mutation=acceptance[2],
                    recorded_states=storage.recorded_states,
                    compressed_intervals=storage.compressed_intervals,
                    compressed_interval_fraction=storage.compressed_intervals / storage.recorded_states,
                    trees_raw_bytes=storage.trees_raw_bytes,
                    trees_gz_bytes=storage.trees_gz_bytes,
                    newick_raw_bytes=storage.newick_raw_bytes,
                    newick_gz_bytes=storage.newick_gz_bytes,
                    fresh_trees_raw_bytes=storage.fresh_trees_raw_bytes,
                    fresh_trees_gz_bytes=storage.fresh_trees_gz_bytes,
                    vanilla_ms_per_step=runtime.vanilla_ms_per_step,
                    record_ms_per_step=runtime.record_ms_per_step,
                    full_ms_per_step=runtime.full_ms_per_step,
                    record_overhead_pct=runtime.record_overhead_pct,
                    full_overhead_pct=runtime.full_overhead_pct,
                    load_trees_and_access_ms=queries.load_trees_and_access_ms,
                    compressed_interval_scan_ms=queries.compressed_interval_scan_ms,
                    compressed_interval_mrca_ms=queries.compressed_interval_mrca_ms,
                    per_sample_ts_at_branch_length_ms=queries.per_sample_ts_at_branch_length_ms,
                    per_sample_recorded_trees_branch_length_ms=queries.per_sample_recorded_trees_branch_length_ms,
                    per_sample_recorded_trees_mrca_ms=queries.per_sample_recorded_trees_mrca_ms,
                    per_sample_newick_branch_length_ms=queries.per_sample_newick_branch_length_ms,
                    random_access_ts_at_ms=queries.random_access_ts_at_ms,
                    random_access_newick_ms=queries.random_access_newick_ms,
                )
            )
    return rows


def write_matrix_outputs(rows: list[MatrixRow], out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "benchmark_matrix.json"
    csv_path = out_path / "benchmark_matrix.csv"

    with open(json_path, "w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary = summarize_matrix(rows)
    with open(out_path / "benchmark_matrix_summary.md", "w") as f:
        f.write(summary)
    with open(out_path / "benchmark_matrix_tables.md", "w") as f:
        f.write(render_matrix_tables(rows))
    with open(out_path / "strong_claims_table.md", "w") as f:
        f.write(render_claims_table(rows, strong=True))
    with open(out_path / "weakened_claims_table.md", "w") as f:
        f.write(render_claims_table(rows, strong=False))


def summarize_matrix(rows: list[MatrixRow]) -> str:
    regimes = sorted(set(row.regime for row in rows))
    lines = ["# Benchmark Matrix Summary", ""]
    lines.append("## Confirmed patterns")
    for regime in regimes:
        regime_rows = [row for row in rows if row.regime == regime]
        record_overhead = np.median([row.record_overhead_pct for row in regime_rows])
        interval_fraction = np.median([row.compressed_interval_fraction for row in regime_rows])
        trees_vs_fresh = np.median([row.trees_raw_bytes / row.fresh_trees_raw_bytes for row in regime_rows])
        lines.append(
            f"- `{regime}`: median record-only overhead {record_overhead:.1f}%, "
            f"median interval fraction {interval_fraction:.3f}, "
            f"median compressed/fresh raw `.trees` ratio {trees_vs_fresh:.3f}."
        )
    lines.append("")
    lines.append("## Query emphasis")
    lines.append("- Strongest empirical query story: compressed interval bulk scans over `ts.trees()`.")
    lines.append("- Weaker story: per-sample `ts.at(...)` versus simple per-state Newick parsing is not consistently dominant in moderate runs.")
    lines.append("- Conceptually strong and measured: compressed-interval MRCA scans avoid revisiting unchanged adjacent states.")
    lines.append("")
    return "\n".join(lines)


def render_matrix_tables(rows: list[MatrixRow]) -> str:
    import statistics
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.regime, row.steps)].append(row)

    lines = ["# Benchmark Matrix Tables", ""]
    lines.append("## Structure and storage")
    lines.append("")
    lines.append("| Regime | Steps | Median interval fraction | Median `.trees` KB | Median fresh `.trees` KB | Median Newick KB | `.trees` / fresh | `.trees` / Newick | `.trees.gz` / `Newick.gz` |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (regime, steps) in sorted(grouped):
        rs = grouped[(regime, steps)]
        med = lambda fn: statistics.median(fn(r) for r in rs)
        lines.append(
            f"| {regime} | {steps} | "
            f"{med(lambda r: r.compressed_interval_fraction):.3f} | "
            f"{med(lambda r: r.trees_raw_bytes) / 1024:.1f} | "
            f"{med(lambda r: r.fresh_trees_raw_bytes) / 1024:.1f} | "
            f"{med(lambda r: r.newick_raw_bytes) / 1024:.1f} | "
            f"{med(lambda r: r.trees_raw_bytes / r.fresh_trees_raw_bytes):.3f} | "
            f"{med(lambda r: r.trees_raw_bytes / r.newick_raw_bytes):.3f} | "
            f"{med(lambda r: r.trees_gz_bytes / r.newick_gz_bytes):.3f} |"
        )

    lines.append("")
    lines.append("## Runtime and query")
    lines.append("")
    lines.append("| Regime | Steps | Record overhead % | Full overhead % | TS compressed scan ms | Recorded trees bulk ms | Newick bulk ms | TS compressed MRCA ms | Recorded trees MRCA ms | Random `ts.at` ms | Random Newick ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (regime, steps) in sorted(grouped):
        rs = grouped[(regime, steps)]
        med = lambda fn: statistics.median(fn(r) for r in rs)
        lines.append(
            f"| {regime} | {steps} | "
            f"{med(lambda r: r.record_overhead_pct):.1f} | "
            f"{med(lambda r: r.full_overhead_pct):.1f} | "
            f"{med(lambda r: r.compressed_interval_scan_ms):.3f} | "
            f"{med(lambda r: r.per_sample_recorded_trees_branch_length_ms):.3f} | "
            f"{med(lambda r: r.per_sample_newick_branch_length_ms):.3f} | "
            f"{med(lambda r: r.compressed_interval_mrca_ms):.3f} | "
            f"{med(lambda r: r.per_sample_recorded_trees_mrca_ms):.3f} | "
            f"{med(lambda r: r.random_access_ts_at_ms):.3f} | "
            f"{med(lambda r: r.random_access_newick_ms):.3f} |"
        )

    lines.append("")
    lines.append("## Sampler diagnostics")
    lines.append("")
    lines.append("| Regime | Steps | Median SPR accept | Median time accept | Median mutation accept |")
    lines.append("|---|---:|---:|---:|---:|")
    for (regime, steps) in sorted(grouped):
        rs = grouped[(regime, steps)]
        med = lambda fn: statistics.median(fn(r) for r in rs)
        lines.append(
            f"| {regime} | {steps} | "
            f"{med(lambda r: r.acceptance_spr):.3f} | "
            f"{med(lambda r: r.acceptance_times):.3f} | "
            f"{med(lambda r: r.acceptance_mutation):.3f} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_claims_table(rows: list[MatrixRow], strong: bool) -> str:
    import statistics
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.regime, row.steps)].append(row)

    if strong:
        lines = ["# Strongest Current Claims", ""]
        lines.append("| Regime | Steps | Interval fraction | `.trees` / fresh | Record overhead % | TS bulk scan / recorded_trees bulk | TS bulk MRCA / recorded_trees MRCA |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for (regime, steps) in sorted(grouped):
            rs = grouped[(regime, steps)]
            med = lambda fn: statistics.median(fn(r) for r in rs)
            lines.append(
                f"| {regime} | {steps} | "
                f"{med(lambda r: r.compressed_interval_fraction):.3f} | "
                f"{med(lambda r: r.trees_raw_bytes / r.fresh_trees_raw_bytes):.3f} | "
                f"{med(lambda r: r.record_overhead_pct):.1f} | "
                f"{med(lambda r: r.compressed_interval_scan_ms / r.per_sample_recorded_trees_branch_length_ms):.3f} | "
                f"{med(lambda r: r.compressed_interval_mrca_ms / r.per_sample_recorded_trees_mrca_ms):.3f} |"
            )
        lines.append("")
        return "\n".join(lines)

    lines = ["# Claims To Weaken Or Avoid", ""]
    lines.append("| Regime | Steps | `.trees` / Newick | `.trees.gz` / `Newick.gz` | Random `ts.at` / Newick | Per-sample `ts.at` / Newick bulk |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for (regime, steps) in sorted(grouped):
        rs = grouped[(regime, steps)]
        med = lambda fn: statistics.median(fn(r) for r in rs)
        lines.append(
            f"| {regime} | {steps} | "
            f"{med(lambda r: r.trees_raw_bytes / r.newick_raw_bytes):.3f} | "
            f"{med(lambda r: r.trees_gz_bytes / r.newick_gz_bytes):.3f} | "
            f"{med(lambda r: r.random_access_ts_at_ms / r.random_access_newick_ms):.3f} | "
            f"{med(lambda r: r.per_sample_ts_at_branch_length_ms / r.per_sample_newick_branch_length_ms):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def print_runtime(result: RuntimeResult) -> None:
    print("\n== Runtime Overhead ==")
    print(
        f"config: n={result.sample_size}, L={result.seq_length}, steps={result.steps}, repeats={result.repeats}"
    )
    print(f"vanilla: {result.vanilla_ms_per_step:.3f} ms/step")
    print(f"record : {result.record_ms_per_step:.3f} ms/step  ({result.record_overhead_pct:+.1f}%)")
    print(f"full   : {result.full_ms_per_step:.3f} ms/step  ({result.full_overhead_pct:+.1f}%)")


def print_storage(results: list[StorageResult]) -> None:
    print("\n== Storage and Structure ==")
    print(
        "steps  states  intervals  .trees      .trees.gz   fresh.trees  fresh.gz    newick      newick.gz"
    )
    for row in results:
        print(
            f"{row.steps:<6} {row.recorded_states:<7} {row.compressed_intervals:<10} "
            f"{fmt_bytes(row.trees_raw_bytes):<11} {fmt_bytes(row.trees_gz_bytes):<11} "
            f"{fmt_bytes(row.fresh_trees_raw_bytes):<12} {fmt_bytes(row.fresh_trees_gz_bytes):<11} "
            f"{fmt_bytes(row.newick_raw_bytes):<11} {fmt_bytes(row.newick_gz_bytes):<11}"
        )


def print_queries(result: QueryResult) -> None:
    print("\n== Query Timings ==")
    print(
        f"config: n={result.sample_size}, L={result.seq_length}, steps={result.steps}, "
        f"states={result.recorded_states}, intervals={result.compressed_intervals}, repeats={result.repeats}"
    )
    print(f"files: .trees={fmt_bytes(result.trees_file_bytes)}, newick={fmt_bytes(result.newick_file_bytes)}, newick.gz={fmt_bytes(result.newick_gz_file_bytes)}")
    print(f"load_trees_and_access_ms:             {result.load_trees_and_access_ms:.3f}")
    print(f"load_newick_and_access_ms:            {result.load_newick_and_access_ms:.3f}")
    print(f"load_newick_gz_and_access_ms:         {result.load_newick_gz_and_access_ms:.3f}")
    print(f"compressed_interval_scan_ms:          {result.compressed_interval_scan_ms:.3f}")
    print(f"compressed_interval_mrca_ms:          {result.compressed_interval_mrca_ms:.3f}")
    print(f"per_sample_ts_at_branch_length_ms:    {result.per_sample_ts_at_branch_length_ms:.3f}")
    print(f"per_sample_recorded_trees_ms:         {result.per_sample_recorded_trees_branch_length_ms:.3f}")
    print(f"per_sample_recorded_trees_mrca_ms:    {result.per_sample_recorded_trees_mrca_ms:.3f}")
    print(f"per_sample_newick_branch_length_ms:   {result.per_sample_newick_branch_length_ms:.3f}")
    print(f"random_access_ts_at_ms:               {result.random_access_ts_at_ms:.3f}")
    print(f"random_access_newick_ms:              {result.random_access_newick_ms:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-sample-size", type=int, default=20)
    parser.add_argument("--runtime-seq-length", type=int, default=50)
    parser.add_argument("--runtime-steps", type=int, default=200)
    parser.add_argument("--storage-sample-size", type=int, default=20)
    parser.add_argument("--storage-seq-length", type=int, default=50)
    parser.add_argument("--storage-steps", type=str, default="100,300,500")
    parser.add_argument("--query-sample-size", type=int, default=20)
    parser.add_argument("--query-seq-length", type=int, default=50)
    parser.add_argument("--query-steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--matrix-preset", type=str, default=None)
    parser.add_argument("--matrix-out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.matrix_preset:
        rows = run_matrix(args.matrix_preset, args.repeats)
        if args.matrix_out_dir is None:
            raise ValueError("--matrix-out-dir is required when using --matrix-preset")
        write_matrix_outputs(rows, args.matrix_out_dir)
        print(f"Wrote matrix outputs to {args.matrix_out_dir}")
        return

    runtime = benchmark_runtime(
        sample_size=args.runtime_sample_size,
        seq_length=args.runtime_seq_length,
        steps=args.runtime_steps,
        repeats=args.repeats,
        seed=args.seed,
    )
    storage = benchmark_storage(
        sample_size=args.storage_sample_size,
        seq_length=args.storage_seq_length,
        steps_list=parse_int_list(args.storage_steps),
        seed=args.seed,
    )
    queries = benchmark_queries(
        sample_size=args.query_sample_size,
        seq_length=args.query_seq_length,
        steps=args.query_steps,
        repeats=args.repeats,
        seed=args.seed,
    )

    print_runtime(runtime)
    print_storage(storage)
    print_queries(queries)

    if args.json_out:
        payload = {
            "runtime": asdict(runtime),
            "storage": [asdict(row) for row in storage],
            "queries": asdict(queries),
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
