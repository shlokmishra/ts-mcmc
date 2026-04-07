"""
Paper-facing benchmark campaign for the usable larger simulation regime.

This script intentionally focuses on a single regime:
    n = 64, L = 1400
using the currently preferred larger-regime sampler defaults.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tskit

from benchmark_recorder import (
    build_fresh_tree_sequence_from_recorded_trees,
    dump_ts_bytes,
    median_timing_ms,
    sum_branch_lengths_from_newick_line,
)
from benchmark_longchain_diagnostics import effective_sample_size, rough_burn_in_fraction
from mcmc import kingman_mcmc
from recorder import Recorder
from tree import coalescence_tree_with_sequences


SAMPLE_SIZE = 64
SEQ_LENGTH = 1400
STEPS = 5000
BURN_IN = 1000
SEEDS = [42, 123, 999]
RUNTIME_REPEATS = 1

TIME_MOVE = "local"
TIME_STEP_SIZE = 1.0
MUTATION_STEP_SIZE = 0.1
SPR_MOVES_PER_STEP = 2


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def summarize_trace(xs):
    arr = np.asarray(xs, dtype=float)
    q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std()),
        "q10": float(q10),
        "median": float(q50),
        "q90": float(q90),
        "ess": effective_sample_size(arr),
        "rough_burn_in_fraction": rough_burn_in_fraction(arr),
    }


def run_chain(record: bool, seed: int):
    seed_all(seed)
    tree, sequences = coalescence_tree_with_sequences(SAMPLE_SIZE, 2, SEQ_LENGTH, 1.0)
    tree.sequences = sequences
    recorder = Recorder(SAMPLE_SIZE, SEQ_LENGTH)
    pi = np.array([0.5, 0.5])
    t0 = time.perf_counter()
    acceptance = kingman_mcmc(
        tree,
        recorder,
        pi,
        steps=STEPS,
        record=record,
        compute_gradients=False,
        print_every=None,
        time_move=TIME_MOVE,
        time_step_size=TIME_STEP_SIZE,
        mutation_step_size=MUTATION_STEP_SIZE,
        spr_moves_per_step=SPR_MOVES_PER_STEP,
    )
    wall = time.perf_counter() - t0
    return recorder, acceptance, wall


def benchmark_runtime(seed: int):
    vanilla = median_timing_ms(lambda: run_chain(False, seed), RUNTIME_REPEATS) / STEPS
    record = median_timing_ms(lambda: run_chain(True, seed), RUNTIME_REPEATS) / STEPS
    return {
        "vanilla_ms_per_step": float(vanilla),
        "record_ms_per_step": float(record),
        "record_overhead_pct": float((record - vanilla) / vanilla * 100.0),
    }


def retained_tree_sequence(full_ts: tskit.TreeSequence) -> tskit.TreeSequence:
    retained = full_ts.keep_intervals([[float(BURN_IN), float(STEPS)]], simplify=False).trim()
    return retained


def benchmark_queries(ts: tskit.TreeSequence, recorded_trees: list[tskit.Tree]):
    newick_lines = [tree.newick() for tree in recorded_trees]
    rng = np.random.default_rng(123)
    random_indices = rng.choice(len(recorded_trees), size=min(100, len(recorded_trees)), replace=False)

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

        load_trees_and_access_ms = median_timing_ms(
            lambda: tskit.load(ts_path).at(ts.sequence_length / 2.0).total_branch_length,
            1,
        )
        load_newick_and_access_ms = median_timing_ms(
            lambda: sum_branch_lengths_from_newick_line(open(nwk_path).read().splitlines()[len(recorded_trees) // 2]),
            1,
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
        1,
    )
    per_sample_newick_branch_length_ms = median_timing_ms(
        lambda: [sum_branch_lengths_from_newick_line(line) for line in newick_lines],
        1,
    )
    random_access_ts_at_ms = median_timing_ms(
        lambda: [ts.at(float(i) + 0.5).total_branch_length for i in random_indices],
        1,
    )
    random_access_newick_ms = median_timing_ms(
        lambda: [sum_branch_lengths_from_newick_line(newick_lines[int(i)]) for i in random_indices],
        1,
    )

    return {
        "trees_file_bytes": trees_file_bytes,
        "newick_file_bytes": newick_file_bytes,
        "newick_gz_file_bytes": newick_gz_file_bytes,
        "load_trees_and_access_ms": float(load_trees_and_access_ms),
        "load_newick_and_access_ms": float(load_newick_and_access_ms),
        "compressed_interval_scan_ms": float(compressed_interval_scan_ms),
        "per_sample_newick_branch_length_ms": float(per_sample_newick_branch_length_ms),
        "random_access_ts_at_ms": float(random_access_ts_at_ms),
        "random_access_newick_ms": float(random_access_newick_ms),
    }


@dataclass
class CampaignRow:
    seed: int
    sample_size: int
    seq_length: int
    steps: int
    burn_in: int
    retained_states: int
    time_move: str
    time_step_size: float
    mutation_step_size: float
    spr_moves_per_step: int
    wall_clock_s: float
    runtime_ms_per_step: float
    vanilla_ms_per_step: float
    record_ms_per_step: float
    record_overhead_pct: float
    acceptance_spr: float
    acceptance_times: float
    acceptance_mutation: float
    loglik_mean: float
    loglik_sd: float
    loglik_q10: float
    loglik_median: float
    loglik_q90: float
    loglik_ess: float
    loglik_rough_burn_in_fraction: float
    mutation_mean: float
    mutation_sd: float
    mutation_q10: float
    mutation_median: float
    mutation_q90: float
    mutation_ess: float
    mutation_rough_burn_in_fraction: float
    compressed_intervals: int
    interval_fraction: float
    trees_raw_bytes: int
    trees_gz_bytes: int
    fresh_trees_raw_bytes: int
    fresh_trees_gz_bytes: int
    newick_raw_bytes: int
    newick_gz_bytes: int
    compressed_interval_scan_ms: float
    per_sample_newick_branch_length_ms: float
    random_access_ts_at_ms: float
    random_access_newick_ms: float


def run_campaign() -> list[CampaignRow]:
    rows = []
    for seed in SEEDS:
        recorder, acceptance, wall = run_chain(True, seed)
        runtime = benchmark_runtime(seed)

        full_ts = recorder.tree_sequence()
        retained_ts = retained_tree_sequence(full_ts)
        retained_trees = recorder.recorded_trees()[BURN_IN:]
        fresh_ts = build_fresh_tree_sequence_from_recorded_trees(retained_trees)

        ts_bytes = dump_ts_bytes(retained_ts)
        fresh_ts_bytes = dump_ts_bytes(fresh_ts)
        newick_bytes = "\n".join(tree.newick() for tree in retained_trees).encode("utf-8")
        queries = benchmark_queries(retained_ts, retained_trees)

        loglik = summarize_trace(recorder.log_likelihoods)
        mutation = summarize_trace(recorder.mutation_rates)

        rows.append(
            CampaignRow(
                seed=seed,
                sample_size=SAMPLE_SIZE,
                seq_length=SEQ_LENGTH,
                steps=STEPS,
                burn_in=BURN_IN,
                retained_states=len(retained_trees),
                time_move=TIME_MOVE,
                time_step_size=TIME_STEP_SIZE,
                mutation_step_size=MUTATION_STEP_SIZE,
                spr_moves_per_step=SPR_MOVES_PER_STEP,
                wall_clock_s=float(wall),
                runtime_ms_per_step=float(wall * 1000.0 / STEPS),
                vanilla_ms_per_step=runtime["vanilla_ms_per_step"],
                record_ms_per_step=runtime["record_ms_per_step"],
                record_overhead_pct=runtime["record_overhead_pct"],
                acceptance_spr=float(acceptance[0]),
                acceptance_times=float(acceptance[1]),
                acceptance_mutation=float(acceptance[2]),
                loglik_mean=loglik["mean"],
                loglik_sd=loglik["sd"],
                loglik_q10=loglik["q10"],
                loglik_median=loglik["median"],
                loglik_q90=loglik["q90"],
                loglik_ess=loglik["ess"],
                loglik_rough_burn_in_fraction=loglik["rough_burn_in_fraction"],
                mutation_mean=mutation["mean"],
                mutation_sd=mutation["sd"],
                mutation_q10=mutation["q10"],
                mutation_median=mutation["median"],
                mutation_q90=mutation["q90"],
                mutation_ess=mutation["ess"],
                mutation_rough_burn_in_fraction=mutation["rough_burn_in_fraction"],
                compressed_intervals=retained_ts.num_trees,
                interval_fraction=float(retained_ts.num_trees / len(retained_trees)),
                trees_raw_bytes=len(ts_bytes),
                trees_gz_bytes=len(gzip.compress(ts_bytes, compresslevel=6)),
                fresh_trees_raw_bytes=len(fresh_ts_bytes),
                fresh_trees_gz_bytes=len(gzip.compress(fresh_ts_bytes, compresslevel=6)),
                newick_raw_bytes=len(newick_bytes),
                newick_gz_bytes=len(gzip.compress(newick_bytes, compresslevel=6)),
                compressed_interval_scan_ms=queries["compressed_interval_scan_ms"],
                per_sample_newick_branch_length_ms=queries["per_sample_newick_branch_length_ms"],
                random_access_ts_at_ms=queries["random_access_ts_at_ms"],
                random_access_newick_ms=queries["random_access_newick_ms"],
            )
        )
    return rows


def render_claims_table(rows: list[CampaignRow]) -> str:
    med = lambda fn: statistics.median(fn(r) for r in rows)
    lines = ["# N64 Main Claims", ""]
    lines.append("| Claim Class | Current n=64 result | How to phrase it |")
    lines.append("|---|---|---|")
    lines.append(
        f"| A. Structure vs fresh TS | interval fraction `{med(lambda r: r.interval_fraction):.3f}`, raw `.trees / fresh` `{med(lambda r: r.trees_raw_bytes / r.fresh_trees_raw_bytes):.3f}` | Foreground this. |"
    )
    lines.append(
        f"| B. Storage vs Newick | raw `.trees / Newick` `{med(lambda r: r.trees_raw_bytes / r.newick_raw_bytes):.3f}`, `.trees.gz / Newick.gz` `{med(lambda r: r.trees_gz_bytes / r.newick_gz_bytes):.3f}` | Usable, but say this is for the n=64, 1400-site regime. |"
    )
    lines.append(
        f"| C. Runtime overhead | record overhead `{med(lambda r: r.record_overhead_pct):.1f}%`, runtime `{med(lambda r: r.runtime_ms_per_step):.3f}` ms/step | Foreground, but only for this tuned larger-regime default. |"
    )
    lines.append(
        f"| D. Analytics | compressed bulk / per-sample Newick `{med(lambda r: r.compressed_interval_scan_ms / r.per_sample_newick_branch_length_ms):.3f}`, random `ts.at` / Newick `{med(lambda r: r.random_access_ts_at_ms / r.random_access_newick_ms):.3f}` | Foreground compressed bulk scans; keep random access secondary. |"
    )
    lines.append("")
    return "\n".join(lines)


def render_limitations_table(rows: list[CampaignRow]) -> str:
    med = lambda fn: statistics.median(fn(r) for r in rows)
    lines = ["# Paper Limitations Table", ""]
    lines.append("| Topic | Limitation | Evidence |")
    lines.append("|---|---|---|")
    lines.append(
        f"| n=128 scaling | still under-mixed under the current default | long-chain diagnostics at `5000` steps only reach log-likelihood ESS `{9.4}` with burn-in fraction `{0.32}` in [`longchain_diagnostics_summary.md`](/Users/shlokmishra/code/ts-mcmc/benchmark_outputs/longchain_diagnostics/longchain_diagnostics_summary.md). |"
    )
    lines.append(
        f"| n=64 burn-in | even usable n=64 chains still need conservative burn-in handling | long-chain diagnostics suggest burn-in fraction near `0.14`; campaign uses fixed `20%` burn-in (`1000/5000` steps). |"
    )
    lines.append(
        f"| Query story | generic per-sample TS access is not the headline win | n=64 random `ts.at` / Newick is `{med(lambda r: r.random_access_ts_at_ms / r.random_access_newick_ms):.3f}`, so the main claim should stay on compressed bulk analytics. |"
    )
    lines.append("")
    return "\n".join(lines)


def render_jere_note(rows: list[CampaignRow]) -> str:
    med = lambda fn: statistics.median(fn(r) for r in rows)
    lines = ["# Results For Jere Now", ""]
    lines.append("Current paper-facing regime:")
    lines.append(f"- `n=64`, `L=1400`, `steps=5000`, fixed burn-in `1000`, seeds `{SEEDS}`")
    lines.append("")
    lines.append("What looks strongest right now:")
    lines.append(f"- shared-edge TS structure compression is very strong: interval fraction `{med(lambda r: r.interval_fraction):.3f}`, raw `.trees / fresh TS` `{med(lambda r: r.trees_raw_bytes / r.fresh_trees_raw_bytes):.3f}`")
    lines.append(f"- raw storage is favorable in this regime: raw `.trees / Newick` `{med(lambda r: r.trees_raw_bytes / r.newick_raw_bytes):.3f}`")
    lines.append(f"- gzipped storage is essentially at parity here rather than a headline win: `.trees.gz / Newick.gz` `{med(lambda r: r.trees_gz_bytes / r.newick_gz_bytes):.3f}`")
    lines.append(f"- recording overhead is small in this tuned default: `{med(lambda r: r.record_overhead_pct):.1f}%`")
    lines.append(f"- compressed bulk analytics are much faster than per-sample baselines: compressed bulk / per-sample Newick `{med(lambda r: r.compressed_interval_scan_ms / r.per_sample_newick_branch_length_ms):.3f}`")
    lines.append("")
    lines.append("What I would flag as limitations:")
    lines.append("- `n=128` is still under-mixed under the same default, so it should stay in the limitations/future-work bucket for now.")
    lines.append("- per-sample random access is not the main TS win; the strongest query story is still bulk scans over compressed intervals.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(rows: list[CampaignRow], out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "paper_n64_campaign.json", "w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)
    with open(out_path / "paper_n64_campaign.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with open(out_path / "n64_main_claims.md", "w") as f:
        f.write(render_claims_table(rows))
    with open(out_path / "n64_limitations_table.md", "w") as f:
        f.write(render_limitations_table(rows))
    with open(out_path / "results_for_jere_now.md", "w") as f:
        f.write(render_jere_note(rows))


def main():
    out_dir = "benchmark_outputs/paper_n64_campaign"
    rows = run_campaign()
    write_outputs(rows, out_dir)
    print(f"Wrote paper-facing n=64 campaign outputs to {out_dir}")


if __name__ == "__main__":
    main()
