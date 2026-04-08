from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mcmc import kingman_mcmc
from recorder import Recorder
from tree import coalescence_tree_with_sequences


TRACE_FIELDS = [
    "iteration",
    "proposal_type",
    "accepted",
    "log_likelihood",
    "log_target",
    "log_alpha",
    "log_hastings",
    "log_q_forward",
    "log_q_reverse",
    "mutation_rate",
    "root_time",
    "cumulative_acceptance_rate",
    "rolling_acceptance_rate",
    "elapsed_s",
    "detached_leaf",
    "chosen_branch",
    "forward_candidate_count",
    "reverse_candidate_count",
    "sampled_time",
    "forward_candidates_json",
    "reverse_candidates_json",
    "time_move_accepted",
    "mutation_move_accepted",
]


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def effective_sample_size(xs) -> float:
    arr = np.asarray(xs, dtype=float)
    n = arr.size
    if n < 4:
        return float(n)
    centered = arr - arr.mean()
    var = np.dot(centered, centered) / n
    if var <= 1e-15:
        return float(n)
    rho_sum = 0.0
    prev_rho = None
    for lag in range(1, min(n - 1, n // 2) + 1):
        rho = np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * var)
        if prev_rho is not None and prev_rho + rho < 0:
            break
        rho_sum += rho
        prev_rho = rho
    return float(n / max(1.0, 1.0 + 2.0 * rho_sum))


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def default_run_name(
    proposal_type: str,
    seed: int,
    sample_size: int,
    seq_length: int,
    steps: int,
) -> str:
    return (
        f"proposal_{sanitize_name(proposal_type)}"
        f"__seed_{seed}"
        f"__n_{sample_size}"
        f"__L_{seq_length}"
        f"__steps_{steps}"
    )


class CSVTraceLogger:
    def __init__(self, path: Path):
        self.path = path
        self.handle = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=TRACE_FIELDS)
        self.writer.writeheader()

    def __call__(self, row: dict) -> None:
        cleaned = {}
        for field in TRACE_FIELDS:
            value = row.get(field)
            if isinstance(value, (list, dict)):
                cleaned[field] = json.dumps(to_jsonable(value))
            else:
                cleaned[field] = to_jsonable(value)
        self.writer.writerow(cleaned)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def load_trace_rows(trace_csv_path: Path) -> list[dict]:
    def parse_numeric(value: str):
        lowered = value.strip().lower()
        if lowered in {"inf", "+inf", "-inf", "nan"}:
            return float(lowered)
        try:
            return int(value)
        except ValueError:
            return float(value)

    rows = []
    with open(trace_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {
                    "proposal_type",
                    "forward_candidates_json",
                    "reverse_candidates_json",
                }:
                    parsed[key] = value
                elif key in {
                    "accepted",
                    "time_move_accepted",
                    "mutation_move_accepted",
                }:
                    parsed[key] = value == "True"
                elif value in ("", None):
                    parsed[key] = None
                else:
                    parsed[key] = parse_numeric(value)
            rows.append(parsed)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def summarize_trace_rows(rows: list[dict], burn_in: int) -> dict:
    post_rows = rows[burn_in:] if burn_in < len(rows) else []
    if not rows:
        raise ValueError("Cannot summarize an empty trace.")
    if not post_rows:
        post_rows = rows

    acceptance = [1.0 if row["accepted"] else 0.0 for row in rows]
    rolling = [float(row["rolling_acceptance_rate"]) for row in rows]
    loglik = [float(row["log_likelihood"]) for row in post_rows]
    target = [float(row["log_target"]) for row in post_rows]
    summary = {
        "proposal_type": rows[-1]["proposal_type"],
        "seed": None,
        "iterations": len(rows),
        "burn_in": burn_in,
        "final_acceptance_rate": float(np.mean(acceptance)),
        "mean_rolling_acceptance_rate": float(np.mean(rolling)),
        "log_likelihood_mean": mean_std(loglik)[0],
        "log_likelihood_sd": mean_std(loglik)[1],
        "log_target_mean": mean_std(target)[0],
        "log_target_sd": mean_std(target)[1],
        "log_likelihood_ess": effective_sample_size(np.asarray(loglik, dtype=float)),
        "log_target_ess": effective_sample_size(np.asarray(target, dtype=float)),
        "total_runtime_s": float(rows[-1]["elapsed_s"]),
    }

    if rows[-1]["proposal_type"] == "local_spr":
        log_hastings = [float(row["log_hastings"]) for row in rows if row["log_hastings"] is not None]
        accepted_local = [float(row["log_hastings"]) for row in rows if row["accepted"] and row["log_hastings"] is not None]
        summary.update(
            {
                "mean_log_hastings": float(np.mean(log_hastings)) if log_hastings else None,
                "accepted_negative_hastings_fraction": (
                    float(np.mean([value < 0 for value in accepted_local])) if accepted_local else None
                ),
                "accepted_positive_hastings_fraction": (
                    float(np.mean([value > 0 for value in accepted_local])) if accepted_local else None
                ),
            }
        )
    else:
        summary.update(
            {
                "mean_log_hastings": None,
                "accepted_negative_hastings_fraction": None,
                "accepted_positive_hastings_fraction": None,
            }
        )
    return summary


@dataclass
class RunConfig:
    proposal_type: str
    seed: int
    sample_size: int
    seq_length: int
    steps: int
    burn_in: int = 0
    mutation_step_size: float = 0.1
    time_move: str = "local"
    time_step_size: float = 1.0
    spr_local_k: int | None = None
    spr_moves_per_step: int = 1
    compute_gradients: bool = False
    record: bool = True
    print_every: int | None = None
    acceptance_window: int = 100


def run_logged_chain(
    output_root: str | Path,
    config: RunConfig,
    make_plots: bool = False,
) -> dict:
    output_root = Path(output_root)
    run_dir = ensure_dir(output_root / default_run_name(
        config.proposal_type,
        config.seed,
        config.sample_size,
        config.seq_length,
        config.steps,
    ))
    traces_dir = ensure_dir(run_dir / "traces")
    plots_dir = ensure_dir(run_dir / "plots")
    summaries_dir = ensure_dir(run_dir / "summaries")

    metadata_path = run_dir / "metadata.json"
    trace_csv_path = traces_dir / "trace.csv"
    summary_json_path = summaries_dir / "summary.json"
    summary_csv_path = summaries_dir / "summary.csv"

    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "output_paths": {
            "run_dir": str(run_dir),
            "trace_csv": str(trace_csv_path),
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "plots_dir": str(plots_dir),
        },
        **asdict(config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    seed_all(config.seed)
    tree, sequences = coalescence_tree_with_sequences(config.sample_size, 2, config.seq_length, 1.0)
    tree.sequences = sequences
    recorder = Recorder(config.sample_size, config.seq_length)
    pi = np.array([0.5, 0.5])

    logger = CSVTraceLogger(trace_csv_path)
    try:
        acceptances = kingman_mcmc(
            tree,
            recorder,
            pi,
            steps=config.steps,
            record=config.record,
            compute_gradients=config.compute_gradients,
            print_every=config.print_every,
            mutation_step_size=config.mutation_step_size,
            time_move=config.time_move,
            time_step_size=config.time_step_size,
            spr_local_k=config.spr_local_k,
            spr_moves_per_step=config.spr_moves_per_step,
            spr_proposal=config.proposal_type,
            iteration_logger=logger,
            acceptance_window=config.acceptance_window,
        )
    finally:
        logger.close()

    rows = load_trace_rows(trace_csv_path)
    summary = summarize_trace_rows(rows, config.burn_in)
    summary.update(
        {
            "seed": config.seed,
            "sample_size": config.sample_size,
            "seq_length": config.seq_length,
            "steps": config.steps,
            "burn_in": config.burn_in,
            "acceptance_spr": float(acceptances[0]),
            "acceptance_times": float(acceptances[1]),
            "acceptance_mutation": float(acceptances[2]),
        }
    )
    summary_json_path.write_text(json.dumps(summary, indent=2))
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    result = {
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "trace_csv_path": trace_csv_path,
        "summary_json_path": summary_json_path,
        "summary_csv_path": summary_csv_path,
        "plots_dir": plots_dir,
        "summary": summary,
    }

    if make_plots:
        from scripts.plot_mcmc_diagnostics import generate_run_plots

        result["plot_paths"] = generate_run_plots(run_dir)
    return result


def load_manifest(path: str | Path) -> list[RunConfig]:
    payload = json.loads(Path(path).read_text())
    runs = payload["runs"] if isinstance(payload, dict) and "runs" in payload else payload
    return [RunConfig(**run) for run in runs]


def write_aggregate_csv(results: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    rows = [result["summary"] for result in results]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path
