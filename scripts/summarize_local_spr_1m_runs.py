from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summaries" / "summary.json").read_text())


def augment_with_root_stats(summary: dict, run_dir: Path) -> dict:
    row = dict(summary)
    if all(key in row for key in ("root_time_min", "root_time_max", "root_time_last")):
        return row
    trace_path = run_dir / "traces" / "trace.csv"
    if trace_path.exists():
        root_min = float("inf")
        root_max = float("-inf")
        root_last = None
        with trace_path.open() as f:
            for trace_row in csv.DictReader(f):
                value = float(trace_row["root_time"])
                root_min = min(root_min, value)
                root_max = max(root_max, value)
                root_last = value
        row["root_time_min"] = root_min
        row["root_time_max"] = root_max
        row["root_time_last"] = root_last
    return row


def collect_rows(root_dirs: list[Path]) -> list[dict]:
    rows = []
    for root_dir in root_dirs:
        run_dirs = list(root_dir.glob("proposal_local_spr__seed_*__n_64__L_1400__steps_1000000"))
        for run_dir in run_dirs:
            rows.append(augment_with_root_stats(read_summary(run_dir), run_dir))
    rows.sort(key=lambda row: int(row["seed"]))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_table(rows: list[dict], path: Path) -> None:
    headers = [
        "seed",
        "final_acceptance_rate",
        "mean_rolling_acceptance_rate",
        "log_likelihood_ess",
        "log_target_ess",
        "total_runtime_s",
        "mean_log_hastings",
        "root_time_min",
        "root_time_max",
        "root_time_last",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def make_variability_plot(rows: list[dict], path: Path) -> None:
    seeds = [int(row["seed"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    axes[0].plot(seeds, [row["final_acceptance_rate"] for row in rows], marker="o")
    axes[0].set_title("Acceptance by Seed")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Final Acceptance")

    axes[1].plot(seeds, [row["log_target_ess"] for row in rows], marker="o")
    axes[1].set_title("Log Target ESS by Seed")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("ESS")

    axes[2].errorbar(
        seeds,
        [row["root_time_last"] for row in rows],
        yerr=[
            [row["root_time_last"] - row["root_time_min"] for row in rows],
            [row["root_time_max"] - row["root_time_last"] for row in rows],
        ],
        fmt="o",
    )
    axes[2].set_title("Root-Time Range by Seed")
    axes[2].set_xlabel("Seed")
    axes[2].set_ylabel("Root Time")

    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dirs", nargs="+", type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows([Path(path) for path in args.root_dirs])
    write_csv(rows, output_dir / "combined_long_run_summaries.csv")
    write_comparison_table(rows, output_dir / "long_run_comparison_table.csv")
    make_variability_plot(rows, output_dir / "long_run_variability.png")


if __name__ == "__main__":
    main()
