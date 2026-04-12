from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text())


def collect_run_summaries(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summaries" / "summary.json"
        rows.append(read_summary(summary_path))
    return rows


def write_rows_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_table(rows: list[dict], output_path: Path) -> None:
    lines = [
        "proposal_type,seed,final_acceptance_rate,mean_rolling_acceptance_rate,"
        "log_likelihood_ess,log_target_ess,total_runtime_s,mean_log_hastings,"
        "root_time_min,root_time_max,root_time_last"
    ]
    for row in rows:
        lines.append(
            f"{row['proposal_type']},{row['seed']},{row['final_acceptance_rate']},"
            f"{row.get('mean_rolling_acceptance_rate')},{row.get('log_likelihood_ess')},"
            f"{row.get('log_target_ess')},{row['total_runtime_s']},{row.get('mean_log_hastings')},"
            f"{row.get('root_time_min')},{row.get('root_time_max')},{row.get('root_time_last')}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def make_comparison_plot(rows: list[dict], output_path: Path) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["proposal_type"], []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for proposal_type, values in grouped.items():
        xs = [row["seed"] for row in values]
        axes[0].plot(xs, [row["final_acceptance_rate"] for row in values], marker="o", label=proposal_type)
        axes[1].plot(xs, [row["log_target_ess"] for row in values], marker="o", label=proposal_type)
        axes[2].plot(xs, [row["total_runtime_s"] for row in values], marker="o", label=proposal_type)
    axes[0].set_title("Final Acceptance by Seed")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Acceptance Rate")
    axes[1].set_title("Log Target ESS by Seed")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("ESS")
    axes[2].set_title("Runtime by Seed")
    axes[2].set_xlabel("Seed")
    axes[2].set_ylabel("Runtime (s)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=str)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(path) for path in args.run_dirs]
    rows = collect_run_summaries(run_dirs)
    write_rows_csv(rows, output_dir / "combined_summaries.csv")
    write_comparison_table(rows, output_dir / "comparison_table.csv")
    make_comparison_plot(rows, output_dir / "comparison_plot.png")


if __name__ == "__main__":
    main()
