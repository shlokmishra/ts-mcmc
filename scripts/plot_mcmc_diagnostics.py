from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ts_mcmc_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ts_mcmc_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcmc_diagnostics import load_trace_rows


def generate_run_plots(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    trace_path = run_dir / "traces" / "trace.csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trace_rows(trace_path)
    if not rows:
        raise ValueError(f"No trace rows found in {trace_path}")

    iterations = np.asarray([row["iteration"] for row in rows], dtype=float)
    proposal_type = rows[0]["proposal_type"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)

    axes[0, 0].plot(iterations, [row["log_likelihood"] for row in rows], lw=1.2)
    axes[0, 0].set_title("Log Likelihood")
    axes[0, 0].set_xlabel("Iteration")

    axes[0, 1].plot(iterations, [row["log_target"] for row in rows], lw=1.2)
    axes[0, 1].set_title("Log Target")
    axes[0, 1].set_xlabel("Iteration")

    axes[1, 0].plot(iterations, [row["cumulative_acceptance_rate"] for row in rows], lw=1.2)
    axes[1, 0].set_title("Cumulative Acceptance")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].plot(iterations, [row["rolling_acceptance_rate"] for row in rows], lw=1.2)
    axes[1, 1].set_title("Rolling Acceptance")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylim(0.0, 1.0)

    if proposal_type == "local_spr":
        hastings = [
            np.nan if row["log_hastings"] is None else float(row["log_hastings"])
            for row in rows
        ]
        axes[2, 0].plot(iterations, hastings, lw=1.0)
        axes[2, 0].axhline(0.0, color="black", lw=0.8, ls="--")
        axes[2, 0].set_title("Log Hastings")
    else:
        axes[2, 0].text(0.5, 0.5, "No Hastings term for baseline spr", ha="center", va="center")
        axes[2, 0].set_title("Log Hastings")
        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])
    axes[2, 0].set_xlabel("Iteration")

    axes[2, 1].plot(iterations, [row["root_time"] for row in rows], lw=1.2)
    axes[2, 1].set_title("Root Time")
    axes[2, 1].set_xlabel("Iteration")

    for ax in axes.ravel():
        ax.grid(alpha=0.2)

    figure_path = plots_dir / "trace_panels.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    if proposal_type == "local_spr":
        fig2, ax2 = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
        ax2.scatter(
            [row["forward_candidate_count"] for row in rows if row["forward_candidate_count"] is not None],
            [row["log_hastings"] for row in rows if row["log_hastings"] is not None],
            s=12,
            alpha=0.7,
        )
        ax2.set_xlabel("Forward Candidate Count")
        ax2.set_ylabel("Log Hastings")
        ax2.set_title("Hastings vs Local Candidate Count")
        ax2.grid(alpha=0.2)
        secondary_path = plots_dir / "hastings_scatter.png"
        fig2.savefig(secondary_path, dpi=180)
        plt.close(fig2)
    else:
        secondary_path = None

    return {
        "trace_panels": figure_path,
        "secondary": secondary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()
    generate_run_plots(args.run_dir)


if __name__ == "__main__":
    main()
