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


def finite_xy(rows: list[dict], y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        x = row["iteration"]
        y = row[y_key]
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def finite_scatter(rows: list[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        x = row[x_key]
        y = row[y_key]
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plot_series(ax, rows: list[dict], y_key: str, title: str, xlabel: str = "Iteration") -> None:
    xs, ys = finite_xy(rows, y_key)
    if xs.size == 0:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.plot(xs, ys, lw=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def generate_run_plots(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    trace_path = run_dir / "traces" / "trace.csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trace_rows(trace_path)
    if not rows:
        raise ValueError(f"No trace rows found in {trace_path}")

    proposal_type = rows[0]["proposal_type"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)

    plot_series(axes[0, 0], rows, "log_likelihood", "Log Likelihood")

    plot_series(axes[0, 1], rows, "log_target", "Log Target")

    plot_series(axes[1, 0], rows, "cumulative_acceptance_rate", "Cumulative Acceptance")
    axes[1, 0].set_ylim(0.0, 1.0)

    plot_series(axes[1, 1], rows, "rolling_acceptance_rate", "Rolling Acceptance")
    axes[1, 1].set_ylim(0.0, 1.0)

    if proposal_type == "local_spr":
        xs, ys = finite_xy(rows, "log_hastings")
        if xs.size == 0:
            axes[2, 0].text(0.5, 0.5, "No finite Hastings data", ha="center", va="center")
            axes[2, 0].set_xticks([])
            axes[2, 0].set_yticks([])
        else:
            axes[2, 0].plot(xs, ys, lw=1.0)
            axes[2, 0].axhline(0.0, color="black", lw=0.8, ls="--")
        axes[2, 0].set_title("Log Hastings")
    else:
        axes[2, 0].text(0.5, 0.5, "No Hastings term for baseline spr", ha="center", va="center")
        axes[2, 0].set_title("Log Hastings")
        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])
    axes[2, 0].set_xlabel("Iteration")

    plot_series(axes[2, 1], rows, "root_time", "Root Time")

    for ax in axes.ravel():
        ax.grid(alpha=0.2)

    figure_path = plots_dir / "trace_panels.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    if proposal_type == "local_spr":
        fig2, ax2 = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
        xs, ys = finite_scatter(rows, "forward_candidate_count", "log_hastings")
        if xs.size == 0:
            ax2.text(0.5, 0.5, "No finite scatter data", ha="center", va="center")
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            ax2.scatter(xs, ys, s=12, alpha=0.7)
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
