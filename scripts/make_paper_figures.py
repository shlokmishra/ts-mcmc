import csv
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", str(Path("benchmark_outputs/.mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/shlokmishra/code/ts-mcmc")
CAMPAIGN_JSON = ROOT / "benchmark_outputs/paper_n64_campaign/paper_n64_campaign.json"
LONGCHAIN_JSON = ROOT / "benchmark_outputs/longchain_diagnostics/longchain_diagnostics.json"
OUT_FIG = ROOT / "benchmark_outputs/paper_n64_campaign/figures"
OUT_TAB = ROOT / "benchmark_outputs/paper_n64_campaign/tables"


def load_json(path: Path):
    return json.loads(path.read_text())


def median(rows, fn):
    return statistics.median([fn(r) for r in rows])


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_structure_storage(rows):
    med_retained = median(rows, lambda r: r["retained_states"])
    med_intervals = median(rows, lambda r: r["compressed_intervals"])
    med_trees = median(rows, lambda r: r["trees_raw_bytes"] / 1024)
    med_fresh = median(rows, lambda r: r["fresh_trees_raw_bytes"] / 1024)
    med_newick = median(rows, lambda r: r["newick_raw_bytes"] / 1024)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    ax = axes[0]
    labels = ["Retained states", "Compressed intervals"]
    vals = [med_retained, med_intervals]
    colors = ["#1f4e5f", "#f08a24"]
    bars = ax.bar(labels, vals, color=colors, width=0.65)
    ax.set_title("A. Structural Compression")
    ax.set_ylabel("Count")
    ax.text(
        0.02,
        0.96,
        "Supports: structure vs fresh TS\nDoes not support: blanket storage claim",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom")

    ax = axes[1]
    size_labels = ["`.trees`", "Fresh TS", "Newick"]
    size_vals = [med_trees, med_fresh, med_newick]
    size_colors = ["#1f4e5f", "#c44e52", "#4c956c"]
    bars = ax.bar(size_labels, size_vals, color=size_colors, width=0.65)
    ax.set_yscale("log")
    ax.set_ylabel("Raw size (KB, log scale)")
    ax.set_title("A. Raw Storage Comparison")
    ax.text(
        0.02,
        0.96,
        "Supports: fresh-TS and raw-Newick comparisons\nDoes not support: gzip win claim",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    for bar, val in zip(bars, size_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom")

    base = OUT_FIG / "figure_1_structure_storage"
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_text(
        OUT_FIG / "figure_1_structure_storage.md",
        "\n".join(
            [
                "# Figure 1 Notes",
                "",
                "Supports:",
                "- A. Structural compression vs fresh-per-state TS",
                "- B. Raw storage vs Newick in the paper-facing n=64 regime",
                "",
                "Strength:",
                "- Strong enough to foreground.",
                "",
                "Does not support:",
                "- blanket gzip storage claims",
                "- any claim beyond the current n=64, L=1400 regime",
            ]
        ),
    )


def plot_runtime_queries(rows):
    med_vanilla = median(rows, lambda r: r["vanilla_ms_per_step"])
    med_record = median(rows, lambda r: r["record_ms_per_step"])
    med_overhead = median(rows, lambda r: r["record_overhead_pct"])
    med_bulk = median(rows, lambda r: r["compressed_interval_scan_ms"])
    med_bulk_newick = median(rows, lambda r: r["per_sample_newick_branch_length_ms"])
    med_rand_ts = median(rows, lambda r: r["random_access_ts_at_ms"])
    med_rand_newick = median(rows, lambda r: r["random_access_newick_ms"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    ax = axes[0]
    labels = ["Vanilla", "Record"]
    vals = [med_vanilla, med_record]
    colors = ["#7a9e9f", "#1f4e5f"]
    bars = ax.bar(labels, vals, color=colors, width=0.65)
    ax.set_ylabel("ms/step")
    ax.set_title("B. Runtime Overhead")
    ax.text(
        0.02,
        0.96,
        f"Supports: modest runtime claim\nCurrent overhead: {med_overhead:.1f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom")

    ax = axes[1]
    labels = ["Bulk TS", "Per-sample\nNewick", "Random\n`ts.at`", "Random\nNewick"]
    vals = [med_bulk, med_bulk_newick, med_rand_ts, med_rand_newick]
    colors = ["#f08a24", "#4c956c", "#6c757d", "#8d99ae"]
    bars = ax.bar(labels, vals, color=colors, width=0.65)
    ax.set_ylabel("ms")
    ax.set_title("C. Query Access Patterns")
    ax.text(
        0.02,
        0.96,
        "Supports: bulk analytics advantage\nDoes not support: blanket query-speed claim",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom")

    base = OUT_FIG / "figure_2_runtime_queries"
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_text(
        OUT_FIG / "figure_2_runtime_queries.md",
        "\n".join(
            [
                "# Figure 2 Notes",
                "",
                "Supports:",
                "- B. modest runtime overhead in the tuned n=64 regime",
                "- D. compressed bulk-query advantage",
                "",
                "Strength:",
                "- Runtime panel is strong enough to foreground for this regime.",
                "- Query panel is strong for bulk scans, not for generic random access.",
                "",
                "Does not support:",
                "- claims about all regimes",
                "- claims that tree sequences are faster for all query patterns",
            ]
        ),
    )


def plot_limitations(long_rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)

    for ax, metric, title in [
        (axes[0], "loglik_ess", "D. Long-Chain ESS"),
        (axes[1], "loglik_rough_burn_in_fraction", "D. Rough Burn-In Fraction"),
    ]:
        for n, color in [(64, "#1f4e5f"), (128, "#c44e52")]:
            xs = [500, 2000, 5000]
            ys = []
            for steps in xs:
                rs = [r for r in long_rows if r["sample_size"] == n and r["steps"] == steps]
                ys.append(median(rs, lambda r: r[metric]))
            ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"n={n}")
        ax.set_xscale("log")
        ax.set_xticks(xs, labels=[str(x) for x in xs])
        ax.set_title(title)
        ax.legend(frameon=False)

    axes[0].set_ylabel("ESS")
    axes[1].set_ylabel("Fraction")
    axes[0].text(
        0.02,
        0.96,
        "Supports: limitation framing\nDoes not support: n=128 as main path",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )

    base = OUT_FIG / "figure_3_limitations"
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_text(
        OUT_FIG / "figure_3_limitations.md",
        "\n".join(
            [
                "# Figure 3 Notes",
                "",
                "Supports:",
                "- D. limitation panel for current scaling boundary",
                "- explicit documentation that n=128 remains future work",
                "",
                "Strength:",
                "- Strong enough to use as a limitation figure or appendix panel.",
                "",
                "Does not support:",
                "- any claim that n=128 is ready now",
                "- any claim that further scaling is already solved",
            ]
        ),
    )


def make_tables(rows, long_rows):
    med = lambda fn: median(rows, fn)
    main_rows = [
        {
            "claim_class": "A. Structure vs fresh TS",
            "metric": "interval fraction / raw .trees to fresh TS",
            "result": f"{med(lambda r: r['interval_fraction']):.3f} / {med(lambda r: r['trees_raw_bytes']/r['fresh_trees_raw_bytes']):.3f}",
            "status": "Foreground",
            "note": "Strongest current result.",
        },
        {
            "claim_class": "B. Storage vs Newick",
            "metric": "raw .trees / raw Newick ; .trees.gz / Newick.gz",
            "result": f"{med(lambda r: r['trees_raw_bytes']/r['newick_raw_bytes']):.3f} ; {med(lambda r: r['trees_gz_bytes']/r['newick_gz_bytes']):.3f}",
            "status": "Cautious",
            "note": "Raw storage favorable; gzip near parity.",
        },
        {
            "claim_class": "C. Runtime overhead",
            "metric": "record overhead %",
            "result": f"{med(lambda r: r['record_overhead_pct']):.1f}",
            "status": "Foreground",
            "note": "For this tuned n=64 regime only.",
        },
        {
            "claim_class": "D. Analytics",
            "metric": "bulk TS / per-sample Newick ; random ts.at / random Newick",
            "result": f"{med(lambda r: r['compressed_interval_scan_ms']/r['per_sample_newick_branch_length_ms']):.3f} ; {med(lambda r: r['random_access_ts_at_ms']/r['random_access_newick_ms']):.3f}",
            "status": "Mixed",
            "note": "Bulk scans strong; random access not headline-worthy.",
        },
    ]
    limitation_rows = [
        {
            "topic": "n=128 scaling",
            "current_position": "Limitation / future work",
            "evidence": "5000-step n=128 long-chain diagnostics reach only modest log-likelihood ESS and still show larger burn-in fraction than n=64.",
        },
        {
            "topic": "Gzipped storage",
            "current_position": "Not a headline claim",
            "evidence": f"n=64 campaign .trees.gz / Newick.gz = {med(lambda r: r['trees_gz_bytes']/r['newick_gz_bytes']):.3f}.",
        },
        {
            "topic": "Random per-sample access",
            "current_position": "Secondary / limitation note",
            "evidence": f"n=64 random ts.at / Newick = {med(lambda r: r['random_access_ts_at_ms']/r['random_access_newick_ms']):.3f}.",
        },
    ]

    write_csv(OUT_TAB / "table_1_main_claims.csv", main_rows, list(main_rows[0].keys()))
    write_csv(OUT_TAB / "table_2_limitations.csv", limitation_rows, list(limitation_rows[0].keys()))

    def md_table(rows_):
        headers = list(rows_[0].keys())
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in rows_:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        return "\n".join(lines) + "\n"

    write_text(
        OUT_TAB / "table_1_main_claims.md",
        "# Table 1 Notes\n\nSupports the main n=64 paper-facing claims.\n\n"
        + md_table(main_rows),
    )
    write_text(
        OUT_TAB / "table_2_limitations.md",
        "# Table 2 Notes\n\nDocuments the current boundary of the paper-facing story.\n\n"
        + md_table(limitation_rows),
    )


def main():
    style()
    rows = load_json(CAMPAIGN_JSON)
    long_rows = load_json(LONGCHAIN_JSON)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    plot_structure_storage(rows)
    plot_runtime_queries(rows)
    plot_limitations(long_rows)
    make_tables(rows, long_rows)

    print(f"Wrote figures to {OUT_FIG}")
    print(f"Wrote tables to {OUT_TAB}")


if __name__ == "__main__":
    main()
