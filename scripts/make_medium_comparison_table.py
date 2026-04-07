from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FIELDS = [
    "proposal_type",
    "sample_size",
    "seed",
    "final_acceptance_rate",
    "mean_rolling_acceptance_rate",
    "log_target_mean",
    "log_target_sd",
    "total_runtime_s",
    "mean_log_hastings",
]


def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summaries" / "summary.json").read_text())


def collect_rows(input_root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("proposal_")):
        summary = load_summary(run_dir)
        rows.append({field: summary.get(field) for field in FIELDS})
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows = collect_rows(Path(args.input_root))
    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
