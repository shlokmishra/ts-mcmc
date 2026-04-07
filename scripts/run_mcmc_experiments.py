from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcmc_diagnostics import RunConfig, load_manifest, run_logged_chain, write_aggregate_csv


def run_configs(configs: list[RunConfig], output_root: str | Path, make_plots: bool) -> list[dict]:
    results = []
    for config in configs:
        results.append(run_logged_chain(output_root, config, make_plots=make_plots))
    return results


def build_single_config(args) -> RunConfig:
    return RunConfig(
        proposal_type=args.proposal_type,
        seed=args.seed,
        sample_size=args.sample_size,
        seq_length=args.seq_length,
        steps=args.steps,
        burn_in=args.burn_in,
        mutation_step_size=args.mutation_step_size,
        time_move=args.time_move,
        time_step_size=args.time_step_size,
        spr_local_k=args.spr_local_k,
        spr_moves_per_step=args.spr_moves_per_step,
        compute_gradients=args.compute_gradients,
        record=not args.no_record,
        print_every=None,
        acceptance_window=args.acceptance_window,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="benchmark_outputs/mcmc_runs")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--proposal-type", type=str, choices=["spr", "local_spr"], default="spr")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--mutation-step-size", type=float, default=0.1)
    parser.add_argument("--time-move", type=str, choices=["global", "local"], default="local")
    parser.add_argument("--time-step-size", type=float, default=1.0)
    parser.add_argument("--spr-local-k", type=int, default=None)
    parser.add_argument("--spr-moves-per-step", type=int, default=1)
    parser.add_argument("--acceptance-window", type=int, default=25)
    parser.add_argument("--compute-gradients", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        configs = load_manifest(args.manifest)
    else:
        configs = [build_single_config(args)]

    results = run_configs(configs, args.output_root, make_plots=args.make_plots)
    aggregate_path = Path(args.output_root) / "aggregate_summary.csv"
    write_aggregate_csv(results, aggregate_path)
    print(json.dumps({"runs": [str(result["run_dir"]) for result in results], "aggregate_csv": str(aggregate_path)}, indent=2))


if __name__ == "__main__":
    main()
