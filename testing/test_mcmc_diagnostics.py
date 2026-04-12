import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcmc_diagnostics import RunConfig, load_trace_rows, run_logged_chain
from scripts.aggregate_mcmc_runs import collect_run_summaries, make_comparison_plot, write_comparison_table, write_rows_csv
from scripts.plot_mcmc_diagnostics import generate_run_plots
from scripts.run_mcmc_experiments import run_configs


def test_logged_run_writes_trace_and_metadata(tmp_path):
    result = run_logged_chain(
        tmp_path,
        RunConfig(
            proposal_type="spr",
            seed=3,
            sample_size=4,
            seq_length=6,
            steps=5,
            burn_in=1,
            spr_moves_per_step=1,
        ),
        make_plots=False,
    )

    assert result["metadata_path"].exists()
    assert result["trace_csv_path"].exists()
    assert result["summary_json_path"].exists()

    metadata = json.loads(result["metadata_path"].read_text())
    assert metadata["proposal_type"] == "spr"
    assert metadata["seed"] == 3


def test_local_spr_trace_includes_hastings_columns(tmp_path):
    result = run_logged_chain(
        tmp_path,
        RunConfig(
            proposal_type="local_spr",
            seed=4,
            sample_size=5,
            seq_length=6,
            steps=5,
            burn_in=1,
            spr_moves_per_step=1,
        ),
        make_plots=False,
    )

    rows = result["trace_csv_path"].read_text().splitlines()
    assert "log_hastings" in rows[0]
    assert "log_q_forward" in rows[0]
    assert "log_q_reverse" in rows[0]
    assert "time_move_node" in rows[0]
    assert "time_move_log_prior_delta" in rows[0]
    assert any("local_spr" in line for line in rows[1:])


def test_baseline_spr_trace_does_not_require_hastings_values(tmp_path):
    result = run_logged_chain(
        tmp_path,
        RunConfig(
            proposal_type="spr",
            seed=5,
            sample_size=4,
            seq_length=6,
            steps=4,
        ),
        make_plots=False,
    )

    with open(result["trace_csv_path"], newline="") as f:
        first_row = next(csv.DictReader(f))
    assert first_row["log_hastings"] == ""


def test_plotting_script_runs_on_tiny_log(tmp_path):
    result = run_logged_chain(
        tmp_path,
        RunConfig(
            proposal_type="local_spr",
            seed=6,
            sample_size=5,
            seq_length=6,
            steps=5,
        ),
        make_plots=False,
    )

    plot_paths = generate_run_plots(result["run_dir"])
    assert plot_paths["trace_panels"].exists()


def test_plotting_script_handles_nonfinite_trace_values(tmp_path):
    run_dir = tmp_path / "run"
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True)
    trace_path = traces_dir / "trace.csv"
    trace_path.write_text(
        "iteration,proposal_type,accepted,log_likelihood,log_target,log_alpha,log_hastings,log_q_forward,log_q_reverse,mutation_rate,root_time,cumulative_acceptance_rate,rolling_acceptance_rate,elapsed_s,detached_leaf,chosen_branch,forward_candidate_count,reverse_candidate_count,sampled_time,forward_candidates_json,reverse_candidates_json,time_move_accepted,mutation_move_accepted\n"
        "0,local_spr,True,inf,nan,-inf,inf,1.0,1.5,0.1,inf,1.0,1.0,0.2,3,4,5,6,7.0,[],[],True,False\n"
        "1,local_spr,False,-10.0,-8.0,-1.0,-0.5,1.0,0.5,0.1,2.0,0.5,0.5,0.4,3,4,5,6,2.5,[],[],False,False\n"
    )

    plot_paths = generate_run_plots(run_dir)
    assert plot_paths["trace_panels"].exists()
    assert plot_paths["secondary"].exists()


def test_experiment_runner_creates_expected_output_structure(tmp_path):
    configs = [
        RunConfig(proposal_type="spr", seed=1, sample_size=4, seq_length=6, steps=3),
        RunConfig(proposal_type="local_spr", seed=2, sample_size=5, seq_length=6, steps=3),
    ]
    results = run_configs(configs, tmp_path, make_plots=False)

    assert len(results) == 2
    for result in results:
        assert (result["run_dir"] / "traces" / "trace.csv").exists()
        assert (result["run_dir"] / "summaries" / "summary.json").exists()
        assert (result["run_dir"] / "metadata.json").exists()
        assert (result["run_dir"] / "run_status.json").exists()


def test_aggregation_helper_writes_outputs(tmp_path):
    results = run_configs(
        [
            RunConfig(proposal_type="spr", seed=10, sample_size=4, seq_length=6, steps=3),
            RunConfig(proposal_type="local_spr", seed=11, sample_size=5, seq_length=6, steps=3),
        ],
        tmp_path / "runs",
        make_plots=False,
    )

    rows = collect_run_summaries([result["run_dir"] for result in results])
    output_dir = tmp_path / "aggregate"
    output_dir.mkdir()
    write_rows_csv(rows, output_dir / "combined_summaries.csv")
    write_comparison_table(rows, output_dir / "comparison_table.csv")
    make_comparison_plot(rows, output_dir / "comparison_plot.png")

    assert (output_dir / "combined_summaries.csv").exists()
    assert (output_dir / "comparison_table.csv").exists()
    assert (output_dir / "comparison_plot.png").exists()


def test_load_trace_rows_handles_inf_and_nan(tmp_path):
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "iteration,proposal_type,accepted,log_likelihood,log_target,log_alpha,log_hastings,log_q_forward,log_q_reverse,mutation_rate,root_time,cumulative_acceptance_rate,rolling_acceptance_rate,elapsed_s,detached_leaf,chosen_branch,forward_candidate_count,reverse_candidate_count,sampled_time,forward_candidates_json,reverse_candidates_json,time_move_accepted,mutation_move_accepted\n"
        "0,local_spr,True,inf,nan,-inf,0.5,1.0,1.5,0.1,2.0,1.0,1.0,0.2,3,4,5,6,7.0,[],[],True,False\n"
    )

    rows = load_trace_rows(trace_path)
    assert rows[0]["log_likelihood"] == float("inf")
    assert str(rows[0]["log_target"]).lower() == "nan"
    assert rows[0]["log_alpha"] == float("-inf")


def test_run_logged_chain_writes_checkpoint(tmp_path):
    result = run_logged_chain(
        tmp_path,
        RunConfig(
            proposal_type="spr",
            seed=7,
            sample_size=4,
            seq_length=6,
            steps=5,
            checkpoint_every=1,
        ),
        make_plots=False,
    )

    checkpoint_path = result["run_dir"] / "checkpoints" / "latest.json"
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text())
    assert "iteration" in payload
    assert payload["iteration"] >= 0


def test_run_logged_chain_guard_aborts_cleanly(tmp_path):
    result = run_logged_chain(
        tmp_path,
        RunConfig(
            proposal_type="spr",
            seed=8,
            sample_size=4,
            seq_length=6,
            steps=5,
            guard_magnitude_limit=0.01,
            checkpoint_every=1,
        ),
        make_plots=True,
    )

    assert result["summary"]["run_status"] == "aborted_by_guard"
    assert result["summary"]["guard_reason"] is not None
    assert (result["run_dir"] / "checkpoints" / "guard_event.json").exists()
    assert (result["run_dir"] / "plots" / "plot_skipped.txt").exists()


def test_trace_every_downsamples_rows_but_keeps_root_summary(tmp_path):
    result = run_logged_chain(
        tmp_path,
        RunConfig(
            proposal_type="local_spr",
            seed=9,
            sample_size=5,
            seq_length=6,
            steps=20,
            trace_every=5,
            checkpoint_every=5,
        ),
        make_plots=False,
    )

    rows = load_trace_rows(result["trace_csv_path"])
    assert len(rows) == 4
    assert result["summary"]["trace_every"] == 5
    assert result["summary"]["trace_rows_written"] == 4
    assert result["summary"]["root_time_min"] is not None
    assert result["summary"]["root_time_max"] is not None
    assert result["summary"]["root_time_last"] is not None
