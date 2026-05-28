"""Tests for the top-level neurodags CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from neurodags.cli import main


def test_tui_subcommand_launches_app():
    with (
        patch("neurodags.cli.NeuroDagsApp", create=True),
        patch("neurodags.cli._cmd_tui") as cmd,
    ):
        cmd.return_value = 0
        assert main(["tui", "pipeline.yml", "--datasets", "datasets.yml"]) == 0


def test_validate_prints_summary(tmp_path, capsys):
    pipe = tmp_path / "pipeline.yml"
    data = tmp_path / "datasets.yml"
    pipe.write_text(
        "datasets: datasets.yml\n"
        "mount_point: null\n"
        "DerivativeDefinitions:\n"
        "  BasicPrep: {}\n"
        "DerivativeList:\n"
        "  - BasicPrep\n"
    )
    data.write_text(
        "demo:\n"
        "  name: Demo\n"
        f"  file_pattern: {tmp_path}/**/*.vhdr\n"
        f"  derivatives_path: {tmp_path}/derivatives\n"
    )

    assert main(["validate", str(pipe)]) == 0
    out = capsys.readouterr().out
    assert "datasets: demo" in out
    assert "derivatives_enabled: BasicPrep" in out


def test_run_uses_derivative_list_when_not_explicit(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.orchestrators.load_configuration", return_value=cfg),
        patch("neurodags.orchestrators.iterate_derivative_pipeline") as iterate,
    ):
        assert main(["run", "pipeline.yml"]) == 0
        called = [call.kwargs["derivative"] for call in iterate.call_args_list]
        assert set(called) == set(cfg["DerivativeList"])


def test_run_with_explicit_derivative(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.orchestrators.load_configuration", return_value=cfg),
        patch("neurodags.orchestrators.iterate_derivative_pipeline") as iterate,
    ):
        assert main(["run", "pipeline.yml", "--derivative", "BasicPrep"]) == 0
        iterate.assert_called_once()
        assert iterate.call_args.kwargs["derivative"] == "BasicPrep"


def test_dry_run_saves_combined_dataframe(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    fake_df = pd.DataFrame({"file_path": ["a"], "plan": [[]]})
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.orchestrators.load_configuration", return_value=cfg),
        patch("neurodags.orchestrators.iterate_derivative_pipeline", return_value=fake_df),
        patch("neurodags.cli._save_dataframe") as save_df,
    ):
        save_df.return_value = Path("dry_run_results.csv")
        assert main(["dry-run", "pipeline.yml", "--derivative", "BasicPrep"]) == 0
        save_df.assert_called_once()


def test_dataframe_command_passes_arguments(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    fake_df = pd.DataFrame({"x": [1]})
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.build_derivative_dataframe", return_value=fake_df) as build_df,
    ):
        assert (
            main(
                [
                    "dataframe",
                    "pipeline.yml",
                    "--include-derivative",
                    "BandPowerMean",
                    "--format",
                    "long",
                ]
            )
            == 0
        )
        _, kwargs = build_df.call_args
        assert kwargs["include_derivatives"] == ["BandPowerMean"]
        assert kwargs["output_format"] == "long"


def test_dataframe_njobs_passed_to_build_derivative_dataframe(dummy_pipeline):
    cfg = dummy_pipeline["config"]
    fake_df = pd.DataFrame({"x": [1]})
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.build_derivative_dataframe", return_value=fake_df) as build_df,
    ):
        assert main(["dataframe", "pipeline.yml", "--n-jobs", "4"]) == 0
        _, kwargs = build_df.call_args
        assert kwargs.get("n_jobs") == 4


def test_dag_pipeline_prints_mermaid(capsys):
    with (
        patch("neurodags.cli._load_pipeline_config", return_value={"DerivativeDefinitions": {}}),
        patch("neurodags.cli.pipeline_to_mermaid", return_value="graph TD\nA-->B"),
    ):
        assert main(["dag", "pipeline.yml"]) == 0
        assert "graph TD" in capsys.readouterr().out


def test_dag_derivative_html_exports():
    cfg = {"DerivativeDefinitions": {"BandPower": {"nodes": []}}}
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.derivative_to_html", return_value=Path("bandpower.html")) as export,
    ):
        assert main(["dag", "pipeline.yml", "--derivative", "BandPower", "--html", "out.html"]) == 0
        export.assert_called_once()


def test_view_launches_visualization_subprocess():
    with patch("neurodags.cli.subprocess.Popen") as popen:
        assert main(["view", "result.nc"]) == 0
        popen.assert_called_once()


# ---------------------------------------------------------------------------
# count command
# ---------------------------------------------------------------------------


def test_count_prints_unique_file_count(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = pd.DataFrame({"file_path": ["a", "a", "b"], "other": [1, 2, 3]})
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        assert main(["count-inputs", "pipeline.yml"]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_count_prints_zero_on_empty_result(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()),
    ):
        assert main(["count-inputs", "pipeline.yml"]) == 0
    assert capsys.readouterr().out.strip() == "0"


# ---------------------------------------------------------------------------
# slurm-script command
# ---------------------------------------------------------------------------


def _make_slurm_config(derivatives: list[str]) -> dict:
    defs = {d: {"nodes": [{"id": 0, "derivative": "SourceFile"}]} for d in derivatives}
    return {
        "datasets": {},
        "mount_point": None,
        "DerivativeDefinitions": defs,
        "DerivativeList": derivatives,
    }


def test_slurm_script_per_file_contains_derivative_names(capsys):
    cfg = _make_slurm_config(["BasicPrep", "Spectrum"])
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "per-file"]) == 0
    out = capsys.readouterr().out
    assert "BasicPrep" in out
    assert "Spectrum" in out
    assert "SLURM_ARRAY_TASK_ID" in out


def test_slurm_script_flat_contains_n_derivatives(capsys):
    cfg = _make_slurm_config(["A", "B", "C"])
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "flat"]) == 0
    out = capsys.readouterr().out
    assert "N_DERIVATIVES=3" in out
    assert '"A"' in out and '"B"' in out and '"C"' in out


def test_slurm_script_chained_chains_dependencies(capsys):
    cfg = _make_slurm_config(["BasicPrep", "Spectrum", "BandPower"])
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "chained"]) == 0
    out = capsys.readouterr().out
    assert "--dependency=afterok:$JOB1" in out
    assert "--dependency=afterok:$JOB2" in out
    assert "submit_pipeline.sh" in out
    assert "run_one_derivative.sh" in out


def test_slurm_script_chained_writes_two_files(tmp_path):
    cfg = _make_slurm_config(["BasicPrep", "Spectrum"])
    submit_path = tmp_path / "submit.sh"
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "chained", "--output", str(submit_path)]) == 0
    assert submit_path.exists()
    assert (tmp_path / "run_one_derivative.sh").exists()
    assert "BasicPrep" in submit_path.read_text()


def test_slurm_script_per_file_writes_file(tmp_path):
    cfg = _make_slurm_config(["BasicPrep"])
    out_path = tmp_path / "run_array.sh"
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "per-file", "--output", str(out_path)]) == 0
    assert out_path.exists()
    assert "SLURM_ARRAY_TASK_ID" in out_path.read_text()


def test_slurm_script_respects_topo_order(capsys):
    # Spectrum depends on BasicPrep — BasicPrep must appear first in pattern 3
    cfg = {
        "datasets": {},
        "mount_point": None,
        "DerivativeDefinitions": {
            "Spectrum": {"nodes": [{"id": 0, "derivative": "BasicPrep.fif"}]},
            "BasicPrep": {"nodes": [{"id": 0, "derivative": "SourceFile"}]},
        },
        "DerivativeList": ["Spectrum", "BasicPrep"],
    }
    with patch("neurodags.cli._load_pipeline_config", return_value=cfg):
        assert main(["slurm-script", "pipeline.yml", "--pattern", "chained"]) == 0
    out = capsys.readouterr().out
    assert out.index("BasicPrep") < out.index("Spectrum")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

def _make_plan(*, cached: bool, has_error: bool, error_path: str | None = None) -> list:
    return [
        {
            "id": "final",
            "kind": "derivative_output",
            "name": "BasicPrep",
            "prefix": "/out/file@BasicPrep",
            "cached": cached,
            "paths": ["/out/file@BasicPrep.fif"] if cached else [],
            "has_error_marker": has_error,
            "error_path": error_path,
        }
    ]


def _status_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_status_no_files_exits_0(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()),
    ):
        assert main(["status", "pipeline.yml"]) == 0
    assert "No files" in capsys.readouterr().out


def test_status_prints_summary_table(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "a.vhdr", "plan": _make_plan(cached=True, has_error=False)},
        {"derivative": "BasicPrep", "file_path": "b.vhdr", "plan": _make_plan(cached=False, has_error=False)},
        {"derivative": "BasicPrep", "file_path": "c.vhdr", "plan": _make_plan(cached=False, has_error=True, error_path="c@BasicPrep.error")},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        rc = main(["status", "pipeline.yml"])
    out = capsys.readouterr().out
    assert rc == 1  # has errors
    assert "BasicPrep" in out
    assert "1" in out  # done count
    assert "Total" in out


def test_status_exit_0_when_complete(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "a.vhdr", "plan": _make_plan(cached=True, has_error=False)},
        {"derivative": "BasicPrep", "file_path": "b.vhdr", "plan": _make_plan(cached=True, has_error=False)},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        assert main(["status", "pipeline.yml"]) == 0


def test_status_exit_1_when_missing(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "a.vhdr", "plan": _make_plan(cached=True, has_error=False)},
        {"derivative": "BasicPrep", "file_path": "b.vhdr", "plan": _make_plan(cached=False, has_error=False)},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        assert main(["status", "pipeline.yml"]) == 1


def test_status_list_errors_prints_file_paths(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "/data/sub-01.vhdr", "plan": _make_plan(cached=False, has_error=True, error_path="/out/sub-01@BasicPrep.error")},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        main(["status", "pipeline.yml", "--list-errors"])
    out = capsys.readouterr().out
    assert "/data/sub-01.vhdr" in out
    assert "/out/sub-01@BasicPrep.error" in out


def test_status_list_missing_prints_file_paths(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "/data/sub-02.vhdr", "plan": _make_plan(cached=False, has_error=False)},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        main(["status", "pipeline.yml", "--list-missing"])
    out = capsys.readouterr().out
    assert "/data/sub-02.vhdr" in out


def test_status_no_list_errors_hint_shown(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "x.vhdr", "plan": _make_plan(cached=False, has_error=True)},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        main(["status", "pipeline.yml"])
    out = capsys.readouterr().out
    assert "--list-errors" in out


def test_status_format_json(dummy_pipeline, capsys):
    import json

    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "a.vhdr", "plan": _make_plan(cached=True, has_error=False)},
        {"derivative": "BasicPrep", "file_path": "b.vhdr", "plan": _make_plan(cached=False, has_error=False)},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df),
    ):
        rc = main(["status", "pipeline.yml", "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert rc == 1  # missing → incomplete
    assert data["complete"] is False
    assert data["grand_total"]["missing"] == 1
    assert data["grand_total"]["done"] == 1
    assert "BasicPrep" in data["derivatives"]


def test_status_njobs_passed_to_run_pipeline(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    fake_df = _status_df([
        {"derivative": "BasicPrep", "file_path": "a.vhdr", "plan": _make_plan(cached=True, has_error=False)},
    ])
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=fake_df) as mock_run,
    ):
        main(["status", "pipeline.yml", "--n-jobs", "4"])
        _, kwargs = mock_run.call_args
        assert kwargs.get("n_jobs") == 4


def test_cmd_run_passes_path_string_to_run_pipeline(dummy_pipeline):
    """_cmd_run must pass args.config (path str) not the loaded dict to run_pipeline.

    If the dict is passed instead, new_definitions and datasets relative paths
    in the YAML are resolved against cwd rather than the pipeline file's location.
    """
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline") as mock_run,
    ):
        mock_run.return_value = None
        main(["run", "pipeline.yml", "--derivative", "BasicPrep"])
        called_config = mock_run.call_args.kwargs.get(
            "pipeline_configuration", mock_run.call_args.args[0] if mock_run.call_args.args else None
        )
        assert called_config == "pipeline.yml", (
            "run_pipeline must receive the path string, not the parsed dict"
        )


# ---------------------------------------------------------------------------
# --log-level / --log-file global flags
# ---------------------------------------------------------------------------


def test_log_level_flag_passed_to_configure_logging(dummy_pipeline, capsys):
    cfg = dummy_pipeline["config"]
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()),
        patch("neurodags.cli.configure_logging") as mock_log,
    ):
        assert main(["--log-level", "WARNING", "count-inputs", "pipeline.yml"]) == 0
        mock_log.assert_called_once()
        _, kwargs = mock_log.call_args
        assert kwargs.get("level") == "WARNING"


def test_log_file_flag_passed_to_configure_logging(dummy_pipeline, tmp_path):
    cfg = dummy_pipeline["config"]
    log_path = str(tmp_path / "run.jsonl")
    with (
        patch("neurodags.cli._load_pipeline_config", return_value=cfg),
        patch("neurodags.cli.run_pipeline", return_value=pd.DataFrame()),
        patch("neurodags.cli.configure_logging") as mock_log,
    ):
        assert main(["--log-file", log_path, "count-inputs", "pipeline.yml"]) == 0
        _, kwargs = mock_log.call_args
        assert kwargs.get("log_file") == log_path
