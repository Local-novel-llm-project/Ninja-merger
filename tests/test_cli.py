import pytest

from main import build_parser, parse_args
from modules.Services import merge_execution
from modules.Utils.merge_types import AppConfig, MergeRunnerOptions, MergeStepResult
from tests.helpers import make_merge_request


def test_build_parser_help_exposes_grouped_cli():
    help_text = build_parser().format_help()

    assert "Input / Output" in help_text
    assert "Execution" in help_text
    assert "Layer Filters" in help_text
    assert "--save-only-last-model" in help_text
    assert "--no-recurrent-mode" in help_text


def test_parse_args_supports_boolean_optional_recurrent_mode():
    options = parse_args(["--no-recurrent-mode", "--out-dir", "./out"])

    assert options.recurrent_mode is False
    assert options.out_dir == "./out"


def test_parse_args_rejects_dump_layers_with_dry_run():
    with pytest.raises(SystemExit):
        parse_args(["--dump-layers", "--dry-run"])


def test_run_merge_respects_recurrent_mode_and_save_last_only(monkeypatch):
    requests = [
        make_merge_request(name="first"),
        make_merge_request(name="second"),
    ]
    seen_calls = []

    monkeypatch.setattr(
        merge_execution,
        "load_config",
        lambda _: AppConfig(models=requests, use_scaling=False),
    )
    monkeypatch.setattr(merge_execution, "parse_layers", lambda value: value)
    monkeypatch.setattr(merge_execution, "_print_run_overview", lambda *args, **kwargs: None)
    monkeypatch.setattr(merge_execution, "_print_run_summary", lambda *args, **kwargs: None)

    def fake_process(
        index,
        total_configs,
        request,
        options,
        should_save_output,
        default_use_scaling,
        cli_include_layers,
        cli_exclude_layers,
        merger_factory,
        current_target_model,
        torch_dtype,
    ):
        seen_calls.append((index, should_save_output, current_target_model))
        return MergeStepResult(
            status="saved",
            target_model=f"merged-{index}",
            request_name=request.name,
            operation=request.operation,
        )

    monkeypatch.setattr(merge_execution, "_process_merge_request", fake_process)

    merge_execution.run_merge(
        MergeRunnerOptions(
            config="dummy.yaml",
            save_only_last_model=True,
            recurrent_mode=False,
        )
    )

    assert seen_calls == [
        (0, False, None),
        (1, True, None),
    ]


def test_run_merge_passes_previous_result_when_recurrent_mode_enabled(monkeypatch):
    requests = [
        make_merge_request(name="first"),
        make_merge_request(name="second"),
    ]
    seen_calls = []

    monkeypatch.setattr(
        merge_execution,
        "load_config",
        lambda _: AppConfig(models=requests, use_scaling=False),
    )
    monkeypatch.setattr(merge_execution, "parse_layers", lambda value: value)
    monkeypatch.setattr(merge_execution, "_print_run_overview", lambda *args, **kwargs: None)
    monkeypatch.setattr(merge_execution, "_print_run_summary", lambda *args, **kwargs: None)

    def fake_process(
        index,
        total_configs,
        request,
        options,
        should_save_output,
        default_use_scaling,
        cli_include_layers,
        cli_exclude_layers,
        merger_factory,
        current_target_model,
        torch_dtype,
    ):
        seen_calls.append(current_target_model)
        return MergeStepResult(
            status="saved",
            target_model=f"merged-{index}",
            request_name=request.name,
            operation=request.operation,
        )

    monkeypatch.setattr(merge_execution, "_process_merge_request", fake_process)

    merge_execution.run_merge(
        MergeRunnerOptions(
            config="dummy.yaml",
            recurrent_mode=True,
        )
    )

    assert seen_calls == [None, "merged-0"]
