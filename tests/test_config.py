from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from poc.cli import main
from poc.config import ConfigError, load_config, validate_config


PILOT = Path(__file__).resolve().parents[1] / "configs/pilot.yaml"


def write_config(tmp_path, changes):
    data = yaml.safe_load(PILOT.read_text())
    for section, values in changes.items():
        if isinstance(values, dict):
            data[section].update(values)
        else:
            data[section] = values
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_pilot_is_typed_but_compile_blocks_unresolved_budget():
    config = load_config(PILOT, phase="exploratory")
    assert config.training.context_tokens == 512
    assert config.teacher.revision is None
    with pytest.raises(ConfigError, match="budget"):
        validate_config(config, command="compile")
    validate_config(config, command="prepare")


@pytest.mark.parametrize("changes", [
    {"typo": 1}, {"training": {"context_token": 512}},
    {"training": {"context_tokens": "512"}},
    {"training": {"context_tokens": True}},
    {"training": {"context_tokens": 0}},
    {"training": {"bank_learning_rate": float("nan")}},
    {"training": {"residual_learning_rate": float("inf")}},
    {"training": {"adam_betas": [0.9, 1.0]}},
    {"training": {"final_training_seeds": []}},
    {"search": {"natural_state_weight": 0.7}},
    {"search": {"teacher_query_budget": -1}},
    {"search": {"total_search_seconds": "2h"}},
    {"teacher": {"trust_remote_code": True}},
    {"teacher": {"revision": "main"}},
    {"teacher": {"weights_sha256": "abc"}},
    {"numerics": {"scientific_dtype": "float16"}},
    {"numerics": {"loss_temperature": 0}},
    {"model": {"ff_residual_rank": -8}},
    {"quantization": {"initial_bits": 3}},
    {"evaluation": {"payload_budget_bytes": 1.5}},
    {"evaluation": {"calibration_manifest": "x"}},
    {"budgets": {"wall_seconds": float("inf")}},
    {"budgets": {"teacher_forward_calls": False}},
])
def test_invalid_types_ranges_and_unknown_fields_fail(tmp_path, changes):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, changes), phase="exploratory")


@pytest.mark.parametrize("section,field,value", [
    ("numerics", "loss_temperature", 2.0),
    ("evaluation", "primary_context_tokens", 1024),
    ("model", "residual_scale", 0.5),
])
def test_loading_rejects_deviations_from_binding_primary_settings(tmp_path, section, field, value):
    with pytest.raises(ConfigError, match=field):
        load_config(write_config(tmp_path, {section: {field: value}}), phase="exploratory")


@pytest.mark.parametrize("phase", ["exploratory", "confirmatory", "replication"])
@pytest.mark.parametrize("section,field,value", [
    ("numerics", "loss_temperature", 2.0),
    ("evaluation", "primary_context_tokens", 1024),
    ("model", "residual_scale", 0.5),
])
def test_prepare_rejects_primary_setting_deviations_through_python_api(phase, section, field, value):
    config = load_config(PILOT, phase="exploratory")
    changed = replace(getattr(config, section), **{field: value})
    with pytest.raises(ConfigError, match=field):
        validate_config(replace(config, phase=phase, **{section: changed}), command="prepare")


def test_missing_and_duplicate_keys_fail(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ConfigError):
        load_config(path, phase="exploratory")
    path.write_text("schema_version: 1\n")
    with pytest.raises(ConfigError):
        load_config(path, phase="exploratory")


def test_phase_and_command_are_checked():
    with pytest.raises(ConfigError):
        load_config(PILOT, phase="confirmatory")
    with pytest.raises(ConfigError):
        validate_config(load_config(PILOT, phase="exploratory"), command="compiel")


def test_manifest_paths_use_repo_root_not_config_directory_or_cwd(tmp_path, monkeypatch):
    path = write_config(tmp_path, {"quantization": {"calibration_manifest": "artifacts/calibration.json"}})
    monkeypatch.chdir(tmp_path)
    config = load_config(path, phase="exploratory")
    assert config.quantization.calibration_manifest == PILOT.parents[1] / "artifacts/calibration.json"


@pytest.mark.parametrize("value", ["../escape.json", "/tmp/outside.json"])
def test_manifest_paths_cannot_escape_repository(tmp_path, value):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, {"evaluation": {"protected_manifest": value}}), phase="exploratory")


def test_compile_invalid_config_does_not_create_attempt(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["compile", "--config", str(PILOT)]) == 2
    assert "budget" in capsys.readouterr().err.lower()
    assert not (tmp_path / "artifacts").exists()


def test_unimplemented_command_reports_prerequisite(capsys):
    assert main(["prepare", "--config", str(PILOT)]) == 3
    assert "not implemented" in capsys.readouterr().err.lower()


def test_cli_integrity_failure_has_exit_four(tmp_path):
    assert main(["prepare", "--config", str(tmp_path / "missing.yaml")]) == 4


def test_cli_usage_errors_return_two():
    assert main(["compile"]) == 2


def resolved_config(tmp_path):
    return load_config(write_config(tmp_path, {
        "teacher": {"revision": "a" * 40, "weights_sha256": "b" * 64, "tokenizer_revision": "c" * 40},
        "training": {"final_target_presentations": 100, "final_training_seeds": [17]},
        "search": {"teacher_query_budget": 3, "total_search_seconds": 30, "final_topology_policy": "fixed_budget"},
        "quantization": {"calibration_manifest": "artifacts/calibration.json", "validated_formats": ["4bit_group64"]},
        "evaluation": {"payload_budget_bytes": 10000, "protected_manifest": "artifacts/protected.json",
                       "independent_cluster_count": 100, "timing_pair_count": 10, "analysis_freeze_sha256": "d" * 64},
        "budgets": {"updates": 10, "valid_target_presentations": 100, "teacher_forward_calls": 3,
                    "suffix_backward_calls": 3, "student_backward_calls": 10, "search_proposals": 3,
                    "search_seconds": 30, "wall_seconds": 60, "disk_bytes": 10000, "memory_bytes": 10000},
    }), phase="exploratory")


@pytest.mark.parametrize("command,section,field", [
    ("compile", "training", "final_target_presentations"),
    ("compile", "training", "final_training_seeds"),
    ("compile", "search", "final_topology_policy"),
    ("compile", "search", "teacher_query_budget"),
    ("compile", "search", "total_search_seconds"),
    ("compile", "evaluation", "payload_budget_bytes"),
    ("fit", "training", "final_target_presentations"),
    ("fit", "teacher", "revision"),
    ("baselines", "quantization", "calibration_manifest"),
    ("freeze", "evaluation", "protected_manifest"),
])
def test_consequential_nulls_block_each_command(tmp_path, command, section, field):
    config = resolved_config(tmp_path)
    validate_config(config, command=command)
    bad_section = replace(getattr(config, section), **{field: None})
    with pytest.raises(ConfigError, match=field):
        validate_config(replace(config, **{section: bad_section}), command=command)


def test_freeze_does_not_require_its_own_output_digest(tmp_path):
    config = resolved_config(tmp_path)
    validate_config(replace(config, evaluation=replace(config.evaluation, analysis_freeze_sha256=None)), command="freeze")


def test_python_api_cannot_bypass_strict_types(tmp_path):
    config = resolved_config(tmp_path)
    with pytest.raises(ConfigError):
        validate_config(replace(config, training=replace(config.training, context_tokens=True)), command="fit")


def test_relative_config_path_resolves_against_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config(Path("configs/pilot.yaml"), phase="exploratory").study == "cgoc_smol135"


def test_symlink_manifest_cannot_escape_repository(tmp_path):
    link = PILOT.parents[1] / "configs" / "test-outside-link"
    link.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(ConfigError):
            load_config(write_config(tmp_path, {"evaluation": {"protected_manifest": "configs/test-outside-link/x"}}), phase="exploratory")
    finally:
        link.unlink()
