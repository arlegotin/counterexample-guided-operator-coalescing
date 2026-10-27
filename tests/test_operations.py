"""Protect operation correctness, explicit backend limitations and measured timing."""
import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest


def doctor():
    assert importlib.util.find_spec("poc.doctor") is not None, "operation probes are not implemented"
    return importlib.import_module("poc.doctor")


@pytest.mark.cpu
def test_cpu_forward_backward_probes_match_fp64_and_record_checksums():
    module = doctor()
    report = module.probe_operations(["torch_cpu"])
    required = {"attention", "rmsnorm", "silu", "gather_patch", "checkpoint", "shared_use", "frozen_suffix", "tiny_decoder"}
    rows = {row.operation: row for row in report.operations if row.dtype == "float32"}
    assert required <= rows.keys()
    for name in required:
        row = rows[name]
        assert row.status == "passed", (name, row.reason)
        assert row.forward_max_abs_error <= row.forward_tolerance
        assert row.gradient_max_abs_error <= row.gradient_tolerance
        assert row.repeat_max_abs_error == 0
        assert len(row.checksum_sha256) == 64
        assert row.elapsed_seconds > 0
    assert rows["frozen_suffix"].details["teacher_parameter_gradients_absent"]
    assert rows["frozen_suffix"].details["input_gradient_norm"] > 0
    assert rows["frozen_suffix"].details["finite_difference_max_abs_error"] < 1e-7
    assert rows["shared_use"].details["summed_gradient_max_abs_error"] < 1e-6
    assert rows["checkpoint"].details["recomputed_forward_calls"] > 1
    assert rows["attention"].details["future_edit_max_abs_error"] == 0
    assert rows["tiny_decoder"].details["tied_head"] is True


@pytest.mark.cpu
def test_reduced_precision_is_measured_separately_from_scientific_fp32():
    report = doctor().probe_operations(["torch_cpu"])
    rows = [row for row in report.operations if row.dtype in {"float16", "bfloat16"}]
    assert {row.dtype for row in rows} == {"float16", "bfloat16"}
    for row in rows:
        assert row.status in {"passed", "failed", "unsupported"}
        assert row.dtype != "float32"
        if row.status != "passed":
            assert row.reason


@pytest.mark.cpu
def test_unavailable_mps_is_limitation_not_cpu_pass(monkeypatch):
    module = doctor()
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    report = module.probe_operations(["torch_mps"])
    assert report.operations
    assert all(row.status == "unavailable" and row.reason for row in report.operations)
    assert all(row.elapsed_seconds is None for row in report.operations)


@pytest.mark.cpu
def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        doctor().probe_operations(["pretend_gpu"])


@pytest.mark.cpu
def test_calibration_distinguishes_measured_tiny_cost_from_unmeasured_model():
    from poc.config import load_config
    config = load_config(Path("configs/pilot.yaml"), phase="exploratory")
    config = replace(config, numerics=replace(config.numerics, scientific_backend="torch_cpu"))
    report = doctor().calibrate_operations(config)
    assert report.measured_tiny_forward_backward_seconds > 0
    assert report.full_model_update_seconds is None
    assert report.full_model_unavailable_reason
    assert report.target_presentations_per_update == 8192
    assert report.extrapolation_method is None


@pytest.mark.mps
def test_real_mps_operations():
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable in this process; run outside sandbox before concluding hardware incompatibility")
    rows = doctor().probe_operations(["torch_mps"]).operations
    assert rows
    assert all(row.status == "passed" for row in rows if row.dtype == "float32"), [(r.operation, r.reason) for r in rows]


@pytest.mark.mlx
def test_real_mlx_operations():
    if importlib.util.find_spec("mlx") is None:
        pytest.skip("MLX not installed; not a qualification pass")
    report = doctor().probe_operations(["mlx"])
    assert report.operations
    assert all(row.status == "passed" for row in report.operations if row.dtype == "float32"), [(r.operation, r.reason) for r in report.operations]


@pytest.mark.cpu
def test_mlx_native_abort_is_reported_as_limitation_in_parent_process(monkeypatch):
    module = doctor()
    import subprocess
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, -6, "", "Metal initialization aborted"))
    rows = module.probe_operations(["mlx"]).operations
    assert all(row.status == "unavailable" for row in rows)
    assert all("-6" in row.reason and "Metal initialization aborted" in row.reason for row in rows)
