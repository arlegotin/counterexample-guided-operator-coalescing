"""Protect missing measurements, architecture refusal and truthful gate reporting."""
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def doctor():
    assert importlib.util.find_spec("poc.doctor") is not None, "hardware doctor is not implemented"
    return importlib.import_module("poc.doctor")


def fake_runner(responses):
    def run(argv):
        stdout, stderr, code = responses.get(tuple(argv), ("", "unavailable in fixture", 1))
        return doctor().CommandResult(tuple(argv), stdout, stderr, code)
    return run


def machine_responses():
    return {
        ("uname", "-m"): ("arm64\n", "", 0),
        ("sw_vers",): ("ProductVersion:\t26.7.1\n", "", 0),
        ("sysctl", "-n", "hw.memsize"): ("38654705664\n", "", 0),
        ("sysctl", "-n", "sysctl.proc_translated"): ("0\n", "", 0),
        ("vm_stat",): ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages inactive: 200.\nPages speculative: 50.\n", "", 0),
        ("memory_pressure", "-Q"): ("System-wide memory free percentage: 45%\n", "", 0),
        ("sysctl", "vm.swapusage"): ("vm.swapusage: total = 8192.00M used = 2048.00M free = 6144.00M (encrypted)\n", "", 0),
        ("df", "-k", "."): ("Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/test 100000 10000 90000 10% /\n", "", 0),
        ("xcode-select", "-p"): ("/Applications/Xcode.app/Contents/Developer\n", "", 0),
        ("system_profiler", "SPHardwareDataType", "SPDisplaysDataType"): ("Chip: Apple M3 Pro\nSerial Number (system): SECRET\nHardware UUID: SECRET-UUID\n", "", 0),
    }


def test_discovery_reports_native_memory_and_safe_headroom_without_identifiers(monkeypatch):
    module = doctor()
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    report = module.discover_machine(fake_runner(machine_responses()))
    assert report.machine_architecture == "arm64"
    assert report.interpreter_architecture == "arm64"
    assert report.physical_memory_bytes.value == 38654705664
    assert report.available_memory_bytes.value == 5734400
    assert report.swap_used_bytes.value == 2147483648
    assert 0 < report.allocation_cap_bytes.value <= 1433600
    encoded = json.dumps(module.to_dict(report))
    assert "SECRET" not in encoded


def test_denied_and_malformed_counters_remain_null_with_reasons():
    module = doctor()
    responses = machine_responses()
    responses[("sysctl", "-n", "hw.memsize")] = ("", "Operation not permitted", 1)
    responses[("vm_stat",)] = ("unexpected output", "", 0)
    report = module.discover_machine(fake_runner(responses))
    assert report.physical_memory_bytes.value is None
    assert "not permitted" in report.physical_memory_bytes.reason
    assert report.available_memory_bytes.value is None
    assert report.available_memory_bytes.reason
    assert report.allocation_cap_bytes.value is None


def test_rosetta_discovery_prevents_native_qualification(monkeypatch):
    module = doctor()
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    responses = machine_responses()
    responses[("sysctl", "-n", "sysctl.proc_translated")] = ("1\n", "", 0)
    report = module.discover_machine(fake_runner(responses))
    assert report.translated.value is True
    assert report.native is False


def test_cpu_module_import_does_not_import_torch_or_mlx():
    doctor()
    result = subprocess.run([sys.executable, "-c", "import sys; import poc.doctor; assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_write_report_is_strict_json_and_cli_does_not_claim_missing_gate(tmp_path, monkeypatch):
    module = doctor()
    report = module.DoctorReport(module.discover_machine(fake_runner({})), module.OperationReport(()), None)
    out = tmp_path / "nested" / "doctor.json"
    module.write_doctor_report(out, report)
    content = json.loads(out.read_text())
    assert content["wp00_status"] == "unqualified"
    monkeypatch.setattr(module, "run_doctor", lambda: report)
    from poc.cli import main
    assert main(["doctor", "--out", str(out)]) == 3
    assert json.loads(out.read_text())["wp00_status"] == "unqualified"


def test_doctor_gate_requires_complete_operation_sets_and_resources(monkeypatch):
    module = doctor()
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    hardware = module.discover_machine(fake_runner(machine_responses()))
    complete = tuple(module.OperationResult(backend, op, "float32", "passed")
                     for backend in ("torch_cpu", "mlx") for op in module.OPERATIONS)
    report = module.DoctorReport(hardware, module.OperationReport(complete), None)
    assert report.compute_path_status == "passed"
    assert report.wp00_status == "pending_asset_provenance"
    assert report.scientific_backend == "torch_cpu"
    incomplete = tuple(row for row in complete if row.operation != "frozen_suffix")
    assert module.DoctorReport(hardware, module.OperationReport(incomplete), None).wp00_status == "unqualified"
