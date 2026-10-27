"""Small local qualification probes. Importing this module loads no ML runtime."""
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Callable

from poc.config import StudyConfig


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


CommandRunner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True)
class Measurement:
    value: int | float | str | bool | None
    reason: str | None = None


@dataclass(frozen=True)
class HardwareManifest:
    machine_architecture: str | None
    interpreter_architecture: str
    python_version: str
    macos_version: Measurement
    translated: Measurement
    native: bool
    chip: Measurement
    physical_memory_bytes: Measurement
    available_memory_bytes: Measurement
    pressure_free_percent: Measurement
    swap_used_bytes: Measurement
    disk_free_bytes: Measurement
    xcode_path: Measurement
    allocation_cap_bytes: Measurement
    command_statuses: tuple[dict, ...]
    available_memory_method: str = "vm_stat (free + inactive + speculative) pages; reclaimability estimate, not guaranteed allocation"


@dataclass(frozen=True)
class OperationResult:
    backend: str
    operation: str
    dtype: str
    status: str
    reason: str | None = None
    forward_max_abs_error: float | None = None
    gradient_max_abs_error: float | None = None
    repeat_max_abs_error: float | None = None
    forward_tolerance: float | None = None
    gradient_tolerance: float | None = None
    checksum_sha256: str | None = None
    elapsed_seconds: float | None = None
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OperationReport:
    operations: tuple[OperationResult, ...]
    reference: str = "PyTorch CPU float64; CPU transfer occurs before casting accelerator results to float64"
    tolerance_method: str = "64 * dtype epsilon * max(1, reference max abs), separately for output/gradient; repeats and FP64 suffix finite differences also reported; tiny-fixture screening only"


@dataclass(frozen=True)
class CalibrationReport:
    backend: str
    measured_tiny_forward_backward_seconds: float | None
    measured_repetitions: int
    target_presentations_per_update: int
    full_model_update_seconds: float | None = None
    full_model_unavailable_reason: str = "No pretrained model or representative context-512 microbatch measured; WP07 owns full-model calibration"
    extrapolation_method: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    hardware: HardwareManifest
    operations: OperationReport
    calibration: CalibrationReport | None
    dependencies: dict = field(default_factory=dict)

    @property
    def scientific_backend(self):
        for backend in ("torch_mps", "torch_cpu"):
            rows = [r for r in self.operations.operations if r.backend == backend and r.dtype == "float32"]
            if {r.operation for r in rows} >= set(OPERATIONS) and all(r.status == "passed" for r in rows):
                return backend
        return None

    @property
    def compute_path_status(self):
        rows = [r for r in self.operations.operations if r.backend == "mlx" and r.dtype == "float32"]
        plausible_mlx = {r.operation for r in rows} >= set(OPERATIONS) and all(r.status == "passed" for r in rows)
        resources = all(m.value is not None and m.value > 0 for m in (
            self.hardware.physical_memory_bytes, self.hardware.allocation_cap_bytes, self.hardware.disk_free_bytes))
        return "passed" if self.hardware.native and resources and self.scientific_backend and plausible_mlx else "unqualified"

    @property
    def wp00_status(self):
        return "pending_asset_provenance" if self.compute_path_status == "passed" else "unqualified"


OPERATIONS = ("attention", "rmsnorm", "silu", "gather_patch", "checkpoint", "shared_use", "frozen_suffix", "tiny_decoder")
BACKENDS = ("torch_cpu", "torch_mps", "mlx")


def to_dict(report):
    result = asdict(report)
    if isinstance(report, DoctorReport):
        result.update(wp00_status=report.wp00_status, compute_path_status=report.compute_path_status, scientific_backend=report.scientific_backend,
                      deferred_prerequisites=["immutable model/data revisions and terms", "actual teacher shapes and numerics", "quantizer formats", "native model export"],
                      scope="WP00 native tiny-operation path only; teacher/assets/quantization/export and later gates remain unqualified")
    return result


def command_runner(argv: Sequence[str]) -> CommandResult:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
        return CommandResult(tuple(argv), result.stdout, result.stderr, result.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(tuple(argv), "", str(exc), 127)


def recording_runner(directory: Path) -> CommandRunner:
    """Raw OS output may contain serial/UUID: directory must be an ignored artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    def run(argv):
        result = command_runner(argv)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", "_".join(argv))
        (directory / f"{name}.json").write_text(json.dumps(asdict(result), indent=2))
        return result
    return run


def discover_machine(runner: CommandRunner) -> HardwareManifest:
    commands = (("uname", "-m"), ("sw_vers",), ("sysctl", "-n", "hw.memsize"),
                ("system_profiler", "SPHardwareDataType", "SPDisplaysDataType"), ("df", "-h", "."),
                ("df", "-k", "."), ("xcode-select", "-p"), ("vm_stat",), ("memory_pressure", "-Q"),
                ("sysctl", "vm.swapusage"), ("sysctl", "-n", "sysctl.proc_translated"))
    results = {args: runner(args) for args in commands}
    def parse(args, pattern, convert=str):
        result = results[args]
        if result.returncode:
            return Measurement(None, f"command exit {result.returncode}: {result.stderr.strip() or 'no stderr'}")
        match = re.search(pattern, result.stdout)
        if not match:
            return Measurement(None, "counter absent or unrecognized command output")
        try:
            return Measurement(convert(match.group(1)))
        except (ValueError, OverflowError):
            return Measurement(None, "counter is malformed")
    machine = parse(("uname", "-m"), r"^\s*(\S+)").value
    translated = parse(("sysctl", "-n", "sysctl.proc_translated"), r"^\s*([01])\s*$", lambda s: s == "1")
    physical = parse(("sysctl", "-n", "hw.memsize"), r"^\s*(\d+)\s*$", int)
    vm = results[("vm_stat",)]
    page_size = re.search(r"page size of (\d+) bytes", vm.stdout)
    pages = [re.search(rf"Pages {kind}:\s*(\d+)\.", vm.stdout) for kind in ("free", "inactive", "speculative")]
    available = (Measurement(int(page_size[1]) * sum(int(m[1]) for m in pages))
                 if not vm.returncode and page_size and all(pages)
                 else Measurement(None, vm.stderr.strip() or "vm_stat page counts unavailable"))
    cap = Measurement(min(64 * 1024**2, available.value // 4)) if available.value is not None else Measurement(None, "measured headroom unavailable; allocation not authorized by this manifest")
    swap = parse(("sysctl", "vm.swapusage"), r"used\s*=\s*([\d.]+[KMG]?)", lambda s: int(float(s[:-1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[s[-1]]) if s[-1] in "KMG" else int(s))
    disk = parse(("df", "-k", "."), r"\n\S+\s+\d+\s+\d+\s+(\d+)", lambda s: int(s) * 1024)
    return HardwareManifest(machine, platform.machine(), platform.python_version(),
        parse(("sw_vers",), r"ProductVersion:\s*(\S+)"), translated,
        machine == "arm64" and platform.machine() == "arm64" and translated.value is False,
        parse(("system_profiler", "SPHardwareDataType", "SPDisplaysDataType"), r"Chip:\s*([^\n]+)"),
        physical, available, parse(("memory_pressure", "-Q"), r"free percentage:\s*(\d+)%", int),
        swap, disk, parse(("xcode-select", "-p"), r"^\s*([^\n]+)"), cap,
        tuple({"argv": list(args), "returncode": result.returncode} for args, result in results.items()))


def _unavailable(backend, reason):
    return [OperationResult(backend, op, "float32", "unavailable", reason) for op in OPERATIONS]


def _probe_backend(backend):
    try:
        import numpy as np
        from poc._probe_torch import execute as torch_execute
        if backend == "mlx":
            from poc._probe_mlx import execute
        else:
            import torch
            if backend == "torch_mps" and not torch.backends.mps.is_available():
                return _unavailable(backend, "MPS unavailable to this process; sandbox denial must be retried outside sandbox before concluding hardware incompatibility")
            if backend == "torch_mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
                return _unavailable(backend, "PYTORCH_ENABLE_MPS_FALLBACK=1 permits silent CPU fallback; unset it for qualification")
            execute = torch_execute
    except (ImportError, OSError, RuntimeError) as exc:
        return _unavailable(backend, f"runtime import/initialization: {type(exc).__name__}: {exc}")
    rows = []
    for dtype in ("float32", "float16", "bfloat16"):
        for operation in OPERATIONS:
            try:
                ref_y, ref_g, _ = torch_execute(operation, "float64", "cpu")
                begin = time.perf_counter()
                y, grad, details = execute(operation, dtype, "mps" if backend == "torch_mps" else "cpu")
                elapsed = time.perf_counter() - begin
                again_y, again_g, _ = execute(operation, dtype, "mps" if backend == "torch_mps" else "cpu")
                eps = {"float32": 2**-23, "float16": 2**-10, "bfloat16": 2**-7}[dtype]
                forward_error = float(np.max(np.abs(y - ref_y)))
                gradient_error = float(np.max(np.abs(grad - ref_g)))
                repeat_error = max(float(np.max(np.abs(y - again_y))), float(np.max(np.abs(grad - again_g))))
                forward_tol = 64 * eps * max(1.0, float(np.max(np.abs(ref_y))))
                gradient_tol = 64 * eps * max(1.0, float(np.max(np.abs(ref_g))))
                finite = all(math.isfinite(v) for v in (forward_error, gradient_error, repeat_error))
                details["reduced_precision_policy"] = "screening only; FP32 remains scientific policy" if dtype != "float32" else "scientific candidate"
                # Structural invariants are independently checked by execute(), raising on failure.
                passed = finite and forward_error <= forward_tol and gradient_error <= gradient_tol and repeat_error <= max(forward_tol, gradient_tol)
                rows.append(OperationResult(backend, operation, dtype, "passed" if passed else "failed",
                    None if passed else "reference/repeat tolerance exceeded or nonfinite result",
                    forward_error, gradient_error, repeat_error, forward_tol, gradient_tol,
                    sha256(y.astype("<f8").tobytes() + grad.astype("<f8").tobytes()).hexdigest(), elapsed, details))
            except (RuntimeError, TypeError, ValueError, AssertionError, NotImplementedError, AttributeError) as exc:
                rows.append(OperationResult(backend, operation, dtype, "unsupported" if isinstance(exc, (NotImplementedError, TypeError, AttributeError)) else "failed", f"{type(exc).__name__}: {exc}"))
    return rows


def probe_operations(backends: Sequence[str]) -> OperationReport:
    if any(backend not in BACKENDS for backend in backends):
        raise ValueError("unknown backend; choose torch_cpu, torch_mps, mlx")
    rows = []
    for backend in backends:
        if backend != "mlx":
            rows.extend(_probe_backend(backend))
            continue
        # Native Metal initialization can abort, bypassing Python exceptions.
        # Preserve the report by isolating MLX; never count a crashed child as pass.
        try:
            child = subprocess.run([sys.executable, "-m", "poc._probe_mlx"],
                                   capture_output=True, text=True, timeout=60, check=False)
            if child.returncode:
                rows.extend(_unavailable(backend, f"MLX probe process exit {child.returncode}: {child.stderr.strip()[-2000:]}; retry outside sandbox before concluding hardware incompatibility"))
            else:
                rows.extend(OperationResult(**row) for row in json.loads(child.stdout))
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            rows.extend(_unavailable(backend, f"MLX probe subprocess: {exc}"))
    return OperationReport(tuple(rows))


def calibrate_operations(config: StudyConfig) -> CalibrationReport:
    backend = config.numerics.scientific_backend
    from poc._probe_torch import execute
    device = "mps" if backend == "torch_mps" else "cpu"
    try:
        execute("tiny_decoder", "float32", device)  # warm-up excluded
        begin = time.perf_counter()
        for _ in range(3):
            execute("tiny_decoder", "float32", device)
        elapsed = (time.perf_counter() - begin) / 3
        return CalibrationReport(backend, elapsed, 3, config.training.target_valid_tokens_per_update)
    except (ImportError, RuntimeError) as exc:
        return CalibrationReport(backend, None, 0, config.training.target_valid_tokens_per_update, reason=str(exc))


def dependency_versions():
    names = ("torch", "mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "numpy", "scipy", "scikit-learn", "datasets", "safetensors", "PyYAML", "pytest", "psutil")
    result = {}
    for name in names:
        try:
            result[name] = {"version": metadata.version(name), "reason": None}
        except metadata.PackageNotFoundError:
            result[name] = {"version": None, "reason": "not installed"}
    return result


def run_doctor() -> DoctorReport:
    hardware = discover_machine(recording_runner(Path("artifacts/doctor-raw")))
    # Probes own <1 MiB tensors. Refuse allocation when headroom is unknown or too small.
    if hardware.allocation_cap_bytes.value is None or hardware.allocation_cap_bytes.value < 1024**2:
        operations = OperationReport(tuple(row for b in BACKENDS for row in _unavailable(b, "insufficient measured allocation headroom (requires 1 MiB)")))
    else:
        operations = probe_operations(BACKENDS)
    provisional = DoctorReport(hardware, operations, None)
    calibration = None
    if provisional.scientific_backend:
        from dataclasses import replace
        from poc.config import PROJECT_ROOT, load_config
        config = load_config(PROJECT_ROOT / "configs/pilot.yaml", phase="exploratory")
        config = replace(config, numerics=replace(config.numerics, scientific_backend=provisional.scientific_backend))
        calibration = calibrate_operations(config)
    return DoctorReport(hardware, operations, calibration, dependency_versions())


def write_doctor_report(path: Path, report: DoctorReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)
