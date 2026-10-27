"""Thin command adapter. Scientific commands await their explicit prerequisites."""
import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from poc.config import COMMANDS, ConfigError, load_config, validate_config
from poc.records import RecordError
from poc.assets import AssetError


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ConfigError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Local CGOC research harness (run doctor for WP00 qualification).",
                     epilog="Exits: 0 completed; 2 invalid config/contract; 3 unmet prerequisite; 4 runtime/integrity failure.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    configured = {"prepare", "baselines", "diagnose", "fit", "compile", "bench", "freeze", "run"}
    for command in sorted(COMMANDS):
        sub = subparsers.add_parser(command, help="pending implementation and qualification")
        if command in configured:
            sub.add_argument("--config", type=Path, required=True)
            sub.add_argument("--phase", choices=("exploratory", "confirmatory", "replication"), default="exploratory")
        if command == "doctor":
            sub.add_argument("--out", type=Path, required=True)
        if command == "validate":
            sub.add_argument("--suite", choices=("component", "teacher_and_harness", "export", "all"), required=True)
            sub.add_argument("--config", type=Path, default=Path('configs/135m.yaml'))
            sub.add_argument("--device", choices=("cpu", "mps"), default="mps")
            sub.add_argument("--out", type=Path, default=Path('artifacts/validation/teacher_and_harness'))
        if command == "validate":
            sub.add_argument("--bundle", type=Path)
            sub.add_argument("--backend", choices=("torch", "mlx"), default="torch")
        if command == "export":
            sources = sub.add_mutually_exclusive_group(required=True)
            sources.add_argument("--source-bundle", type=Path)
            sources.add_argument("--teacher-assets", type=Path)
            sub.add_argument("--teacher-manifest-sha256")
            sub.add_argument("--runtime-spec", type=Path)
            sub.add_argument("--out", type=Path, required=True)
        if command == "baselines":
            sub.add_argument("--stage", choices=("format_and_quant",), required=True)
        if command == "diagnose":
            sub.add_argument("--kind", choices=("single_merge",), required=True)
        if command == "fit":
            sub.add_argument("--method", required=True, choices=("T", "Q", "R", "RC", "CKA", "BS", "SV", "P", "H", "QCE", "M"))
        if command in {"export", "reproduce"}:
            sub.add_argument("--run", required=True)
            sub.add_argument("--attempt", required=True)
        if command in {"evaluate", "analyze"}:
            sub.add_argument("--freeze", type=Path, required=True)
        if command in {"bench", "evaluate"}:
            sub.add_argument("--split", required=True)
        if command == "report":
            sub.add_argument("--include-failures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        try:
            args = _parser().parse_args(argv)
        except SystemExit as exc:
            return int(exc.code)
        if args.command == "doctor":
            from poc.doctor import run_doctor, write_doctor_report
            report = run_doctor()
            write_doctor_report(args.out, report)
            print(f"doctor: WP00 compute-path {report.compute_path_status}; full gate={report.wp00_status}; scientific backend={report.scientific_backend}; report={args.out}")
            return 0 if report.compute_path_status == "passed" else 3
        if args.command == "validate" and args.suite == "component":
            from poc.validation import run_component_validation
            return run_component_validation()
        if args.command == "validate" and args.suite == "teacher_and_harness":
            from poc.teacher_validation import run_teacher_validation
            return run_teacher_validation(config_path=args.config, device=args.device, out=args.out)
        if args.command == "export":
            from poc.export import export_command
            return export_command(args)
        if args.command == "validate" and args.suite == "export":
            if args.bundle is not None:
                from poc.export import validate_artifact
                return validate_artifact(args.bundle, args.backend)
            from poc.validation import run_export_validation
            return run_export_validation(backend=args.backend)
        if args.command == "baselines" and args.stage == "format_and_quant":
            from poc.quantization_stage import run_format_and_quant
            return run_format_and_quant(load_config(args.config,phase=args.phase))
        if hasattr(args, "config") and args.command != "validate":
            config = load_config(args.config, phase=args.phase)
            validate_config(config, command=args.command)
        if args.command == "prepare" and config.preparation is not None:
            import json
            from poc.data.prepare import run_prepare
            summary = run_prepare(config)
            print(json.dumps({"status":summary["status"], "counts":summary["counts"],
                              "manifest":str(config.preparation.output_dir / "preparation.json")}, sort_keys=True))
            return 0
        print(f"{args.command}: not implemented; WP00 and command-specific evidence are unmet prerequisites", file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(f"invalid configuration/contract: {exc}", file=sys.stderr)
        return 2
    except (OSError, RecordError, AssetError, ValueError, RuntimeError) as exc:
        print(f"runtime/data-integrity failure: {exc}", file=sys.stderr)
        return 4
