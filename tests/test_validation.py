"""The component CLI must execute controls before reporting a WP01 pass."""
import json
from pathlib import Path
import subprocess
import sys


def test_failed_control_cannot_publish_a_passing_wp01_record(tmp_path):
    root = Path(__file__).resolve().parents[1]
    control = tmp_path / 'test_intentionally_failed_control.py'
    control.write_text('def test_broken_control():\n    assert False, "controlled component failure"\n')
    script = ("import sys; import poc.validation as v; "
              f"v.COMPONENT_CONTROLS=({str(control)!r},); "
              "sys.exit(v.run_component_validation())")
    result = subprocess.run([sys.executable, '-c', script], cwd=root, text=True, capture_output=True)
    assert result.returncode == 4
    record = json.loads((root/'artifacts/validation/component.json').read_text())
    assert record['status'] == 'failed' and record['returncode'] != 0
    assert 'WP01 component passed' not in result.stdout


def test_component_cli_runs_real_controls_and_records_wp01():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'poc', 'validate', '--suite', 'component'],
                            cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    record_path = root / 'artifacts/validation/component.json'
    assert record_path.exists(), 'component command must persist its actual execution result'
    record = json.loads(record_path.read_text())
    assert record['status'] == 'passed'
    assert record['gate'] == 'WP01'
    assert record['wp00_status'] == 'pending_asset_provenance'
    assert record['returncode'] == 0 and 'passed' in record['stdout']
    assert all(len(digest) == 64 for digest in record['source_sha256'].values())
    assert 'WP01 component passed' in result.stdout


def test_teacher_cli_routes_explicit_local_device_and_output(tmp_path,monkeypatch):
    import poc.cli as cli
    try:
        import poc.teacher_validation as validation
    except ImportError:
        raise AssertionError('teacher validation route is missing')
    def execute(**kwargs):
        assert kwargs['device']=='cpu'
        assert kwargs['out']==tmp_path
        return 4
    monkeypatch.setattr(validation,'run_teacher_validation',execute)
    assert cli.main(['validate','--suite','teacher_and_harness','--device','cpu','--out',str(tmp_path)])==4


import pytest


@pytest.mark.parametrize('suite',['export','all'])
def test_unimplemented_validation_suites_keep_explicit_prerequisite_status(suite):
    from poc.cli import main
    assert main(['validate','--suite',suite])==3
