"""Capture final reporting/CPU checks with exact outputs and tested byte bindings.

No pretrained assets, corpus readers, native requalification, or scientific reruns.
The final manifest refresh follows the verification record so its hash is current.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from poc.records import canonical_json
from poc.report import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT/'provenance/task25-verification-2026-10-28.json'


def source_bindings():
    paths = [ROOT/'.gitignore', ROOT/'pyproject.toml', ROOT/'README.md', ROOT/'PROGRESS.md', ROOT/'DECISIONS.md',
             ROOT/'docs/reproduction.md', ROOT/'docs/terminal-report-narrative.md',
             ROOT/'docs/analysis/2026-10-28-decision-appendix.md']
    for tree in ['poc','tests','configs','protocol','provenance']:
        paths.extend(p for p in (ROOT/tree).rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.yml','.json','.md','.txt'}
                     and p != DESTINATION and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(set(paths))}


def run(argv):
    env = {**os.environ,'PATH':str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH']}
    started = time.monotonic()
    result = subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    record = {'argv':argv,'command':' '.join(argv),'cwd':str(ROOT), 'exit_code':result.returncode,
              'stdout':result.stdout,'stderr':result.stderr,'wall_seconds':time.monotonic()-started,
              'interpreter':str(ROOT/'.venv/bin/python'), 'scope':'software/reporting verification; not scientific cost or native qualification'}
    print(json.dumps({'command':record['command'],'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}))
    if result.returncode:
        raise RuntimeError('verification failed; retained output above')
    return record


def main():
    before = source_bindings()
    base = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    commands = [run(['python','-m','pytest','tests/test_report.py','-q']),
                run(['python','-m','pytest']),
                run(['python','-m','poc','report','--include-failures'])]
    if source_bindings()!=before:raise RuntimeError('tested source changed during verification')
    tdd_path=Path('/tmp/task25-red.txt')
    record={'schema_version':1,'status':'passed','tested_git_base':base,
            'source_binding_scope':'Exact current code/test/config/owned-provenance/document bytes; uncommitted additions identified by hashes, not falsely attributed to base commit. This record and generated report outputs are downstream, excluded from tested-source hashes to avoid cycles.',
            'source_sha256':before,'source_unchanged_during_commands':True,'commands':commands,
            'tdd_initial_red':{'command':'python -m pytest tests/test_report.py -q',
                              'failure':'20 authored cases failed because terminal classification/traceability module did not exist',
                              'stdout':tdd_path.read_text() if tdd_path.exists() else None,
                              'provenance':'actual initial pytest stdout captured in /tmp/task25-red.txt; later red-green cases are covered by final source/tests'},
            'native_and_assets':'Not rerun; retained prerequisite qualifications and independent unchanged-byte audits are in task25-downstream-audits-2026-10-28.json.',
            'no_model_or_data_work':'No pretrained-model loading, fitting, inference, protected reads or new downloads performed.'}
    DESTINATION.write_bytes(canonical_json(record))
    # The manifest can now hash the actual final verification record. The report
    # stdout must be identical; no scientific calculation or suite is repeated.
    refreshed=run(['python','-m','poc','report','--include-failures'])
    if refreshed['stdout'] != commands[-1]['stdout']:
        raise RuntimeError('report changed while refreshing downstream manifest')
    print(json.dumps({'verification':str(DESTINATION.relative_to(ROOT)), 'tested_files':len(before), 'manifest_refresh':'passed'}))


if __name__=='__main__':main()
