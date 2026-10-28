"""Capture this final reporting fix only; preserve all historical verification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from poc.records import canonical_json
from poc.report import sha256_file

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'provenance/final-fix-verification-2026-10-28.json'


def bindings():
    paths=[ROOT/'.gitignore',ROOT/'pyproject.toml',ROOT/'docs/reproduction.md']
    for tree in ('poc','tests','configs','protocol','provenance'):
        paths.extend(p for p in (ROOT/tree).rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.yml','.json','.md','.txt'}
                     and '__pycache__' not in p.parts and p!=OUT)
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(set(paths))}


def command(argv):
    started=time.monotonic()
    env={**os.environ,'PATH':str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH']}
    result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    value={'command':' '.join(argv),'argv':argv,'cwd':str(ROOT),'exit_code':result.returncode,
           'stdout':result.stdout,'stderr':result.stderr,'wall_seconds':time.monotonic()-started}
    print(json.dumps(value),flush=True)
    if result.returncode:raise RuntimeError('final fix verification failed')
    return value


def main():
    base=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    prior_names=['provenance/task25-verification-2026-10-28.json','provenance/report-review-2026-10-28.json',
                 'provenance/report-evidence-2026-10-28.json','provenance/report-metric-sources-2026-10-28.json',
                 'provenance/report-bundle-census-2026-10-28.json','provenance/task25-downstream-audits-2026-10-28.json',
                 'results/regressions.parquet','REPORT.md','docs/analysis/2026-10-28-decision-appendix.md']
    evidence=json.loads((ROOT/'provenance/report-evidence-2026-10-28.json').read_text())
    prior_names+= [r['path'] for r in evidence['records'].values()]
    prior={name:sha256_file(ROOT/name) for name in sorted(set(prior_names))}
    source=bindings()
    commands=[command(['python','-m','pytest','tests/test_report.py','-q']),
              command(['python','-m','pytest']),
              command(['python','-m','poc','report','--include-failures'])]
    assert source==bindings(),'tested sources changed during verification'
    assert prior=={name:sha256_file(ROOT/name) for name in prior},'original scientific/quality evidence changed'
    manifest=json.loads((ROOT/'artifacts/release_manifest.json').read_text())
    assert all(sha256_file(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size==r['bytes'] for r in manifest['files'])
    summary=json.loads((ROOT/'results/summary.json').read_text())
    assert summary['outcome']=='diagnosed_inconclusive' and len(summary['claims'])==290 and len(summary['records'])==20
    value={'schema_version':1,'status':'passed','fix_base':base,'findings':['F1','F2'],
           'scope':'Reporting gate/schema fix only; no pretrained model, corpus/protected reader, native qualification, bundle census, download or experiment.',
           'source_sha256':source,'source_binding':'Current tested bytes on fix_base, including uncommitted fixes; source bytes checked unchanged across commands. New downstream proof is excluded from its own source binding and release inventory.',
           'commands':commands,'historical_evidence_sha256_unchanged':prior,
           'release_inventory_files_verified':len(manifest['files']),
           'derived_outputs':{p:sha256_file(ROOT/p) for p in ['REPORT.md','results/summary.json','results/regressions.parquet','artifacts/release_manifest.json']},
           'representation_change':'Scientific index/claims/metrics/records unchanged. Derived summary gates add category=unknown, record_id=null, supersedes=[] defaults; analysis adds validity_confirmed=null. These are API metadata, not relabeled scientific receipts.',
           'red_proof':{'command':'python -m pytest tests/test_report.py -q','exit_code':1,
                        'stdout':Path('/tmp/final-fix-both-red.txt').read_text(),
                        'scope':'Actual authored fixture output before gate/schema fixes; 33 failed,29 passed.'},
           'quality_provenance':'Original f07b3fb Task25 verification and c69ed9c review remain historical; this downstream proof does not modify or re-certify their source bindings.'}
    OUT.write_bytes(canonical_json(value))
    print(json.dumps({'verification':str(OUT.relative_to(ROOT)),'source_files':len(source),'prior_files_unchanged':len(prior)}),flush=True)


if __name__=='__main__':main()
