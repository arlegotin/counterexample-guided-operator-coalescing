"""Executable asset-free WP1 runner; scientific pilot remains gated."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

from poc.continuation import load_config, resolve_config, make_spec, pilot_blockers, installed_requirements, sha256, ROOT
from poc.continuation_budget import PilotLedger
from poc.continuation_controls import mechanism_controls, owned_fitting, numeric_fixture
from poc.continuation_evidence import export_numeric
from poc.contracts import CostRecord
from poc.records import canonical_json
from poc.runtime.torch_decoder import EXECUTION_PROFILE


def environment():
    import torch
    return dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        torch=str(torch.__version__),execution_device='cpu',execution_dtype='float64',
        execution_profile=EXECUTION_PROFILE,
        packages={d.metadata['Name']:d.version for d in sorted(importlib.metadata.distributions(),
            key=lambda d:d.metadata['Name'].lower())},
        installed_requirements=installed_requirements(),
        installed_requirements_sha256=hashlib.sha256(installed_requirements().encode()).hexdigest(),
        historical_wheel_source_sha256={name:sha256(ROOT/'provenance/dependency-locks'/name)
            for name in ('qualified-darwin-arm64-py312.requirements.txt','wheel-provenance.json','upstream-commits.json')},
        wheel_hashes='NOT RECONSTRUCTED from installed packages; native lock not requalified',
        linux_proc_visible=Path('/proc/cpuinfo').exists(),native_qualified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('validate-config','owned-controls','profile','pilot'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():parser.error('--out already exists; preserve receipts and use a new path')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();ledger=None
    receipt=dict(schema_version=1,mode=args.mode,execution_profile=EXECUTION_PROFILE,status='invalid',
        archive_gates={'download':'UNAVAILABLE','sha256_verification':'NOT EXECUTED',
            'internal_manifest':'NOT EXECUTED','mechanism':'BLOCKED','acquisition':'BLOCKED'},
        scientific_scope='WP1 owned CPU infrastructure only; historical WP04 remains INCONCLUSIVE')
    code=4
    try:
        if sys.version_info[:2]!=(3,12):raise ValueError('Python 3.12 required')
        resolved=resolve_config(load_config(args.config))
        receipt['resolved_config']=resolved
        receipt['environment']=environment()
        env_hash=hashlib.sha256(canonical_json(receipt['environment'])).hexdigest()
        spec=make_spec(resolved,mode=args.mode,environment_sha256=env_hash)
        receipt['run_spec']=spec.to_dict()
        if args.mode=='pilot':
            receipt.update(status='BLOCKED',blockers=pilot_blockers(resolved));code=3
        elif args.mode=='validate-config':
            receipt.update(status='valid_config_only',blockers=pilot_blockers(resolved));code=0
        else:
            directory=args.out.parent/(args.out.stem+'-private')
            # Refuse to replace any prior private checkpoints/cost ledger.
            directory.mkdir(exist_ok=False)
            ledger=PilotLedger(spec,directory/'ledger',observe_resources=True)
            import psutil
            rss=psutil.Process().memory_info().rss
            ledger.tracker.check(CostRecord(memory_bytes=rss))
            ledger.tracker.record(CostRecord(memory_bytes=rss))
            controls={}
            if args.mode=='owned-controls':
                captured={}
                ledger.shared('owned-mechanisms',CostRecord(teacher_forward_calls=24,student_forward_calls=24),
                    lambda:captured.update(mechanism_controls()))
                controls['mechanisms']=captured
            controls['fitting']=owned_fitting(resolved,ledger,env_hash,directory/'owned-fit')
            receipt['controls']=controls
            receipt['numeric_fixture']=export_numeric(numeric_fixture())
            receipt['query_cost_graph']=ledger.graph()
            receipt.update(status='complete_owned_only',
                profile_scope='tiny CPU fixture; cannot forecast pretrained 48-update pilot',
                memory_policy='kernel CPU process-lifetime RSS high-water; includes pre-run imports, not device memory')
            code=0
    except Exception as exc:
        receipt.update(status='invalid',error_type=type(exc).__name__,error=str(exc))
        code=4
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started
        if ledger is not None:
            try:
                prior=ledger.tracker.attempt_cost.wall_seconds
                ledger.tracker.record(CostRecord(wall_seconds=max(0,receipt['elapsed_seconds']-prior)))
                # The entire private output tree is live, including checkpoints,
                # ledgers and metadata; report high-water disk for the parent.
                disk=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
                import psutil
                ledger.tracker.record(CostRecord(disk_bytes=disk,memory_bytes=psutil.Process().memory_info().rss))
                receipt['aggregate_cost']=ledger.tracker.validate().to_dict()
            except Exception as exc:
                receipt.update(status='invalid',accounting_error=str(exc));code=4
        # Exclusive creation, finite JSON, no model weights or raw probe records.
        with args.out.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':receipt['status'],'receipt':str(args.out),'exit_code':code}))
    return code


if __name__=='__main__':raise SystemExit(main())
