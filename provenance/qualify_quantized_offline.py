"""Fresh-process native proof using actual immutable Task9 artifacts; no corpus I/O."""
import json
from pathlib import Path
import sys
import time
import resource
import tempfile
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from poc.contracts import Budget,CostRecord,RunSpec
from poc.budget import BudgetTracker
from poc.records import AttemptRecord,append_event
from poc.bundle import _sha,_canonical,verify_bundle,measure_bundle
from poc.quantization_stage import _prior_costs
from test_offline_bundle import run_offline_bundle
from test_export import ordinary_artifact_inference


def main(summary):
    summary=Path(summary);report=json.loads(summary.read_text())
    policy=json.loads((ROOT/'protocol/quantization-pilot-caps.json').read_text())
    prior=_prior_costs(ROOT/'artifacts/quantization',policy['synthetic_qualification']['prior_observed_wall_seconds'])
    if prior['wall_seconds']+60>policy['aggregate_caps']['wall_seconds']: raise RuntimeError('insufficient aggregate offline-proof headroom')
    raw=json.loads(Path(report['run_record']).read_text());spec=RunSpec(**{**raw['spec'],'budgets':Budget(**raw['spec']['budgets'])})
    attempt=AttemptRecord(raw['run_id'],raw['attempt_id'],Path(report['run_record']).parent,spec)
    tracker=BudgetTracker(spec.budgets,attempt);tracker.check(CostRecord(wall_seconds=60))
    best=next(r for r in report['rows'] if r['variant']==report['baseline_star'])
    anchor=report['anchor'];rtn8=next(r for r in report['rows'] if r['variant']=='frontier-rtn-b8-g64')
    outputs=[];started=time.monotonic();status='running'
    output=summary.parent/('native-offline-proof-'+uuid4().hex+'.json')
    try:
        for label,artifact in (('anchor',anchor),('baseline_star',best),('rtn8_fidelity',rtn8)):
            root=Path(artifact['artifact']);verify_bundle(root)
            before={p.relative_to(root).as_posix():_sha(p) for p in root.rglob('*') if p.is_file()}
            work=Path(tempfile.mkdtemp(prefix='cgoc-task9-offline-'+label+'-'))
            row=run_offline_bundle(root,work,backend='mlx',os_denial=True,blocked_source_dirs=('/private/tmp/cgoc-task8-floating-attempt5',))
            ordinary_artifact_inference(root,work,backend='mlx')
            after={p.relative_to(root).as_posix():_sha(p) for p in root.rglob('*') if p.is_file()}
            if before!=after: raise RuntimeError('offline proof mutated immutable payload')
            outputs.append({'label':label,'owned_workdir':str(work),'manifest_sha256':_sha(root/'manifest.json'),'payload_bytes':measure_bundle(root).payload_bytes,'ordinary_import_unchanged_payload':True,**row})
        status='passed'
    finally:
        seconds=time.monotonic()-started
        memory=max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss]+[r['peak_rss_bytes'] for r in outputs])
        tracker.record(CostRecord(wall_seconds=seconds,memory_bytes=memory,memory_unavailable_reason=None))
        proof={'status':status,'scope':'actual native offline generation; repository teacher/data/training and Task8 floating source denied; installed environment and only each artifact allowed; networking/download hooks denied','rows':outputs,'wall_seconds':seconds,'largest_process_peak_rss_bytes':memory,'student_forward_calls':len(outputs)*5,'source_sha256':{'tests/test_offline_bundle.py':_sha(ROOT/'tests/test_offline_bundle.py'),'tests/test_export.py':_sha(ROOT/'tests/test_export.py'),'provenance/qualify_quantized_offline.py':_sha(Path(__file__))},'original_stage_summary_sha256':_sha(summary)}
        output.write_bytes(_canonical(proof));append_event(attempt,{'kind':'native_offline_qualification','status':status,'evidence':str(output),'evidence_sha256':_sha(output)})
        print(json.dumps(proof))

if __name__=='__main__': main(sys.argv[1])
