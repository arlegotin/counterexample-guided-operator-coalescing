"""One exact-census benchmark on the already frozen copied ledger inventory."""
import hashlib,json,shutil,tempfile,time
from pathlib import Path
from poc.budget import BudgetTracker
from poc.checkpoint import _spec
from poc.records import AttemptRecord
import provenance.measure_single_merge_precision as m


def main():
    started=time.monotonic();origin=m.ROOT;out=m.OUT;original=m.ORIGINAL
    before=json.loads((out/'accounting-profile.json').read_text());latest=json.loads((out/'latest.json').read_text());attempt=origin/latest['attempt'];plan=json.loads((out/'plan.json').read_text())
    timings={};results={}
    with tempfile.TemporaryDirectory(prefix='task12-disk-benchmark-') as temp:
        copy=Path(temp)
        for item in before['frozen_directory_inventory']:
            relative=item['path'];target=copy/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if relative in before['copied_ledgers_sha256']:
                shutil.copyfile(origin/relative,target)
                if hashlib.sha256(target.read_bytes()).hexdigest()!=before['copied_ledgers_sha256'][relative]:raise ValueError('frozen copied ledger changed')
            else:target.touch()
        copied_bytes=sum(p.stat().st_size for p in copy.rglob('*') if p.is_file());assert copied_bytes==before['temporary_copy_bytes']
        m.ROOT=copy;m.ORIGINAL=copy/original.relative_to(origin);m.OUT=copy/out.relative_to(origin)
        copied_attempt=copy/attempt.relative_to(origin);spec=_spec(json.loads((copied_attempt.parents[1]/'spec.json').read_text()))
        rec=m.PrecisionRecorder(BudgetTracker(spec.budgets,AttemptRecord(copied_attempt.parents[1].name,copied_attempt.name,copied_attempt,spec)),plan)
        assert rec.aggregate().to_dict()==before['unchanged_aggregate_totals']
        old=lambda:sum(p.stat().st_size for p in m.ORIGINAL.rglob('*') if p.is_file())
        assert old()==rec.disk()
        for label,fn in [('pathlib_original',old),('scandir_exact',rec.disk)]:
            start=time.monotonic();values=[fn() for _ in range(200)];elapsed=time.monotonic()-start
            assert len(set(values))==1;results[label]=values[0];timings[label]={'calls':200,'seconds':elapsed,'seconds_per_call':elapsed/200}
        assert len(set(results.values()))==1 and rec.aggregate().to_dict()==before['unchanged_aggregate_totals']
        result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'precision_source_sha256':hashlib.sha256((origin/'provenance/measure_single_merge_precision.py').read_bytes()).hexdigest(),'parent_profile_sha256':hashlib.sha256((out/'accounting-profile.json').read_bytes()).hexdigest(),'no_model_text_or_prediction_calls':True,'identical_frozen_inventory_count':len(before['frozen_directory_inventory']),'exact_same_bytes':results,'exact_same_aggregate_totals':True,'timings':timings,'no_caching_or_skipped_checks':True,'temporary_copy_bytes':copied_bytes,'combined_disk_peak_bound':before['combined_disk_peak_bound']}
    result['wall_seconds']=time.monotonic()-started
    if result['wall_seconds']>30:result['budget_overrun']=True
    (out/'disk-census-benchmark.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
