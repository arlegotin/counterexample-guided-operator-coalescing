"""Bounded read-only CPU diagnosis on copied cost ledgers and directory metadata."""
import cProfile,gc,hashlib,io,json,shutil,pstats,tempfile,time
from pathlib import Path
from poc.budget import BudgetTracker
from poc.checkpoint import _spec
from poc.contracts import CostRecord
from poc.records import AttemptRecord
import provenance.measure_single_merge_precision as m


def main():
    started=time.monotonic();origin=m.ROOT;out=m.OUT;original=m.ORIGINAL
    latest=json.loads((out/'latest.json').read_text());actual=origin/latest['attempt'];plan=json.loads((out/'plan.json').read_text())
    files=[p for p in original.rglob('*') if p.is_file()];inventory=[{'path':str(p.relative_to(origin)),'bytes':p.stat().st_size} for p in files]
    hashes={};timings={};profiles={}
    with tempfile.TemporaryDirectory(prefix='task12-accounting-cpu-') as temp:
        copy=Path(temp)
        for path in files:
            target=copy/path.relative_to(origin);target.parent.mkdir(parents=True,exist_ok=True)
            if path.name in ('spec.json','costs.jsonl') or path.name.startswith('caps-'):
                shutil.copyfile(path,target);hashes[str(path.relative_to(origin))]=hashlib.sha256(target.read_bytes()).hexdigest()
            else:target.touch()  # exact directory/file inventory; stat-only placeholders, never content read
        copied_bytes=sum(p.stat().st_size for p in copy.rglob('*') if p.is_file())
        m.ROOT=copy;m.ORIGINAL=copy/original.relative_to(origin);m.OUT=copy/out.relative_to(origin)
        copied_attempt=copy/actual.relative_to(origin);spec=_spec(json.loads((copied_attempt.parents[1]/'spec.json').read_text()))
        tracker=BudgetTracker(spec.budgets,AttemptRecord(copied_attempt.parents[1].name,copied_attempt.name,copied_attempt,spec))
        rec=m.PrecisionRecorder(tracker,plan)
        initial=rec.aggregate().to_dict();warm=time.monotonic()
        for label,fn in [('aggregate_check',lambda:rec.check_aggregate(CostRecord())),('recursive_disk_stat',rec.disk)]:
            profile=cProfile.Profile();count=0;before=time.monotonic();profile.enable()
            while count<200 and time.monotonic()-started<25:
                fn();count+=1
            profile.disable();elapsed=time.monotonic()-before;timings[label]={'calls':count,'seconds':elapsed,'seconds_per_call':elapsed/count}
            stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(18);profiles[label]=stream.getvalue()
        assert rec.aggregate().to_dict()==initial
        result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_model_text_or_prediction_calls':True,'copied_ledgers_sha256':hashes,'frozen_directory_inventory':inventory,'stat_only_placeholders_explicit':True,'original_bytes':sum(x['bytes'] for x in inventory),'temporary_copy_bytes':copied_bytes,'combined_disk_peak_bound':sum(x['bytes'] for x in inventory)+copied_bytes,'setup_seconds':warm-started,'timings':timings,'profiles':profiles,'unchanged_aggregate_totals':initial,'temporary_duplicates_removed_after_profile':True}
    result['wall_seconds']=time.monotonic()-started
    if result['wall_seconds']>30:result['budget_overrun']=True
    (out/'accounting-profile.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','frozen_directory_inventory','copied_ledgers_sha256','unchanged_aggregate_totals')}))

if __name__=='__main__':main()
