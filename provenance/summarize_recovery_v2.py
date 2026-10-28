"""Safe numeric reduction of completed continuation measurements, no corpus I/O."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/continuation-v2'


def load(path):return json.loads(Path(path).read_text())


def main():
    pilot=load(ART/'recovery/result.json');frontier=load(ART/'frontier/result.json')
    legacy=load(ART/'legacy-frontier-retry1/result.json');replay=load(ART/'legacy-construction-replay/result.json')
    if any(r['status']!='complete_exploratory' for r in (pilot,frontier,legacy)):raise ValueError('incomplete scientific result')
    if replay['status']!='complete_construction_replay':raise ValueError('incomplete construction replay')
    if set(pilot['population']['fit_components'])&set(pilot['population']['validation_components']):raise ValueError('fit/dev overlap')
    if pilot['arms']['A']['initialized_digest']!=pilot['arms']['B']['initialized_digest']:raise ValueError('A/B initial state differs')
    curves=[];arms={}
    for name,arm in pilot['arms'].items():
        if arm['updates']!=16 or arm['targets']!=131072 or len(arm['observations'])!=5:raise ValueError('incomplete arm')
        before=arm['observations'][0];after=arm['observations'][-1]
        initial={r['component_id']:r['kl_sum']/r['valid_targets'] for r in before['validation']['rows']}
        final={r['component_id']:r['kl_sum']/r['valid_targets'] for r in after['validation']['rows']}
        for row in arm['fit_metrics']:
            weights=pilot['run_spec']['resolved_config']['config']['arms'][name]['mixture']
            reconstructed=sum(weights[s]*row['source_kl_sums'][s]/row['source_counts'][s] for s in weights)
            if abs(reconstructed-row['weighted_objective'])>1e-12:raise ValueError('source objective mismatch')
        arms[name]=dict(initial_kl=before['validation']['kl_per_target'],final_kl=after['validation']['kl_per_target'],
            relative_recovery=(before['validation']['kl_per_target']-after['validation']['kl_per_target'])/before['validation']['kl_per_target'],
            improved_components=sum(final[c]<initial[c] for c in initial),components=len(initial),
            step8_to16_kl_reduction=arm['observations'][-2]['validation']['kl_per_target']-after['validation']['kl_per_target'],
            initialized_digest=arm['initialized_digest'],fitted_digest=arm['fitted_digest'],
            unique_parameter_bytes=arm['unique_parameter_bytes'],cost=arm['cost'])
        for observation in arm['observations']:
            curves.append(dict(arm=name,step=observation['step'],fit_kl=observation['fit']['kl_per_target'],
                validation_kl=observation['validation']['kl_per_target'],validation_nll=observation['validation']['nll_per_target']))
    native=[];reference=None
    for row in [*frontier['rows'],*legacy['rows']]:
        n=sum(r['valid_targets'] for r in row['rows'])
        if n!=4088 or abs(sum(r['kl_sum'] for r in row['rows'])/n-row['kl_per_target'])>1e-12:raise ValueError('native reduction mismatch')
        coverage=sorted((r['component_id'],r['window_sha256'],r['valid_targets']) for r in row['rows'])
        if reference is None:reference=coverage
        if coverage!=reference:raise ValueError('native populations differ')
        native.append(dict(label=row['label'],payload_bytes=row['payload_bytes'],kl_per_target=row['kl_per_target'],
            nll_per_target=row['nll_per_target'],resident_bytes=row['resident']['resident_bytes'],
            load_seconds=row['native_load_seconds'],prefill_median_seconds=statistics.median(t['prefill_512_next_token_seconds'] for t in row['timing_samples']),
            decode16_median_seconds=statistics.median(t['decode_16_teacher_forced_seconds'] for t in row['timing_samples']),
            manifest_sha256=row['manifest_sha256'],execution_profile=row['execution_profile']))
    native.sort(key=lambda r:r['payload_bytes'])
    for row in native:
        row['dominated_in_measured_bytes_and_kl_by']=[q['label'] for q in native if q is not row
            and q['payload_bytes']<=row['payload_bytes'] and q['kl_per_target']<=row['kl_per_target']
            and (q['payload_bytes']<row['payload_bytes'] or q['kl_per_target']<row['kl_per_target'])]
    scores=load(ART/'recovered-scores.json');score_summary={}
    for phase in ('original','fresh_exploratory'):
        value=scores[phase]
        score_summary[phase]=dict(point_estimates=value['point_estimates'],formula_max_error=value['formula_max_abs_error'],
            bootstrap={k:{f:v for f,v in b.items() if f!='contrast_draws'} for k,b in value['bootstrap'].items()},
            scope='conditional descriptive component resampling on fixed one-update states; no scientific replication/equivalence')
    # The recorded maximum contains each parent observation AND supplied child
    # observation. Twice that maximum is an intentionally loose concurrent bound
    # for recorded worker counters, without assuming peaks occur together.
    groups=defaultdict(lambda:dict(memory=0,teacher=0,wall=0))
    ledger=next((ART/'recovery/ledger').glob('*/costs.jsonl'))
    for line in ledger.open():
        row=json.loads(line);cost=row['cost'];g=groups[row['attempt_id']]
        g['memory']=max(g['memory'],cost['memory_bytes'] or 0);g['teacher']+=cost['teacher_forward_calls'];g['wall']+=cost['wall_seconds']
    reconciled={}
    for name,count in [('frontier',680),('legacy_native',408)]:
        g=next(v for v in groups.values() if v['teacher']==count)
        reconciled[name]=dict(recorded_stage_max_bytes=g['memory'],conservative_sum_of_recorded_parent_and_child_bounds=2*g['memory'])
    summary=dict(schema_version=2,scope='exploratory, known historical data, no protected/capability/transfer/replication claim',
        arms=arms,native=native,historical_scores=score_summary,
        aggregate_new_campaign_cost=replay['aggregate_cost'],recorded_worker_memory_reconciliation=reconciled,
        memory_limitations='worker phase originally used max(parent,child); bounds above reconcile recorded counters, not absolute physical peak proof. Conversion repeated with summed parent/child accounting and byte-identical artifacts. Unified counters overlap.',
        construction_replay=dict(byte_identical=replay['byte_identical_manifest_and_all_files'],memory=replay['construction']['memory']),
        prior_historical_training_cost='separate repository-reported1800s main-global ceiling plus other old runs; not zero and not counted as newly repeated training',
        limitations=['one sentinel pair/seed,12fit/8previously exposed development components',
            '16updates do not establish convergence or adequate fitting;8-to16validation curves still improve',
            'A/B compares exposure allocation as well as source content under equal logical targets; no isolated donor-harm claim',
            'native table is a frozen subset of eligible dense artifacts, not an exhaustive newly tuned frontier',
            '3-context fixed-token timings are descriptive, not stable tails or production speed proof',
            'all new scientific/profiling/failed/replay work remains charged; generic engineering/tests are separately logged',
            'legacy zero FLOP/unique-token counters mean uninstrumented; coverage comes from target plans',
            'no protected confirmation, competent-task retention, repeated real merges, new distribution or second environment'])
    out=ROOT/'provenance/continuation-v2';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    for name,rows in [('learning-curves.csv',curves),('native-frontier.csv',native)]:
        with (out/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'arms':arms,'native_points':len(native),'new_updates':summary['aggregate_new_campaign_cost']['updates'],
        'new_execution_seconds':summary['aggregate_new_campaign_cost']['wall_seconds']},indent=2))


if __name__=='__main__':main()
