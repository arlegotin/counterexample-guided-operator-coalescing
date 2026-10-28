"""Materialize permitted metric metadata from retained evidence; no model/data reader.

This is a local evidence-preservation step. Public report regeneration uses only the
resulting metadata and receipts via `python -m poc report --include-failures`.
"""
from dataclasses import asdict
import json
from pathlib import Path

from poc.report import EvidenceRecord, Claim, GateRecord, AnalysisResult, inspect_existing_bundle, sha256_file
from poc.records import canonical_json

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.superpowers/sdd/2026-10-27-cgoc'
TEACHER = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def preserve_audits():
    names = ['task-12-review.md', 'task-12-inherited-prerequisites.json',
             'task-12-final-verification.json', 'task-12-controller-final-evidence.json',
             'task-12-controller-statistics-evidence.json', 'task-11-controller-final-evidence.json',
             'task-11-controller-canonical-evidence.json', 'task-11-controller-global-evidence.json',
             'task-11-controller-source-snapshots.json', 'task-11-controller-fix1-evidence.json',
             'task-11-rereview.md']
    records = {}
    for name in names:
        path = SCRATCH/name
        records[name] = {'source_path': str(path.relative_to(ROOT)), 'source_sha256': sha256_file(path),
                         'content': read(path) if path.suffix == '.json' else path.read_text()}
    value = {'schema_version': 1, 'scope': 'Preserved controller audits/review, not independent model replication; original source records are unchanged.',
             'review_status': 'Approved', 'current_gate': 'WP04 inconclusive terminal',
             'historical_pending_flag': 'Raw WP04 was written before review; subsequent actual review is preserved here.',
             'records': records}
    save('provenance/task25-downstream-audits-2026-10-28.json', value)
    (ROOT/'provenance/task12-review-2026-10-28.md').write_text((SCRATCH/'task-12-review.md').read_text())


def preserve_decisions():
    source = SCRATCH/'progress.md'
    lines = [(i,l) for i,l in enumerate(source.read_text().splitlines(),1) if 'Ruling:' in l]
    explanations = [
        ('Execute the approved plan on the existing branch.', 'Reversible implementation rework; no remote integration was authorized.'),
        ('Bootstrap the missing test project before measuring it.', 'A fictitious baseline pass would invalidate verification history.'),
        ('Keep the tiny ownership gate independent of later real bundles.', 'A circular prerequisite could block qualification or falsely certify export.'),
        ('Measure the native anchor before allocating the final payload ceiling.', 'A guessed ceiling would invalidate all later byte comparisons.'),
        ('Permit an evidence-limited terminal report at any gate.', 'Mislabeling unexecuted experiments as incomplete or successful research.'),
        ('Allow qualified tiny CPU work before real-asset provenance.', 'Premature asset use would invalidate source qualification.'),
        ('Treat a boundary digest as the prefix producer identity.', 'Rejecting valid student states or accepting stale states would invalidate causal scoring.'),
        ('Enforce independent mutable caches at each depth.', 'Aliased cache writes could corrupt autoregressive results.'),
        ('Use bounded public asset preparation after source qualification.', 'Rights/provenance errors or uncontrolled local spending.'),
        ('Expand the pool before scoring using an outcome-blind rule.', 'Exposure leakage or a sample size chosen from favorable results.'),
        ('Distinguish a competence screen from a smoke test.', 'Near-chance proxy results could be mistaken for preserved capability.'),
        ('Verify guard changes separately without relabeling old numerics.', 'False source attribution or unnecessary repeated model work.'),
        ('Keep early export provenance explicit until fitted-run resolution exists.', 'Invented training lineage or an unsupported native-export claim.'),
        ('Package the runtime deterministically to prevent import writes.', 'Changed immutable bundle bytes and misleading offline proofs.'),
        ('Leave complete large archives unknown under the actual cap.', 'Partial archives could be advertised as complete deployment sizes.'),
        ('Pin the untrained anchor initializer and retain its artifact.', 'An unstable or missing ceiling would invalidate downstream allocation.'),
        ('Use a separate bounded quantization stage with thin CLI adapters.', 'Reversible module rework or unrecorded calibration costs.'),
        ('Reuse the cost-record contract and qualify each numerical backend.', 'Undercounted costs or applying the wrong negative-KL floor.'),
        ('Freeze the larger preliminary frontier population before outcomes.', 'Selection bias or mistaking tiny fixtures for competitive evidence.'),
        ('Version conversion caps while retaining all prior spending.', 'Erased failures or a format allocation chosen by quality outcomes.'),
        ('Include the byte-feasible tied-embedding quantization variants.', 'Omitting a strong eligible comparator; unresolved dynamic allocation reproduction.'),
        ('Canonicalize tensor digest bytes without changing arithmetic.', 'Hash failures or stale-cache acceptance; no need to relabel numerical evidence.'),
        ('Bound the operation-only probe pilot and require calibrated depths.', 'Uncalibrated interventions could be mistaken for an active advantage.'),
        ('Verify all checkpoint state before caller-owned restoration.', 'A partial restore could silently change optimizer/data/RNG continuation.'),
        ('Use only full saved-policy fit windows with recorded exclusions.', 'A new source-bound run would be needed if eligibility were wrong.'),
        ('Keep the intended horizon and charge failed/carry-in work once.', 'A reset budget or false claim of completed training.'),
        ('Qualify one common local mixture operation within finite caps.', 'Unsupported renormalization or a local metric mislabeled as global quality.'),
        ('Observe the last safe checkpoint without extending training.', 'A selected endpoint or hidden extension of the fixed training budget.'),
        ('Give scoring methods the same recipients and fitted states.', 'Confounding active selection with different data or post hoc damage tuning.'),
        ('Choose aligned eligible records before predictions and retain rejects.', 'Outcome-driven pool changes or stale candidate hidden-state reuse.'),
        ('Count profiling inside aggregate diagnostic caps.', 'Uncharged work or infeasible continuation based on blind extrapolation.'),
        ('Freeze the complete protocol before final damage predictions.', 'Damage leakage or falsely claiming efficient random-only acquisition.'),
        ('Retain the original sample and append limited shortlist coverage.', 'Post hoc sample replacement or a false exhaustive shortlist-miss rate.'),
        ('Diagnose recorder overhead on copies without neural reruns.', 'Uncharged diagnosis or an integrity change presented as faster model execution.'),
        ('Preserve the full decision history and downstream review evidence.', 'Loss of reconstructible approval/audit history or circular provenance.'),
    ]
    if len(lines) != len(explanations):
        raise ValueError('ledger changed: review every new literal ruling before preservation')
    out = ['# Controller decision appendix', '',
           f'Exactly {len(lines)} lines containing the literal `Ruling:` occur in the saved ledger, in original order. '
           'This includes the later directive about preserving the earlier rulings. These are controller decisions under existing user authorization, not requests for additional user permission.', '',
           f'Source: `.superpowers/sdd/2026-10-27-cgoc/progress.md`; SHA256 `{sha256_file(source)}`. '
           'Historical counts, budgets and pending states below remain verbatim; later receipts govern current scientific claims.', '']
    for index, ((number,line), (explanation,cost)) in enumerate(zip(lines,explanations),1):
        out.extend([f'## {index}. Ledger line {number}', '', '```text', line, '```', '',
                    f'Explanation: {explanation}', '', f'Cost if wrong: {cost}', ''])
    (ROOT/'docs/analysis/2026-10-28-decision-appendix.md').write_text('\n'.join(out))


def metric_sources(original, precision, quant, training):
    metrics = []; snapshots = []
    def add_source(path, expected, rows, metadata):
        actual = sha256_file(ROOT/path)
        if actual != expected:
            raise ValueError(f'metric artifact hash mismatch: {path}')
        source_index = len(snapshots)
        safe = []
        for pointer, value in rows:
            row = {key:value[key] for key in ('kl_sum','nll_sum','valid_targets') if key in value}
            if row.get('valid_targets',0) <= 0 or 'kl_sum' not in row:
                raise ValueError('metric source lacks genuine sums/denominators')
            safe.append({'source_pointer': pointer, **row})
            metrics.append({**metadata, **row, 'metric_id': f'm{len(metrics):05d}', 'status':'measured',
                            'record':'metric_sources', 'pointer':f'/sources/{source_index}/rows/{len(safe)-1}',
                            'source_path':path, 'source_pointer':pointer,'source_sha256':actual,
                            'component_id':value.get('component',value.get('record',{}).get('recipient_component')),
                            'document_id':value.get('document',value.get('record',{}).get('recipient_id')),
                            'probe_id':value.get('record',{}).get('probe_id'),
                            'probe_source':value.get('source'), 'depth':value.get('record',{}).get('depth'),
                            'model_revision':TEACHER,'seed':17})
        snapshots.append({'path':path, 'sha256':actual, 'rows':safe})
    for study, parent, pattern in [('original_primary',original,'artifacts/diagnostics/task12/trials-final/*.json'),
                                   ('fresh_exploratory',precision,'artifacts/diagnostics/task12/precision-v1/trials/*.json')]:
        for path in sorted(ROOT.glob(pattern)):
            relative = str(path.relative_to(ROOT)); value=read(path)
            if value.get('damage_status') not in (None,'complete','observed_after_full_phase_protocol_freeze'):
                raise ValueError('incomplete damage record')
            allowed = ['natural_damage_rows','natural_score_rows'] + (['secondary_rows','intervention_damage_rows'] if study=='original_primary' else [])
            for key in allowed:
                rows = [(f'/{key}/{i}',row) for i,row in enumerate(value[key])]
                add_source(relative, parent['artifact_sha256'][relative], rows,
                           {'study':study, 'scope':key+('; conditional teacher-suffix intervention, average depth then source components' if key=='intervention_damage_rows' else '; whole-model natural KL, equal document target means')+'; one fitted K29/r8 state; not capability preservation',
                            'pair':json.dumps(value['identity']['pair']), 'candidate_digest':value['identity']['fitted_digest']})
    for run in training['runs']:
        if run['run_id'] not in {training['main_global']['run_id'],training['main_global']['final_observation_run_id'],training['local_operation']['run_id']}:
            continue
        rows=[]
        for i,row in enumerate(run['observations']):
            if 'sources' in row:
                for source,v in row['sources'].items(): rows.append((f'/observations/{i}/sources/{source}',v))
            else: rows.append((f'/observations/{i}',row))
        add_source(run['result'],run['result_sha256'],rows,
                   {'study':'local_conditional_operation' if run['run_id']==training['local_operation']['run_id'] else 'global_canonical_curve',
                    'scope':'source-bound observation; endpoint/horizon/role in source pointer; no final preservation claim',
                    'candidate_digest':training['main_global']['candidate_digest'] if run.get('source_step')==29 else None})
    # Quantization sums are already retained publicly, so the normalized snapshot is
    # source-bound to the receipt rather than reopening a model or its predictions.
    path='provenance/quantization-135m-2026-10-27.json'
    for i,row in enumerate(quant['rows']):
        if 'development' in row:
            add_source(path,sha256_file(ROOT/path),[(f'/rows/{i}/development',row['development'])],
                       {'study':'preliminary_dense_Q', 'scope':row['variant']+f"; {'4 smoke' if row['phase']=='smoke' else '32 development'} components, {row['development']['valid_targets']} targets; no comparison to global8doc",
                        'candidate_digest':row.get('manifest_sha256')})
    for stage in ['compressed_capability_regressions','protected_confirm_A','protected_confirm_B','native_shared_trained_at_Bstar','runtime_speed_comparison','360M_replication']:
        metrics.append({'metric_id':stage,'status':'unexecuted','study':stage,'scope':'Task13–24 unexecuted',
                        'unavailable_reason':'WP04 terminal inconclusive; no measurement exists'})
    save('provenance/report-metric-sources-2026-10-28.json',
         {'schema_version':1,'scope':'Whitelist-only metric preservation; no private text/token IDs/masks/weights. Source hashes bind original permitted raw artifacts.',
          'sources':snapshots,'metrics':metrics})
    return metrics


def bundle_census(quant):
    records=[]
    anchor=read(ROOT/'provenance/payload-anchor-135m.json')
    sources=[('anchor',Path(anchor['artifact']),anchor['manifest_sha256'],anchor['payload_bytes'])]
    sources.extend((row['variant'],Path(row['artifact']),row['manifest_sha256'],row['payload_bytes']) for row in quant['rows'] if row.get('artifact') and row.get('manifest_sha256'))
    floating=read(ROOT/'provenance/bundle-deployment-2026-10-27.json')
    sources.extend((f'floating_{kind}',Path('/private/tmp/cgoc-task8-floating-attempt5')/kind,
                    row['manifest_sha256'],row['payload_bytes']) for kind,row in floating['artifacts'].items())
    for label,path,digest,expected in sources:
        if not path.exists():
            records.append({'label':label,'path':str(path),'status':'unavailable_local_bundle','expected_payload_bytes':expected,
                            'manifest_sha256':digest,'reason':'historical evidence retained; no recreation or model load'})
            continue
        proof=inspect_existing_bundle(path,digest)
        if proof['payload_bytes'] != expected:raise ValueError('bundle actual byte sum differs from receipt')
        records.append({'label':label, 'path':str(path), **proof})
    offline=quant['offline_proof']
    if sha256_file(ROOT/offline['path'])!=offline['sha256']:raise ValueError('pinned offline proof hash mismatch')
    save('provenance/report-bundle-census-2026-10-28.json',
         {'schema_version':1,'scope':'File/hash/byte verification of existing bundles only; no model instantiated or new qualification.',
          'bundles':records,'existing_offline_proof':{'path':offline['path'],'sha256':offline['sha256'],'status':offline['result']['status']},
          'floating_offline_receipt':{'path':'provenance/bundle-deployment-2026-10-27.json','sha256':sha256_file(ROOT/'provenance/bundle-deployment-2026-10-27.json')},
          'all_existing_bytes_match':True})


def main():
    preserve_audits(); preserve_decisions()
    original=read(ROOT/'provenance/single-merge-original-135m-2026-10-27.json')
    precision=read(ROOT/'provenance/single-merge-precision-135m-2026-10-27.json')
    training=read(ROOT/'provenance/training-135m-2026-10-27.json')
    quant=read(ROOT/'provenance/quantization-135m-2026-10-27.json')
    metric_sources(original,precision,quant,training); bundle_census(quant)
    plan_path='artifacts/diagnostics/task12/precision-v1/plan.json'
    plan=read(ROOT/plan_path)
    save('provenance/report-protocol-2026-10-28.json', {'schema_version':1,
         'source_path':plan_path,'source_sha256':sha256_file(ROOT/plan_path),
         'teacher_revision':TEACHER,
         **{key:plan[key] for key in ('bootstrap_seed','seed_data_order','seed_model_and_probes','data_manifest_sha256')},
         'scope':'Same primary fitted states; fresh source components/data order only; no independent fitting seed or second teacher'})
    records={}; claims=[]; sections=[]
    def register(name,path):
        records[name]=asdict(EvidenceRecord(path,sha256_file(ROOT/path)))
    for name,path in {'original':'provenance/single-merge-original-135m-2026-10-27.json',
                      'precision':'provenance/single-merge-precision-135m-2026-10-27.json',
                      'training':'provenance/training-135m-2026-10-27.json',
                      'quant':'provenance/quantization-135m-2026-10-27.json',
                      'gate':'provenance/task12-wp04-2026-10-27.json',
                      'teacher':'provenance/teacher-qualification-2026-10-27.json',
                      'anchor':'provenance/payload-anchor-135m.json',
                      'deployment':'provenance/bundle-deployment-2026-10-27.json',
                      'review':'provenance/task25-downstream-audits-2026-10-28.json',
                      'metric_sources':'provenance/report-metric-sources-2026-10-28.json',
                      'bundle_census':'provenance/report-bundle-census-2026-10-28.json',
                      'blueprint':'docs/blueprint.md',
                      'protocol_meta':'provenance/report-protocol-2026-10-28.json',
                      'config':'configs/135m.yaml',
                      'assets':'provenance/assets-135m-sources.json',
                      'preparation':'provenance/preparation-135m-2026-10-27.json',
                      'development_protocol':'protocol/development.md',
                      'design':'docs/superpowers/specs/2026-10-27-cgoc-design.md',
                      'original_upstream_audits':'provenance/task12-controller-audits-2026-10-27.json',
                      'locks':'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'}.items():
        if not (ROOT/path).exists() and name=='locks':continue
        register(name,path)
    def section(title):
        rows=[];sections.append([title,rows]);return rows
    def claim(rows,label,record,pointer,scope):
        from poc.report import resolve_pointer
        identity=f'C{len(claims)+1:03d}'
        value=resolve_pointer(read(ROOT/records[record]['path']),pointer)
        claims.append(asdict(Claim(identity,value,record,pointer,scope)))
        rows.append([label,identity])
    rows=section('Terminal decision and subsequent review')
    for label,rec,ptr in [('Gate','gate','/decision'),('Gate diagnosis','gate','/reason'),('Subsequent independent review','review','/review_status'),('Unexecuted task IDs','gate','/unexecuted_tasks')]:
        claim(rows,label,rec,ptr,'WP04 original gate predates review; subsequent review establishes Approved current state')
    rows=section('Teacher development competence and model scope')
    for label,ptr in [('Teacher architecture','/teacher'),('ARC Easy','/development/arc/groups/ARC-Easy'),('ARC Challenge','/development/arc/groups/ARC-Challenge'),('BLiMP accuracy','/development/blimp/accuracy'),('BLiMP items','/development/blimp/items'),('Synthetic families','/development/synthetic')]:
        claim(rows,label,'teacher',ptr,'Teacher development screen only; no compressed-model capability outcome or protected evaluation')
    for key in ['teacher_revision','bootstrap_seed','seed_data_order','seed_model_and_probes','data_manifest_sha256']:
        claim(rows,key,'protocol_meta','/'+key,'Fixed exploratory fresh repeat; original fitted states and seed17 methods retained')
    studies=[('Original primary','original','/primary_summary',8,16),
             ('Original eighteen-pair secondary','original','/primary_summary/secondary_original18',8,16),
             ('Fresh fixed exploratory repeat','precision','/summaries/fresh_repeat',16,32),
             ('Fresh eighteen-pair secondary','precision','/summaries/fresh_original18_secondary',16,32),
             ('Pooled exploratory secondary','precision','/summaries/pooled_exploratory_secondary',24,48)]
    for title,record,base,nscore,ndamage in studies:
        rows=section(title)
        scope=f'{title}; {nscore} score / {ndamage} natural-damage source components; seed17; same one-update fitted states; nats/target; sampled-set oracle only'
        for method in ['weight_distance','activation_mse','cka','natural_kl','random_donor','active_1','active_01']:
            claim(rows,method+' selected pair / damage / oracle / regret',record,base+'/regret/'+method,scope)
        for method in ['natural_kl','active_1']:
            claim(rows,method+' source-component bootstrap interval and choice counts',record,base+'/uncertainty/methods/'+method,scope+'; 2000 draws; selection and oracle recomputed')
        for comparator in ['natural_kl','random_donor']:
            claim(rows,'active minus '+comparator+' discrete contrast',record,base+'/uncertainty/paired_discrete_contrasts/'+comparator,scope+'; favorable negative / adverse positive; zero spikes are not equivalence')
    rows=section('Original intervention endpoint and sampled shortlist coverage')
    for method in ['natural_kl','active_1','active_01','random_donor']:
        claim(rows,method+' conditional intervention regret','original','/primary_summary/fresh_intervention_regret/'+method,'Original separate sixteen intervention-damage components; frozen teacher suffix; no fresh-repeat intervention endpoint')
    for label,pointer in [('Coverage','/actual_shortlist_coverage'),('Deduplicated shortlist size','/actual_shortlist_count'),('Sampled shortlist misses','/primary_summary/shortlist_misses'),('Permutation initializer control','/primary_summary/initializer_control')]:
        claim(rows,label,'original',pointer,'Finite sampled pairs only; not exhaustive global coverage or a measured trained joint-composition failure')
    for i in range(2):
        for key in ['pair','fit_before','fit_after']:
            claim(rows,f'Extra-step sensitivity {i+1} {key}','original',f'/sensitivity_refits/{i}/{key}','Separate second updates, fit checks only; primary states remain one update')
    rows=section('Canonical learning curve and local conditional operation')
    for key in ['planned_updates','durable_updates','durable_training_targets','partial_update_charged_targets','fit_windows','stop']:
        claim(rows,key,'training','/main_global/'+key,'One K15/r8 seed17 bounded global run; failed/carry-in work remains separate from durable learning')
    for i,run in enumerate(training['runs']):
        if run['run_id'] not in {training['main_global']['run_id'],training['main_global']['final_observation_run_id'],training['local_operation']['run_id']}:continue
        for j,observation in enumerate(run['observations']):
            if run['run_id']==training['local_operation']['run_id']:
                claim(rows,'Local '+observation['label']+' weighted KL','training',f'/runs/{i}/observations/{j}/weighted_kl','Conditional .5 natural/.25 student/.25 donor KL through frozen teacher suffix; not full-model post-local KL')
            else:
                claim(rows,'Global '+observation.get('label',observation['role']+'_step'+str(observation['step'])),'training',f'/runs/{i}/observations/{j}',
                      'Global full-model fidelity; development eight components/4088 targets, fit checks three/1533; never compare to Q32doc as matched superiority')
    rows=section('Measured bytes and preliminary dense frontier')
    claim(rows,'Immutable untrained prescribed anchor B*','anchor','/payload_bytes','Complete K15/r8 SEQUENCE/SVD/RTN4 g64 BF16-residual format; no trained quality at B*')
    for kind in ['dense','compact']:
        for key in ['payload_bytes','resident_bytes','archive']:
            claim(rows,'Untrained floating '+kind+' '+key,'deployment',f'/artifacts/{kind}/{key}','Storage-format operation only; not a trained deployment or latency advantage')
    claim(rows,'Common measured install bytes','deployment','/install/bytes','Conservative entire pinned environment with caches; not minimal install')
    for i,row in enumerate(quant['rows']):
        if row.get('development'):
            for key in ['status','at_or_below_anchor','payload_bytes','development']:
                claim(rows,row['variant']+' '+key,'quant',f'/rows/{i}/{key}',
                      f"{'Operation smoke only; 4 components/2044 targets' if row['phase']=='smoke' else 'Preliminary Q frontier; 32 dev components/16352 targets'}; all oversized points retained; no trained shared counterpart at B*")
    for i,row in enumerate(quant['rows']):
        if 'development' not in row:claim(rows,row['variant']+' unavailable status','quant',f'/rows/{i}','Unsupported format is explicit, not an invented zero quality outcome')
    census=read(ROOT/'provenance/report-bundle-census-2026-10-28.json')
    for i,bundle in enumerate(census['bundles']):
        for key in ['status','payload_bytes']:
            if key in bundle:claim(rows,bundle['label']+' current '+key,'bundle_census',f'/bundles/{i}/{key}','Current byte/hash checks only; existing pinned offline proofs inherited, no new model loads')
    rows=section('Construction, failures and unavailable counters')
    for record,base,label in [('training','/unique_attempt_cost_totals','Task11 unique attempts'),('precision','/aggregate_cost','Task12 all actual attempts'),('precision','/followup_cost','Task12 follow-up subset')]:
        for key in ['wall_seconds','teacher_forward_calls','student_forward_calls','student_backward_calls','suffix_backward_calls','updates','valid_target_presentations','search_proposals','search_seconds','disk_bytes','memory_bytes']:
            claim(rows,label+' '+key,record,base+'/'+key,'Dispatches include head chunks; presentations include repeated/replayed targets; memory maxima nonadditive and disk highwaters are not summed across stages')
    for key in ['overall_wall_seconds_including_this_audit','followup_wall_seconds_including_this_audit','source_component_counts','data_eligibility','cap_history','separately_charged_cpu_diagnoses','attempts']:
        claim(rows,key,'precision','/'+key,'Historical failures/caps and fully charged strict replay; CPU diagnoses already in aggregate; no independent neural replication')
    claim(rows,'All Task11 attempt statuses/costs','training','/cost_scope','Sum each actual attempt once; inherited aggregate values never added twice')
    overview=[]
    for title,record,base in [('Original primary','original','/primary_summary'),('Fresh exploratory','precision','/summaries/fresh_repeat'),('Pooled secondary','precision','/summaries/pooled_exploratory_secondary')]:
        ids=[]
        for method in ['natural_kl','random_donor','active_1','active_01']:
            ids.append(next(c['id'] for c in claims if c['record']==record and c['pointer']==base+'/regret/'+method))
        overview.append([title,*ids])
    narrative=(ROOT/'docs/terminal-report-narrative.md').read_text()
    failures=[
        'One-step fitting remains unsaturated; both extra-step controls improve. Raw unaligned truncated SVD fails a function-preserving permutation fixture. Neither diagnoses a universal absence of reusable computation.',
        'Original logging-only failure and the fresh preparation memory failure remain in source-bound receipts. The old 8 GiB guard failed at 8618934272 bytes RSS+driver; future 9 GiB caps do not rewrite that failure. Reader/serialization growth remains unresolved; process isolation is not a leak fix.',
        'Only one model and one primary method/probe seed were tested. Fresh seed19 orders new source components; exact same-host reconstruction of eighteen fitted states is not independent replication or transfer.',
        'Task11 ends at its wall cap with twenty-nine durable updates; the planned thirty-one horizon and the incomplete update are explicit. Strong-Q and global learning use different development populations.',
        'No trained native shared model at B*, protected A/B, full capability comparison, runtime speed comparison, full SV implementation, compiler, random-only efficient pipeline, or second-scale replication exists.',
        'Full dense/compact archives and unpack times remain unavailable under historical budgets. Dynamic allocation reproduction for the original two targets remains unresolved because original scores/thresholds are absent, despite fixed-artifact fidelity.',
        'Unique learning-token and estimated-FLOP counters are unavailable, not measured zeros. Resource observations overlap; no physical peak is obtained by adding RSS, allocated and driver counters.',
        'Teacher arithmetic and composition screens were near zero; they cannot establish preservation of reasoning. Pretraining contamination is unknown. B is from the same FineWeb source; WikiText card/prose licensing conflict remains unresolved.',
        'Novelty is unresolved. Existing overlap is replication/engineering unless later source comparisons justify a narrower claim; comparisons must be refreshed before any external publication.'
    ]
    release_files=['README.md','PROGRESS.md','DECISIONS.md','docs/reproduction.md','docs/terminal-report-narrative.md',
                   'docs/analysis/2026-10-28-decision-appendix.md','provenance/task25-downstream-audits-2026-10-28.json',
                   'provenance/task12-review-2026-10-28.md','provenance/report-metric-sources-2026-10-28.json',
                   'provenance/report-bundle-census-2026-10-28.json','provenance/prepare_terminal_report.py',
                   'poc/report.py','poc/cli.py','tests/test_report.py','pyproject.toml']
    release_files += [r['path'] for r in records.values()]
    release_files += [str(p.relative_to(ROOT)) for tree in ['poc','tests','configs','protocol','provenance/dependency-locks'] for p in (ROOT/tree).rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.yml','.md','.json','.txt'}]
    release_files += ['.gitignore','provenance/verify_terminal_report.py','provenance/task25-verification-2026-10-28.json']
    save('provenance/report-evidence-2026-10-28.json',
         {'schema_version':1, 'records':records,'claims':claims,'sections':sections,'overview':overview,
          'stages':{**{f'Task{i}':'executed; reviewed qualification/diagnostic evidence, scope limited by receipts' for i in range(1,13)},
                    **{f'Task{i}':'unexecuted; terminal WP04 inconclusive' for i in range(13,25)},
                    'Task25':'local evidence-only terminal reporting; independent Task25 review is subsequent'},
          'gates':[asdict(GateRecord('WP04','inconclusive',read(ROOT/'provenance/task12-wp04-2026-10-27.json')['reason'],('gate','review')))],
          'analysis':asdict(AnalysisResult(budget_limited_curve=True, insufficient_sensitivity=True,unresolved_sv_comparison=True)),
          'metrics_record':'metric_sources','metrics':[], 'narrative':narrative,'failures':failures,
          'release_files':sorted(set(release_files))})
    print(json.dumps({'status':'metadata_preserved','rulings':35,'claims':len(claims),'records':len(records)}))


if __name__=='__main__':main()
