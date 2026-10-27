"""Outcome-blind Task12 text/token preflight; no model is loaded."""
import hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import psutil
from poc.data.manifest import load_manifest
from poc.data.access import open_records
from poc.data.windows import TokenDocument,WindowPolicy,make_windows
from poc.teacher_validation import _restricted

ROOT=Path(__file__).resolve().parents[1]


def prepare():
    start=time.monotonic();out=ROOT/'artifacts/diagnostics/task12';p=json.loads((out/'proposal.json').read_text())
    m=load_manifest(ROOT/'data/prepared-135m-v2/data-manifest.json',expected_sha256=p['data_manifest_sha256'],roles={'fit','dev','probe_score'})
    if hashlib.sha256((ROOT/'data/prepared-135m-v2/windows.json').read_bytes()).hexdigest()!=p['saved_windows_sha256']:raise ValueError('window manifest changed')
    used=set();original=p['selected'];chosen={};windows={};exclusions=[]
    for label,rows in original.items():
        role=rows[0]['role'];n=len(rows)
        candidates=[next(r for r in m.records if r.document_id==x['document_id']) for x in rows]
        candidates+=sorted((r for r in m.records if m.roles[r.document_id]==role and r.source=='HuggingFaceFW/fineweb-edu' and r.metadata.get('token_count',0)>=512 and r.document_id not in {x['document_id'] for v in original.values() for x in v}),key=lambda r:hashlib.sha256(('task12-v1:'+r.document_id).encode()).digest())
        chosen[label]=[]
        for r in candidates:
            if m.membership[r.document_id] in used:continue
            list(open_records(_restricted(m,[r]),purpose={'fit':'fit','dev':'selection','probe_score':'ranking'}[role]))
            data=Path(r.token_path).read_bytes()
            if hashlib.sha256(data).hexdigest()!=r.token_sha256:raise ValueError('token file changed')
            raw=json.loads(data)
            if raw['document_id']!=r.document_id:raise ValueError('token document mismatch')
            tokens=tuple(raw['fields']['text']);w=make_windows(TokenDocument(r.document_id,tokens),WindowPolicy())[0]
            if len(tokens)<512 or len(w.token_ids)!=512:
                exclusions.append(dict(document_id=r.document_id,label=label,reason='verified_pre_prediction_canonical_length',tokens=len(tokens),window=len(w.token_ids)));continue
            chosen[label].append(dict(document_id=r.document_id,component_id=m.membership[r.document_id],role=role,token_sha256=r.token_sha256))
            windows[r.document_id]=asdict(w);used.add(m.membership[r.document_id])
            if len(chosen[label])==n:break
        if len(chosen[label])!=n:raise ValueError('insufficient allowed eligible components')
    v=psutil.virtual_memory();memory=8*1024**3 if v.available>=10*1024**3 else 6*1024**3
    p.update(stage='approved_profile',selected=chosen,original_selected=original,pre_prediction_exclusions=exclusions,updates_cap=22,target_cap=180224,teacher_call_cap=350000,student_call_cap=30000,memory_bytes_cap=memory,memory_at_plan={'available_bytes':v.available,'total_bytes':v.total},bootstrap=2000,
      main_score_context='same probe_score recipient components, first512/reset0; ordinary natural targets remain teacher natural outputs',
      methods={'weight_distance':'squared projection distance / projection element count','activation_mse':'mean FFN residual-output squared error, equal depths and documents','cka':'1-centered linear CKA, equal depths and documents','natural_kl':'full candidate KL(T natural || candidate natural), full vocabulary T1',
      'random_donor':'natural_kl + .5*student_conditional_KL + .5*seeded_random_donor_conditional_KL','active_1':'natural_kl + .5*student_conditional_KL + .5*max_donor_conditional_KL','active_01':'natural_kl + .1*(.5*student_conditional_KL + .5*max_donor_conditional_KL)'},
      primary_contrast='active_1 minus natural_kl selected damage/regret; all predeclared variants reported',
      secondary='dev8 canonical natural downstream KL practical comparator, separately component-resampled',
      primary_probe_filter='unfiltered stress; fit-threshold filtered rows are separately labeled, never replace primary',
      source_generation='at most2edit attempts per document; freeze alignment availability before candidate outcomes',
      task12_carry_in={'audit_wall_seconds':time.monotonic()-start,'spec_paths':[],'evidence':{}})
    (out/'windows.json').write_text(json.dumps(windows,sort_keys=True)+'\n')
    p['selected_windows_sha256']=hashlib.sha256((out/'windows.json').read_bytes()).hexdigest()
    (out/'plan.json').write_text(json.dumps(p,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'selected':{k:len(v) for k,v in chosen.items()},'excluded':exclusions,'memory':p['memory_at_plan'],'memory_cap':memory,'preflight_seconds':p['task12_carry_in']['audit_wall_seconds']}))

if __name__=='__main__':prepare()
