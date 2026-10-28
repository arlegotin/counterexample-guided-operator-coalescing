"""Historical exploratory inputs with preserved exposure history; no protected reads."""
from dataclasses import replace
from pathlib import Path
import json
import shutil

import torch

from poc.contracts import ProbeRecord
from poc.continuation_plan import digest
from poc.data.access import open_records
from poc.data.manifest import load_manifest
from poc.data.windows import TokenDocument,WindowPolicy,make_windows
from poc.teacher import sha256_file
from poc.teacher_validation import _batch


def restricted_population(manifest, ids, *, role):
    if role not in ('fit','dev'):raise PermissionError('only exploratory fit/dev permitted')
    ids=tuple(ids)
    if len(set(ids))!=len(ids) or any(manifest.roles.get(i)!=role for i in ids):
        raise PermissionError('population contains missing or forbidden roles')
    rows={r.document_id:r for r in manifest.records}
    return replace(manifest,records=tuple(rows[i] for i in ids),roles={i:role for i in ids},
        membership={i:manifest.membership[i] for i in ids},quarantine={})


def prepare_populations(asset_root, out, *, device):
    """Metadata frame fixed before content; known historic dev remains exploratory."""
    root=Path(asset_root);out=Path(out)
    historical=json.loads((root/'artifacts/diagnostics/task12/plan.json').read_text())
    global_plan=json.loads((root/'artifacts/training/task11/learn31-diverse-v2/plan.json').read_text())
    fit_ids=[r['document_id'] for r in historical['selected']['medoid_and_fit']]
    dev_ids=global_plan['dev_ids']
    manifest=load_manifest(root/'data/prepared-135m-v2/data-manifest.json',
        expected_sha256=historical['data_manifest_sha256'],roles={'fit','dev'})
    fit=restricted_population(manifest,fit_ids,role='fit');dev=restricted_population(manifest,dev_ids,role='dev')
    if set(fit.membership.values())&set(dev.membership.values()):raise ValueError('fit/dev component overlap')
    prior=Path(manifest.exposure_path)
    if not prior.is_file():raise ValueError('original exposure history unavailable')
    target=out/'exposures.jsonl'
    if not target.exists():shutil.copyfile(prior,target)
    lineage=dict(original_exposure_sha256=sha256_file(prior),data_manifest_sha256=historical['data_manifest_sha256'],
        fit_ids=fit_ids,validation_ids=dev_ids,fit_components=[fit.membership[i] for i in fit_ids],
        validation_components=[dev.membership[i] for i in dev_ids],status='historically exposed exploratory fit/dev; no protected or fresh confirmation claim')
    (out/'population.json').write_text(json.dumps(lineage,sort_keys=True,indent=2)+'\n')
    fit=replace(fit,exposure_path=str(target),lineage_sha256=lineage['original_exposure_sha256'])
    dev=replace(dev,exposure_path=str(target),lineage_sha256=lineage['original_exposure_sha256'])
    fit_docs=tuple(open_records(fit,purpose='fit'));dev_docs=tuple(open_records(dev,purpose='selection'))
    def batches(docs,population):
        result=[];windows=[]
        for doc in docs:
            path=Path(doc.token_path)
            if sha256_file(path)!=doc.token_sha256:raise ValueError('token hash mismatch')
            raw=json.loads(path.read_text())
            if raw['document_id']!=doc.document_id:raise ValueError('token identity mismatch')
            window=make_windows(TokenDocument(doc.document_id,tuple(raw['fields']['text'])),WindowPolicy())[0]
            if len(window.token_ids)!=512:raise ValueError('insufficient declared context')
            result.append(_batch(window.token_ids,device=device,document_id=doc.document_id,
                cluster_id=population.membership[doc.document_id],targets=window.target_indices))
            windows.append(dict(document_id=doc.document_id,component_id=population.membership[doc.document_id],
                window_sha256=digest(window.token_ids),source_start=window.source_start,targets=len(window.target_indices)))
        return tuple(result),windows
    fit_batches,fit_windows=batches(fit_docs,fit);dev_batches,dev_windows=batches(dev_docs,dev)
    lineage.update(fit_windows=fit_windows,validation_windows=dev_windows)
    return fit,fit_docs,fit_batches,dev_batches,lineage


def fit_templates(asset_root, fit, pair):
    path=Path(asset_root)/'artifacts/diagnostics/task12/fit-pool.json'
    raw=json.loads(path.read_text());selected={}
    for value in raw['rows']:
        if value['recipient_role']!='fit' or value['recipient_id'] not in fit.roles:
            raise PermissionError('historical pool contains unapproved source')
        if (value['status']=='accepted' and value['depth'] in pair and any(value['target_mask'])
                and value['recipient_tokens']!=value['donor_tokens'] and len(value['recipient_tokens'])==512):
            record=ProbeRecord(**value);key=(record.recipient_id,record.depth)
            if key not in selected or record.probe_id<selected[key].probe_id:selected[key]=record
    if set(selected)!={(i,d) for i in fit.roles for d in pair}:raise ValueError('pool lacks complete fit/depth coverage')
    return tuple(selected[k] for k in sorted(selected)),dict(path=str(path),sha256=sha256_file(path),
        origin='historical raw input templates; current activations/profile identities recomputed',
        historical_acquisition_cost='not isolated; excluded from current execution cost; no acquisition efficiency claim')
