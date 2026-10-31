"""New natural-only populations from an explicit, authorized raw inventory.

No benchmark readers, historical donor archives, parent preparation manifests,
model scoring, or implicit home-cache discovery. Raw text stays in ignored local
storage. Missing historical overlap information remains an explicit limitation.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import re
import time

from poc.assets import LogicalDiskBudget
from poc.data.dedup import DedupPolicy, build_components
from poc.data.manifest import DataManifest, DocumentRecord, load_manifest
from poc.data.access import open_records
from poc.records import canonical_json
from poc.teacher import sha256_file


def prepare_natural_records(records,tokenizer,out,*,tokenizer_digest,inventory_sha256,
        fit_components=512,tuning_components=128,seed=17,max_document_bytes,max_records,max_tokens,max_disk_bytes,prior_records=(),prior_manifest_hashes=()):
    for h in (tokenizer_digest,inventory_sha256):
        if not re.fullmatch('[0-9a-f]{64}',h):raise ValueError('pinned inventory/tokenizer hash required')
    if any(type(v)!=int or v<1 for v in (fit_components,tuning_components,max_document_bytes,max_records,max_tokens,max_disk_bytes)):
        raise ValueError('positive preparation counts and ceilings required')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();disk=LogicalDiskBudget(out,max_disk_bytes)
    def write(path,data):
        disk.reserve(path,len(data));path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    records=tuple(records)
    if len(records)>max_records:raise ValueError('natural inventory record ceiling exceeded')
    if not records or any(not re.fullmatch('[0-9a-f]{40}',r.revision) for r in records):raise ValueError('explicit pinned natural source inventory required')
    if any(r.source in ('allenai/ai2_arc','alexwarstadt/blimp') for r in records):raise PermissionError('natural preparation refuses benchmark sources')
    # Full-document tokenization determines eligibility. Component assignment is
    # made before any window is constructed, and never depends on model outputs.
    eligible=[];tokens={};exclusions=[];total_tokens=0
    for r in records:
        text=r.read_text()
        if len(text.encode())>max_document_bytes:
            exclusions.append({'document_id':r.document_id,'reason':'document_byte_ceiling'});continue
        ids=tuple(tokenizer.encode(text,add_special_tokens=False))+(tokenizer.eos_token_id,)
        total_tokens+=len(ids)
        if total_tokens>max_tokens:raise ValueError('natural tokenization ceiling exceeded')
        if len(ids)<512:
            exclusions.append({'document_id':r.document_id,'reason':'fewer_than_512_tokens'});continue
        eligible.append(replace(r,text=text));tokens[r.document_id]=ids
    prior_records=tuple(prior_records);prior_ids={r.document_id for r in prior_records}
    combined=tuple(r for r in eligible if r.document_id not in prior_ids)+prior_records
    components=build_components(combined,DedupPolicy(seed=seed))
    prior_components={components.membership[i] for i in prior_ids}
    overlapping={r.document_id for r in eligible if r.document_id in prior_ids or components.membership[r.document_id] in prior_components}
    exclusions.extend({'document_id':i,'reason':'known_prior_overlap'} for i in sorted(overlapping))
    retained=tuple(r for r in components.records if r.document_id in tokens and r.document_id not in overlapping)
    components=replace(components,records=retained,membership={r.document_id:components.membership[r.document_id] for r in retained})
    grouped={}
    for r in components.records:grouped.setdefault(components.membership[r.document_id],[]).append(r)
    order=sorted(grouped);random.Random(seed).shuffle(order)
    count=fit_components+tuning_components
    summary=dict(schema_version=1,inventory_sha256=inventory_sha256,tokenizer_digest=tokenizer_digest,
        requested_components={'fit':fit_components,'tuning':tuning_components},eligible_components=len(order),
        eligibility='at least 512 tokens including one real EOS; first512 reset context; one window/component',
        role_policy='dedup components then seeded shuffle; fit then component-disjoint tuning; no outcome replacement',
        historical_exposure='unknown; exploratory only',known_prior_records=len(prior_records),
        known_prior_manifest_sha256=list(prior_manifest_hashes),dedup_policy=asdict(DedupPolicy(seed=seed)),
        exclusions=exclusions,token_ids_processed=total_tokens,
        sources=sorted({(r.source,r.revision) for r in records}),seed=seed,
        source_terms='inherited from explicitly authorized source inventory; no raw redistribution')
    if len(order)<count:
        summary.update(status='shortage',elapsed_seconds=time.monotonic()-started)
        write(out/'preparation.json',canonical_json(summary))
        raise ValueError(f'natural population shortage: {len(order)} eligible components for {count}')
    for name,role,ids in (('fit','fit',order[:fit_components]),('tuning','dev',order[fit_components:count])):
        selected=[];membership={}
        for component in ids:
            r=min(grouped[component],key=lambda r:r.document_id)
            path=out/'text'/f'{r.document_id}.txt';tokenpath=out/'tokens'/f'{r.document_id}.json'
            if not re.fullmatch('[a-zA-Z0-9_-]+',r.document_id):raise ValueError('safe document IDs required')
            text=r.read_text().encode();payload=canonical_json({'document_id':r.document_id,'fields':{'text':tokens[r.document_id]},'tokenizer_digest':tokenizer_digest})
            write(path,text);write(tokenpath,payload)
            selected.append(replace(r,text=None,text_path=str(path),text_sha256=hashlib.sha256(text).hexdigest(),
                token_path=str(tokenpath),token_sha256=hashlib.sha256(payload).hexdigest(),
                metadata={**r.metadata,'tokenizer_digest':tokenizer_digest,'inventory_sha256':inventory_sha256,
                          'window_policy':'first512_reset0','prior_exposure':'unknown'}))
            membership[r.document_id]=component
        manifest=DataManifest(tuple(selected),{r.document_id:role for r in selected},membership,
                              f'recovery-natural-v1-{seed}',str(out/'exposures.jsonl'))
        path=out/f'{name}-manifest.json';payload=canonical_json(asdict(manifest));write(path,payload)
        summary[name]={'path':str(path),'sha256':hashlib.sha256(payload).hexdigest(),'components':len(ids)}
    summary.update(status='prepared_exploratory',elapsed_seconds=time.monotonic()-started)
    write(out/'preparation.json',canonical_json(summary))
    return summary


def load_natural_population(path,expected_sha256,*,role,components,tokenizer_digest):
    if role not in ('fit','dev'):raise PermissionError('only fit/tuning populations permitted')
    # Inspect metadata roles before restricting, so a wrong-role manifest cannot
    # silently turn into an empty population or expose any token/text content.
    if sha256_file(Path(path))!=expected_sha256:raise ValueError('population manifest hash mismatch')
    raw=json.loads(Path(path).read_text())
    if set(raw['roles'].values())!={role}:raise PermissionError('population role mismatch')
    manifest=load_manifest(Path(path),expected_sha256=expected_sha256,roles={role})
    if len(manifest.records)!=components or len(set(manifest.membership.values()))!=components:
        raise ValueError('exact one-window/component population size required')
    from poc.recovery_study import NaturalWindow
    from poc.teacher_validation import _batch
    batches=[]
    for row in open_records(manifest,purpose='fit' if role=='fit' else 'selection'):
        if row.metadata.get('tokenizer_digest')!=tokenizer_digest:raise ValueError('population tokenizer mismatch')
        if sha256_file(Path(row.token_path))!=row.token_sha256:raise ValueError('natural token content hash mismatch')
        raw=json.loads(Path(row.token_path).read_text())
        if raw['document_id']!=row.document_id or raw.get('tokenizer_digest')!=tokenizer_digest:raise ValueError('natural token identity mismatch')
        ids=tuple(raw['fields']['text'])
        if len(ids)<512:raise ValueError('natural eligibility changed')
        b=_batch(ids[:512],device='cpu',document_id=row.document_id,cluster_id=manifest.membership[row.document_id],targets=range(1,512))
        NaturalWindow.from_batch(b)
        batches.append(b)
    return manifest,tuple(batches)


def read_fineweb_inventory(path,expected_sha256,out,*,source_manifest,max_range_bytes,max_request_bytes,
                           max_disk_bytes,max_records,max_document_bytes):
    """Read only prospective shard/row-group/row IDs through capped ranges.

    This verifies every downloaded interval, not the full 2 GB parent shard.
    Pin, terms, inventory and duplicate-row checks precede all network I/O.
    The caller supplies the inventory; no adaptive replacement or model score
    influences retrieval. Transfer receipts and rejected rows stay local.
    """
    from poc.assets import RangeAsset,TransferBudget
    from poc.data.prepare import check_source_gate,_request,FINEWEB
    if sha256_file(Path(path))!=expected_sha256:raise ValueError('natural source inventory hash mismatch')
    raw=json.loads(Path(path).read_text())
    if set(raw)!={'schema_version','repository','revision','shards'} or raw['schema_version']!=1:
        raise ValueError('unknown natural inventory fields/version')
    gate=check_source_gate(Path(source_manifest));source=gate['sources'][FINEWEB]
    if raw['repository']!=FINEWEB or raw['revision']!='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9' or raw['revision']!=source['revision']:
        raise ValueError('natural source repository/revision mismatch')
    entries={e['path']:e for e in source['files']};seen=set();seen_groups=set()
    if not isinstance(raw['shards'],list) or not raw['shards']:raise ValueError('explicit shard inventory required')
    for shard in raw['shards']:
        if set(shard)!={'path','row_groups'} or shard['path'] not in entries:raise ValueError('unknown inventory shard')
        for group in shard['row_groups']:
            if set(group)!={'index','rows'} or type(group['index'])!=int or group['index']<0:
                raise ValueError('invalid row-group inventory')
            key=(shard['path'],group['index'])
            if key in seen_groups:raise ValueError('duplicate row group')
            seen_groups.add(key)
            if not group['rows'] or any(type(i)!=int or i<0 for i in group['rows']):raise ValueError('explicit nonnegative row IDs required')
            for i in group['rows']:
                row_key=(*key,i)
                if row_key in seen:raise ValueError('duplicate inventory row; unique rows required')
                seen.add(row_key)
    if len(seen)>max_records:raise ValueError('inventory exceeds frozen record cap')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    disk=LogicalDiskBudget(out,max_disk_bytes);transfer=TransferBudget(max_range_bytes)
    # The exact inventory is saved before content is opened.
    disk.write_bytes(out/'inventory.json',Path(path).read_bytes())
    import pyarrow.parquet as pq
    started=time.monotonic();records=[];exclusions=[];receipts=[]
    try:
        for shard in raw['shards']:
            entry=entries[shard['path']]
            reader=RangeAsset(_request(FINEWEB,source,entry,entry['bytes']),out/Path(entry['path']).stem,
                transfer,max_request_bytes=max_request_bytes,disk_budget=disk)
            try:
                parquet=pq.ParquetFile(reader)
                sizes=[parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)]
                for group in shard['row_groups']:
                    g=group['index']
                    if g>=len(sizes) or any(i>=sizes[g] for i in group['rows']):raise ValueError('prospective row inventory outside pinned shard')
                    table=parquet.read_row_group(g)
                    for i in group['rows']:
                        item=table.slice(i,1).to_pylist()[0];text=item['text']
                        row=sum(sizes[:g])+i
                        identity=hashlib.sha256(canonical_json({'source':FINEWEB,'revision':raw['revision'],'shard':entry['path'],'row':row})).hexdigest()
                        if len(text.encode())>max_document_bytes:
                            exclusions.append({'document_id':identity,'row':row,'reason':'document_byte_ceiling'});continue
                        records.append(DocumentRecord(identity,FINEWEB,raw['revision'],entry['path'],row,text=text,
                            text_sha256=hashlib.sha256(text.encode()).hexdigest(),url=item.get('url'),
                            source_record_id=item.get('id'),row_group=g,metadata={'inventory_sha256':expected_sha256,
                            'parent_sha256':entry['sha256'],'parent_full_hash_locally_verified':False,
                            'attribution':'HuggingFaceFW/FineWeb-Edu; ODC-By database terms and underlying web rights'}))
            finally:
                receipts.append({'shard':entry['path'],'ranges':reader.ranges,'parent_sha256':entry['sha256'],'parent_full_hash_locally_verified':False})
                reader.close()
    finally:
        evidence={'schema_version':1,'inventory_sha256':expected_sha256,'source_manifest_sha256':sha256_file(Path(source_manifest)),
            'range_bytes':transfer.bytes,'elapsed_seconds':time.monotonic()-started,'receipts':receipts,'exclusions':exclusions,
            'records_retained':len(records),'raw_redistribution':False}
        disk.write_bytes(out/'acquisition.json',canonical_json(evidence))
    return tuple(records),evidence
