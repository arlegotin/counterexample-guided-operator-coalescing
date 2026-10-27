"""Bounded local preparation. This module never imports a model or scores data."""
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

from poc.assets import (AssetError, AssetRequest, RangeAsset, TransferBudget, fetch_asset,
                        fetch_git_blob, file_sha256, resolve_asset)
from poc.config import ConfigError, PROJECT_ROOT
from poc.data.benchmarks import blimp_roles, official_role
from poc.data.dedup import DedupPolicy, assign_roles, build_components, quarantine_overlap
from poc.data.manifest import DataManifest, DocumentRecord, ExposureLedger
from poc.data.windows import TokenDocument, WindowPolicy, make_windows, parse_wikitext_articles, rolling_windows
from poc.records import canonical_json


TEACHER='HuggingFaceTB/SmolLM2-135M'
FINEWEB='HuggingFaceFW/fineweb-edu'
WIKITEXT='Salesforce/wikitext'
ARC='allenai/ai2_arc'
BLIMP='alexwarstadt/blimp'


def sampled_rows(group_sizes, *, seed: int, group_count: int, rows_per_group: int):
    rng=random.Random(seed)
    groups=sorted(rng.sample(range(len(group_sizes)),min(group_count,len(group_sizes))))
    return [(g,sorted(rng.sample(range(group_sizes[g]),min(rows_per_group,group_sizes[g])))) for g in groups]


def check_source_gate(path: Path):
    data=json.loads(Path(path).read_text())
    if data.get('source_gate',{}).get('status') != 'passed_local_preparation':
        raise AssetError('source gate is not qualified for local preparation')
    if set(data.get('sources',{})) != {TEACHER,FINEWEB,WIKITEXT,ARC,BLIMP} or not data.get('terms_evidence'):
        raise AssetError('source gate lacks required sources/terms evidence')
    return data




def check_exposure_history(root: Path):
    """Changing output directories/omitting a parent cannot reset prior access.

    Scan ledger metadata only. With any prior preparation exposure, fresh pool
    construction fails closed; future explicit reconciliation must use the
    immutable fingerprint history, never simply choose a new directory/salt.
    """
    def fail(error):
        raise AssetError('cannot verify historical exposure ledgers') from error
    for directory, _subdirs, filenames in os.walk(root, onerror=fail):
        if 'exposures.jsonl' in filenames and (Path(directory)/'exposures.jsonl').stat().st_size:
            raise AssetError('historical exploratory exposure blocks fresh pool preparation; reconcile immutable lineage first')

def check_parent_lineage(path: Path, expected_sha256: str):
    if file_sha256(path) != expected_sha256:
        raise AssetError('parent preparation manifest hash mismatch')
    parent=json.loads(Path(path).read_text())
    if parent['status']!='prepared_outcome_blind' or parent['protected_predictions'] or parent['training_updates']:
        raise AssetError('parent pool is not fresh outcome-blind preparation')
    exposure=Path(path).parent/'exposures.jsonl'
    if exposure.exists() and exposure.stat().st_size:
        raise AssetError('parent exploratory exposure prevents fresh protected reassignment; preserve exposure lineage')
    return {'parent_manifest_sha256':expected_sha256,'parent_preparation_manifest':str(path),
            'parent_exploratory_exposures':0,'expansion_phase':'fresh initial pool before any exploratory/scoring access'}


def _plain(value):
    if isinstance(value,Path): return str(value)
    if isinstance(value,dict): return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_plain(v) for v in value]
    return value


def _preparation_policy(prep):
    return _plain(asdict(prep))


def _execution_identity(config):
    import subprocess
    code={str(p.relative_to(PROJECT_ROOT)):file_sha256(p) for p in sorted((PROJECT_ROOT/'poc').rglob('*.py'))}
    return {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),
            'dirty_patch_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--'],cwd=PROJECT_ROOT)).hexdigest(),
            'code_sha256':code,'complete_config':_plain(asdict(config)),
            'dependency_lock_sha256':file_sha256(PROJECT_ROOT/'provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt'),
            'qualification_sha256':file_sha256(PROJECT_ROOT/'provenance/qualification-2026-10-27.json')}

def _write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=canonical_json(value)
    if path.exists():
        if path.read_bytes()!=encoded: raise AssetError(f'immutable output changed: {path}')
    else:
        with path.open('xb') as stream: stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _request(repo, source, entry, cap):
    prefix='datasets/' if repo != TEACHER else ''
    url=f"https://huggingface.co/{prefix}{repo}/resolve/{source['revision']}/{entry['path']}"
    return resolve_asset(AssetRequest(repo,source['revision'],entry['path'],url,entry['sha256'],entry['bytes'],
                                      source['license'],source['terms_url'],True,cap))


def run_prepare(config):
    prep=config.preparation
    if prep is None: raise ConfigError('resolved preparation paths and byte/record/token ceilings required')
    source_gate=check_source_gate(prep.source_manifest)  # before output creation and all network
    sources=source_gate['sources']
    if config.teacher.repository != TEACHER or config.teacher.revision != sources[TEACHER]['revision'] or config.teacher.tokenizer_revision != sources[TEACHER]['revision']:
        raise ConfigError('teacher/tokenizer pins disagree with qualified source manifest')
    qualification=json.loads((PROJECT_ROOT/'provenance/qualification-2026-10-27.json').read_text())
    component=json.loads((PROJECT_ROOT/'provenance/component-validation-2026-10-27.json').read_text())
    if qualification['compute_path_status']!='passed' or component['status']!='passed':
        raise ConfigError('WP00 compute-path and WP01 component gates required')
    lineage=(check_parent_lineage(prep.parent_preparation_manifest,prep.parent_manifest_sha256)
             if prep.parent_preparation_manifest is not None else {'parent_manifest_sha256':None,'parent_exploratory_exposures':0})
    out=prep.output_dir
    final=out/'preparation.json'
    if final.exists():
        summary=json.loads(final.read_text())
        if summary['source_manifest_sha256']!=file_sha256(prep.source_manifest) or {'parent_preparation_manifest':None,'parent_manifest_sha256':None,**summary['preparation_policy']}!=_preparation_policy(prep):
            raise AssetError('existing preparation policy/source changed; fresh output and lineage review required')
        if config.teacher.weights_sha256 is not None and config.teacher.weights_sha256!=summary['teacher_asset_manifest_sha256']:
            raise AssetError('teacher asset manifest disagrees with configured hash')
        for relative,digest in summary['output_files'].items():
            if file_sha256(out/relative)!=digest: raise AssetError('prepared output hash changed')
        import uuid
        verification={'schema_version':1,'status':'integrity_verified','preparation_manifest_sha256':file_sha256(final),
                      'verified_file_count':len(summary['output_files']),'execution_identity':_execution_identity(config)}
        _write(PROJECT_ROOT/'artifacts/preparation-verifications'/f'{uuid.uuid4().hex}.json',verification)
        return summary
    # Reject rebuilding after any exploratory access, even in a different output directory.
    check_exposure_history(PROJECT_ROOT)
    exposure=out/'exposures.jsonl'
    if exposure.exists() and exposure.stat().st_size:
        raise AssetError('unsafe rebuild after exploratory exposure: preserve lineage and quarantine against prior exposures')
    if shutil.disk_usage(PROJECT_ROOT).free < prep.max_disk_bytes:
        raise AssetError('preparation disk ceiling exceeds current free disk')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    execution_identity=_execution_identity(config)
    budget=TransferBudget(prep.max_download_bytes)
    ranges_budget=TransferBudget(prep.max_range_bytes)
    assets=[]
    def fetch(repo,entry):
        source=sources[repo]; local=out/'assets'/repo.replace('/','--')/entry['path']
        if not local.exists(): budget.reserve(entry['bytes'])
        if entry['sha256']:
            m=fetch_asset(_request(repo,source,entry,prep.max_download_bytes),local)
        else:
            m=fetch_git_blob(repo,source['revision'],entry['path'],entry['git_blob_sha1'],entry['bytes'],local,
                             license=source['license'],terms_url=source['terms_url'],max_bytes=min(prep.max_download_bytes,8*1024**2))
        assets.append(asdict(m)); return local
    print('prepare: source/terms and compute/component prerequisites passed; fetching pinned teacher assets',flush=True)
    model_dir=out/'assets'/TEACHER.replace('/','--')
    for entry in sources[TEACHER]['files']: fetch(TEACHER,entry)
    teacher_assets=[m for m in assets if m['source']['repository']==TEACHER]
    teacher_sha=_write(out/'teacher-assets.json',{'schema_version':1,'assets':[{**m,'local_path':str(Path(m['local_path']).relative_to(out))} for m in teacher_assets]})
    if config.teacher.weights_sha256 is not None and config.teacher.weights_sha256!=teacher_sha:
        raise AssetError('teacher asset manifest disagrees with configured hash')
    # Tokenizer is Rust tokenizers only. It has no remote-code import or model path.
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(model_dir/'tokenizer.json'))
    tokenizer_config=json.loads((model_dir/'tokenizer_config.json').read_text())
    eos=tokenizer_config['eos_token']
    if isinstance(eos,dict): eos=eos['content']
    eos_id=tokenizer.token_to_id(eos)
    if eos_id is None: raise AssetError('original tokenizer EOS token missing')
    # Native SmolLM2 tokenizer inserts no BOS/EOS on encode. Deliberately add one
    # real EOS per independent text sequence and record this convention.
    probe=tokenizer.encode('Owned tokenizer convention probe.',add_special_tokens=True).ids
    plain=tokenizer.encode('Owned tokenizer convention probe.',add_special_tokens=False).ids
    if probe!=plain: raise AssetError('unexpected original tokenizer special-token insertion')
    records=[]; natural=[]; benchmark=[]; windows=[]; token_ids_count=0; exclusions=[]; output_files={}
    def save_record(repo,entry,row,text,*,source_record_id=None,url=None,article_id=None,row_group=None,metadata=None,token_fields=None):
        nonlocal token_ids_count
        if len(records)>=prep.max_total_records: raise AssetError('record ceiling exceeded')
        encoded=text.encode()
        if len(encoded)>prep.max_document_bytes: raise AssetError('document byte ceiling exceeded')
        identity=hashlib.sha256(canonical_json({'source':repo,'revision':sources[repo]['revision'],'shard':entry['path'],'row':row,'record_id':source_record_id})).hexdigest()
        base=out/'records'/identity[:2]/identity
        text_path=base.with_suffix('.txt');token_path=base.with_suffix('.tokens.json')
        text_path.parent.mkdir(parents=True,exist_ok=True)
        if text_path.exists():
            if text_path.read_bytes()!=encoded: raise AssetError('record content changed')
        else: text_path.write_bytes(encoded)
        fields={'text':text} if token_fields is None else token_fields
        tokens={k:tokenizer.encode(v,add_special_tokens=False).ids+[eos_id] for k,v in fields.items()}
        token_ids_count+=sum(map(len,tokens.values()))
        if token_ids_count>prep.max_token_ids: raise AssetError('token ID ceiling exceeded')
        th=_write(token_path,{'document_id':identity,'fields':tokens})
        output_files[str(text_path.relative_to(out))]=hashlib.sha256(encoded).hexdigest()
        output_files[str(token_path.relative_to(out))]=th
        record=DocumentRecord(identity,repo,sources[repo]['revision'],entry['path'],row,text_path=str(text_path),
            text_sha256=hashlib.sha256(encoded).hexdigest(),url=url,article_id=article_id,source_record_id=source_record_id,row_group=row_group,
            local_source_sha256=entry.get('local_sha256'),token_path=str(token_path),token_sha256=th,metadata=metadata or {})
        records.append(record)
        return record,tokens
    print('prepare: sampling FineWeb shard/row-group/row IDs without first-N selection',flush=True)
    rng=random.Random(prep.seed)
    shards=sorted(rng.sample(sources[FINEWEB]['files'],min(prep.fineweb_shards,len(sources[FINEWEB]['files']))),key=lambda x:x['path'])
    sampling=[]; fineweb_text_bytes=0; fineweb_tokens=0
    import pyarrow.parquet as pq
    for index,entry in enumerate(shards):
        range_reader=RangeAsset(_request(FINEWEB,sources[FINEWEB],entry,entry['bytes']),out/'ranges'/Path(entry['path']).stem,
                                 ranges_budget,max_request_bytes=prep.max_range_request_bytes)
        parquet=pq.ParquetFile(range_reader)
        sizes=[parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)]
        choices=sampled_rows(sizes,seed=prep.seed+index+1,group_count=prep.row_groups_per_shard,rows_per_group=prep.rows_per_group)
        offsets=[sum(sizes[:g]) for g in range(len(sizes))]
        for group,rows in choices:
            table=parquet.read_row_group(group)
            for local_row in rows:
                item=table.slice(local_row,1).to_pylist()[0]
                if len(natural)>=prep.max_fineweb_records: raise AssetError('FineWeb record ceiling exceeded')
                if len(item['text'].encode())>prep.max_document_bytes:
                    exclusions.append({'source':FINEWEB,'shard':entry['path'],'row_group':group,'row':offsets[group]+local_row,'reason':'predeclared document byte ceiling'});continue
                source_metadata={k:v for k,v in item.items() if k!='text'}
                r,t=save_record(FINEWEB,entry,offsets[group]+local_row,item['text'],source_record_id=item['id'],url=item.get('url'),row_group=group,metadata=source_metadata)
                natural.append(r);fineweb_text_bytes+=len(item['text'].encode());fineweb_tokens+=len(t['text'])
                windows.extend(asdict(w) for w in make_windows(TokenDocument(r.document_id,tuple(t['text'])),WindowPolicy(seed=prep.seed)))
        sampling.append({'shard':entry['path'],'revision':sources[FINEWEB]['revision'],'parent_sha256':entry['sha256'],
                         'parent_full_hash_locally_verified':False,'group_sizes':sizes,'selected_rows':choices,'range_files':range_reader.ranges})
        budget.reserve(range_reader.budget.bytes-sum(x['bytes'] for s in sampling[:-1] for x in s['range_files']))
        range_reader.close()
        print(f'prepare: FineWeb shard {index+1}/{len(shards)}; records={len(natural)} range_bytes={ranges_budget.bytes}',flush=True)
    # Hash exact retained range bytes in each document's local-source evidence.
    for i,r in enumerate(natural):
        evidence=next(s for s in sampling if s['shard']==r.shard)
        natural[i]=replace(r,local_source_sha256=hashlib.sha256(canonical_json({'ranges':evidence['range_files']})).hexdigest())
    print('prepare: downloading official WikiText validation/test and ARC splits',flush=True)
    benchmark_roles={}; benchmark_membership={}; wiki_records=[]; blimp_records=[]
    for repo in (WIKITEXT,ARC):
        for entry in sources[repo]['files']:
            path=fetch(repo,entry); entry={**entry,'local_sha256':file_sha256(path)}
            split=Path(entry['path']).name.split('-')[0]
            items=pq.read_table(path).to_pylist()
            if repo==WIKITEXT:
                for article,text in enumerate(parse_wikitext_articles(item['text'] for item in items)):
                    r,t=save_record(repo,entry,article,text,article_id=f'{repo}:{split}:{article}',metadata={'official_split':split,'config':'wikitext-103-raw-v1'})
                    wiki_records.append(r);benchmark.append(r)
                    windows.extend(asdict(w) for w in rolling_windows(TokenDocument(r.document_id,tuple(t['text']))))
                    benchmark_roles[r.document_id]=official_role(split);benchmark_membership[r.document_id]=r.document_id
            else:
                for row,item in enumerate(items):
                    text=json.dumps(item,ensure_ascii=False,separators=(',',':'))
                    fields={'question':item['question'],**{f'choice_{i}':v for i,v in enumerate(item['choices']['text'])}}
                    r,t=save_record(repo,entry,row,text,source_record_id=item['id'],metadata={'official_split':split,'config':entry['path'].split('/')[0]},token_fields=fields)
                    benchmark.append(r);benchmark_roles[r.document_id]=official_role(split);benchmark_membership[r.document_id]=r.document_id
    print('prepare: downloading all 67 original BLiMP paradigms; allocating fixed 10% dev per paradigm',flush=True)
    for entry in sources[BLIMP]['files']:
        path=fetch(BLIMP,entry);entry={**entry,'local_sha256':file_sha256(path)}
        paradigm=Path(entry['path']).stem
        for row,line in enumerate(path.read_text().splitlines()):
            item=json.loads(line)
            r,t=save_record(BLIMP,entry,row,line,source_record_id=str(item['pairID']),metadata={'paradigm':paradigm,'pair_id':str(item['pairID'])},
                            token_fields={key:item[key] for key in ('sentence_good','sentence_bad')})
            blimp_records.append(r);benchmark.append(r)
    paradigms={}
    for r in blimp_records: paradigms.setdefault(r.metadata['paradigm'],[]).append(r.metadata['pair_id'])
    roles=blimp_roles(paradigms,prep.salt)
    for r in blimp_records:
        benchmark_roles[r.document_id]=roles[r.metadata['paradigm'],r.metadata['pair_id']];benchmark_membership[r.document_id]=r.document_id
    print('prepare: building duplicate components, cross-source quarantine, and sensitivity manifests',flush=True)
    # Project minimal synthetic records are supplied by the existing generator modules;
    # this manifest is solely the named public-source preparation population.
    def overlap_texts():
        for r in benchmark:
            if r.source==WIKITEXT: yield r
            else:
                item=json.loads(r.read_text())
                pieces=([item['sentence_good'],item['sentence_bad']] if r.source==BLIMP else [item['question']]+item['choices']['text'])
                for i,text in enumerate(pieces): yield replace(r,document_id=f'{r.document_id}:{i}',text=text,text_path=None,text_sha256=None)
    sensitivities={}; primary=None; component_primary=None
    for threshold in (.7,.8,.9):
        policy=DedupPolicy(threshold=threshold,seed=prep.seed)
        components=build_components(natural,policy)
        manifest=assign_roles(components,prep.salt)
        manifest=quarantine_overlap(manifest,overlap_texts(),policy)
        sensitivities[str(threshold)]={'component_count':len(set(components.membership.values())), 'near_edges':len(components.near_edges),
            'candidate_pairs':components.candidate_pairs,'quarantine_records':len(manifest.quarantine),'role_counts':dict(Counter(manifest.roles.values()))}
        if threshold==.8: primary=manifest;component_primary=components
    all_manifest=DataManifest(primary.records+tuple(benchmark),primary.roles|benchmark_roles,primary.membership|benchmark_membership,
        prep.salt,exposure_path=str(exposure),lineage_sha256=None,quarantine=primary.quarantine)
    ExposureLedger(exposure).validate_rebuild(all_manifest,DedupPolicy())
    outputs={'data-manifest.json':asdict(all_manifest),'components.json':asdict(component_primary),
             'windows.json':{'schema_version':1,'natural_policy':asdict(WindowPolicy(seed=prep.seed)),'wikitext_policy':{'context':512,'stride':256},'windows':windows},
             'sampling.json':{'schema_version':1,'seed':prep.seed,'shards':sampling,'exclusions':exclusions},
             'assets.json':{'schema_version':1,'assets':assets},'sensitivity.json':{'thresholds':sensitivities,'screen':'64 seeded MinHash permutations;32 bands of2; exact Jaccard verified candidates; approximate recall; exact cross-source inverted-index screening'}}
    for filename,value in outputs.items(): output_files[filename]=_write(out/filename,value)
    output_files['teacher-assets.json']=file_sha256(out/'teacher-assets.json')
    for asset in assets: output_files[str(Path(asset['local_path']).relative_to(out))]=asset['sha256']
    for sample in sampling:
        for part in sample['range_files']: output_files[str(Path(part['local_path']).relative_to(out))]=part['sha256']
    disk=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if disk>prep.max_disk_bytes: raise AssetError('actual preparation disk ceiling exceeded')
    summary={'schema_version':1,'status':'prepared_outcome_blind','source_manifest_sha256':file_sha256(prep.source_manifest),
       'preparation_policy':_preparation_policy(prep),'lineage':lineage,'execution_identity':execution_identity,
       'teacher_asset_manifest_sha256':teacher_sha,'tokenizer_revision':sources[TEACHER]['revision'],
       'tokenizer_convention':{'native_add_special_tokens_equal_plain':True,'bos_inserted':False,'eos_appended_once_per_sequence':True,'eos_token_id':eos_id},
       'counts':{'FineWeb':len(natural),'WikiText_articles':len(wiki_records),'ARC':sum(r.source==ARC for r in benchmark),'BLiMP_pairs':len(blimp_records),'BLiMP_paradigms':len(paradigms),'token_ids':token_ids_count,'windows':len(windows),'roles':dict(Counter(all_manifest.roles.values()))},
       'storage':{'fineweb_original_text_bytes':fineweb_text_bytes,'fineweb_token_ids':fineweb_tokens,'utf8_bytes_per_token_id':fineweb_text_bytes/fineweb_tokens if fineweb_tokens else None,
                  'uint32_token_bytes_per_utf8_byte':4*fineweb_tokens/fineweb_text_bytes if fineweb_text_bytes else None,'actual_file_bytes':disk,'download_bytes_this_attempt':budget.bytes,'fineweb_range_bytes':ranges_budget.bytes},
       'dedup_sensitivity':sensitivities,'wall_seconds':time.monotonic()-started,'protected_predictions':0,'training_updates':0,
       'final_study_budget_status':'unresolved; routine preparation ceilings only','output_files':output_files}
    # The final receipt is written last: partial failures cannot resemble completed preparation.
    _write(final,summary)
    return summary
