#!/usr/bin/env python3
"""Metadata-only entry point for the new dense baseline campaign.

Only validate is implemented here. Native controls/profile/run/analyze are added
and qualified before they can be advertised or dispatched. No Metal import,
teacher/model construction or separate text/token-file read occurs in validate.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.dense_frontier import validate_contract_shape,calibration_plan,installed_algorithm_sources
from poc.contracts import thaw
from poc.recovery_endpoints import finite_json,verify_refs,file_sha,hash_value


def read(reference):
    verify_refs([reference]);return finite_json(reference['path'])


def metadata_validate(path):
    import yaml
    from poc.data.manifest import load_manifest
    c=thaw(validate_contract_shape(yaml.safe_load(Path(path).read_text()),root=ROOT))
    acceptance=read(c['reference_acceptance']);parent=read(c['reference_config']) if c['reference_config']['path'].endswith('.json') else None
    if acceptance.get('status')!='accepted':raise ValueError('independent old frontier admission required')
    receipt=read(acceptance['result']);independent=read(acceptance['contract'])
    if receipt.get('status')!='accepted' or receipt.get('configuration')!=c['reference_config'] or independent.get('config')!=c['reference_config']:
        raise ValueError('reference admission/configuration identity differs')
    verify_refs([c['reference_config']]);parent=yaml.safe_load(Path(c['reference_config']['path']).read_text())
    if c['inputs']!=parent['inputs'] or set(receipt.get('artifacts',{}))!=set(c['reference_artifacts']):
        raise ValueError('same complete reference population/grid required')
    prepared=finite_json(Path(parent['out'])/'preparation.json')
    if prepared['study_contract_sha256']!=hash_value(parent):raise ValueError('reference prepared contract differs')
    for label,ref in c['reference_artifacts'].items():
        old=prepared['artifacts'][label];verified=receipt['artifacts'][label]
        if (ref!={k:old[k] for k in ('directory','manifest_sha256')}
                or old['manifest_sha256']!=verified['manifest_sha256'] or old['complete_bytes']!=verified['complete_bytes']):
            raise ValueError('complete independently admitted reference artifact differs')
        manifest={'path':str(Path(ref['directory'])/'manifest.json'),'sha256':ref['manifest_sha256']}
        read(manifest)
    verify_refs([c['fit_manifest'],c['worker'],{k:c['inputs'][k] for k in ('path','sha256')}])
    if (c['worker']['sha256']!=parent['worker']['sha256']
            or c['worker']['sha256']!='7c062d1cd6686c247690a7367f021cdbad8a8a2b45aba358085d88c2507828b5'):
        raise ValueError('qualified offline native worker changed')
    scientific=read(parent['reference']['scientific_observation']);rows=scientific.get('populations',{}).get('tuning',[])
    if len(rows)!=128 or len({r['component_id'] for r in rows})!=128 or sum(r['full']['valid_targets'] for r in rows)!=65408:
        raise ValueError('complete fixed exploratory128component population required')
    if c['schema_version']==2:
        from poc.dense_frontier import verify_dense_parent_derivation
        fit,plan=verify_dense_parent_derivation(c,evaluation_components={r['component_id'] for r in rows})
    else:
        fit=load_manifest(Path(c['fit_manifest']['path']),expected_sha256=c['fit_manifest']['sha256'],roles={'fit'})
        plan=calibration_plan(fit,evaluation_components={r['component_id'] for r in rows})
    # Hash the whole installed original package without importlib.find_spec:
    # its parent __init__ would initialize Metal even in metadata-only mode.
    algorithms=installed_algorithm_sources()
    return {'schema_version':1,'status':'metadata_valid; executable native campaign/source/backend/profile/serial gates pending',
        'contract_sha256':file_sha(path),'calibration_plan':plan,'algorithm_sources':algorithms,
        'algorithm_sources_sha256':hash_value({'files':algorithms}),'reference_points':len(c['reference_artifacts']),'new_points':len(c['recipes']),
        'numerical_profile':c['numerics'],'model_loaded':False,'native_imports':False,'separate_text_token_reads':0,
        'resource_freeze':c['resource_freeze'],'scope':'reference/data-role/source metadata preparation; no new native scientific or practical result'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('validate',),required=True);args=parser.parse_args(argv)
    report=metadata_validate(args.config.absolute())
    print(json.dumps({k:v for k,v in report.items() if k not in ('algorithm_sources','calibration_plan')},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
