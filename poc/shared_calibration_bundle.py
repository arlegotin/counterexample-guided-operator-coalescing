"""Versioned, content-verified research statistics; no executable model payload."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,re,tempfile
from uuid import uuid4
import torch
from poc.records import canonical_json
from poc.shared_calibration import SharedCalibrationSpec,SharedCalibrationResult,OwnerInputStatistics

_NAMES=('moment_sum','abs_sum','max_abs')
_COUNTERS={'statistics_bytes','raw_target_presentations','unique_context_target_positions','unique_contexts',
           'processed_context_tokens','operator_input_presentations','projection_input_presentations'}


def _digest(value):
 if type(value)is not str or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA256 required')
 return value


def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()


def _canonical_path(path):
 p=Path(path)
 if not p.is_absolute() or p.resolve()!=p or any(part.is_symlink() for part in (p,*p.parents)):raise ValueError('canonical statistics path without symlink ancestors required')
 return p


def _regular(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file() or p.stat().st_nlink!=1:raise ValueError('regular unlinked artifact files required')
 return p


@dataclass(frozen=True)
class CalibrationRef:
 directory:str
 manifest_sha256:str
 def __post_init__(self):
  if type(self.directory)is not str or not self.directory:raise ValueError('explicit statistics directory required')
  _digest(self.manifest_sha256)


def _layout(identity):
 if set(identity)!={'schema_version','spec_sha256','spec','owner_layout','owners','counters','hidden_sha256','window_records'} or type(identity['schema_version'])is not int or identity['schema_version']!=1:raise ValueError('closed version1 calibration result identity required')
 spec=SharedCalibrationSpec.from_mapping(identity['spec'])
 if identity['spec_sha256']!=spec.digest():raise ValueError('calibration spec digest differs')
 rows=identity['owner_layout'];owners=identity['owners'];counters=identity['counters']
 if not isinstance(rows,dict) or not rows or set(rows)!=set(owners) or len(rows)%3:raise ValueError('complete canonical projection owners required')
 if set(counters)!=_COUNTERS or any(type(v)is not int or v<=0 for v in counters.values()):raise ValueError('explicit positive calibration counters required')
 groups={};statistics_bytes=0;shapes={}
 for key,row in rows.items():
  match=re.fullmatch(r'bank\.(0|[1-9][0-9]*)\.(gate_proj|up_proj|down_proj)\.weight',key)
  if not match or set(row)!={'use_sites','shape'}:raise ValueError('invalid canonical projection layout')
  bank=int(match[1]);projection=match[2];uses=row['use_sites'];shape=row['shape']
  if (not isinstance(uses,list) or not uses or any(type(d)is not int or d<0 for d in uses) or uses!=sorted(set(uses))
      or not isinstance(shape,list) or len(shape)!=2 or any(type(d)is not int or d<=0 for d in shape)):raise ValueError('explicit use sites and projection matrix shape required')
  if bank in groups and groups[bank]!=uses:raise ValueError('one use group per canonical bank required')
  groups[bank]=uses;shapes.setdefault(bank,{})[projection]=shape
  width=shape[1];statistics_bytes+=8*(width*width+2*width)
  owner=owners[key]
  if set(owner)!={'moment_sum_sha256','abs_sum_sha256','max_abs_sha256','count','use_counts'}:raise ValueError('closed statistics owner identity required')
  for name in _NAMES:_digest(owner[name+'_sha256'])
  raw=counters['raw_target_presentations']
  if (type(owner['count'])is not int or owner['count']!=raw*len(uses)
      or owner['use_counts']!={str(d):raw for d in uses}
      or any(type(n)is not int for n in owner['use_counts'].values())):raise ValueError('canonical owner/all-use counts differ')
 if set(groups)!=set(range(len(groups))):raise ValueError('canonical bank indices must be contiguous')
 depths=[d for g in groups.values() for d in g]
 if sorted(depths)!=list(range(len(depths))):raise ValueError('use groups must partition all logical depths')
 for projections in shapes.values():
  if set(projections)!={'gate_proj','up_proj','down_proj'} or projections['gate_proj']!=projections['up_proj'] or projections['down_proj']!=list(reversed(projections['gate_proj'])):raise ValueError('complete compatible SwiGLU projection shapes required')
 windows=identity['window_records'];hidden=identity['hidden_sha256']
 if not isinstance(windows,list) or not windows or len(windows)>spec.max_windows or len(hidden)!=len(windows):raise ValueError('finite complete window records required')
 if statistics_bytes>spec.max_statistics_bytes or statistics_bytes!=counters['statistics_bytes']:raise ValueError('statistics storage bound/census differs')
 if (counters['processed_context_tokens']!=len(windows)*512 or counters['unique_contexts']>len(windows)
     or counters['unique_context_target_positions']>counters['raw_target_presentations']
     or counters['operator_input_presentations']!=counters['raw_target_presentations']*len(depths)
     or counters['projection_input_presentations']!=3*counters['operator_input_presentations']):raise ValueError('raw/operator/projection/context units differ')
 raw_total=0
 for index,(row,hidden_sha) in enumerate(zip(windows,hidden,strict=True)):
  if set(row)!={'spec_sha256','source_sha256','window_plan_sha256','window_sha256','presentation_index','raw_targets','projection_inputs','hidden_sha256'}:raise ValueError('closed window record required')
  _digest(hidden_sha);_digest(row['window_sha256'])
  if (row['spec_sha256']!=spec.digest() or row['source_sha256']!=spec.source_sha256
      or row['window_plan_sha256']!=spec.window_plan_sha256 or row['hidden_sha256']!=hidden_sha
      or type(row['presentation_index'])is not int or row['presentation_index']!=index
      or type(row['raw_targets'])is not int or not 0<row['raw_targets']<=511
      or type(row['projection_inputs'])is not int or row['projection_inputs']!=row['raw_targets']*len(depths)*3):raise ValueError('window/source/mask/work identity differs')
  raw_total+=row['raw_targets']
 if raw_total!=counters['raw_target_presentations']:raise ValueError('window raw-target sum differs')
 return spec


def _validate(result):
 if not isinstance(result,SharedCalibrationResult):raise ValueError('typed collected statistics required')
 identity=result.identity();spec=_layout(identity)
 for key,stats in result.owners.items():
  width=result.owner_layout[key]['shape'][1]
  for name,shape in (('moment_sum',(width,width)),('abs_sum',(width,)),('max_abs',(width,))):
   value=getattr(stats,name)
   if value.device.type!='cpu' or value.dtype!=torch.float64 or tuple(value.shape)!=shape or not torch.isfinite(value).all().item():raise ValueError('finite CPU FP64 projection statistics with exact shape required')
   if name!='moment_sum' and (value<0).any().item():raise ValueError('absolute statistics must be nonnegative')
 return identity,spec


def load_calibration(ref:CalibrationRef,*,expected_contract_sha256:str)->SharedCalibrationResult:
 from safetensors.torch import load_file
 _digest(expected_contract_sha256)
 if not isinstance(ref,CalibrationRef):raise ValueError('explicit calibration reference required')
 root=_canonical_path(ref.directory)
 if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()}!={'manifest.json','statistics.safetensors'}:raise ValueError('exact regular statistics artifact inventory required')
 manifest=_regular(root/'manifest.json');tensor_path=_regular(root/'statistics.safetensors')
 if manifest.stat().st_size>16*1024**2 or _sha(manifest)!=ref.manifest_sha256:raise ValueError('manifest byte identity differs')
 raw=json.loads(manifest.read_bytes())
 if (set(raw)!={'schema_version','study_contract_sha256','result_sha256','result_identity','files'}
     or type(raw['schema_version'])is not int or raw['schema_version']!=1
     or raw['study_contract_sha256']!=expected_contract_sha256):raise ValueError('closed statistics schema/contract mismatch')
 _digest(raw['result_sha256']);spec=_layout(raw['result_identity']);files=raw['files']
 if set(files)!={'statistics.safetensors'} or set(files['statistics.safetensors'])!={'bytes','sha256'}:raise ValueError('exact tensor file census required')
 entry=files['statistics.safetensors'];_digest(entry['sha256'])
 if (type(entry['bytes'])is not int or entry['bytes']!=tensor_path.stat().st_size
     or not 0<entry['bytes']<=spec.max_statistics_bytes+16*1024**2 or _sha(tensor_path)!=entry['sha256']):raise ValueError('tensor bytes/hash/storage bound differs')
 try:values=load_file(str(tensor_path),device='cpu')
 except Exception as exc:raise ValueError('invalid safe statistics tensor file') from exc
 identity=raw['result_identity'];keys={key+'.'+name for key in identity['owners'] for name in _NAMES}
 if set(values)!=keys:raise ValueError('missing/extra statistics tensors')
 owners={key:OwnerInputStatistics(*(values[key+'.'+name] for name in _NAMES),row['count'],{int(d):n for d,n in row['use_counts'].items()}) for key,row in identity['owners'].items()}
 result=SharedCalibrationResult(identity['spec_sha256'],spec,owners,identity['owner_layout'],identity['counters'],tuple(identity['hidden_sha256']),tuple(identity['window_records']))
 actual,_=_validate(result)
 if actual!=identity or result.digest()!=raw['result_sha256']:raise ValueError('statistics values/result identity differ')
 if _sha(_regular(manifest))!=ref.manifest_sha256 or _sha(_regular(tensor_path))!=entry['sha256']:raise ValueError('statistics changed during load')
 return result


def save_calibration(result:SharedCalibrationResult,destination:Path,*,expected_result_sha256:str,study_contract_sha256:str)->CalibrationRef:
 from safetensors.torch import save_file
 _digest(expected_result_sha256);_digest(study_contract_sha256)
 identity,_=_validate(result)
 if result.digest()!=expected_result_sha256:raise ValueError('collected statistics changed before serialization')
 target=_canonical_path(destination)
 if target.exists() or target.is_symlink():raise FileExistsError(target)
 target=target.resolve();target.parent.mkdir(parents=True,exist_ok=True)
 temporary=Path(tempfile.mkdtemp(prefix='.calibration-',dir=target.parent))
 try:
  values={key+'.'+name:getattr(stats,name).detach().contiguous().clone() for key,stats in result.owners.items() for name in _NAMES}
  tensor_path=temporary/'statistics.safetensors';save_file(values,str(tensor_path))
  raw={'schema_version':1,'study_contract_sha256':study_contract_sha256,'result_sha256':expected_result_sha256,
       'result_identity':identity,'files':{'statistics.safetensors':{'bytes':tensor_path.stat().st_size,'sha256':_sha(tensor_path)}}}
  (temporary/'manifest.json').write_bytes(canonical_json(raw));digest=_sha(temporary/'manifest.json')
  load_calibration(CalibrationRef(str(temporary),digest),expected_contract_sha256=study_contract_sha256)
  if result.digest()!=expected_result_sha256:raise ValueError('collected statistics mutated during serialization')
  if target.exists() or target.is_symlink():raise FileExistsError(target)
  temporary.rename(target)
  return CalibrationRef(str(target),digest)
 except BaseException as exc:
  if temporary.exists():
   failed=target.with_name(target.name+'.failed-'+uuid4().hex);temporary.rename(failed)
   (failed/'failure.json').write_bytes(canonical_json({'status':'failed','error':repr(exc),'expected_result_sha256':expected_result_sha256,'study_contract_sha256':study_contract_sha256,'scope':'failed statistics work retained; never a published calibration ref'}))
  raise
