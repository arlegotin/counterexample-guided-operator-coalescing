"""Canonical model storage: external-base tiny fixtures and standalone deployment.

The legacy bank-state APIs require an exact dense base. Deployment bundles carry
all canonical floating or packed owners, tokenizer, licenses, and offline runtime.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from poc.bank import BankDecoder, canonicalize_partition, owner_census
from poc.contracts import BankStateRef
from poc.runtime.torch_decoder import TorchDecoder


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _state_categories(bank):
    # Enumerate actual registered owners instead of matching arbitrary name prefixes.
    records = owner_census(bank)
    parameters = dict(bank.named_parameters())
    return {category: {record.name: parameters[record.name] for record in records if record.category == category}
            for category in ('bank', 'residual')}


def write_bank_state(bank: BankDecoder, destination: Path) -> BankStateRef:
    if not isinstance(bank, BankDecoder):
        raise ValueError('bank state requires BankDecoder')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bank-state-', dir=destination.parent))
    try:
        files = {}
        for category, parameters in _state_categories(bank).items():
            if not parameters:
                continue
            filename = 'bank.safetensors' if category == 'bank' else 'residuals.safetensors'
            values = {name: parameter.detach().cpu().contiguous().clone() for name, parameter in parameters.items()}
            if any(not torch.isfinite(value).all().item() for value in values.values()):
                raise ValueError('cannot serialize nonfinite bank state')
            save_file(values, temporary / filename)
            files[filename] = _sha(temporary / filename)
        manifest = {'schema': 1, 'kind': 'external-base-bank-state', 'base_digest': bank.base_digest,
                    'checkpoint_digest': bank.manifest().checkpoint_digest,
                    'depth': bank.partition.depth, 'groups': bank.partition.groups, 'rank': bank.rank,
                    'files': files}
        (temporary / 'state.json').write_text(json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n')
        digest = _sha(temporary / 'state.json')
        # Do not overwrite an existing fixture; callers use a fresh destination.
        temporary.rename(destination)
        return BankStateRef(str(destination.resolve()), digest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('duplicate state metadata key')
        result[name] = value
    return result


def load_bank_state(ref: BankStateRef, base: TorchDecoder) -> BankDecoder:
    if not isinstance(ref, BankStateRef):
        raise ValueError('hashed BankStateRef required')
    directory = Path(ref.directory)
    manifest_path = directory / 'state.json'
    if manifest_path.is_symlink() or _sha(manifest_path) != ref.manifest_sha256:
        raise ValueError('state manifest hash mismatch')
    try:
        manifest = json.loads(manifest_path.read_text(), object_pairs_hook=_unique_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid state metadata') from exc
    expected_keys = {'schema', 'kind', 'base_digest', 'checkpoint_digest', 'depth', 'groups', 'rank', 'files'}
    if (not isinstance(manifest, dict) or set(manifest) != expected_keys or type(manifest['schema']) is not int
            or manifest['schema'] != 1 or manifest['kind'] != 'external-base-bank-state'):
        raise ValueError('unsupported bank state schema')
    if (not isinstance(base, TorchDecoder) or isinstance(base, BankDecoder)
            or manifest['base_digest'] != base.manifest().checkpoint_digest):
        raise ValueError('bank state requires the exact external dense base')
    partition = canonicalize_partition(manifest['groups'], manifest['depth'])
    rank = manifest['rank']
    if type(rank) is not int or rank < 0:
        raise ValueError('invalid rank in state')
    expected_files = {'bank.safetensors'} | ({'residuals.safetensors'} if rank else set())
    if not isinstance(manifest['files'], dict) or set(manifest['files']) != expected_files:
        raise ValueError('state weight files do not match rank')
    for filename, digest in manifest['files'].items():
        path = directory / filename
        if path.is_symlink() or _sha(path) != digest:
            raise ValueError('state weight hash mismatch')
    # Validate actual dimensions/partition/rank before model allocation and tensor load.
    result = BankDecoder(base, partition, rank, 0, initialize=False)
    with torch.no_grad():
        for category, parameters in _state_categories(result).items():
            if not parameters:
                continue
            filename = 'bank.safetensors' if category == 'bank' else 'residuals.safetensors'
            values = load_file(directory / filename, device='cpu')
            if set(values) != set(parameters):
                raise ValueError('state tensor keys differ from canonical owners')
            for name, parameter in parameters.items():
                value = values[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all().item():
                    raise ValueError('state tensor shape, dtype or finite values invalid')
                parameter.copy_(value.to(parameter.device))
    if result.manifest().checkpoint_digest != manifest['checkpoint_digest']:
        raise ValueError('reconstructed bank checkpoint digest mismatch')
    return result

# Standalone deployment format. Legacy external-base fixtures above remain distinct.
from pathlib import PurePosixPath
import stat
from poc.contracts import BundleManifest, BundleRef, ByteAccounting, ArchiveRecord, RuntimeSpec, DecoderAdapter, thaw

# Only this explicit inference code set is copied. No training/evaluation/data code.
RUNTIME_FILES = ('__init__.py', 'contracts.py', 'bundle.py', 'bank.py', 'export.py',
                 'runtime/__init__.py', 'runtime/torch_decoder.py', 'runtime/mlx_decoder.py')
RUNTIME_ARCHIVE = 'runtime/poc-runtime.zip'
WEIGHT_FILES = {'non_ffn': 'weights/non_ffn.safetensors', 'dense_ffn': 'weights/bank.safetensors',
                'bank': 'weights/bank.safetensors', 'residual': 'weights/residuals.safetensors'}


def _canonical(value):
    return (json.dumps(thaw(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _relative(name):
    if (type(name) is not str or not name or '\\' in name or str(PurePosixPath(name)) != name
            or PurePosixPath(name).is_absolute() or any(p in ('.', '..') for p in name.split('/'))):
        raise ValueError('unsafe bundle relative path')
    return name


def _regular_files(root):
    if root.is_symlink() or not root.is_dir(): raise ValueError('bundle must be a real directory')
    files = {}
    for path in root.rglob('*'):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode): continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('bundle rejects symlinks, hardlinks and nonregular files')
        name = _relative(path.relative_to(root).as_posix())
        files[name] = path
    return files


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid bundle JSON') from exc


def _packed_layout(shape, bits, group_size):
    if (len(shape)!=2 or any(type(n) is not int or n<1 for n in shape)
            or bits not in (2,3,4,8) or group_size not in (32,64,128)
            or shape[1]%group_size or shape[1]*bits%32):
        raise ValueError('unsupported native affine bit/group/shape combination; no padding fallback')
    # Three-bit groups require a multiple of 32 input values and packing in
    # blocks of 32. MLX supports it; GPTQ separately restricts to 2/4/8.
    rows,cols=shape
    return {'weight_shape':(rows,cols*bits//32),'scale_shape':(rows,cols//group_size),
            'packed_bytes':rows*cols*bits//8,'scale_bytes':rows*cols//group_size*4,
            'bias_bytes':rows*cols//group_size*4,'padding_elements':0}


def _expected_schema(metadata):
    """Derive permitted tensor owners/shapes from executable architecture, not tensor claims."""
    from poc.runtime.torch_decoder import TinyConfig
    config = metadata['config']
    dims = {k: config[k] for k in TinyConfig.__dataclass_fields__ if k in config}
    TinyConfig(**dims)  # finite dimensions/GQA/norm/RoPE validation
    h, f, v, n = (config[k] for k in ('hidden_size', 'intermediate_size', 'vocab_size', 'num_hidden_layers'))
    hd = metadata['head_dim']; q, kv = config['num_attention_heads'], config['num_key_value_heads']
    if type(hd) is not int or hd < 2 or hd % 2: raise ValueError('invalid head dimension')
    if config.get('model_type') != 'llama' or config.get('hidden_act', 'silu') != 'silu':
        raise ValueError('unsupported deployment architecture')
    rope = config.get('rope_parameters') or config.get('rope_scaling') or {}
    if rope.get('rope_type', rope.get('type', 'default')) != 'default': raise ValueError('unsupported RoPE')
    import math
    if type(metadata['rope_theta']) not in (int, float) or not math.isfinite(metadata['rope_theta']) or metadata['rope_theta'] <= 0:
        raise ValueError('invalid RoPE theta')
    kind, rank = metadata['kind'], metadata['rank']
    if type(rank) is not int or rank < 0 or rank > min(h, f): raise ValueError('invalid residual rank')
    groups = metadata['groups']
    if kind == 'dense':
        if groups != [[i] for i in range(n)] or rank != 0 or metadata['base_digest'] is not None:
            raise ValueError('invalid dense ownership')
    elif kind == 'compact':
        from poc.contracts import Partition
        Partition(tuple(tuple(g) for g in groups), n)
        if not isinstance(metadata['base_digest'], str) or len(metadata['base_digest']) != 64: raise ValueError('missing base provenance')
    else: raise ValueError('unknown bundle kind')
    expected = {'embed_tokens.weight': ((v,h), 'non_ffn'), 'norm.weight': ((h,), 'non_ffn')}
    if not config.get('tie_word_embeddings', False): expected['lm_head.weight'] = ((v,h), 'non_ffn')
    norm_sites = {'norm'}
    for i in range(n):
        for site in ('input_layernorm', 'post_attention_layernorm'):
            expected[f'layers.{i}.{site}.weight'] = ((h,), 'non_ffn'); norm_sites.add(f'layers.{i}.{site}')
        for name, shape in (('q_proj',(q*hd,h)), ('k_proj',(kv*hd,h)), ('v_proj',(kv*hd,h)), ('o_proj',(h,q*hd))):
            expected[f'layers.{i}.self_attn.{name}.weight'] = (shape, 'non_ffn')
            if config.get('attention_bias', False): expected[f'layers.{i}.self_attn.{name}.bias'] = ((shape[0],), 'non_ffn')
    if set(metadata['norm_epsilons']) != norm_sites: raise ValueError('missing norm epsilon sites')
    if any(type(e) not in (int,float) or not math.isfinite(e) or e <= 0 for e in metadata['norm_epsilons'].values()):
        raise ValueError('invalid actual norm epsilon')
    if config.get('mlp_bias', False): raise ValueError('FFN bias is unsupported')
    for i in range(n if kind == 'dense' else len(groups)):
        for name, shape in (('gate_proj',(f,h)), ('up_proj',(f,h)), ('down_proj',(h,f))):
            prefix = f'layers.{i}.mlp' if kind == 'dense' else f'bank.{i}'
            expected[f'{prefix}.{name}.weight'] = (shape, 'dense_ffn' if kind == 'dense' else 'bank')
    if rank:
        for i in range(n):
            for name, shape in (('gate_proj',(f,h)), ('up_proj',(f,h)), ('down_proj',(h,f))):
                prefix = f'layers.{i}.mlp.residuals.{name}'
                expected[prefix+'.A'] = ((rank,shape[1]), 'residual')
                expected[prefix+'.B'] = ((shape[0],rank), 'residual')
    return expected


def _verify_runtime_archive(path):
    import zipfile
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            expected = {'poc/'+f for f in RUNTIME_FILES}
            if len(entries) != len(expected) or {e.filename for e in entries} != expected:
                raise ValueError('unexpected or duplicate runtime archive entries')
            if [e.filename for e in entries] != sorted(expected):
                raise ValueError('runtime archive entries must be sorted')
            for entry in entries:
                _relative(entry.filename)
                if (entry.compress_type != zipfile.ZIP_STORED or entry.date_time != (1980,1,1,0,0,0)
                        or entry.create_system != 3 or entry.external_attr != (stat.S_IFREG|0o644)<<16
                        or entry.flag_bits & 1 or entry.extra or entry.comment):
                    raise ValueError('unsafe or noncanonical runtime archive entry')
            if archive.comment or archive.testzip() is not None: raise ValueError('runtime archive checksum mismatch')
    except (zipfile.BadZipFile,NotImplementedError) as exc:
        raise ValueError('invalid runtime archive') from exc


def verify_bundle(path: Path) -> BundleManifest:
    root = Path(path); files = _regular_files(root)
    if 'manifest.json' not in files: raise ValueError('missing bundle manifest')
    raw = _read_json(files['manifest.json'])
    keys = {'schema_version','file_entries','owners','config','kind','groups','rank','base_digest',
            'head_dim','rope_theta','norm_epsilons','tokenizer_digest','source_checkpoint_digest',
            'dtype','source_backend','dependency_lock_sha256','install_bytes','install_unknown_reason','tokenizer_files',
            'license_files','omitted_weight_files','execution_profile'}
    if not isinstance(raw,dict) or set(raw) != keys or type(raw['schema_version']) is not int or raw['schema_version'] != 1:
        raise ValueError('unsupported deployment manifest schema')
    entries = raw['file_entries']
    if not isinstance(entries,dict) or set(files) != set(entries) | {'manifest.json'} or 'manifest.json' in entries:
        raise ValueError('undeclared or missing bundle file')
    whitelist = {'config.json','quantization.json','requirements.lock'} | set(WEIGHT_FILES.values())
    whitelist.add(RUNTIME_ARCHIVE)
    for category in ('tokenizer','license'):
        names = raw[category+'_files']
        if not isinstance(names,list) or not names or len(set(names)) != len(names): raise ValueError('missing tokenizer/licenses')
        prefix = 'tokenizer/' if category == 'tokenizer' else 'LICENSES/'
        for name in names:
            _relative(name)
            if PurePosixPath(name).name != name: raise ValueError('invalid ancillary filename')
            whitelist.add(prefix+name)
    if 'tokenizer.json' not in raw['tokenizer_files'] or set(entries) != whitelist - set(raw['omitted_weight_files']):
        raise ValueError('file whitelist or omitted categories differ')
    for name, entry in entries.items():
        _relative(name)
        if not isinstance(entry,dict) or set(entry) != {'bytes','sha256'} or type(entry['bytes']) is not int or entry['bytes'] < 0:
            raise ValueError('invalid file entry')
        if files[name].stat().st_size != entry['bytes'] or _sha(files[name]) != entry['sha256']:
            raise ValueError('bundle file size/hash mismatch')
    _verify_runtime_archive(root/RUNTIME_ARCHIVE)
    if _sha(root/'requirements.lock') != raw['dependency_lock_sha256']: raise ValueError('dependency lock mismatch')
    quant = _read_json(root/'quantization.json')
    packed = quant.get('format') == 'mlx-affine'
    if packed:
        if set(quant) != {'format','schema_version','method','owners','calibration_manifest_sha256'} or quant['schema_version'] != 1 or quant['method'] not in ('rtn','gptq','awq','dwq','dynamic'):
            raise ValueError('invalid native quantization metadata')
    if _read_json(root/'config.json') != raw['config'] or (not packed and quant != {'format':'floating','schema_version':1}):
        raise ValueError('config or floating format metadata mismatch')
    import re
    if raw['source_backend'] not in ('torch_cpu','torch_mps'):
        raise ValueError('unsupported scientific source backend')
    for name in ('source_checkpoint_digest','dependency_lock_sha256','tokenizer_digest','base_digest'):
        value = raw[name]
        if value is None and name in ('tokenizer_digest','base_digest'): continue
        if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value): raise ValueError('invalid source provenance digest')
    expected = _expected_schema(raw)
    owners = raw['owners']
    if not isinstance(owners,list) or len(owners) != len(expected) or {o.get('name') for o in owners} != set(expected):
        raise ValueError('missing or duplicate canonical owners')
    from safetensors import safe_open
    actual = {}
    for file in set(WEIGHT_FILES.values()) & set(entries):
        with safe_open(root/file, framework='numpy') as stream:
            for key in stream.keys():
                if key in actual: raise ValueError('duplicate tensor owner across files')
                view = stream.get_slice(key); actual[key] = (file,tuple(view.get_shape()),view.get_dtype())
    wanted_tensors=set()
    if packed and (not isinstance(quant['owners'],dict) or not set(quant['owners']) <= set(expected)):
        raise ValueError('unknown packed owner')
    for name in expected:
        wanted_tensors.update([name+'.packed',name+'.scales',name+'.biases'] if packed and name in quant['owners'] else [name])
    if set(actual) != wanted_tensors: raise ValueError('missing or extra tensor owner')
    aliases = set()
    for owner in owners:
        if set(owner) != {'name','shape','dtype','category','file','aliases','use_sites','trainable'}:
            raise ValueError('invalid owner record')
        name = owner['name']; shape, category = expected[name]
        if packed and name in quant['owners']:
            fmt=quant['owners'][name]
            if set(fmt)!={'bits','group_size'} or category=='residual': raise ValueError('invalid packed owner format')
            layout=_packed_layout(shape,fmt['bits'],fmt['group_size'])
            if owner['dtype']!='uint32': raise ValueError('packed owner dtype mismatch')
            for suffix,key,code in (('.packed','weight_shape','U32'),('.scales','scale_shape','F32'),('.biases','scale_shape','F32')):
                if actual[name+suffix]!=(owner['file'],layout[key],code): raise ValueError('packed tensor shape/dtype mismatch')
        else:
            dtype = 'bfloat16' if packed and category=='residual' else raw['dtype']
            if dtype not in ('float32','float64','bfloat16') or owner['dtype'] != dtype: raise ValueError('unsupported floating dtype')
            code = {'float32':'F32','float64':'F64','bfloat16':'BF16'}[dtype]
            if actual[name] != (owner['file'],shape,code): raise ValueError('tensor owner shape/dtype mismatch')
        if (tuple(owner['shape']) != shape or owner['category'] != category or owner['file'] != WEIGHT_FILES[category]
                or type(owner['trainable']) is not bool):
            raise ValueError('tensor owner file/shape/dtype/category mismatch')
        wanted_aliases = [name]
        if name == 'embed_tokens.weight' and raw['config'].get('tie_word_embeddings',False): wanted_aliases.append('lm_head.weight')
        if owner['aliases'] != wanted_aliases or aliases.intersection(owner['aliases']): raise ValueError('invalid tied alias ownership')
        aliases.update(owner['aliases'])
        wanted_sites = wanted_aliases
        if category == 'bank':
            index = int(name.split('.')[1]); projection = name.split('.')[2]
            wanted_sites = [f'layers.{d}.mlp.{projection}.weight' for d in raw['groups'][index]]
        if owner['use_sites'] != wanted_sites: raise ValueError('invalid executed owner use sites')
    used = {owner['file'] for owner in owners}
    if set(raw['omitted_weight_files']) != set(WEIGHT_FILES.values()) - used:
        raise ValueError('invalid omitted categories')
    ByteAccounting(sum(p.stat().st_size for p in files.values()), install_bytes=raw['install_bytes'], install_unknown_reason=raw['install_unknown_reason'])
    return BundleManifest(1, entries, raw)


def _write_runtime(root):
    import importlib.resources
    import zipfile
    source_root = importlib.resources.files('poc')
    (root/'runtime').mkdir()
    with zipfile.ZipFile(root/RUNTIME_ARCHIVE,mode='w',compression=zipfile.ZIP_STORED) as archive:
        for filename in sorted(RUNTIME_FILES):
            info = zipfile.ZipInfo('poc/'+filename,date_time=(1980,1,1,0,0,0))
            info.create_system = 3; info.external_attr = (stat.S_IFREG|0o644)<<16
            archive.writestr(info,source_root.joinpath(filename).read_bytes())


def write_bundle(decoder: DecoderAdapter, destination: Path, runtime: RuntimeSpec) -> BundleRef:
    if not isinstance(runtime, RuntimeSpec): raise ValueError('explicit RuntimeSpec required')
    if not isinstance(decoder, TorchDecoder): raise ValueError('floating export requires TorchDecoder')
    # Checks mapping and actual norm epsilon before taking a snapshot.
    source_manifest = decoder.manifest()
    if list(decoder.named_buffers()): raise ValueError('unmodeled persistent buffers are unsupported')
    census = owner_census(decoder); params = dict(decoder.named_parameters())
    aliases = {}
    for name,p in decoder.named_parameters(remove_duplicate=False): aliases.setdefault(id(p),[]).append(name)
    is_bank = isinstance(decoder, BankDecoder)
    meta = {'schema_version':1, 'kind':'compact' if is_bank else 'dense',
            'config':thaw(source_manifest.config), 'head_dim':decoder.head_dim,'rope_theta':decoder.rope_theta,
            'rank':decoder.rank if is_bank else 0,
            'groups':[list(g) for g in decoder.partition.groups] if is_bank else [[i] for i in range(len(decoder.layers))],
            'base_digest':decoder.base_digest if is_bank else None, 'dtype':source_manifest.dtype,
            'source_checkpoint_digest':source_manifest.checkpoint_digest,'source_backend':source_manifest.backend,'tokenizer_digest':decoder.tokenizer_digest,
            'norm_epsilons':{'norm':decoder.norm.variance_epsilon}, 'owners':[],
            'install_bytes':runtime.install_bytes,'install_unknown_reason':runtime.install_unknown_reason,
            'tokenizer_files':sorted(runtime.tokenizer_files), 'license_files':sorted(runtime.license_files)}
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    meta['execution_profile'] = EXECUTION_PROFILE
    for i,layer in enumerate(decoder.layers):
        for site in ('input_layernorm','post_attention_layernorm'):
            meta['norm_epsilons'][f'layers.{i}.{site}'] = getattr(layer,site).variance_epsilon
    for record in census:
        meta['owners'].append({'name':record.name,'shape':list(record.shape),'dtype':source_manifest.dtype,
                               'category':record.category,'file':WEIGHT_FILES[record.category],
                               'aliases':aliases[id(params[record.name])], 'use_sites':list(record.use_sites), 'trainable':record.trainable})
    _expected_schema(meta)
    destination = Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bundle-',dir=destination.parent))
    try:
        used = {o['file'] for o in meta['owners']}
        (temporary/'weights').mkdir()
        for filename in sorted(used):
            values = {o['name']:params[o['name']].detach().cpu().contiguous().clone()
                      for o in meta['owners'] if o['file'] == filename}
            if any(not torch.isfinite(t).all().item() for t in values.values()): raise ValueError('nonfinite weights')
            save_file(values, temporary/filename)
            del values
        meta['omitted_weight_files'] = sorted(set(WEIGHT_FILES.values())-used)
        for filename,value in (('config.json',meta['config']),('quantization.json',{'format':'floating','schema_version':1})):
            (temporary/filename).write_bytes(_canonical(value))
        for folder,mapping in (('tokenizer',runtime.tokenizer_files),('LICENSES',runtime.license_files)):
            (temporary/folder).mkdir()
            for name,source in mapping.items():
                path = Path(source)
                if path.is_symlink() or not path.is_file(): raise ValueError('export input must be a regular local file')
                shutil.copyfile(path,temporary/folder/name)
        # Reject tokenizers requiring source files or remote code; tokenizer.json is the standalone engine.
        from tokenizers import Tokenizer
        Tokenizer.from_file(str(temporary/'tokenizer/tokenizer.json'))
        if 'tokenizer_config.json' in runtime.tokenizer_files:
            if _read_json(temporary/'tokenizer/tokenizer_config.json').get('auto_map'): raise ValueError('remote tokenizer code unsupported')
        shutil.copyfile(runtime.dependency_lock,temporary/'requirements.lock')
        meta['dependency_lock_sha256'] = _sha(temporary/'requirements.lock')
        _write_runtime(temporary)
        meta['file_entries'] = {name:{'bytes':p.stat().st_size,'sha256':_sha(p)}
                                for name,p in sorted(_regular_files(temporary).items())}
        (temporary/'manifest.json').write_bytes(_canonical(meta))
        verify_bundle(temporary)
        digest = _sha(temporary/'manifest.json')
        temporary.rename(destination)
        return BundleRef(str(destination.resolve()),digest)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def load_bundle(path: Path, *, backend: str, offline: bool = True) -> DecoderAdapter:
    if offline is not True: raise ValueError('bundle loading is local-only; offline must be true')
    manifest = verify_bundle(Path(path))
    if backend == 'torch':
        if _read_json(Path(path)/'quantization.json')['format']!='floating': raise ValueError('packed bundles require native MLX; scientific dequantization is not a deployment fallback')
        from poc.export import _load_scientific
        return _load_scientific(Path(path),manifest)
    if backend == 'mlx':
        from poc.runtime.mlx_decoder import MLXDecoder
        return MLXDecoder(Path(path),manifest)
    raise ValueError('unsupported bundle backend; choose torch or mlx')


def measure_bundle(path: Path) -> ByteAccounting:
    manifest = verify_bundle(Path(path))
    return ByteAccounting(sum(p.stat().st_size for p in _regular_files(Path(path)).values()),
                          install_bytes=manifest.metadata['install_bytes'],
                          install_unknown_reason=manifest.metadata['install_unknown_reason'])


def measure_resident(decoder) -> ByteAccounting:
    if isinstance(decoder,TorchDecoder):
        # Count real backing storage, reject views that conceal allocation overhead.
        allocations = {}
        for p in decoder.parameters():
            storage = p.untyped_storage(); key = (str(p.device),storage.data_ptr())
            allocations[key] = storage.nbytes()
        return ByteAccounting(0,resident_bytes=sum(allocations.values()),resident_unknown_reason=None,
                              owner_count=len(list(decoder.parameters())))
    from poc.runtime.mlx_decoder import MLXDecoder
    if not isinstance(decoder,MLXDecoder): raise ValueError('unsupported resident census')
    return decoder.resident_accounting()


def archive_bundle(path: Path, destination: Path) -> ArchiveRecord:
    import gzip
    import tarfile
    import time
    import zlib
    verify_bundle(Path(path)); files = _regular_files(Path(path))
    destination = Path(destination)
    if destination.exists() or destination.resolve().is_relative_to(Path(path).resolve()): raise ValueError('archive requires a new external destination')
    start = time.monotonic()
    with destination.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',compresslevel=9,mtime=0,fileobj=raw) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
                for name,p in sorted(files.items()):
                    info = tarfile.TarInfo(name); info.size = p.stat().st_size
                    info.mode = 0o644; info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with p.open('rb') as stream: archive.addfile(info,stream)
    unpack_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.bundle-unpack-',dir=destination.parent) as directory:
        unpacked = Path(directory)
        with tarfile.open(destination,mode='r:gz') as archive:
            archive.extractall(unpacked,filter='data')
        verify_bundle(unpacked)
        unpacked_bytes = sum(p.stat().st_size for p in _regular_files(unpacked).values())
    unpack_seconds = time.monotonic()-unpack_start
    return ArchiveRecord(str(destination.resolve()),destination.stat().st_size,_sha(destination),
                         f'gzip-level9/zlib-{zlib.ZLIB_RUNTIME_VERSION}',time.monotonic()-start,
                         destination.stat().st_size+unpacked_bytes,unpack_seconds=unpack_seconds,unpack_unknown_reason=None)
