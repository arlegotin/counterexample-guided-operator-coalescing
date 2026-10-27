"""Catch duplicated owners, unsafe payloads and non-deterministic accounting."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest
import torch
from test_model_parity import batch

pytestmark = pytest.mark.cpu


def api():
    from poc import bundle
    assert hasattr(bundle, 'write_bundle'), 'standalone bundle serializer is missing'
    return bundle


def runtime(tmp_path):
    from poc.contracts import RuntimeSpec
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({f't{i}': i for i in range(41)}, unk_token='t0'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = tmp_path / 'fixture-tokenizer.json'; tok.save(str(tokenizer))
    license = tmp_path / 'LICENSE'; license.write_text('Fixture weights and tokenizer: public domain.\n')
    lock = tmp_path / 'requirements.lock'; lock.write_text('torch==2.14.1\nsafetensors==0.8.0\nmlx==0.32.4\n')
    return RuntimeSpec(tokenizer_files={'tokenizer.json': str(tokenizer)},
                       license_files={'fixture.txt': str(license)}, dependency_lock=str(lock))


def model(kind):
    from poc.runtime.torch_decoder import TinyConfig, make_tiny_decoder
    from poc.bank import BankDecoder, canonicalize_partition
    dense = make_tiny_decoder(TinyConfig(), seed=39, dtype=torch.float32)
    # Preserve actual mutable epsilons rather than trusting original config.
    dense.norm.variance_epsilon = 3e-5
    dense.layers[1].input_layernorm.variance_epsilon = 2e-5
    if kind == 'dense': return dense
    return BankDecoder(dense, canonicalize_partition([[0, 2], [1]], 3), int(kind), 9)


@pytest.mark.parametrize('kind', ['dense', '0', '2'])
def test_complete_payload_roundtrip_and_canonical_owners(tmp_path, kind):
    io = api(); source = model(kind)
    ref = io.write_bundle(source, tmp_path / 'bundle', runtime(tmp_path))
    manifest = io.verify_bundle(Path(ref.directory))
    assert 'manifest.json' not in manifest.file_entries
    assert ref.manifest_sha256 == hashlib.sha256((Path(ref.directory) / 'manifest.json').read_bytes()).hexdigest()
    measured = io.measure_bundle(Path(ref.directory))
    files = [p for p in Path(ref.directory).rglob('*') if p.is_file()]
    assert measured.payload_bytes == sum(p.stat().st_size for p in files)
    assert measured.install_bytes is None and measured.install_unknown_reason
    loaded = io.load_bundle(Path(ref.directory), backend='torch')
    from poc.bank import owner_census
    assert owner_census(loaded) == owner_census(source)
    assert loaded.lm_head.weight is loaded.embed_tokens.weight
    assert loaded.norm.variance_epsilon == 3e-5
    assert loaded.layers[1].input_layernorm.variance_epsilon == 2e-5
    torch.testing.assert_close(loaded.logits(batch(offset=7)), source.logits(batch(offset=7)), atol=1e-7, rtol=0)
    cache = loaded.new_cache(start_position=7)
    parts = [loaded.logits(batch([[token]], offset=7+i), cache=cache) for i, token in enumerate([1,4,7,2,3])]
    torch.testing.assert_close(torch.cat(parts, 1), loaded.logits(batch(offset=7)), atol=2e-6, rtol=0)
    assert len({id(c) for c in cache.layers}) == 3
    resident = io.measure_resident(loaded)
    assert resident.owner_count == len(owner_census(source))
    assert resident.resident_bytes == sum(p.numel()*p.element_size() for p in source.parameters())
    assert (Path(ref.directory) / 'weights/residuals.safetensors').exists() == (kind == '2')
    first = io.archive_bundle(Path(ref.directory), tmp_path / 'first.tar.gz')
    second = io.archive_bundle(Path(ref.directory), tmp_path / 'second.tar.gz')
    assert Path(first.path).read_bytes() == Path(second.path).read_bytes()
    assert first.archive_bytes == Path(first.path).stat().st_size


@pytest.mark.parametrize('attack', ['missing', 'extra', 'changed', 'symlink', 'hardlink', 'traversal', 'external', 'owner'])
def test_bundle_rejects_integrity_and_accounting_tricks(tmp_path, attack):
    io = api(); ref = io.write_bundle(model('2'), tmp_path / 'bundle', runtime(tmp_path))
    root = Path(ref.directory); path = root / 'weights/bank.safetensors'
    if attack == 'missing': path.unlink()
    elif attack == 'extra': (root / 'accidental.ckpt').write_bytes(b'external weights')
    elif attack == 'changed': path.write_bytes(path.read_bytes() + b'x')
    elif attack in ('symlink', 'hardlink'):
        outside = tmp_path / 'outside'; path.rename(outside)
        if attack == 'symlink': path.symlink_to(outside)
        else: os.link(outside, path)
    else:
        meta = json.loads((root / 'manifest.json').read_text())
        if attack == 'traversal': meta['file_entries']['../outside'] = meta['file_entries'].pop('weights/bank.safetensors')
        elif attack == 'external': meta['owners'][0]['file'] = '/external/weights.safetensors'
        else: meta['owners'].append(meta['owners'][0])
        (root / 'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError): io.verify_bundle(root)


def test_load_rejects_rehashed_tensor_shape_and_unavailable_backend(tmp_path):
    from safetensors.torch import load_file, save_file
    io = api(); ref = io.write_bundle(model('dense'), tmp_path / 'bundle', runtime(tmp_path))
    root = Path(ref.directory); path = root / 'weights/non_ffn.safetensors'
    values = load_file(path); key = next(iter(values)); values[key] = values[key][:1]; save_file(values, path)
    raw = json.loads((root / 'manifest.json').read_text())
    raw['file_entries']['weights/non_ffn.safetensors'] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / 'manifest.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError): io.load_bundle(root, backend='torch')


def test_export_native_is_complete_and_import_is_mlx_lazy(tmp_path):
    io = api(); export = importlib.import_module('poc.export')
    ref = export.export_native(model('0'), tmp_path / 'bundle', runtime(tmp_path))
    assert io.verify_bundle(Path(ref.directory)).schema_version == 1
    import subprocess, sys
    result = subprocess.run([sys.executable, '-c', "import sys; import poc.bundle, poc.export; assert 'mlx.core' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cpu_source_digest_reconstruction_is_checked_and_backend_is_explicit(tmp_path):
    io = api(); ref = io.write_bundle(model('dense'),tmp_path/'bundle',runtime(tmp_path))
    root = Path(ref.directory)
    with pytest.raises(ValueError,match='backend'): io.load_bundle(root,backend='cuda')
    raw = json.loads((root/'manifest.json').read_text()); raw['source_checkpoint_digest'] = '0'*64
    (root/'manifest.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='digest'): io.load_bundle(root,backend='torch')


def test_export_cli_receipt_and_validation_routes(tmp_path,capsys):
    from poc.cli import main
    io = api(); ref = io.write_bundle(model('0'),tmp_path/'source',runtime(tmp_path))
    code = main(['export','--run','fixture-run','--attempt','fixture-attempt',
                 '--source-bundle',ref.directory,'--out',str(tmp_path/'reexport')])
    assert code == 0
    receipt = json.loads((tmp_path/'reexport.receipt.json').read_text())
    assert receipt['run_label'] == 'fixture-run' and receipt['attempt_label'] == 'fixture-attempt'
    assert receipt['source_manifest_sha256'] == ref.manifest_sha256
    assert receipt['source_kind'] == 'standalone_bundle'
    assert not receipt['fitted_checkpoint_resolved']
    assert main(['validate','--suite','export','--bundle',str(tmp_path/'reexport'),'--backend','torch']) == 0
    assert 'scientific' in capsys.readouterr().out
    (tmp_path/'reexport/weights/bank.safetensors').unlink()
    assert main(['validate','--suite','export','--bundle',str(tmp_path/'reexport'),'--backend','torch']) == 4


def test_archive_reports_measured_unpack_and_temporary_disk_peak(tmp_path):
    io = api(); ref = io.write_bundle(model('dense'),tmp_path/'bundle',runtime(tmp_path))
    record = io.archive_bundle(Path(ref.directory),tmp_path/'bundle.tar.gz')
    assert record.unpack_seconds is not None and record.unpack_seconds > 0
    assert record.temporary_disk_peak_bytes >= record.archive_bytes+io.measure_bundle(Path(ref.directory)).payload_bytes


def ordinary_artifact_inference(directory,workdir,backend='torch'):
    """Normal bytecode settings must not mutate the verified deployment payload."""
    import subprocess,sys
    root = Path(directory)
    before = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    script = '''import sys
from pathlib import Path
root = Path(sys.argv[1]); package = root/'runtime/poc-runtime.zip'
sys.path.insert(0,str(package if package.exists() else root/'runtime'))
from poc.export import validate_artifact
validate_artifact(root,sys.argv[2])
'''
    env = dict(os.environ); env.pop('PYTHONDONTWRITEBYTECODE',None)
    result = subprocess.run([sys.executable,'-I','-c',script,str(root),backend],cwd=workdir,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stderr
    after = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    assert after == before,'ordinary runtime imports changed canonical bundle files'
    api().verify_bundle(root)


def test_ordinary_artifact_import_preserves_payload_bytes(tmp_path):
    io = api(); ref = io.write_bundle(model('0'),tmp_path/'bundle',runtime(tmp_path))
    ordinary_artifact_inference(ref.directory,tmp_path)


@pytest.mark.parametrize('attack',['extra','duplicate','traversal','symlink'])
def test_runtime_archive_rejects_unsafe_entries(tmp_path,attack):
    import zipfile
    import warnings
    io = api(); ref = io.write_bundle(model('0'),tmp_path/'bundle',runtime(tmp_path))
    root = Path(ref.directory); archive = root/'runtime/poc-runtime.zip'
    assert archive.exists(),'bytecode-safe runtime archive is missing'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        with zipfile.ZipFile(archive,'a') as zipped:
            if attack == 'duplicate': zipped.writestr('poc/__init__.py','pass')
            elif attack == 'traversal': zipped.writestr('../escape.py','pass')
            elif attack == 'symlink':
                info = zipfile.ZipInfo('unsafe'); info.external_attr = 0o120777<<16; zipped.writestr(info,'/external')
            else: zipped.writestr('weights.safetensors',b'hidden')
    raw = json.loads((root/'manifest.json').read_text())
    raw['file_entries']['runtime/poc-runtime.zip'] = {'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (root/'manifest.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError): io.verify_bundle(root)


def test_accounting_contracts_reject_nonfinite_and_ambiguous_unknowns(tmp_path):
    from dataclasses import replace
    from poc.contracts import ByteAccounting, ArchiveRecord
    spec = runtime(tmp_path)
    with pytest.raises(ValueError): replace(spec,install_bytes=4,install_unknown_reason=3)
    with pytest.raises(ValueError): ByteAccounting(0,resident_bytes=2,resident_unknown_reason=3)
    with pytest.raises(ValueError): ByteAccounting(0,phase_peaks={'load':{'bytes':float('nan')}})
    with pytest.raises(ValueError): ArchiveRecord('a',1,'a'*64,'',0,0)


def test_materialized_owner_count_matches_compact_declaration(tmp_path):
    io = api(); compact = model('2')
    assert io.measure_resident(compact).owner_count == compact.owner_count
