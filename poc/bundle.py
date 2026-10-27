"""Safe tiny bank-state fixtures, requiring a separately supplied exact dense base.

This is not the standalone deployment bundle/exporter specified by R17.
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
