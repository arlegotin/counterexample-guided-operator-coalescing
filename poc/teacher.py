"""Pinned, verified, local-only safe teacher assets and original HF reference."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from poc.runtime.torch_decoder import TorchDecoder


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetManifest:
    path: Path
    expected_sha256: str

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str):
        asset=cls(Path(path).resolve(),expected_sha256)
        asset.verify()
        return asset

    def verify(self) -> Path:
        if not re.fullmatch(r'[0-9a-f]{64}',self.expected_sha256) or sha256_file(self.path)!=self.expected_sha256:
            raise ValueError('teacher asset manifest hash mismatch')
        raw=json.loads(self.path.read_text()); files=raw.get('assets',[])
        if raw.get('schema_version')!=1 or not files:
            raise ValueError('invalid teacher asset schema')
        parents=set(); names=set()
        for entry in files:
            path=(self.path.parent/entry['local_path']).resolve()
            if not path.is_relative_to(self.path.parent) or path.name in names:
                raise ValueError('asset path escapes manifest root or duplicates a file')
            names.add(path.name); parents.add(path.parent)
            source=entry['source']
            if not source.get('terms_permitted') or not re.fullmatch(r'[0-9a-f]{40}',source.get('revision','')):
                raise ValueError('unqualified teacher source terms/revision')
            if path.stat().st_size!=entry['bytes'] or sha256_file(path)!=entry['sha256']:
                raise ValueError('teacher source size/hash mismatch')
            if 'sha256' in source and source['sha256']!=entry['sha256']:
                raise ValueError('source digest differs from local digest')
        if len(parents)!=1 or not {'config.json','model.safetensors','tokenizer.json','tokenizer_config.json'}<=names:
            raise ValueError('incomplete or inconsistent safe teacher asset set')
        directory=parents.pop()
        # Reject unverified sidecars that Transformers might otherwise discover.
        if any(p.is_file() and p.name not in names and p.resolve()!=self.path
               for p in directory.iterdir()):
            raise ValueError('unverified file in teacher directory')
        return directory


def load_reference(asset: AssetManifest, config=None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    directory=asset.verify()  # every byte verified before any config/model load
    raw=json.loads((directory/'config.json').read_text())
    if raw.get('auto_map') or raw.get('model_type')!='llama':
        raise ValueError('remote code or unsupported teacher architecture')
    manifest=json.loads(asset.path.read_text())
    if config is not None:
        pins={(f['source']['repository'],f['source']['revision']) for f in manifest['assets']}
        if pins!={(config.teacher.repository,config.teacher.revision)} or config.teacher.trust_remote_code:
            raise ValueError('teacher source differs from configured pins')
        if config.teacher.weights_sha256!=asset.expected_sha256:
            raise ValueError('configured teacher manifest hash mismatch')
    hf_config=AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(directory,config=hf_config,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='eager')
    # Transformers injects this loader location; it changes neither execution
    # nor source identity and must never enter checkpoint/probe/cache identity.
    model.config._name_or_path = ''
    model.requires_grad_(False); model.eval()
    return model


def load_tokenizer(asset: AssetManifest):
    from transformers import AutoTokenizer
    directory=asset.verify()
    raw=json.loads((directory/'tokenizer_config.json').read_text())
    if raw.get('auto_map'): raise ValueError('tokenizer remote code forbidden')
    return AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False,use_fast=True)


def load_teacher(asset: AssetManifest, config=None) -> TorchDecoder:
    return TorchDecoder.from_transformers(load_reference(asset,config),tokenizer_digest=asset.expected_sha256)
