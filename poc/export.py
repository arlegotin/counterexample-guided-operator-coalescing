"""Canonical floating export and direct scientific reconstruction, without a dense base."""
from pathlib import Path
from poc.contracts import BundleRef, DecoderAdapter, RuntimeSpec


def export_native(decoder: DecoderAdapter, destination: Path, config: RuntimeSpec) -> BundleRef:
    """Export FP32 canonical owners for MLX; quantized formats are a separate stage."""
    if str(decoder.dtype) != 'torch.float32':
        raise ValueError('native floating deployment requires scientific FP32 weights')
    from poc.bundle import write_bundle
    return write_bundle(decoder, destination, config)


class _Config:
    def __init__(self, raw):
        self.__dict__.update(raw)

    def to_dict(self):
        return dict(self.__dict__)


def _load_scientific(path, manifest, *, values=None):
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from poc.bank import BankDecoder, _BankUse, _Factors
    from poc.contracts import Partition, thaw
    from poc.runtime.torch_decoder import TorchDecoder, EXECUTION_PROFILE
    raw = thaw(manifest.metadata)
    if raw['execution_profile'] != EXECUTION_PROFILE:
        raise ValueError('unsupported scientific execution profile')
    compact = raw['kind'] == 'compact'
    model = (BankDecoder if compact else TorchDecoder).__new__(BankDecoder if compact else TorchDecoder)
    nn.Module.__init__(model)
    model.config = _Config(raw['config'])
    model.head_dim, model.rope_theta = raw['head_dim'], raw['rope_theta']
    model.tokenizer_digest = raw['tokenizer_digest']
    if values is None:
        values = {}
        for file in sorted({o['file'] for o in raw['owners']}): values.update(load_file(path/file,device='cpu'))
    owners = {}
    for record in raw['owners']:
        tensor = values[record['name']]
        if not torch.isfinite(tensor).all().item(): raise ValueError('nonfinite deployment owner')
        owners[record['name']] = nn.Parameter(tensor,requires_grad=record['trainable'])
    del values
    def module(prefix):
        result = nn.Module()
        for suffix in ('weight','bias'):
            if prefix+'.'+suffix in owners: result.register_parameter(suffix,owners[prefix+'.'+suffix])
        return result
    def linear(prefix):
        weight = owners[prefix+'.weight']
        result = nn.Linear(weight.shape[1],weight.shape[0],bias=prefix+'.bias' in owners,device='meta',dtype=weight.dtype)
        result.weight = weight
        if prefix+'.bias' in owners: result.bias = owners[prefix+'.bias']
        return result
    def norm(prefix):
        result = module(prefix); result.variance_epsilon = raw['norm_epsilons'][prefix]; return result
    model.embed_tokens = nn.Embedding(model.config.vocab_size,model.config.hidden_size,device='meta')
    model.embed_tokens.weight = owners['embed_tokens.weight']
    model.layers = nn.ModuleList()
    if compact:
        model.partition = Partition(tuple(tuple(g) for g in raw['groups']),model.config.num_hidden_layers)
        model.rank, model.base_digest = raw['rank'],raw['base_digest']
        # Registration order follows the scientific adapter, bank after readout.
    class FFN(nn.Module):
        def __init__(self,prefix):
            super().__init__()
            for name in ('gate_proj','up_proj','down_proj'): setattr(self,name,linear(prefix+'.'+name))
        def forward(self,x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))
    for depth in range(model.config.num_hidden_layers):
        layer = nn.Module(); prefix = f'layers.{depth}'
        # HF Llama registers attention, MLP, then the two norms.
        layer.self_attn = nn.Module()
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            setattr(layer.self_attn,name,linear(prefix+'.self_attn.'+name))
        layer.mlp = nn.Module() if compact else FFN(prefix+'.mlp')
        layer.input_layernorm = norm(prefix+'.input_layernorm')
        layer.post_attention_layernorm = norm(prefix+'.post_attention_layernorm')
        model.layers.append(layer)
    model.norm = norm('norm')
    model.lm_head = nn.Linear(model.config.hidden_size,model.config.vocab_size,bias=False,device='meta')
    model.lm_head.weight = owners['embed_tokens.weight' if model.config.tie_word_embeddings else 'lm_head.weight']
    if compact:
        model.bank = nn.ModuleList([FFN(f'bank.{i}') for i in range(len(raw['groups']))])
        for depth,index in enumerate(model.partition.reuse_map):
            factors = {}
            if model.rank:
                for name in ('gate_proj','up_proj','down_proj'):
                    prefix = f'layers.{depth}.mlp.residuals.{name}'
                    factors[name] = _Factors(owners[prefix+'.A'],owners[prefix+'.B'])
                    # _Factors makes Parameter wrappers; preserve canonical object and trainability.
                    factors[name].A, factors[name].B = owners[prefix+'.A'],owners[prefix+'.B']
            model.layers[depth].mlp = _BankUse(model.bank,index,factors)
    model.eval()
    # CPU source identity includes actual scalar state, aliases and canonical values.
    # A device move changes scientific identity, so other source backends retain
    # their immutable source provenance while the loaded CPU model gets its own.
    if raw['source_backend'] == 'torch_cpu' and raw['source_checkpoint_digest'] != model.manifest().checkpoint_digest:
        raise ValueError('reconstructed scientific checkpoint digest mismatch')
    return model


def export_command(args) -> int:
    """Resolve an owned fitted generation, or retain explicit-source provenance."""
    import hashlib
    import json
    from poc.bundle import _canonical, _read_json, load_bundle, verify_bundle
    from poc.contracts import thaw
    receipt_path = args.out.with_name(args.out.name+'.receipt.json')
    if receipt_path.exists(): raise FileExistsError(receipt_path)
    fitted = None
    if args.run_root is not None:
        if args.runtime_spec is None:
            from poc.config import ConfigError
            raise ConfigError('--run-root requires --runtime-spec')
        from poc.checkpoint import resolve_fitted_run, checkpoint_decoder
        fitted, checkpoint_ref = resolve_fitted_run(args.run_root,args.run,args.attempt)
        decoder = checkpoint_decoder(fitted)
        runtime = RuntimeSpec(**_read_json(args.runtime_spec))
        source_digest = checkpoint_ref.manifest_sha256
        kind = 'verified_fitted_checkpoint'
    elif args.source_bundle is not None:
        manifest = verify_bundle(args.source_bundle)
        source_digest = hashlib.sha256((args.source_bundle/'manifest.json').read_bytes()).hexdigest()
        decoder = load_bundle(args.source_bundle,backend='torch')
        runtime = RuntimeSpec(
            {n:str(args.source_bundle/'tokenizer'/n) for n in manifest.metadata['tokenizer_files']},
            {n:str(args.source_bundle/'LICENSES'/n) for n in manifest.metadata['license_files']},
            str(args.source_bundle/'requirements.lock'),manifest.metadata['install_bytes'],manifest.metadata['install_unknown_reason'])
        kind = 'standalone_bundle'
    else:
        if args.teacher_manifest_sha256 is None or args.runtime_spec is None:
            from poc.config import ConfigError
            raise ConfigError('--teacher-assets requires --teacher-manifest-sha256 and --runtime-spec')
        from poc.teacher import AssetManifest, load_teacher
        asset = AssetManifest.load(args.teacher_assets,expected_sha256=args.teacher_manifest_sha256)
        decoder = load_teacher(asset); source_digest = asset.expected_sha256
        runtime = RuntimeSpec(**_read_json(args.runtime_spec)); kind = 'verified_teacher_assets'
    ref = export_native(decoder,args.out,runtime)
    receipt = {'schema_version':1,'run_label':args.run,'attempt_label':args.attempt,
               'source_kind':kind,'source_manifest_sha256':source_digest,
               'bundle_manifest_sha256':ref.manifest_sha256,'fitted_checkpoint_resolved':False,
               'scope':'floating canonical export; native deployment requires separate MLX qualification'}
    if fitted is not None:
        receipt.update(run_id=args.run,attempt_id=args.attempt,fitted_checkpoint_resolved=True,
            checkpoint_manifest_sha256=checkpoint_ref.manifest_sha256,
            fitted_candidate_digest=fitted.candidate_digest,fit_step=fitted.step,
            fit_valid_target_presentations=fitted.target_cursor)
        receipt.pop('run_label');receipt.pop('attempt_label')
    receipt_path.write_bytes(_canonical(receipt))
    print(json.dumps({'bundle':ref.directory,'manifest_sha256':ref.manifest_sha256,'receipt':str(receipt_path)},sort_keys=True))
    return 0


def validate_artifact(path,backend):
    """Load and execute a tokenizer-generated prompt with the explicitly selected backend."""
    import json
    from tokenizers import Tokenizer
    from poc.bundle import load_bundle, measure_bundle, measure_resident
    from poc.contracts import Batch
    decoder = load_bundle(path,backend=backend)
    tokenizer = Tokenizer.from_file(str(path/'tokenizer/tokenizer.json'))
    ids = tokenizer.encode('A small model reads a sentence.').ids
    if not ids: raise ValueError('bundle tokenizer returned an empty validation prompt')
    ids = ids[:min(16,decoder.config.max_position_embeddings)]
    if backend == 'torch':
        import torch
        x = torch.tensor([ids],dtype=torch.int64)
        batch = Batch(x,torch.ones_like(x,dtype=torch.bool),torch.ones((1,len(ids)-1),dtype=torch.bool),
                      torch.arange(len(ids))[None],('validation',),('validation',))
        with torch.no_grad(): logits = decoder.logits(batch)
        if not torch.isfinite(logits).all().item(): raise ValueError('nonfinite scientific bundle output')
    else:
        import mlx.core as mx
        x = mx.array([ids],dtype=mx.int32)
        batch = Batch(x,mx.ones_like(x,dtype=mx.bool_),mx.ones((1,len(ids)-1),dtype=mx.bool_),
                      mx.arange(len(ids))[None],('validation',),('validation',))
        logits = decoder.logits(batch); mx.eval(logits)
        if not bool(mx.all(mx.isfinite(logits)).item()): raise ValueError('nonfinite native bundle output')
    print(json.dumps({'status':'passed','scope':'scientific artifact execution' if backend == 'torch' else 'native MLX Metal artifact execution',
                      'backend':backend,'prompt_tokens':len(ids),'owner_count':measure_resident(decoder).owner_count,
                      'payload_bytes':measure_bundle(path).payload_bytes,'wp03':'not established by a single artifact smoke test'},sort_keys=True))
    return 0


def make_payload_anchor(teacher: DecoderAdapter, runtime: RuntimeSpec, *, recorder=None) -> BundleRef:
    """SEQUENCE K15/rank8, first-depth medoids, CPU FP64 SVD seed17, RTN4/g64.

    Owned temporary output is staging only. The stage driver verifies and
    atomically publishes a separate immutable durable copy before recording B*.
    No fitting or quality claim is implied by this prescribed format anchor.
    """
    import tempfile
    import shutil
    from poc.bank import BankDecoder,canonicalize_partition
    from poc.quantize import QuantizationSpec,quantize_rtn
    from poc.bundle import write_bundle
    depth=len(teacher.layers)
    if depth!=30: raise ValueError('prescribed primary anchor requires actual depth30')
    directory=Path(tempfile.mkdtemp(prefix='cgoc-payload-anchor-'))
    try:
        compact=BankDecoder(teacher,canonicalize_partition([list(range(i,i+2)) for i in range(0,30,2)],30),8,17)
        floating=write_bundle(compact,directory/'floating',runtime)
        result=quantize_rtn(floating,QuantizationSpec('rtn',4,64,str(directory/'packed')),recorder)
        shutil.rmtree(directory/'floating')
        return result.bundle
    except BaseException:
        shutil.rmtree(directory);raise
