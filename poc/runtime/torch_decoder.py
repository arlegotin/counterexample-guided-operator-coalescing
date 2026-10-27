"""Offline, eager Llama execution and cache-free differentiable FFN boundaries.

The adapter owns the supplied modules, preserving parameter sharing and actual
config dimensions. It deliberately rejects unsupported RoPE/architecture modes.
"""
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from poc.contracts import (Batch, Boundary, CacheState, ContinuationRequest,
                           ContinuationScore, FFNOperator, LayerCache, ModelManifest)

EXECUTION_PROFILE = 'eager-llama-hf-fp32-reciprocal-true-fp64-v2'


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 41
    hidden_size: int = 16
    intermediate_size: int = 32
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                     'num_attention_heads', 'num_key_value_heads', 'max_position_embeddings'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.hidden_size % self.num_attention_heads or (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError('hidden width must produce even RoPE head dimensions')
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError('query heads must be divisible by KV heads')
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
               for x in (self.rms_norm_eps, self.rope_theta)):
            raise ValueError('norm epsilon and RoPE base must be positive and finite')


class TorchDecoder(nn.Module):
    """Frozen teacher by default; candidate inputs remain in the autograd graph."""

    def __init__(self, model: nn.Module, *, tokenizer_digest: str | None = None):
        super().__init__()
        config = model.config
        if getattr(config, 'model_type', None) != 'llama':
            raise ValueError('only eager Llama architecture is implemented')
        rope = getattr(config, 'rope_parameters', None) or getattr(config, 'rope_scaling', None) or {}
        if rope.get('rope_type', rope.get('type', 'default')) != 'default':
            raise ValueError('scaled/dynamic RoPE is not qualified')
        if getattr(config, 'hidden_act', 'silu') != 'silu' or getattr(config, 'attention_dropout', 0):
            raise ValueError('only deterministic SiLU Llama is implemented')
        self.config = config
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.tokenizer_digest = tokenizer_digest
        self.head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        self.rope_theta = rope.get('rope_theta', getattr(config, 'rope_theta', 10000.0))
        if self.head_dim % 2 or config.num_attention_heads % config.num_key_value_heads:
            raise ValueError('unsupported RoPE head shape or grouped-query head ratio')
        if len(self.layers) != config.num_hidden_layers:
            raise ValueError('actual layer count differs from config')
        for layer in self.layers:
            expected = (config.num_attention_heads * self.head_dim, config.hidden_size)
            if tuple(layer.self_attn.q_proj.weight.shape) != expected:
                raise ValueError('actual query projection shape differs from config')
            if tuple(layer.self_attn.k_proj.weight.shape) != (config.num_key_value_heads*self.head_dim, config.hidden_size):
                raise ValueError('actual key projection shape differs from config')
            if tuple(layer.mlp.gate_proj.weight.shape) != (config.intermediate_size, config.hidden_size):
                raise ValueError('actual FFN projection shape differs from config')
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_transformers(cls, model: nn.Module, *, tokenizer_digest: str | None = None) -> 'TorchDecoder':
        return cls(model, tokenizer_digest=tokenizer_digest)

    @property
    def device(self):
        return self.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def invalidate_state_snapshot(self) -> None:
        """Invalidate after external storage writes (including unsafe .data/NumPy).

        Normal no-grad in-place edits, optimizers, state loads, module replacement,
        config edits and dtype/device moves are also detected by the signature.
        External writers must call this before any digest/probe/cache operation.
        """
        self._state_snapshot = None

    def _digest(self) -> str:
        config = json.dumps(self.config.to_dict(), sort_keys=True, default=str)
        tensors = list(self.named_parameters(remove_duplicate=False)) + list(self.named_buffers(remove_duplicate=False))
        signature = (EXECUTION_PROFILE, config, self.head_dim, self.rope_theta,
                     tuple((name, id(t), t.data_ptr(), t._version, str(t.dtype), str(t.device),
                            tuple(t.shape), tuple(t.stride())) for name, t in tensors))
        snapshot = getattr(self, '_state_snapshot', None)
        if snapshot is not None and snapshot[0] == signature:
            return snapshot[1]
        digest = hashlib.sha256(config.encode())
        digest.update(json.dumps([EXECUTION_PROFILE, self.head_dim,
                                  self.rope_theta, str(self.device)]).encode())
        # Hash detached snapshots only; live candidate activations never cross NumPy.
        for name, tensor in self.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        # Equal values with different sharing are different executable ownership.
        owners = {}
        for name, parameter in self.named_parameters(remove_duplicate=False):
            owners.setdefault(id(parameter), []).append(name)
        digest.update(json.dumps(list(owners.values()), sort_keys=True).encode())
        result = digest.hexdigest()
        self._state_snapshot = (signature, result)
        return result

    def manifest(self) -> ModelManifest:
        owners = {}
        for name, parameter in self.named_parameters(remove_duplicate=False):
            owners.setdefault(id(parameter), []).append(name)
        sites = {names[0]: names for names in owners.values()}
        return ModelManifest(json.loads(json.dumps(self.config.to_dict(), default=str)), sites, self.tokenizer_digest, self._digest(),
                             str(self.dtype).removeprefix('torch.'), f'torch_{self.device.type}')

    def new_cache(self, *, start_position: int = 0) -> CacheState:
        return CacheState(tuple(LayerCache() for _ in self.layers), start_position=start_position,
                          model_digest=self._digest())

    def _validate_batch(self, batch: Batch) -> None:
        if not isinstance(batch, Batch):
            raise ValueError('expected Batch')
        ids, mask, targets, positions = batch.input_ids, batch.attention_mask, batch.target_mask, batch.positions
        if any(not isinstance(x, torch.Tensor) for x in (ids, mask, targets, positions)):
            raise ValueError('torch adapter requires torch batch tensors')
        if ids.dtype != torch.int64 or positions.dtype not in (torch.int64, torch.int32):
            raise ValueError('input IDs must be int64 and positions integer')
        if mask.dtype != torch.bool or targets.dtype != torch.bool:
            raise ValueError('attention and target masks must be bool')
        if any(x.device != self.device for x in (ids, mask, targets, positions)):
            raise ValueError('batch tensors must be on model device')
        if torch.any((ids < 0) | (ids >= self.config.vocab_size)).item():
            raise ValueError('token ID outside vocabulary')
        if not mask[:,0].all().item() or ((~mask[:,:-1]) & mask[:,1:]).any().item():
            raise ValueError('attention mask must right-pad nonempty prompts')
        if (targets & ~(mask[:,:-1] & mask[:,1:])).any().item():
            raise ValueError('target mask may score only consecutive real tokens')
        if ((positions < 0) | (positions >= self.config.max_position_embeddings)).any().item():
            raise ValueError('model-context position outside configured range')
        if ((positions[:,1:] - positions[:,:-1] != 1) & mask[:,1:]).any().item():
            raise ValueError('model-context positions must be consecutive')

    def _rms(self, x, norm):
        # FP64 CPU is a numerical reference; scientific FP32 matches Llama RMS.
        work = x if x.dtype == torch.float64 else x.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + norm.variance_epsilon)).to(x.dtype) * norm.weight

    def _rope(self, x, positions):
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        exponents = torch.arange(0, self.head_dim, 2, device=x.device, dtype=work_dtype)/self.head_dim
        # Match original HF operation order in scientific FP32. Negative powers
        # differ by an ulp at real head widths and amplify through 30 blocks.
        # Preserve the true FP64 reference's original high-precision arithmetic.
        inv = (self.rope_theta ** -exponents if work_dtype == torch.float64
               else 1.0 / (self.rope_theta ** exponents))
        angles = positions.to(work_dtype)[..., None] * inv
        angles = torch.cat((angles, angles), dim=-1)[:, None]
        left, right = x.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)

    def _attention(self, x, batch, depth, layer_cache=None):
        layer = self.layers[depth]
        attention = layer.self_attn
        b, s, _ = x.shape
        z = self._rms(x, layer.input_layernorm)
        def project(proj, heads):
            return proj(z).view(b,s,heads,self.head_dim).transpose(1,2)
        q = self._rope(project(attention.q_proj, self.config.num_attention_heads), batch.positions)
        k = self._rope(project(attention.k_proj, self.config.num_key_value_heads), batch.positions)
        v = project(attention.v_proj, self.config.num_key_value_heads)
        past = 0
        if layer_cache is not None and layer_cache.key is not None:
            past = layer_cache.key.shape[-2]
            k = torch.cat((layer_cache.key, k), dim=-2)
            v = torch.cat((layer_cache.value, v), dim=-2)
        if layer_cache is not None:
            layer_cache.key, layer_cache.value = k.detach(), v.detach()
        repeat = self.config.num_attention_heads // self.config.num_key_value_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        scores = q @ k.transpose(-1,-2) * self.head_dim**-0.5
        causal = torch.arange(k.shape[-2], device=x.device)[None,:] <= (past + torch.arange(s, device=x.device))[:,None]
        key_mask = (torch.ones((b,past+s), dtype=torch.bool, device=x.device)
                    if layer_cache is not None else batch.attention_mask)
        allowed = causal[None,None] & key_mask[:,None,None,:]
        scores = scores.masked_fill(~allowed, -float('inf'))
        work = scores if scores.dtype == torch.float64 else scores.float()
        probs = torch.softmax(work, dim=-1).to(x.dtype)
        output = (probs @ v).transpose(1,2).contiguous().view(b,s,-1)
        return x + attention.o_proj(output)

    def _block(self, x, batch, depth, layer_cache=None):
        u = self._attention(x, batch, depth, layer_cache)
        return u + self.layers[depth].mlp(self._rms(u, self.layers[depth].post_attention_layernorm))

    def _validate_cache(self, cache, batch):
        if not isinstance(cache, CacheState) or len(cache.layers) != len(self.layers):
            raise ValueError('cache must contain one buffer per executed depth')
        if (any(not isinstance(layer, LayerCache) for layer in cache.layers)
                or len({id(layer) for layer in cache.layers}) != len(cache.layers)):
            raise ValueError('cache requires distinct per-depth LayerCache owners')
        if batch.input_ids.shape[0] != 1 or not batch.attention_mask.all().item():
            raise ValueError('cached execution supports one unpadded batch row only')
        if not torch.equal(batch.positions[0], torch.arange(cache.start_position+cache.length,
                           cache.start_position+cache.length+batch.input_ids.shape[1], device=self.device)):
            raise ValueError('cache position must continue exactly once from previous tokens')
        if cache.model_digest != self._digest():
            raise ValueError('cache belongs to different model state')
        for layer in cache.layers:
            if cache.length == 0:
                if layer.key is not None or layer.value is not None:
                    raise ValueError('empty cache contains stale buffers')
            else:
                expected = (1, self.config.num_key_value_heads, cache.length, self.head_dim)
                if any(not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected
                       or tensor.dtype != self.dtype or tensor.device != self.device
                       for tensor in (layer.key, layer.value)):
                    raise ValueError('cache layer shape, dtype or device is inconsistent')

    def final_hidden(self, batch: Batch, *, cache: CacheState | None = None):
        self._validate_batch(batch)
        if cache is not None:
            self._validate_cache(cache, batch)
        x = self.embed_tokens(batch.input_ids)
        # Commit the cache atomically after all depths succeed, so an exception
        # cannot silently leave some layers advanced and others unchanged.
        pending = None if cache is None else tuple(LayerCache(c.key,c.value) for c in cache.layers)
        for depth in range(len(self.layers)):
            x = self._block(x, batch, depth, None if pending is None else pending[depth])
        result = self._rms(x, self.norm)
        if cache is not None:
            for destination, source in zip(cache.layers, pending, strict=True):
                destination.key, destination.value = source.key, source.value
            cache.length += batch.input_ids.shape[1]
        return result

    def readout(self, hidden):
        self._validate_hidden(hidden)
        return self.lm_head(hidden)

    def logits(self, batch: Batch, *, cache: CacheState | None = None):
        return self.readout(self.final_hidden(batch, cache=cache))

    def _validate_hidden(self, hidden):
        if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError('hidden tensor must end in model hidden width')
        if hidden.device != self.device or hidden.dtype != self.dtype:
            raise ValueError('hidden tensor dtype/device must match model')

    def _validate_boundary(self, boundary):
        if not isinstance(boundary, Boundary) or not 0 <= boundary.depth < len(self.layers):
            raise ValueError('boundary depth outside model')
        self._validate_batch(boundary.batch)
        self._validate_hidden(boundary.u)
        # The producer digest is provenance, not a restriction on the executor.
        # A teacher suffix must also accept a detached current-candidate state.
        # Probe consumers own staleness checks against the expected producer.

    def prefix_boundary(self, batch: Batch, depth: int) -> Boundary:
        self._validate_batch(batch)
        if type(depth) is not int or not 0 <= depth < len(self.layers):
            raise ValueError('depth outside executed model')
        x = self.embed_tokens(batch.input_ids)
        for index in range(depth):
            x = self._block(x, batch, index)
        u = self._attention(x, batch, depth)
        return Boundary(u, batch, depth, self._digest())

    def ff_residual(self, boundary: Boundary, operator: FFNOperator | None = None):
        self._validate_boundary(boundary)
        layer = self.layers[boundary.depth]
        normalized = self._rms(boundary.u, layer.post_attention_layernorm)
        delta = (layer.mlp if operator is None else operator)(normalized)
        if delta.shape != boundary.u.shape or delta.dtype != self.dtype or delta.device != self.device:
            raise ValueError('FFN operator must preserve [B,S,d], dtype and device')
        return boundary.u + delta

    def suffix_hidden(self, after_ffn, boundary: Boundary):
        self._validate_boundary(boundary)
        self._validate_hidden(after_ffn)
        if after_ffn.shape != boundary.u.shape:
            raise ValueError('suffix input must match boundary shape')
        x = after_ffn
        for depth in range(boundary.depth+1, len(self.layers)):
            x = self._block(x, boundary.batch, depth)
        return self._rms(x, self.norm)

    def suffix_logits(self, after_ffn, boundary: Boundary):
        return self.readout(self.suffix_hidden(after_ffn, boundary))

    def score_continuations(self, requests: Sequence[ContinuationRequest]) -> list[ContinuationScore]:
        scores = []
        for request in requests:
            ids = torch.tensor([request.prompt_ids + request.continuation_ids], dtype=torch.int64, device=self.device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            targets = torch.zeros((1, ids.shape[1]-1), dtype=torch.bool, device=self.device)
            targets[:,len(request.prompt_ids)-1:] = True
            batch = Batch(ids, mask, targets, torch.arange(ids.shape[1], device=self.device)[None],
                          (request.item_id,), (request.item_id,))
            with torch.no_grad():
                hidden = self.final_hidden(batch)
                total = 0.0
                for start in range(0, targets.shape[1], 32):
                    end = min(start+32, targets.shape[1])
                    logits = self.readout(hidden[:,start:end])
                    work = logits if logits.dtype == torch.float64 else logits.float()
                    selected = work.log_softmax(-1).gather(-1, ids[:,start+1:end+1,None]).squeeze(-1)
                    total += selected.masked_select(targets[:,start:end]).cpu().double().sum().item()
            scores.append(ContinuationScore(request.item_id, total, int(targets.sum().item())))
        return scores


def make_tiny_decoder(config: TinyConfig, *, seed: int, dtype: torch.dtype) -> TorchDecoder:
    """Initialize an offline deterministic tiny model, preserving caller RNG state."""
    from transformers import LlamaConfig, LlamaForCausalLM
    if not isinstance(config, TinyConfig):
        raise ValueError('make_tiny_decoder requires TinyConfig')
    if type(seed) is not int or seed < 0 or dtype not in (torch.float64, torch.float32):
        raise ValueError('tiny fixtures require nonnegative seed and CPU FP32/FP64')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LlamaForCausalLM(LlamaConfig(**asdict(config), tie_word_embeddings=True,
                                           attention_dropout=0.0, attn_implementation='eager')).to(dtype=dtype)
    return TorchDecoder(model)
