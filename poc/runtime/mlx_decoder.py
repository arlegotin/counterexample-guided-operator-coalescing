"""Canonical FP32 MLX Metal inference. No per-depth FFN reconstruction or CPU fallback.

MLX imports are deliberately inside construction. Arrays are materialized once
per canonical owner; depth dispatch and KV buffers remain independent.
"""
import hashlib
import json
from types import SimpleNamespace

from poc.contracts import (Batch, Boundary, ByteAccounting, CacheState, ContinuationScore,
                           LayerCache, ModelManifest, thaw)


class MLXDecoder:
    def __init__(self, path, manifest):
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError('MLX Metal unavailable: qualify with native sandbox escalation; CPU fallback forbidden')
        if mx.default_device() != mx.gpu: mx.set_default_device(mx.gpu)
        from poc.runtime.torch_decoder import EXECUTION_PROFILE
        if manifest.metadata['execution_profile'] != EXECUTION_PROFILE or manifest.metadata['dtype'] != 'float32':
            raise ValueError('native floating format requires the qualified FP32 execution profile')
        self.mx, self.metadata = mx, manifest.metadata
        self.config = SimpleNamespace(**thaw(self.metadata['config']))
        self.head_dim, self.rope_theta = self.metadata['head_dim'],self.metadata['rope_theta']
        self.norm_epsilons = self.metadata['norm_epsilons']
        self.weights = {}
        for filename in sorted({o['file'] for o in self.metadata['owners']}):
            self.weights.update(mx.load(str(path/filename)))
        mx.eval(*self.weights.values())
        self.aliases = {alias:o['name'] for o in self.metadata['owners'] for alias in o['aliases']}
        self.layer_map = tuple(next(i for i,g in enumerate(self.metadata['groups']) if depth in g)
                               for depth in range(self.config.num_hidden_layers))
        self._digest()

    def _require_metal(self):
        if not self.mx.metal.is_available() or self.mx.default_device() != self.mx.gpu:
            raise RuntimeError('native inference requires Metal GPU; CPU substitution is forbidden')

    def owner(self,name):
        return self.weights[self.aliases.get(name,name)]

    def _digest(self):
        self._require_metal()
        import numpy as np
        from poc.bundle import _canonical
        mx = self.mx
        # MLX permits item assignment; object IDs alone do not protect stale caches.
        # Hash real materialized canonical values, one owner at a time.
        digest = hashlib.sha256(_canonical(self.metadata))
        digest.update(_canonical(vars(self.config)))
        digest.update(_canonical(self.aliases))
        expected_aliases = {a:o['name'] for o in self.metadata['owners'] for a in o['aliases']}
        if self.aliases != expected_aliases: raise ValueError('native alias state differs from artifact')
        digest.update(_canonical([self.head_dim,self.rope_theta,self.layer_map,dict(self.norm_epsilons)]))
        if self.layer_map != tuple(next(i for i,g in enumerate(self.metadata['groups']) if d in g) for d in range(self.config.num_hidden_layers)):
            raise ValueError('native bank mapping differs from artifact state')
        if set(self.weights) != {o['name'] for o in self.metadata['owners']}:
            raise ValueError('native owner state differs from artifact')
        for owner in self.metadata['owners']:
            name = owner['name']; value = self.weights[name]
            if not isinstance(value,mx.array) or tuple(value.shape) != tuple(owner['shape']) or value.dtype != mx.float32:
                raise ValueError('native owner shape/dtype differs from artifact')
            if not bool(mx.all(mx.isfinite(value)).item()): raise ValueError('nonfinite native state')
            digest.update(name.encode()); digest.update(np.asarray(value).tobytes(order='C'))
        return digest.hexdigest()

    def manifest(self):
        return ModelManifest(vars(self.config),{o['name']:o['use_sites'] for o in self.metadata['owners']},
                             self.metadata['tokenizer_digest'],self._digest(),'float32','mlx_metal')

    def resident_accounting(self):
        self.mx.eval(*self.weights.values())
        if len({id(a) for a in self.weights.values()}) != len(self.weights):
            raise ValueError('duplicate native owner allocation')
        return ByteAccounting(0,resident_bytes=sum(a.nbytes for a in self.weights.values()),resident_unknown_reason=None,
                              owner_count=len(self.weights))

    def new_cache(self, *, start_position=0):
        return CacheState(tuple(LayerCache() for _ in range(self.config.num_hidden_layers)),
                          start_position=start_position,model_digest=self._digest())

    def _validate_batch(self,batch):
        self._require_metal()
        mx = self.mx
        if not isinstance(batch,Batch) or any(not isinstance(t,mx.array) for t in (batch.input_ids,batch.positions,batch.attention_mask,batch.target_mask)):
            raise ValueError('native adapter requires MLX Batch arrays')
        ids,positions,mask,targets = batch.input_ids,batch.positions,batch.attention_mask,batch.target_mask
        if ids.dtype not in (mx.int32,mx.int64) or positions.dtype not in (mx.int32,mx.int64): raise ValueError('native IDs/positions must be integer')
        if mask.dtype != mx.bool_ or targets.dtype != mx.bool_: raise ValueError('native masks must be boolean')
        if bool(mx.any((ids<0)|(ids>=self.config.vocab_size)).item()): raise ValueError('token outside native vocabulary')
        if not bool(mx.all(mask[:,0]).item()) or bool(mx.any((~mask[:,:-1])&mask[:,1:]).item()): raise ValueError('native prompts must be nonempty and right-padded')
        if bool(mx.any(targets & ~(mask[:,:-1]&mask[:,1:])).item()): raise ValueError('invalid native target mask')
        if bool(mx.any((positions<0)|(positions>=self.config.max_position_embeddings)).item()): raise ValueError('native position outside context')
        if bool(mx.any(((positions[:,1:]-positions[:,:-1])!=1)&mask[:,1:]).item()): raise ValueError('native positions must be consecutive')

    def _linear(self,x,prefix):
        result = x @ self.owner(prefix+'.weight').T
        if prefix+'.bias' in self.weights: result = result + self.weights[prefix+'.bias']
        return result

    def _rms(self,x,prefix):
        mx = self.mx
        return x * mx.rsqrt(mx.mean(mx.square(x),axis=-1,keepdims=True)+self.norm_epsilons[prefix]) * self.owner(prefix+'.weight')

    def _rope(self,x,positions):
        mx = self.mx
        exponents = mx.arange(0,self.head_dim,2,dtype=mx.float32)/self.head_dim
        inv = 1.0 / mx.power(self.rope_theta,exponents)
        angles = positions.astype(mx.float32)[...,None]*inv
        angles = mx.concatenate((angles,angles),axis=-1)[:,None]
        left,right = mx.split(x,2,axis=-1)
        return x*mx.cos(angles)+mx.concatenate((-right,left),axis=-1)*mx.sin(angles)

    def _attention(self,x,batch,depth,cache=None):
        mx = self.mx; b,s,_ = x.shape; prefix = f'layers.{depth}'
        z = self._rms(x,prefix+'.input_layernorm')
        def project(name,heads):
            return self._linear(z,prefix+'.self_attn.'+name).reshape(b,s,heads,self.head_dim).transpose(0,2,1,3)
        q = self._rope(project('q_proj',self.config.num_attention_heads),batch.positions)
        k = self._rope(project('k_proj',self.config.num_key_value_heads),batch.positions)
        v = project('v_proj',self.config.num_key_value_heads); past = 0
        if cache is not None and cache.key is not None:
            past = cache.key.shape[-2]; k = mx.concatenate((cache.key,k),axis=-2); v = mx.concatenate((cache.value,v),axis=-2)
        if cache is not None: cache.key,cache.value = k,v
        repeat = self.config.num_attention_heads//self.config.num_key_value_heads
        k,v = mx.repeat(k,repeat,axis=1),mx.repeat(v,repeat,axis=1)
        scores = (q @ k.swapaxes(-1,-2))*self.head_dim**-.5
        causal = mx.arange(k.shape[-2])[None,:] <= (past+mx.arange(s))[:,None]
        mask = mx.ones((b,past+s),dtype=mx.bool_) if cache is not None else batch.attention_mask
        scores = mx.where(causal[None,None]&mask[:,None,None,:],scores,-float('inf'))
        probs = mx.softmax(scores,axis=-1,precise=True)
        out = (probs@v).transpose(0,2,1,3).reshape(b,s,-1)
        return x+self._linear(out,prefix+'.self_attn.o_proj')

    def _ffn(self,x,depth):
        mx = self.mx
        def projection(z,name):
            if self.metadata['kind'] == 'dense': return self._linear(z,f'layers.{depth}.mlp.{name}')
            result = self._linear(z,f'bank.{self.layer_map[depth]}.{name}')
            if self.metadata['rank']:
                prefix = f'layers.{depth}.mlp.residuals.{name}'
                result = result + (z@self.weights[prefix+'.A'].T)@self.weights[prefix+'.B'].T
            return result
        gate = projection(x,'gate_proj')
        return projection((gate*mx.sigmoid(gate))*projection(x,'up_proj'),'down_proj')

    def _block(self,x,batch,depth,cache=None):
        u = self._attention(x,batch,depth,cache)
        return u+self._ffn(self._rms(u,f'layers.{depth}.post_attention_layernorm'),depth)

    def _validate_cache(self,cache,batch):
        mx = self.mx
        if not isinstance(cache,CacheState) or len(cache.layers) != self.config.num_hidden_layers or len({id(c) for c in cache.layers}) != len(cache.layers):
            raise ValueError('native cache requires distinct buffers per executed depth')
        if cache.model_digest != self._digest(): raise ValueError('native cache belongs to different model state')
        if batch.input_ids.shape[0] != 1 or not bool(mx.all(batch.attention_mask).item()): raise ValueError('native cache requires one unpadded prompt')
        wanted = mx.arange(cache.start_position+cache.length,cache.start_position+cache.length+batch.input_ids.shape[1])
        if not bool(mx.array_equal(batch.positions[0],wanted).item()): raise ValueError('native cache position must continue once')
        for layer in cache.layers:
            if not isinstance(layer,LayerCache): raise ValueError('invalid native cache layer')
            if not cache.length:
                if layer.key is not None or layer.value is not None: raise ValueError('stale native cache buffers')
            else:
                shape = (1,self.config.num_key_value_heads,cache.length,self.head_dim)
                if any(not isinstance(t,mx.array) or t.shape != shape or t.dtype != mx.float32 for t in (layer.key,layer.value)):
                    raise ValueError('native cache shape/dtype inconsistent')

    def final_hidden(self,batch, *, cache=None):
        self._validate_batch(batch)
        if cache is not None: self._validate_cache(cache,batch)
        x = self.owner('embed_tokens.weight')[batch.input_ids]
        pending = None if cache is None else tuple(LayerCache(c.key,c.value) for c in cache.layers)
        for depth in range(self.config.num_hidden_layers):
            x = self._block(x,batch,depth,None if pending is None else pending[depth])
            # Bound lazy graph lifetime and temporary memory at actual depth.
            self.mx.eval(x)
        result = self._rms(x,'norm'); self.mx.eval(result)
        if cache is not None:
            for dst,src in zip(cache.layers,pending,strict=True): dst.key,dst.value = src.key,src.value
            cache.length += batch.input_ids.shape[1]
        return result

    def _validate_hidden(self,x):
        self._require_metal()
        if not isinstance(x,self.mx.array) or x.ndim < 2 or x.shape[-1] != self.config.hidden_size or x.dtype != self.mx.float32:
            raise ValueError('native hidden must have model width and FP32 dtype')

    def readout(self,hidden):
        self._validate_hidden(hidden)
        return self._linear(hidden,'lm_head')

    def logits(self,batch, *, cache=None): return self.readout(self.final_hidden(batch,cache=cache))

    def prefix_boundary(self,batch,depth):
        self._validate_batch(batch)
        if type(depth) is not int or not 0 <= depth < self.config.num_hidden_layers: raise ValueError('native depth outside model')
        x = self.owner('embed_tokens.weight')[batch.input_ids]
        for d in range(depth): x = self._block(x,batch,d); self.mx.eval(x)
        return Boundary(self._attention(x,batch,depth),batch,depth,self._digest())

    def _validate_boundary(self,boundary):
        if not isinstance(boundary,Boundary) or not 0 <= boundary.depth < self.config.num_hidden_layers: raise ValueError('invalid native boundary')
        self._validate_batch(boundary.batch); self._validate_hidden(boundary.u)

    def ff_residual(self,boundary,operator=None):
        self._validate_boundary(boundary)
        z = self._rms(boundary.u,f'layers.{boundary.depth}.post_attention_layernorm')
        delta = self._ffn(z,boundary.depth) if operator is None else operator(z)
        self._validate_hidden(delta)
        if delta.shape != boundary.u.shape: raise ValueError('native FFN must preserve boundary shape')
        return boundary.u+delta

    def suffix_hidden(self,after_ffn,boundary):
        self._validate_boundary(boundary); self._validate_hidden(after_ffn)
        if after_ffn.shape != boundary.u.shape: raise ValueError('native suffix shape differs')
        x = after_ffn
        for depth in range(boundary.depth+1,self.config.num_hidden_layers):
            x = self._block(x,boundary.batch,depth); self.mx.eval(x)
        return self._rms(x,'norm')

    def suffix_logits(self,after_ffn,boundary): return self.readout(self.suffix_hidden(after_ffn,boundary))

    def score_continuations(self,requests):
        mx = self.mx; results = []
        for request in requests:
            ids = mx.array([request.prompt_ids+request.continuation_ids],dtype=mx.int32)
            n = ids.shape[1]; mask = mx.ones_like(ids,dtype=mx.bool_)
            target = mx.arange(n-1)[None] >= len(request.prompt_ids)-1
            batch = Batch(ids,mask,target,mx.arange(n)[None],(request.item_id,),(request.item_id,))
            hidden = self.final_hidden(batch); total = 0.
            for start in range(0,n-1,32):
                end = min(start+32,n-1); logits = self.readout(hidden[:,start:end])
                log_probs = logits-mx.logsumexp(logits,axis=-1,keepdims=True)
                selected = mx.take_along_axis(log_probs,ids[:,start+1:end+1,None],axis=-1).squeeze(-1)
                # Python floats aggregate the selected FP32 token scores in FP64.
                values = mx.where(target[:,start:end],selected,0); mx.eval(values)
                total += sum(float(v) for v in values.reshape(-1).tolist())
            results.append(ContinuationScore(request.item_id,total,len(request.continuation_ids)))
        return results
