# Floating canonical deployment bundles

`poc.bundle.write_bundle(decoder, destination, runtime)` exports a dense
`TorchDecoder` or a canonical `BankDecoder`. `RuntimeSpec` explicitly supplies
local tokenizer and license files plus a pinned dependency lock. These input
locations never enter output identity. `poc.export.export_native` restricts this
format to scientific FP32, the qualified MLX floating format. Quantization and
the measured B* anchor are separate work.

Every artifact carries all backbone, FFN bank and private residual owners,
actual per-site RMSNorm epsilons, attention configuration, tokenizer, licenses,
and candidate-specific inference code. Gate/up/down and residual factors retain
`[out,in]` orientation and unit residual scale. Rank zero has no residual file.
Tied embedding/readout shares one loaded owner. No source checkpoint, base
model, fitting directory or model cache is required.

```python
from pathlib import Path
from poc.bundle import write_bundle, load_bundle, verify_bundle, measure_bundle
from poc.contracts import RuntimeSpec

runtime = RuntimeSpec(
    tokenizer_files={"tokenizer.json": "/local/verified/tokenizer.json"},
    license_files={"Apache-2.0.txt": "/local/LICENSE"},
    dependency_lock="/local/pinned.requirements.txt",
)
ref = write_bundle(decoder, Path("/tmp/candidate"), runtime)
manifest = verify_bundle(Path(ref.directory))
scientific = load_bundle(Path(ref.directory), backend="torch", offline=True)
native = load_bundle(Path(ref.directory), backend="mlx", offline=True)
```

The scientific loader directly allocates canonical owners on CPU; it never
constructs a temporary dense base. MLX imports lazily and requires real Metal.
There is no CPU fallback for a native request. Inference/state boundaries also
reject a caller switching the global MLX default to CPU after bundle load. Native cache identity hashes
actual materialized weights, because MLX permits item assignment without a
public tensor version counter. This has a measurable load/cache overhead and
is intentionally conservative. Cache buffers are distinct per executed depth.

To use only artifact code, prepend
`/tmp/candidate/runtime/poc-runtime.zip` to `sys.path`, then import `poc.bundle`.
The deterministic ZIP_STORED source archive prevents bytecode caches from
changing the immutable payload under ordinary caller settings. The pinned
common environment is separately installed; its Torch, MLX, safetensors,
tokenizer and transitive libraries count toward installation cost. Both loaders
are local-only. Candidate inference never uses a download API.

The manifest hashes every other regular file and omits its own entry. The
returned `BundleRef.manifest_sha256` is the detached hash of the actual manifest
bytes. Payload bytes include the manifest. Verification rejects unknown files,
missing owners, inconsistent aliases/shapes/use sites, external owner paths,
symlinks and hardlinks. `archive_bundle` uses sorted fixed-metadata USTAR plus
gzip level 9/mtime zero; it measures extraction verification time and temporary
archive-plus-extracted disk bytes. Compressor identity is recorded.
`measure_bundle` reports payload and installation accounting; `measure_resident`
counts materialized canonical storage owners, excluding caller-owned KV caches.
Unmeasured archive/resident/phase values carry explicit unknown reasons. RSS
and MLX allocation counters overlap on unified memory and must not be added.

CLI export requires an explicit source. Run and attempt are external provenance
labels, not a resolver for a fitted training checkpoint:

```sh
python -m poc export --run my-run --attempt export1 \
  --source-bundle /tmp/source --out /tmp/reexport
python -m poc export --run teacher --attempt export1 \
  --teacher-assets data/prepared-135m-v2/teacher-assets.json \
  --teacher-manifest-sha256 9f027c4d1f9423fa1341c88a180ef08bc7fd45f7f3b144f64ed83de273d51083 \
  --runtime-spec /tmp/runtime-inputs.json --out /tmp/teacher-bundle
python -m poc validate --suite export --bundle /tmp/reexport --backend torch
python -m poc validate --suite export --bundle /tmp/reexport --backend mlx
```

`runtime-inputs.json` contains the fields shown in `RuntimeSpec` above. The
external `*.receipt.json` binds source kind/digest, supplied labels and exported
manifest digest. Compact candidates can be exported through the Python API
before the fitting-run checkpoint schema is implemented. Bundle validation
executes a tokenizer-generated prompt. Without `--bundle`, the export suite
runs CPU component tests by default, or native controls with `--backend mlx`.
A CPU result never establishes native deployment or WP03; a single artifact
smoke test also does not establish the complete WP03 gate.

`provenance/measure_bundle_deployment.py /private/tmp/new-attempt` performs
actual-shape dense/K=15 rank-8 floating parity at 16/128/512 tokens with cache
offset 7, frozen synthetic continuation choices, owner/byte census and fresh-process offline generation. Deterministic archive
equality and safe unpack/accounting run on complete tiny dense/rank-zero/rank-two
fixtures. Full-model archive costs are explicitly unmeasured after a bounded
level-9 gzip attempt established that repeated compression exceeds this task's
remaining aggregate time ceiling. Its nested OS sandbox
denies repository/source/training file contents and networking while allowing
installed common dependencies. Python audit and native `open` negative controls
check source denial. Prompts are tokenizer-generated; protected data stays sealed.
The finite resource preflight is printed before conversion, and failed attempts
remain attributable. This is identity/format evidence, not a trained compressed
candidate or a quality-frontier claim.
