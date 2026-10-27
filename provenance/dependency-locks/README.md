# Tested native dependency resolution

These artifacts describe the resolution that passed tiny CPU/MPS/MLX tests on 2026-10-27. They are measured pins, not a suggested latest-version template.

- `qualified-darwin-arm64-py312.requirements.txt`: 67 exact public wheel URLs and SHA-256 hashes, including scientific/deployment/analysis/test transitive dependencies. Every selected wheel tag matches the installed distribution's `WHEEL` metadata; every archive was fetched and its hash independently checked against PyPI metadata. The local editable project is excluded.
- `wheel-provenance.json`: matching wheel filenames/tags, verified hashes, PyPI source-archive hashes, upstream project URLs and declared license metadata. The installed distributions retain complete license text.
- `upstream-commits.json`: immutable public release-tag commits for core runtime, analysis and test sources. Transformers uses tag `v5.21`; its source version was verified as `5.21.0`. Transitive package sources are pinned by their source-archive hashes where supplied rather than invented Git commits.
- `environment.json`: tested interpreter/compiler/platform, SDK, lock digest, PyTorch's own embedded source commit and pending clean-rebuild status.

Core versions: torch 2.14.1, mlx/​mlx-metal 0.32.4, mlx-lm 0.32.2 with train extra dependencies, transformers 5.21.0, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, datasets 5.1.0, safetensors 0.8.0, psutil 7.2.2, PyYAML 6.0.3 and pytest 9.1.1. No nightly, source build, system-wide installation or third-party patch was used. MLX's selected wheel targets macOS 26+; this lock is specific to this machine class and Python 3.12.

A later rebuild must create a second environment, install the exact wheel lock with hash checking, then install this checkout without resolving/updating dependencies:

```sh
uv venv --python 3.12 /tmp/cgoc-rebuild
uv pip sync --python /tmp/cgoc-rebuild/bin/python --require-hashes provenance/dependency-locks/qualified-darwin-arm64-py312.requirements.txt
uv pip install --python /tmp/cgoc-rebuild/bin/python --no-deps -e .
/tmp/cgoc-rebuild/bin/python -m pytest
/tmp/cgoc-rebuild/bin/python -m pytest tests/test_operations.py -m 'mps or mlx'
/tmp/cgoc-rebuild/bin/python -m poc doctor --out artifacts/rebuild-doctor.json
```

Use a writable cache, e.g. `--cache-dir /tmp/cgoc-uv-cache`, in this sandbox. Package fetching and host Metal access may require authorized escalation. The second rebuild has **not** been performed; Task 24 must record it before WP07/confirmation. Exact Python 3.12.9 provisioning must also be retained for a strict rebuild. Dependency versions alone do not qualify actual teacher, data, quantization, training or native export.
