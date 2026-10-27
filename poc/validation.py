"""Execute owned, asset-free component controls; record only actual outcomes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


COMPONENT_CONTROLS = (
    'tests/test_model_parity.py', 'tests/test_attention_causality.py',
    'tests/test_suffix_gradients.py', 'tests/test_bank.py', 'tests/test_aliasing.py',
    'tests/test_initialization.py', 'tests/test_kl.py', 'tests/test_chunked_loss.py',
    'tests/reference_math_checks.py',
)


def run_component_validation() -> int:
    """Run Tasks 3–5 CPU controls, persist WP01 outcome, preserve full WP00 status.

    This requires the source checkout and its test extra; a runtime wheel alone
    does not contain the qualification controls. Never implies teacher, assets,
    deployment, or full-model memory qualification.
    """
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-m', 'pytest', *COMPONENT_CONTROLS, '-q', '-m', 'not mps and not mlx']
    sources = [*COMPONENT_CONTROLS, 'tests/conftest.py', 'pyproject.toml',
               'provenance/decoder-tiny-numerics-2026-10-27.json',
               'provenance/loss-tiny-numerics-2026-10-27.json',
               *[str(p.relative_to(root)) for p in sorted((root/'poc').rglob('*.py'))]]
    before = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    after = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    passed = result.returncode == 0 and before == after
    record = dict(schema_version=1, gate='WP01', suite='component', status='passed' if passed else 'failed',
                  wp00_status='pending_asset_provenance', scope='Tasks 3–5 asset-free CPU controls only',
                  timestamp=datetime.now(timezone.utc).isoformat(),
                  command=['python', *command[1:]], returncode=result.returncode,
                  source_sha256=after, sources_unchanged=before == after,
                  stdout=result.stdout, stderr=result.stderr)
    path = root / 'artifacts/validation/component.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    print(f"WP01 component {'passed' if passed else 'failed'}; full WP00 pending asset provenance; record=artifacts/validation/component.json")
    return 0 if passed else 4


def run_export_validation(*, backend: str) -> int:
    """CPU/component and real MLX controls have separate explicit invocations."""
    if backend not in ('torch','mlx'): raise ValueError('unknown export validation backend')
    root = Path(__file__).resolve().parents[1]
    files = ['tests/test_export.py','tests/test_offline_bundle.py'] if backend == 'torch' else ['tests/test_native_parity.py']
    command = [sys.executable,'-m','pytest',*files,'-m','cpu' if backend == 'torch' else 'mlx','-q']
    result = subprocess.run(command,cwd=root)
    print('scientific export/component controls; native deployment unqualified' if backend == 'torch' else 'explicit native MLX export controls')
    return 0 if result.returncode == 0 else 4
