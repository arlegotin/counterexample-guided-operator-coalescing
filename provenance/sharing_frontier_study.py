"""Complete-artifact preparation for the prospectively fixed RTN frontier.

Guarded preparation, outcome-blind profiles, fixed evaluation and paired
analysis preserve sources and never chain quantization across frontier points.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import shutil
import tempfile
from uuid import uuid4

from poc.contracts import BundleRef, thaw
from poc.quantize import QuantizationSpec, _verified_ref


def refresh_runtime(bundle: BundleRef, destination: Path) -> BundleRef:
    """Make a separately verified artifact with the current qualified runtime.

    Every non-runtime file and all owner/model metadata retain their exact source
    content. Failed partial copies remain beside the intended output, without an
    accepted BundleRef. Caller profiles/charges copying, hashing and disk storage.
    """
    from poc.bundle import _canonical, _sha, _regular_files, _write_runtime, verify_bundle

    source, manifest = _verified_ref(bundle)
    source, destination = source.resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('artifact source and destination may not overlap')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.build-', dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        if _sha(temporary / 'manifest.json') != bundle.manifest_sha256:
            raise ValueError('copied source manifest changed')
        verify_bundle(temporary)
        shutil.rmtree(temporary / 'runtime')
        _write_runtime(temporary)
        raw = thaw(manifest.metadata)
        raw['file_entries'] = {
            name: {'bytes': path.stat().st_size, 'sha256': _sha(path)}
            for name, path in sorted(_regular_files(temporary).items()) if name != 'manifest.json'}
        # Runtime packaging cannot quietly modify weights/tokenizer/config/etc.
        for name, entry in manifest.file_entries.items():
            if name != 'runtime/poc-runtime.zip' and raw['file_entries'].get(name) != thaw(entry):
                raise ValueError('repack changed a non-runtime dependency')
        (temporary / 'manifest.json').write_bytes(_canonical(raw))
        verify_bundle(temporary)
        digest = _sha(temporary / 'manifest.json')
        temporary.rename(destination)
        return BundleRef(str(destination), digest)
    except BaseException as exc:
        if temporary.exists():
            failed = destination.with_name(destination.name + '.failed-' + uuid4().hex)
            temporary.rename(failed)
            (failed / 'failure.json').write_text(json.dumps({
                'source_manifest_sha256': bundle.manifest_sha256,
                'intended_output': str(destination), 'error': repr(exc)}, indent=2) + '\n')
        raise


def rtn_jobs(teacher_ref: BundleRef, shared_ref: BundleRef, destination: Path):
    """Six fixed uniform-precision points, each from its own floating source."""
    _, teacher = _verified_ref(teacher_ref)
    _, shared = _verified_ref(shared_ref)
    if teacher.metadata['kind'] != 'dense' or shared.metadata['kind'] != 'compact':
        raise ValueError('dense teacher and compact shared floating sources required')
    jobs = []
    for kind, source in (('dense', teacher_ref), ('shared', shared_ref)):
        for bits in (8, 4, 2):
            label = f'{kind}_rtn{bits}'
            jobs.append((label, source, QuantizationSpec(
                'rtn', bits, 64, str(Path(destination) / label), embedding_bits=bits)))
    return tuple(jobs)


def metric_work_plan(rows, *, mode):
    """Budget the existing chunk32 worker on the entire fixed input inventory.

    Masks reduce scored targets; every complete context still runs one hidden
    forward and all16 readout chunks for each model. This is an accounting
    preflight; the source-bound worker additionally verifies vocabulary/hashes.
    """
    if mode not in ('profile', 'run') or not isinstance(rows, list) or len(rows) != 128:
        raise ValueError('fixed128-component inventory and profile/run mode required')
    documents, components = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('explicit window records required')
        doc, component = row.get('document_id'), row.get('component_id')
        if (type(doc) is not str or not doc or type(component) is not str or not component
                or doc in documents or component in components):
            raise ValueError('one distinct window per source document/component required')
        documents.add(doc); components.add(component)
        tokens, positions = row.get('tokens'), row.get('positions')
        if (not isinstance(tokens, list) or len(tokens) != 512
                or any(type(t) is not int or t < 0 for t in tokens)
                or not isinstance(positions, list) or positions != list(range(512))
                or any(type(t) is not int for t in positions)):
            raise ValueError('complete context512 with reset positions required')
        attention, target = row.get('attention_mask'), row.get('target_mask')
        if (not isinstance(attention, list) or len(attention) != 512
                or not isinstance(target, list) or len(target) != 511
                or any(type(t) is not bool for t in attention + target)
                or not attention[0] or not any(target)
                or any(not a and b for a, b in zip(attention, attention[1:]))
                or any(valid and not (attention[i] and attention[i+1]) for i, valid in enumerate(target))):
            raise ValueError('explicit eligible shifted target masks required')
    selected = rows[:1] if mode == 'profile' else rows
    calls = len(selected) * (1 + len(range(0, 511, 32)))
    return {'components': len(selected),
            'valid_target_presentations': sum(sum(row['target_mask']) for row in selected),
            'teacher_forward_calls': calls, 'student_forward_calls': calls}


def forecast_worker(profile, rows, *, teacher_complete_bytes, memory_bytes,
                    output_disk_bytes, wall_ceiling):
    """Resource-only completed schedule from an already admitted native profile.

    The caller must separately verify the profile's source/protocol/guard hashes.
    All profile copies and work remain separately charged. The fixed policy is
    twice(load/copy/setup +128 context costs)+120s, rounded upward to30s. Bounds
    are rejected when infeasible, never clipped to manufacture a runnable plan.
    """
    import math

    for name, value in (('teacher bytes', teacher_complete_bytes),
                        ('memory cap', memory_bytes), ('disk cap', output_disk_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError('positive integer ' + name + ' required')
    if type(wall_ceiling) not in (int, float) or not math.isfinite(wall_ceiling) or wall_ceiling <= 0:
        raise ValueError('finite positive wall ceiling required')
    profile_work = metric_work_plan(rows, mode='profile')
    run_work = metric_work_plan(rows, mode='run')
    expected = {k: v for k, v in profile_work.items() if k != 'components'}
    if (not isinstance(profile, dict) or profile.get('mode') != 'profile' or 'rows' in profile
            or profile.get('counts') != expected
            or any(type(v) is not int for v in profile.get('counts', {}).values())):
        raise ValueError('quality-free profile with exact input/query counts required')
    durations = profile.get('context_metric_seconds')
    wall = profile.get('wall_seconds')
    if (not isinstance(durations, list) or len(durations) != 1
            or type(durations[0]) not in (int, float) or not math.isfinite(durations[0]) or durations[0] <= 0
            or type(wall) not in (int, float) or not math.isfinite(wall) or wall < durations[0]):
        raise ValueError('finite actual context/setup timing required')
    memory = profile.get('memory')
    if (not isinstance(memory, dict) or set(memory) != {
            'process_peak_rss_bytes', 'torch_mps_driver_bytes', 'mlx_peak_bytes'}
            or any(type(v) is not int or v < 0 for v in memory.values())):
        raise ValueError('separately observed memory counters required')
    observed = max(memory.values())
    artifact_bytes = profile.get('complete_bytes')
    if type(artifact_bytes) is not int or artifact_bytes <= 0:
        raise ValueError('complete actual artifact bytes required')
    copies = teacher_complete_bytes + artifact_bytes + 16 * 1024**2
    forecast = wall - durations[0] + len(rows) * durations[0]
    allowance = math.ceil((2 * forecast + 120) / 30) * 30
    if allowance > wall_ceiling or observed > memory_bytes or copies > output_disk_bytes:
        raise ValueError('completed schedule exceeds its declared resource envelope')
    resource = {k: v for k, v in run_work.items() if k != 'components'}
    resource.update(wall_seconds=allowance, memory_bytes=memory_bytes, output_disk_bytes=output_disk_bytes)
    return {'resource': resource, 'forecast_seconds': forecast,
            'required_copy_bytes': copies, 'observed_memory_max_bytes': observed}


def load_contract(path):
    """Parse the closed version1 study identity, without creating any output.

    File/model/role/receipt verification is a separate mandatory preflight before
    actual preparation. Successfully parsing a draft does not permit execution.
    """
    from dataclasses import fields
    import math
    import re
    import yaml
    from poc.contracts import Budget, freeze

    def closed(value, names, label):
        if not isinstance(value, dict) or set(value) != set(names):
            raise ValueError('missing or unknown ' + label + ' fields')

    def digest(value):
        if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError('exact SHA256 reference required')

    def absolute(value):
        if type(value) is not str or not value or '\x00' in value or not Path(value).is_absolute():
            raise ValueError('explicit absolute path required')

    def reference(value):
        closed(value, ('path', 'sha256'), 'file reference')
        absolute(value['path']); digest(value['sha256'])

    def limits(value, names, label):
        closed(value, names, label)
        for key, item in value.items():
            integer = key not in ('wall_seconds', 'wall_ceiling', 'campaign_wall_seconds', 'deadline_unix')
            if (type(item) not in ((int,) if integer else (int, float))
                    or not math.isfinite(item) or item <= 0):
                raise ValueError('finite positive resource limits required: ' + key)

    raw = yaml.safe_load(Path(path).read_text())
    closed(raw, ('schema_version', 'study', 'out', 'teacher', 'shared', 'inputs',
                 'qualified_source', 'worker', 'reference', 'serial_gate', 'resource'), 'frontier contract')
    if type(raw['schema_version']) is not int or raw['schema_version'] != 1:
        raise ValueError('only frontier schema1 is implemented')
    if type(raw['study']) is not str or re.fullmatch('[a-z0-9][a-z0-9-]{0,99}', raw['study']) is None:
        raise ValueError('explicit ASCII study identity required')
    absolute(raw['out'])
    for key in ('teacher', 'shared'):
        closed(raw[key], ('directory', 'manifest_sha256'), key)
        absolute(raw[key]['directory']); digest(raw[key]['manifest_sha256'])
    closed(raw['inputs'], ('path', 'sha256', 'population_manifest_sha256'), 'compiled inputs')
    absolute(raw['inputs']['path']); digest(raw['inputs']['sha256']); digest(raw['inputs']['population_manifest_sha256'])
    for key in ('qualified_source', 'worker'):
        reference(raw[key])
    ref = raw['reference']
    closed(ref, ('teacher_mps_digest', 'source_run_id', 'preparation',
                 'preparation_attempt', 'preparation_protocol', 'source_contract', 'scientific_observation'), 'source reference')
    for key in ('teacher_mps_digest', 'source_run_id'):
        digest(ref[key])
    for key in ('preparation', 'preparation_attempt', 'preparation_protocol', 'source_contract', 'scientific_observation'):
        reference(ref[key])
    gate = raw['serial_gate']
    closed(gate, ('directory', 'registry_sha256', 'runner', 'result_relative', 'attempt_field'), 'serial gate')
    absolute(gate['directory']); absolute(gate['runner']); digest(gate['registry_sha256'])
    relative = gate['result_relative']
    if (type(relative) is not str or not relative or Path(relative).is_absolute()
            or '..' in Path(relative).parts or Path(relative).suffix != '.json'
            or gate['attempt_field'] not in ('attempt_sha256', 'analysis_attempt_sha256')):
        raise ValueError('closed relative accepted-analysis binding required')
    resource = raw['resource']
    closed(resource, ('preparation', 'profile', 'evaluation'), 'resource')
    closed(resource['preparation'], {field.name for field in fields(Budget)}, 'preparation budget')
    preparation = Budget(**resource['preparation'])
    if (any(getattr(preparation, key) != 0 for key in (
            'updates', 'valid_target_presentations', 'suffix_backward_calls',
            'student_backward_calls', 'search_proposals', 'search_seconds'))
            or any(getattr(preparation, key) <= 0 for key in ('wall_seconds', 'memory_bytes', 'disk_bytes'))):
        raise ValueError('finite inference-only preparation budget required')
    limits(resource['profile'], ('wall_seconds', 'memory_bytes', 'output_disk_bytes'), 'profile caps')
    limits(resource['evaluation'], ('memory_bytes', 'output_disk_bytes', 'wall_ceiling',
           'campaign_wall_seconds', 'family_disk_bytes', 'deadline_unix'), 'evaluation caps')
    return freeze(raw)


def file_sha256(path):
    import hashlib
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('regular unlinked frozen file required: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_reference(reference):
    if set(reference) != {'path', 'sha256'} or file_sha256(reference['path']) != reference['sha256']:
        raise ValueError('frozen JSON reference changed')
    path = Path(reference['path'])
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError('metadata reference exceeds its finite64MiB bound')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('metadata object required')
    return value


def verify_source_receipts(contract):
    """Admit the already accepted dev-input compiler BEFORE token content I/O.

    This verifies its custom host-supervisor format, frozen source ancestry and
    exact output references. It neither reads compiled tokens nor loads models.
    Actual current-source/environment/serial admission is additionally required.
    """
    import math
    c = thaw(contract)
    ref = c['reference']
    source = read_json_reference(ref['source_contract'])
    protocol = read_json_reference(ref['preparation_protocol'])
    attempt = read_json_reference(ref['preparation_attempt'])
    prep = read_json_reference(ref['preparation'])
    bindings = protocol.get('frozen_files', [])
    if (ref['source_contract'] not in bindings
            or attempt.get('protocol_sha256') != ref['preparation_protocol']['sha256']
            or type(attempt.get('exit_code')) is not int or attempt['exit_code'] != 0
            or attempt.get('failure', 'missing') is not None
            or attempt.get('caps') != protocol.get('caps')
            or attempt.get('command') != [protocol.get('python'), protocol.get('worker')]
            or attempt.get('study') != protocol.get('study')
            or attempt.get('supervisor_sha256') not in {r.get('sha256') for r in bindings}
            or not any(r.get('path') == protocol.get('worker') for r in bindings)):
        raise ValueError('source compiler lacks exact successful guarded acceptance')
    caps = protocol['caps']
    if set(caps) != {'wall_seconds', 'process_rss_bytes', 'family_disk_bytes'}:
        raise ValueError('source compiler resource contract differs')
    for limit, measured in (('wall_seconds', 'elapsed_seconds'),
                            ('process_rss_bytes', 'sampled_peak_process_tree_rss_bytes'),
                            ('family_disk_bytes', 'sampled_family_output_peak_bytes')):
        cap, actual = caps[limit], attempt.get(measured)
        if (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0
                or type(actual) not in (int, float) or not math.isfinite(actual)
                or actual < 0 or actual > cap):
            raise ValueError('source compiler failed its resource envelope')
    for item in [*bindings, *source.get('frozen_files', [])]:
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('accepted source compiler dependency changed')
    population = source.get('population', {})
    if (population.get('role') != 'dev' or population.get('purpose') != 'selection'
            or type(population.get('components')) is not int or population['components'] != 128
            or population.get('sha256') != c['inputs']['population_manifest_sha256']
            or source.get('source_run_id') != ref['source_run_id']
            or source.get('teacher_mps_digest') != ref['teacher_mps_digest']):
        raise ValueError('only the declared exploratory dev/selection population is admitted')
    output = Path(protocol['output']).resolve()
    if (Path(ref['preparation']['path']).resolve() != output / 'preparation.json'
            or Path(ref['preparation_attempt']['path']).parent.resolve() != Path(protocol['provenance']).resolve()
            or not Path(c['inputs']['path']).resolve().is_relative_to(output)
            or prep.get('study_contract_sha256') != ref['source_contract']['sha256']
            or prep.get('inputs') != c['inputs'] or prep.get('components') != 128
            or prep.get('valid_targets') != 65408
            or prep.get('teacher_mps_digest') != ref['teacher_mps_digest']):
        raise ValueError('accepted input compiler outputs differ from the declared study')
    for target, original in (('teacher', 'teacher_noop'), ('shared', 'shared_full17_float')):
        old = prep.get('artifacts', {}).get(original, {})
        if any(old.get(k) != v for k, v in c[target].items()) or not Path(c[target]['directory']).resolve().is_relative_to(output):
            raise ValueError('floating source artifact differs from accepted preparation')
    return prep


NATIVE_WORKER_SHA256 = '7c062d1cd6686c247690a7367f021cdbad8a8a2b45aba358085d88c2507828b5'


def contract_digest(contract):
    import hashlib
    from poc.records import canonical_json
    return hashlib.sha256(canonical_json(thaw(contract))).hexdigest()


def native_worker(contract):
    import importlib.util
    ref = contract['worker']
    if ref['sha256'] != NATIVE_WORKER_SHA256 or file_sha256(ref['path']) != NATIVE_WORKER_SHA256:
        raise ValueError('only the qualified native fidelity worker is admitted')
    spec = importlib.util.spec_from_file_location('_shared_rtn_native_worker', ref['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_sources(contract):
    """Read-only complete metadata/input preflight; no model execution/output."""
    import poc
    from poc.runtime.torch_decoder import EXECUTION_PROFILE
    from poc.bundle import measure_bundle

    c = thaw(contract)
    prep = verify_source_receipts(c)  # role/output provenance BEFORE token I/O
    if Path(poc.__file__).resolve().parents[1] != ROOT:
        raise ValueError('project import namespace differs from the driver source')
    qualification = read_json_reference(c['qualified_source'])
    cpu = qualification.get('tests', {}).get('default_cpu', {})
    if (type(cpu.get('passed')) is not int or cpu['passed'] <= 0
            or cpu.get('guard', {}).get('exit_code') != 0
            or cpu.get('guard', {}).get('failure', 'missing') is not None):
        raise ValueError('baseline CPU qualification did not pass')
    sources = qualification.get('source_files')
    needed = {str(p.relative_to(ROOT)) for p in (ROOT / 'poc').rglob('*.py')}
    if not isinstance(sources, dict) or not needed <= set(sources):
        raise ValueError('baseline qualification omits current runtime sources')
    for name, digest in sources.items():
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or file_sha256(ROOT / name) != digest):
            raise ValueError('qualified baseline source changed')
    output = Path(c['out']).resolve()
    for label in ('teacher', 'shared'):
        original = Path(c[label]['directory']).resolve()
        if output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError('study output overlaps an immutable source')
    if output.exists():
        owner = output / '.frontier-owner.json'
        if not owner.is_file():
            raise FileExistsError('output is not owned by this exact study')
        if json.loads(owner.read_text()) != {'schema_version': 1, 'study': c['study'], 'contract_sha256': contract_digest(c)}:
            raise ValueError('output belongs to another study identity')
    teacher_ref, shared_ref = (BundleRef(**c[k]) for k in ('teacher', 'shared'))
    teacher_path, teacher = _verified_ref(teacher_ref)
    shared_path, shared = _verified_ref(shared_ref)
    if teacher.metadata['kind'] != 'dense' or shared.metadata['kind'] != 'compact':
        raise ValueError('dense and compact floating sources required')
    for base, manifest in ((teacher_path, teacher), (shared_path, shared)):
        quantization = read_json_reference({'path': str(base / 'quantization.json'),
                                            'sha256': manifest.file_entries['quantization.json']['sha256']})
        if (quantization.get('format') != 'floating'
                or manifest.metadata['execution_profile'] != EXECUTION_PROFILE
                or manifest.metadata['config']['max_position_embeddings'] < 512):
            raise ValueError('floating context512 sources with qualified profile required')
    for label, old_label in (('teacher', 'teacher_noop'), ('shared', 'shared_full17_float')):
        if measure_bundle(Path(c[label]['directory'])).payload_bytes != prep['artifacts'][old_label]['complete_bytes']:
            raise ValueError('complete source bytes differ from accepted preparation')
    worker = native_worker(c)
    worker.compatible_tokenizers(thaw(teacher.metadata), thaw(shared.metadata))
    rows = worker.read_windows(c['inputs']['path'], c['inputs']['sha256'],
                               population_sha256=c['inputs']['population_manifest_sha256'],
                               tokenizer_digest=teacher.metadata['tokenizer_digest'],
                               vocabulary_size=teacher.metadata['config']['vocab_size'])
    run = metric_work_plan(rows, mode='run')
    if run['valid_target_presentations'] != prep['valid_targets']:
        raise ValueError('input masks differ from accepted source population')
    return {'windows': rows, 'run_work': run, 'profile_work': metric_work_plan(rows, mode='profile'),
            'teacher': teacher_ref, 'shared': shared_ref, 'preparation': prep,
            'baseline_sources': sources, 'worker': worker}


def write_exclusive(path, value):
    import os
    from poc.records import canonical_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_json(value)); stream.flush(); os.fsync(stream.fileno())


def execute_stage(command, *, root, output, stage, caps, frozen_files, execution_env=None):
    """Existing finite supervisor plus qualified process-group cleanup/file logs."""
    import importlib.util
    import math

    if (not isinstance(command, list) or not command or any(type(x) is not str for x in command)
            or set(caps) != {'wall_seconds', 'memory_bytes', 'disk_bytes'}):
        raise ValueError('explicit command and complete finite caps required')
    for key, value in caps.items():
        if (type(value) not in ((int, float) if key == 'wall_seconds' else (int,))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('invalid finite stage resource cap')
    helper_root = ROOT / 'provenance/shared-rtn-frontier-v1'
    guard = helper_root / 'scientific-stage-supervisor-v1.py'
    durable = helper_root / 'durable-stage-v1.py'
    cleanup = helper_root / 'qualified-controller-v4.py'
    if file_sha256(guard) != 'a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba':
        raise ValueError('qualified stage supervisor changed')
    refs = list(frozen_files) + [{'path': str(p), 'sha256': file_sha256(p)} for p in (guard, durable, cleanup)]
    seen = {}
    for ref in refs:
        if ref['path'] in seen and seen[ref['path']] != ref['sha256']:
            raise ValueError('conflicting frozen stage reference')
        if file_sha256(ref['path']) != ref['sha256']:
            raise ValueError('frozen stage source changed')
        seen[ref['path']] = ref['sha256']
    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=False)
    protocol = {'root': str(Path(root).resolve()), 'output': str(Path(output).resolve()),
                'provenance': str(stage), 'command': command, 'caps': dict(caps),
                'supervisor_sha256': file_sha256(guard),
                'frozen_files': [{'path': p, 'sha256': h} for p, h in sorted(seen.items())],
                'execution_env': {'HF_HUB_OFFLINE': None, 'PYTORCH_ENABLE_MPS_FALLBACK': None,
                                  **(execution_env or {})},
                'scope': 'fixed RTN artifact study; finite source-bound stage; no retries'}
    path = stage / 'protocol.json'
    write_exclusive(path, protocol)
    protocol_sha = file_sha256(path)
    spec = importlib.util.spec_from_file_location('_rtn_durable_stage', durable)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rc = module.run_stage([sys.executable, str(guard), '--protocol', str(path)],
                          cwd=str(Path(root).resolve()), log_path=stage / 'supervisor.log')
    if file_sha256(path) != protocol_sha:
        raise ValueError('stage protocol mutated during execution')
    attempt_path = stage / 'attempt.json'
    if not attempt_path.is_file():
        raise RuntimeError('stage lacks an accepted guard receipt; inspect retained logs')
    attempt = json.loads(attempt_path.read_text())
    if attempt.get('protocol_sha256') != protocol_sha:
        raise ValueError('guard receipt does not bind the executed protocol')
    for source_path, digest in seen.items():
        if file_sha256(source_path) != digest:
            raise ValueError('frozen source changed during the stage; receipt retained')
    return {'exit_code': attempt['exit_code'], 'supervisor_exit_code': rc,
            'attempt': attempt, 'attempt_path': str(attempt_path),
            'attempt_sha256': file_sha256(attempt_path), 'protocol_path': str(path),
            'protocol_sha256': protocol_sha}


def require_stage_success(result):
    """A child result is eligible only with a successful, unchanged guard."""
    if result.get('supervisor_exit_code') != 0 or result.get('exit_code') != 0:
        raise ValueError('stage or its supervisor failed; retained work is not accepted')
    attempt = read_json_reference({'path': result['attempt_path'], 'sha256': result['attempt_sha256']})
    protocol = read_json_reference({'path': result['protocol_path'], 'sha256': result['protocol_sha256']})
    if (attempt.get('exit_code') != 0 or attempt.get('failure', 'missing') is not None
            or attempt.get('protocol_sha256') != result['protocol_sha256']
            or attempt != result['attempt']):
        raise ValueError('guarded stage receipt is incomplete or changed')
    for limit, measured in (('wall_seconds', 'wall_seconds'),
                            ('memory_bytes', 'sampled_process_tree_rss_bytes'),
                            ('disk_bytes', 'sampled_scratch_peak_bytes')):
        if attempt.get(measured, float('inf')) > protocol['caps'][limit]:
            raise ValueError('completed stage exceeds its frozen cap')
    for ref in protocol['frozen_files']:
        if file_sha256(ref['path']) != ref['sha256']:
            raise ValueError('accepted stage source changed')


def require_serial_gate(contract):
    import importlib.util
    path = ROOT / 'provenance/shared-rtn-frontier-v1/continuation-gates-v2.py'
    if file_sha256(path) != '5d4b2a2100fb8594b4d958a57c2463ab3b7050b51e33c3b5cc721a9b865582d3':
        raise ValueError('qualified serial admission implementation changed')
    spec = importlib.util.spec_from_file_location('_rtn_continuation_gates', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.admit_analysis(**thaw(contract['serial_gate']))


def implementation_sources(contract):
    qualification = read_json_reference(contract['qualified_source'])
    refs = {str(ROOT / p): h for p, h in qualification['source_files'].items()}
    for path in [Path(__file__).resolve(), ROOT / 'tests/test_sharing_frontier.py',
                 *sorted((ROOT / 'provenance/shared-rtn-frontier-v1').glob('*.py'))]:
        refs[str(path)] = file_sha256(path)
    return refs


def require_own_controls(contract):
    path = ROOT / 'provenance' / contract['study'] / 'controls.json'
    if not path.is_file():
        raise ValueError('current source-bound owned controls have not passed')
    data = json.loads(path.read_text())
    if (data.get('contract_sha256') != contract_digest(contract)
            or data.get('sources') != implementation_sources(contract)):
        raise ValueError('owned controls bind another source/configuration')
    require_stage_success(data['stage'])
    return data


def prepare_campaign(contract):
    """Create complete fixed RTN artifacts only after actual serial admission."""
    import gc
    import importlib.util
    import time
    from dataclasses import replace
    from poc.contracts import Budget, CostRecord, RunSpec
    from poc.records import start_attempt
    from poc.recovery_family import RetainedFamily
    from poc.train import _call, finish_attempt_cost

    started = time.monotonic()
    require_serial_gate(contract)
    controls = require_own_controls(contract)
    require_current_qualification(contract)
    inspected = inspect_sources(contract)
    c = thaw(contract); output = Path(c['out'])
    if (output / 'preparation.json').exists():
        raise FileExistsError('preparation already completed; no automatic repeat')
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        write_exclusive(output / '.frontier-owner.json', {
            'schema_version': 1, 'study': c['study'], 'contract_sha256': contract_digest(c)})
    family = RetainedFamily(str(output), [str(Path(c['reference']['preparation']['path']).parent), str(ROOT / 'provenance')],
                            required_sources=[str(Path(c['teacher']['directory']) / 'manifest.json'),
                                              str(Path(c['shared']['directory']) / 'manifest.json'), c['inputs']['path']])
    spec = importlib.util.spec_from_file_location('_rtn_common_driver', ROOT / 'provenance/recovery_diagnosis.py')
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    sources = implementation_sources(c)
    budget = Budget(**c['resource']['preparation'])
    identity = RunSpec('Q', c['study'] + ':preparation', {'contract': c, 'expected_artifact_forward_calls': 64,
                       'retained_family': family.identity(), 'sources': sources},
                       __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), common.hashed(sources), common.hashed(common.environment()),
                       {'contract': contract_digest(c), 'teacher': c['teacher']['manifest_sha256'],
                        'shared': c['shared']['manifest_sha256'], 'population': c['inputs']['population_manifest_sha256'],
                        'compiled_inputs': c['inputs']['sha256']},
                       {'device': 'native MLX packing and artifact generation; no quality scoring'},
                       {'source_fit': 17, 'quantization': 17}, (c['reference']['source_run_id'],), budget)
    attempt = start_attempt(identity, output / 'records')

    class Tracker(common.StudyTracker):
        def disk(self):
            return family.disk()
        def observed(self, cost):
            cost = super().observed(cost)
            if 'mlx.core' in sys.modules:
                cost = replace(cost, memory_bytes=max(cost.memory_bytes or 0, int(sys.modules['mlx.core'].get_peak_memory())))
            return cost
        def check(self, cost):
            if self.totals.student_forward_calls + cost.student_forward_calls > 64:
                raise ValueError('fixed offline-proof artifact-forward allowance exceeded')
            super().check(cost)

    recorder = Tracker(budget, attempt, output)
    try:
        import mlx.core as mx
        import torch
        from poc.quantize import quantize_rtn
        from poc.bundle import verify_bundle, measure_bundle
        torch.set_num_threads(1); mx.set_default_device(mx.gpu)
        if not mx.metal.is_available():
            raise RuntimeError('actual Metal backend is unavailable')
        artifacts = {}
        floating = {}
        for label, source in (('teacher_noop', inspected['teacher']), ('shared_float', inspected['shared'])):
            floating[label] = _call(recorder, lambda source=source, label=label:
                                     refresh_runtime(source, output / 'artifacts' / label))
        refs = dict(floating)
        for label, source, qspec in rtn_jobs(floating['teacher_noop'], floating['shared_float'], output / 'artifacts'):
            with retain_packing_failures(qspec.destination):
                refs[label] = quantize_rtn(source, qspec, recorder).bundle
        for label, ref in refs.items():
            metadata = verify_bundle(Path(ref.directory)).metadata
            artifacts[label] = {'study_contract_sha256': contract_digest(c), 'directory': ref.directory, 'manifest_sha256': ref.manifest_sha256,
                                'complete_bytes': measure_bundle(Path(ref.directory)).payload_bytes,
                                'runtime_sha256': file_sha256(Path(ref.directory) / 'runtime/poc-runtime.zip'),
                                'execution_profile': metadata['execution_profile'], 'owners': len(metadata['owners']),
                                'teacher_needed_at_inference': False}
        if len({a['runtime_sha256'] for a in artifacts.values()}) != 1:
            raise ValueError('all eight artifacts must execute the same frozen runtime')
        gc.collect(); mx.clear_cache()
        test_spec = importlib.util.spec_from_file_location('_rtn_offline_proof', ROOT / 'tests/test_offline_bundle.py')
        test_module = importlib.util.module_from_spec(test_spec)
        sys.path.insert(0, str(ROOT / 'tests')); test_spec.loader.exec_module(test_module)
        proofs = {}
        for label, ref in refs.items():
            directory = output / 'offline-proofs' / label; directory.mkdir(parents=True)
            proofs[label] = _call(recorder, lambda ref=ref, directory=directory:
                                  test_module.run_offline_bundle(ref.directory, directory, backend='mlx',
                                     blocked_source_dirs=(str(ROOT), str(Path(c['teacher']['directory']).parent))),
                                  student_forward_calls=8, processed_inputs=4)
            recorder.record(CostRecord(processed_tokens=len(proofs[label]['initial_ids']) + 3))
        finish_attempt_cost(recorder, elapsed=time.monotonic() - started)
        result = {'schema_version': 1, 'study_contract_sha256': contract_digest(c), 'source_identity': common.hashed(sources),
                  'artifacts': artifacts, 'inputs': c['inputs'], 'teacher_mps_digest': c['reference']['teacher_mps_digest'],
                  'components': 128, 'valid_targets': 65408, 'offline_proofs': proofs,
                  'blocked_source_roots': sorted({__import__('os').fsdecode(row[9:]) for row in
                     __import__('subprocess').check_output(['git', 'worktree', 'list', '--porcelain', '-z'], cwd=ROOT).split(b'\0')
                     if row.startswith(b'worktree ')}),
                  'retained_family': family.identity(), 'cost': recorder.totals.to_dict(),
                  'scope': 'complete fixed artifacts and offline generation proofs; no new quality scoring'}
        recorder.check(CostRecord())
        write_exclusive(output / 'preparation.json', result)
        common.finish_attempt(attempt, 'exhausted', reason='all eight fixed complete artifacts prepared')
        return result
    except BaseException as exc:
        try:
            finish_attempt_cost(recorder, elapsed=time.monotonic() - started)
        finally:
            common.finish_attempt(attempt, 'failed', reason=repr(exc))
        raise


POINT_LABELS = ('teacher_noop', 'shared_float', 'dense_rtn8', 'dense_rtn4',
                'dense_rtn2', 'shared_rtn8', 'shared_rtn4', 'shared_rtn2')


def native_protocol(contract, prepared, label, rows, *, mode, resource, output, ledger, profile_binding=None):
    """Compile the already qualified worker's exact unchanged protocol schema."""
    import math
    c = thaw(contract)
    if (prepared.get('study_contract_sha256') != contract_digest(c)
            or prepared.get('inputs') != c['inputs'] or set(prepared.get('artifacts', {})) != set(POINT_LABELS)
            or label not in POINT_LABELS
            or prepared.get('teacher_mps_digest') != c['reference']['teacher_mps_digest']):
        raise ValueError('prepared artifacts/inputs do not belong to this fixed study')
    work = metric_work_plan(rows, mode=mode)
    counts = {k: v for k, v in work.items() if k != 'components'}
    if set(resource) != {'wall_seconds', 'memory_bytes', 'output_disk_bytes', *counts}:
        raise ValueError('complete closed native resource plan required')
    for key, value in resource.items():
        if (type(value) not in ((int, float) if key == 'wall_seconds' else (int,))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('finite positive native resource limits required')
    if any(resource[k] != value for k, value in counts.items()):
        raise ValueError('resource plan changes raw target or model-query work')
    if mode == 'profile':
        if profile_binding is not None or any(resource[k] != v for k, v in c['resource']['profile'].items()):
            raise ValueError('profile may not import quality results or alter its caps')
    elif (not isinstance(profile_binding, dict) or set(profile_binding) != {'path', 'sha256'}
          or resource['wall_seconds'] > c['resource']['evaluation']['wall_ceiling']
          or any(resource[k] != c['resource']['evaluation'][k] for k in ('memory_bytes', 'output_disk_bytes'))):
        raise ValueError('evaluation requires a frozen profile and bounded completed schedule')
    teacher = prepared['artifacts']['teacher_noop']
    artifact = prepared['artifacts'][label]
    if artifact['runtime_sha256'] != teacher['runtime_sha256']:
        raise ValueError('teacher and candidate embedded runtimes differ')
    return {'schema_version': 1, 'study': c['study'], 'study_contract_sha256': contract_digest(c), 'mode': mode,
            'artifact': {'path': artifact['directory'], 'manifest_sha256': artifact['manifest_sha256']},
            'teacher': {'path': teacher['directory'], 'manifest_sha256': teacher['manifest_sha256'],
                        'mps_checkpoint_digest': prepared['teacher_mps_digest']},
            'inputs': c['inputs'], 'numerical': {'chunk_size': 32, 'temperature': 1,
                                               'negative_floor': 2e-6, 'decision_resolution': 2e-5},
            'worker_sha256': c['worker']['sha256'], 'runtime_sha256': artifact['runtime_sha256'],
            'blocked_source_roots': prepared['blocked_source_roots'], 'output': str(Path(output).resolve()),
            'ledger': str(Path(ledger).resolve()), 'resource': dict(resource), 'profile_binding': profile_binding}


def check_native_result(protocol, value, *, rows=None):
    """Check accepted-worker output identity/work before publishing its index."""
    import math
    expected = {'schema_version': 1, 'mode': protocol['mode'],
                'study_contract_sha256': protocol['study_contract_sha256'],
                'protocol_sha256': contract_digest(protocol),
                'artifact_manifest_sha256': protocol['artifact']['manifest_sha256'],
                'teacher_manifest_sha256': protocol['teacher']['manifest_sha256'],
                'runtime_sha256': protocol['runtime_sha256'], 'worker_sha256': protocol['worker_sha256'],
                'inputs_sha256': protocol['inputs']['sha256']}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('native result belongs to another source/protocol/artifact')
    counts = {k: protocol['resource'][k] for k in ('valid_target_presentations', 'teacher_forward_calls', 'student_forward_calls')}
    if value.get('counts') != counts:
        raise ValueError('native result did not complete its exact target/query plan')
    noop = protocol['artifact']['manifest_sha256'] == protocol['teacher']['manifest_sha256']
    if value.get('teacher_equivalent_control') is not noop or (noop and value.get('teacher_equivalent_passed') is not True):
        raise ValueError('teacher-equivalent numerical qualification failed')
    if protocol['mode'] == 'profile':
        if 'rows' in value:
            raise ValueError('profiles must not disclose candidate quality')
        return
    actual = value.get('rows')
    if not isinstance(rows, list) or not isinstance(actual, list) or len(actual) != len(rows):
        raise ValueError('complete prespecified component population required')
    for expected_row, row in zip(rows, actual, strict=True):
        n = sum(expected_row['target_mask'])
        if (any(row.get(k) != expected_row[k] for k in ('document_id', 'component_id', 'window_sha256'))
                or row.get('study_contract_sha256') != protocol['study_contract_sha256']
                or row.get('protocol_sha256') != contract_digest(protocol)
                or type(row.get('valid_targets')) is not int or row['valid_targets'] != n
                or type(row.get('disagreements')) is not int or not 0 <= row['disagreements'] <= n):
            raise ValueError('native component/window/mask bindings differ')
        for key in ('kl_sum', 'nll_sum', 'teacher_nll_sum'):
            number = row.get(key)
            if type(number) not in (int, float) or not math.isfinite(number):
                raise ValueError('finite raw full-vocabulary metric sums required')
        if row['kl_sum'] < -2e-6 * n or row['nll_sum'] < 0 or row['teacher_nll_sum'] < 0:
            raise ValueError('raw metrics exceed the qualified numerical floor')


def study_family(contract):
    from poc.recovery_family import RetainedFamily
    c = thaw(contract)
    return RetainedFamily(c['out'], [str(Path(c['reference']['preparation']['path']).parent), str(ROOT / 'provenance')],
                          required_sources=[str(Path(c['teacher']['directory']) / 'manifest.json'),
                                            str(Path(c['shared']['directory']) / 'manifest.json'), c['inputs']['path']])


def execution_references(contract, config_path):
    c = thaw(contract)
    refs = [{'path': p, 'sha256': h} for p, h in implementation_sources(c).items()]
    refs += [dict(c['qualified_source']), dict(c['worker']),
             {'path': str(Path(config_path).resolve()), 'sha256': file_sha256(config_path)},
             {'path': c['inputs']['path'], 'sha256': c['inputs']['sha256']}]
    refs += [dict(v) for v in c['reference'].values() if isinstance(v, dict)]
    refs += [{'path': str(Path(c[k]['directory']) / 'manifest.json'), 'sha256': c[k]['manifest_sha256']}
             for k in ('teacher', 'shared')]
    controls = ROOT / 'provenance' / c['study'] / 'controls.json'
    refs.append({'path': str(controls), 'sha256': file_sha256(controls)})
    qualification = ROOT / 'provenance' / c['study'] / 'qualification.json'
    require_current_qualification(c)
    refs.append({'path': str(qualification), 'sha256': file_sha256(qualification)})
    refs += require_serial_gate(c)
    return refs


def read_prepared(contract, *, config_path=None):
    c = thaw(contract); output = Path(c['out'])
    path = output / 'preparation-acceptance.json'
    if not path.is_file():
        raise ValueError('no guarded accepted artifact preparation')
    acceptance = json.loads(path.read_text())
    config_path = Path(config_path or ROOT / 'configs' / (c['study'] + '.yaml')).resolve()
    caps = c['resource']['preparation']
    check_stage_binding(acceptance['stage'],
        command=[sys.executable, str(Path(__file__).resolve()), '--config', str(config_path), '--internal-prepare'],
        root=ROOT, output=output,
        caps={'wall_seconds': caps['wall_seconds'] + 30, 'memory_bytes': caps['memory_bytes'], 'disk_bytes': caps['disk_bytes']},
        required_refs=execution_references(c, config_path))
    if acceptance['output'].get('path') != str(output / 'preparation.json'):
        raise ValueError('preparation acceptance names another output')
    result = read_json_reference(acceptance['output'])
    if (result.get('study_contract_sha256') != contract_digest(c)
            or set(result.get('artifacts', {})) != set(POINT_LABELS)
            or result.get('inputs') != c['inputs']):
        raise ValueError('prepared artifact index changed')
    for row in result['artifacts'].values():
        _verified_ref(BundleRef(row['directory'], row['manifest_sha256']))
    return result


def native_job(contract, prepared, label, rows, *, mode, resource, config_path, profile_binding=None, expected_protocol_ref=None):
    c = thaw(contract); output = Path(c['out']); case = output / 'workers' / (mode + '-' + label)
    protocol = native_protocol(c, prepared, label, rows, mode=mode, resource=resource,
                               output=case / 'result.json', ledger=case / 'costs.jsonl', profile_binding=profile_binding)
    if mode == 'run':
        if expected_protocol_ref is None or read_json_reference(expected_protocol_ref) != protocol:
            raise ValueError('evaluation differs from its frozen protocol before dispatch')
    elif expected_protocol_ref is not None:
        raise ValueError('profiles cannot import an evaluation protocol')
    protocol_path = output / 'protocols' / (mode + '-' + label + '.json')
    write_exclusive(protocol_path, protocol)
    refs = execution_references(c, config_path)
    refs += [{'path': str(protocol_path), 'sha256': file_sha256(protocol_path)},
             {'path': str(output / 'preparation.json'), 'sha256': file_sha256(output / 'preparation.json')}]
    for item in (protocol['teacher'], protocol['artifact']):
        refs.append({'path': str(Path(item['path']) / 'manifest.json'), 'sha256': item['manifest_sha256']})
    if profile_binding is not None:
        refs.append(dict(profile_binding))
    if expected_protocol_ref is not None:
        refs.append(dict(expected_protocol_ref))
    stage = execute_stage([sys.executable, c['worker']['path'], '--protocol', str(protocol_path)],
                          root=Path('/private/tmp'), output=case,
                          stage=output / 'stages' / (mode + '-' + label),
                          caps={'wall_seconds': resource['wall_seconds'] + 30,
                                'memory_bytes': resource['memory_bytes'], 'disk_bytes': resource['output_disk_bytes']},
                          frozen_files=refs, execution_env={'PYTHONPATH': ''})
    require_stage_success(stage)
    result_ref = {'path': str(case / 'result.json'), 'sha256': file_sha256(case / 'result.json')}
    item = {'stage': stage, 'protocol': {'path': str(protocol_path), 'sha256': file_sha256(protocol_path)},
            'output': result_ref}
    accept_native_job(c, prepared, item, label, rows, mode=mode, config_path=config_path,
                      expected_protocol_ref=expected_protocol_ref)
    return item


def profile_campaign(contract, *, config_path):
    """One-shot preparation/profile/freeze, with no candidate quality disclosed."""
    c = thaw(contract)
    require_serial_gate(c); require_own_controls(c); require_current_qualification(c)
    inspected = inspect_sources(c); output = Path(c['out'])
    if output.exists():
        raise FileExistsError('profile/preparation already attempted; preserve it under this identity')
    output.mkdir(parents=True, exist_ok=False)
    write_exclusive(output / '.frontier-owner.json', {
        'schema_version': 1, 'study': c['study'], 'contract_sha256': contract_digest(c)})
    refs = execution_references(c, config_path)
    caps = c['resource']['preparation']
    stage = execute_stage([sys.executable, str(Path(__file__).resolve()), '--config', str(Path(config_path).resolve()),
                           '--internal-prepare'], root=ROOT, output=output, stage=output / 'stages/preparation',
                          caps={'wall_seconds': caps['wall_seconds'] + 30, 'memory_bytes': caps['memory_bytes'],
                                'disk_bytes': caps['disk_bytes']}, frozen_files=refs)
    require_stage_success(stage)
    prep_ref = {'path': str(output / 'preparation.json'), 'sha256': file_sha256(output / 'preparation.json')}
    write_exclusive(output / 'preparation-acceptance.json', {'stage': stage, 'output': prep_ref})
    prepared = read_prepared(c, config_path=config_path)
    records = {}
    profile_work = {k: v for k, v in inspected['profile_work'].items() if k != 'components'}
    for label in POINT_LABELS:
        records[label] = native_job(c, prepared, label, inspected['windows'], mode='profile',
                                    resource={**c['resource']['profile'], **profile_work}, config_path=config_path)
    index = {'schema_version': 1, 'study_contract_sha256': contract_digest(c), 'profiles': records}
    write_exclusive(output / 'profiles.json', index)
    protocols = {}; budgets = {}; total_wall = 0; future_copies = 0
    for label in POINT_LABELS:
        profile_ref = records[label]['output']; value = read_json_reference(profile_ref)
        plan = forecast_worker(value, inspected['windows'],
                               teacher_complete_bytes=prepared['artifacts']['teacher_noop']['complete_bytes'],
                               memory_bytes=c['resource']['evaluation']['memory_bytes'],
                               output_disk_bytes=c['resource']['evaluation']['output_disk_bytes'],
                               wall_ceiling=c['resource']['evaluation']['wall_ceiling'])
        budgets[label] = plan; total_wall += plan['resource']['wall_seconds'] + 30; future_copies += plan['required_copy_bytes']
        case = output / 'workers' / ('run-' + label)
        protocol = native_protocol(c, prepared, label, inspected['windows'], mode='run', resource=plan['resource'],
                                   output=case / 'result.json', ledger=case / 'costs.jsonl', profile_binding=profile_ref)
        path = output / 'protocols' / ('planned-run-' + label + '.json')
        write_exclusive(path, protocol)
        protocols[label] = {'path': str(path), 'sha256': file_sha256(path)}
    family = study_family(c)
    if (total_wall > c['resource']['evaluation']['campaign_wall_seconds']
            or family.disk() + future_copies > c['resource']['evaluation']['family_disk_bytes']):
        raise ValueError('full eight-point evaluation is infeasible under the frozen campaign caps')
    freeze = {'schema_version': 1, 'study_contract_sha256': contract_digest(c),
              'sources': implementation_sources(c), 'order': list(POINT_LABELS), 'protocols': protocols,
              'budgets': budgets, 'planned_outer_wall_seconds': total_wall,
              'current_family_bytes': family.disk(), 'future_copy_reserve_bytes': future_copies,
              'profile_index': {'path': str(output / 'profiles.json'), 'sha256': file_sha256(output / 'profiles.json')},
              'preparation_acceptance': {'path': str(output / 'preparation-acceptance.json'),
                                         'sha256': file_sha256(output / 'preparation-acceptance.json')}}
    write_exclusive(output / 'evaluation-freeze.json', freeze)
    return freeze


def run_campaign(contract, *, config_path):
    c = thaw(contract)
    require_serial_gate(c); require_own_controls(c); require_current_qualification(c)
    output = Path(c['out']); freeze_path = output / 'evaluation-freeze.json'
    if not freeze_path.is_file():
        raise ValueError('evaluation requires its immutable accepted profile/resource freeze')
    if (output / 'evaluation-index.json').exists():
        raise FileExistsError('evaluation already completed; no repeat under this identity')
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get('study_contract_sha256') != contract_digest(c) or freeze.get('sources') != implementation_sources(c)
            or freeze.get('order') != list(POINT_LABELS)):
        raise ValueError('evaluation freeze belongs to another source/configuration')
    profiles = read_json_reference(freeze['profile_index'])
    read_json_reference(freeze['preparation_acceptance'])
    inspected = inspect_sources(c); prepared = read_prepared(c, config_path=config_path); records = {}
    for label in freeze['order']:
        planned = read_json_reference(freeze['protocols'][label])
        profile = profiles['profiles'][label]
        require_stage_success(profile['stage'])
        records[label] = native_job(c, prepared, label, inspected['windows'], mode='run',
                                    resource=planned['resource'], config_path=config_path,
                                    profile_binding=profile['output'], expected_protocol_ref=freeze['protocols'][label])
        actual = read_json_reference(records[label]['protocol'])
        if actual != planned:
            raise ValueError('actual evaluation protocol differs from its frozen completed schedule')
        if study_family(c).disk() > c['resource']['evaluation']['family_disk_bytes']:
            raise ValueError('retained evaluation family exceeds its frozen disk cap')
    index = {'schema_version': 1, 'study_contract_sha256': contract_digest(c),
             'evaluation_freeze': {'path': str(freeze_path), 'sha256': file_sha256(freeze_path)}, 'results': records}
    write_exclusive(output / 'evaluation-index.json', index)
    return index


def run_controls(contract, *, config_path):
    """Asset-free owned CPU controls; their guarded attempts remain separate."""
    c = thaw(contract); provenance = ROOT / 'provenance' / c['study']
    receipt = provenance / 'controls.json'
    if receipt.exists():
        return require_own_controls(c)
    sources = implementation_sources(c)
    scratch = Path(tempfile.mkdtemp(prefix='cgoc-rtn-owned-controls-', dir='/private/tmp'))
    refs = [{'path': p, 'sha256': h} for p, h in sources.items()]
    refs.append({'path': str(Path(config_path).resolve()), 'sha256': file_sha256(config_path)})
    stage = execute_stage([sys.executable, '-m', 'pytest', 'tests/test_sharing_frontier.py', '-q',
                           '--basetemp=' + str(scratch / 'pytest')], root=ROOT, output=scratch,
                          stage=provenance / 'controls-stage',
                          caps={'wall_seconds': 180, 'memory_bytes': 2 * 1024**3, 'disk_bytes': 2 * 1024**3},
                          frozen_files=refs)
    require_stage_success(stage)
    result = {'schema_version': 1, 'contract_sha256': contract_digest(c), 'sources': sources,
              'stage': stage, 'scope': 'owned CPU implementation controls only; no pretrained/native qualification'}
    write_exclusive(receipt, result)
    return result


def require_preparation_guard(contract, config_path):
    """The internal worker must be a direct child of its recorded supervisor."""
    import os
    c = thaw(contract); stage = Path(c['out']) / 'stages/preparation'
    protocol_path = stage / 'protocol.json'; start_path = stage / 'stage-start.json'
    if not protocol_path.is_file() or not start_path.is_file():
        raise ValueError('internal preparation requires its verified finite supervisor')
    protocol = json.loads(protocol_path.read_text()); start = json.loads(start_path.read_text())
    caps = c['resource']['preparation']
    expected_command = [str(Path(__file__).resolve()), '--config', str(Path(config_path).resolve()), '--internal-prepare']
    if (start.get('pid') != os.getppid() or start.get('protocol_sha256') != file_sha256(protocol_path)
            or protocol.get('command', [])[1:] != expected_command
            or protocol.get('root') != str(ROOT) or protocol.get('output') != c['out']
            or protocol.get('provenance') != str(stage)
            or protocol.get('supervisor_sha256') != 'a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba'
            or protocol.get('caps') != {'wall_seconds': caps['wall_seconds'] + 30,
                                      'memory_bytes': caps['memory_bytes'], 'disk_bytes': caps['disk_bytes']}):
        raise ValueError('internal preparation is not under its exact source-bound resource supervisor')


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Fixed complete-artifact RTN fidelity/byte study')
    parser.add_argument('--config', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mode', choices=('validate', 'controls', 'profile', 'run', 'analyze', 'queue'))
    group.add_argument('--internal-prepare', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        c = load_contract(args.config)
        if args.internal_prepare:
            require_preparation_guard(c, args.config)
            prepare_campaign(c)
            summary = {'phase': 'preparation', 'status': 'complete', 'quality_disclosed': False}
        elif args.mode == 'validate':
            inspected = inspect_sources(c)
            summary = {'mode': 'validate', 'study': c['study'], 'contract_sha256': contract_digest(c),
                       'components': len(inspected['windows']),
                       'valid_targets': inspected['run_work']['valid_target_presentations'],
                       'points': list(POINT_LABELS), 'native_work': False,
                       'execution_requires': 'current owned controls and accepted serial device gate'}
        elif args.mode == 'controls':
            run_controls(c, config_path=args.config)
            summary = {'mode': 'controls', 'status': 'complete', 'native_work': False}
        elif args.mode == 'profile':
            profile_campaign(c, config_path=args.config)
            summary = {'mode': 'profile', 'status': 'complete', 'evaluation_resources_frozen': True,
                       'quality_disclosed': False}
        elif args.mode == 'analyze':
            analyze_campaign(c, config_path=args.config)
            summary = {'mode': 'analyze', 'status': 'complete', 'scope': 'exploratory RTN fidelity/bytes'}
        elif args.mode == 'queue':
            queue_campaign(c, config_path=args.config)
            summary = {'mode': 'queue', 'status': 'complete'}
        else:
            run_campaign(c, config_path=args.config)
            summary = {'mode': 'run', 'status': 'complete', 'points': len(POINT_LABELS)}
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


def summarize_frontier(prepared, protocols, results, windows, scientific):
    """Fixed eight-point exploratory analysis; whole components move together."""
    import math
    import numpy as np
    if any(set(packet) != set(POINT_LABELS) for packet in (prepared['artifacts'], protocols, results)):
        raise ValueError('all eight prespecified points, including adverse outcomes, are required')
    metric_work_plan(windows, mode='run')
    points = {}; targets = np.asarray([sum(row['target_mask']) for row in windows], dtype=np.int64)
    teacher_rows = results['teacher_noop']['rows']
    for label in POINT_LABELS:
        value = results[label]; artifact = prepared['artifacts'][label]
        check_native_result(protocols[label], value, rows=windows)
        if (type(value.get('complete_bytes')) is not int or value['complete_bytes'] <= 0
                or value['complete_bytes'] != artifact['complete_bytes']
                or artifact.get('teacher_needed_at_inference') is not False):
            raise ValueError('actual complete independent artifact census differs')
        for row, teacher_row in zip(value['rows'], teacher_rows, strict=True):
            if abs(row['teacher_nll_sum'] - teacher_row['teacher_nll_sum']) > 2e-5 * row['valid_targets']:
                raise ValueError('teacher targets/numerics differ between artifacts')
        sums = {key: math.fsum(row[key] for row in value['rows'])
                for key in ('kl_sum', 'nll_sum', 'teacher_nll_sum')}
        sums.update(valid_targets=int(targets.sum()), disagreements=sum(row['disagreements'] for row in value['rows']))
        n = sums['valid_targets']; component_kl = [row['kl_sum'] / row['valid_targets'] for row in value['rows']]
        worst = sorted(value['rows'], key=lambda row: row['kl_sum'] / row['valid_targets'], reverse=True)[:10]
        points[label] = {'complete_bytes': value['complete_bytes'], 'within_named_ceiling': value['complete_bytes'] <= 66064401,
                         'raw_sums': sums, 'kl': sums['kl_sum'] / n, 'nll': sums['nll_sum'] / n,
                         'delta_nll': (sums['nll_sum'] - sums['teacher_nll_sum']) / n,
                         'perplexity': math.exp(sums['nll_sum'] / n) if sums['nll_sum'] / n < 709 else None,
                         'disagreement_rate': sums['disagreements'] / n,
                         'macro_component_kl': float(np.mean(component_kl)),
                         'component_kl_quantiles': dict(zip(('q50', 'q90', 'q95', 'q99'), np.quantile(component_kl, [.5, .9, .95, .99]).tolist())),
                         'worst_components': [{'component_id': r['component_id'], 'valid_targets': r['valid_targets'],
                                               'kl': r['kl_sum'] / r['valid_targets']} for r in worst]}
    for label, point in points.items():
        point['dominated_by'] = [other for other, candidate in points.items() if other != label
                                and candidate['complete_bytes'] <= point['complete_bytes'] and candidate['kl'] <= point['kl']
                                and (candidate['complete_bytes'] < point['complete_bytes'] or candidate['kl'] < point['kl'])]
    old = scientific.get('populations', {}).get('tuning', [])
    if scientific.get('step') != 256 or len(old) != len(windows):
        raise ValueError('complete scientific source observation required')
    errors = []
    for previous, window, current in zip(old, windows, results['shared_float']['rows'], strict=True):
        if (any(previous.get(k) != window[k] for k in ('document_id', 'component_id', 'window_sha256'))
                or previous.get('target_positions') != [i for i, valid in enumerate(window['target_mask']) if valid]
                or previous['full'].get('status') != 'ok' or previous['full'].get('valid_targets') != current['valid_targets']):
            raise ValueError('scientific/native source windows, masks or status differ')
        error = abs(previous['full']['kl'] - current['kl_sum']) / current['valid_targets']; errors.append(error)
        if error > 2e-5 or abs(previous['full']['nll'] - current['nll_sum']) / current['valid_targets'] > 2e-5:
            raise ValueError('floating source scientific/native parity exceeds frozen resolution')
    difference = np.asarray([a['kl_sum'] - b['kl_sum'] for a, b in
                             zip(results['shared_rtn4']['rows'], results['dense_rtn4']['rows'], strict=True)])
    draw = np.random.default_rng(17).integers(0, len(windows), size=(10000, len(windows)))
    contrasts = difference[draw].sum(axis=1) / targets[draw].sum(axis=1)
    return {'schema_version': 1, 'study_contract_sha256': prepared['study_contract_sha256'], 'named_ceiling_bytes': 66064401,
            'components': len(windows), 'valid_targets': int(targets.sum()), 'points': points,
            'primary': {'label': 'shared_rtn4_minus_dense_rtn4', 'direction': 'negative favors shared',
                        'kl_difference': float(difference.sum() / targets.sum()),
                        'paired_component_percentile_95': np.quantile(contrasts, [.025, .975]).tolist(),
                        'better_components': int(np.sum(difference < 0)), 'worse_components': int(np.sum(difference > 0)),
                        'tied_components': int(np.sum(difference == 0)), 'bootstrap_draws': 10000, 'bootstrap_seed': 17,
                        'uncertainty': 'paired source-component uncertainty conditional on these fixed fitted artifacts; no training-seed replication'},
            'float_source_parity': {'max_abs_component_kl_difference': max(errors), 'resolution': 2e-5},
            'scope': {'population': 'already exposed development/selection; no protected confirmation',
                      'capability_acceptability_established': False, 'learned_dense_quantization_qualified': False,
                      'production_runtime_established': False,
                      'frontier': 'descriptive RTN-only byte/fidelity points; point dominance is not an uncertainty-qualified practical win',
                      'stored_private_corrections': 'packed shared factors stored BF16 and computed FP32; floating source storage counted as exported'}}


def analyze_campaign(contract, *, config_path):
    """Only complete guarded runs can enter the prespecified component analysis."""
    c = thaw(contract); require_serial_gate(c); require_own_controls(c); require_current_qualification(c)
    output = Path(c['out']); index_path = output / 'evaluation-index.json'
    if not index_path.is_file():
        raise ValueError('analysis requires all accepted evaluations')
    if (output / 'analysis.json').exists():
        raise FileExistsError('analysis already completed; no silent replacement')
    index = json.loads(index_path.read_text())
    if index.get('study_contract_sha256') != contract_digest(c) or set(index.get('results', {})) != set(POINT_LABELS):
        raise ValueError('complete same-study evaluation index required')
    freeze = read_json_reference(index['evaluation_freeze'])
    if freeze.get('sources') != implementation_sources(c) or freeze.get('study_contract_sha256') != contract_digest(c):
        raise ValueError('analysis source differs from frozen evaluation')
    prepared = read_prepared(c, config_path=config_path); inspected = inspect_sources(c); protocols = {}; results = {}
    profiles = read_json_reference(freeze['profile_index']); costs = {}
    for mode, records in (('profile', profiles['profiles']), ('run', index['results'])):
        if set(records) != set(POINT_LABELS):
            raise ValueError('all fixed profiles and evaluations required')
        for label in POINT_LABELS:
            item = records[label]
            protocol, value = accept_native_job(c, prepared, item, label, inspected['windows'], mode=mode,
                config_path=config_path, expected_protocol_ref=freeze['protocols'][label] if mode == 'run' else None)
            ledger = Path(protocol['ledger'])
            costs[mode + '-' + label] = {'stage': item['stage'],
                'worker_wall_seconds': value['wall_seconds'], 'memory': value['memory'],
                'counts': value['counts'], 'ledger': {'path': str(ledger), 'sha256': file_sha256(ledger)},
                'snapshots_retained': [value['artifact_snapshot'], value['teacher_snapshot']]}
            if mode == 'run':
                protocols[label] = protocol; results[label] = value
    scientific = read_json_reference(c['reference']['scientific_observation'])
    result = summarize_frontier(prepared, protocols, results, inspected['windows'], scientific)
    result.update(evaluation_index={'path': str(index_path), 'sha256': file_sha256(index_path)},
                  preparation_acceptance=freeze['preparation_acceptance'], execution_costs=costs,
                  source_preparation=c['reference']['preparation'], source_run_id=c['reference']['source_run_id'],
                  retained_family_bytes=study_family(c).disk(),
                  cost_policy='all preparations, offline proofs, profiles, worker copies and guards retained; ancestor training/preparation linked, not assigned zero cost; overlapping stage/worker wall and memory counters never added')
    if result['retained_family_bytes'] > c['resource']['evaluation']['family_disk_bytes']:
        raise ValueError('complete retained family exceeds its cap')
    write_exclusive(output / 'analysis.json', result)
    lines = ['# Fixed RTN sharing frontier', '',
             'Exploratory fidelity/bytes on 128 already exposed components (65,408 targets). No capability tolerance or practical equivalence established. Learned dense quantization is outside this RTN-only comparison.', '',
             '| Artifact | Complete bytes | KL nats/target | NLL nats/target | Within 66,064,401 bytes |',
             '|---|---:|---:|---:|---|']
    for label in POINT_LABELS:
        p = result['points'][label]
        lines.append(f"| {label} | {p['complete_bytes']} | {p['kl']:.10f} | {p['nll']:.10f} | {p['within_named_ceiling']} |")
    p = result['primary']
    lines += ['', f"Prespecified primary shared RTN4 minus dense RTN4: {p['kl_difference']:.10f} nats/target; paired component 95% interval {p['paired_component_percentile_95']}. Negative favors sharing. This interval conditions on one fitted seed/source; it is not algorithmic replication.", '',
              'All raw sums, component tails, adverse outcomes, exact artifact and source identities, guard receipts, costs and lineage are retained in analysis.json and its linked files. Packed private corrections are BF16 stored and FP32 computed. No speedup, capability preservation or counterexample benefit follows from this study.']
    with (output / 'REPORT.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    return result


def require_current_qualification(contract):
    c = thaw(contract); path = ROOT / 'provenance' / c['study'] / 'qualification.json'
    if not path.is_file():
        raise ValueError('new driver requires current full CPU/public/historical qualification')
    value = json.loads(path.read_text())
    if (value.get('status') != 'complete' or value.get('contract_sha256') != contract_digest(c)
            or value.get('source_files') != implementation_sources(c)):
        raise ValueError('full qualification binds another source/configuration')
    require_stage_success(value['default_cpu'])
    protocol = read_json_reference({'path': value['default_cpu']['protocol_path'],
                                    'sha256': value['default_cpu']['protocol_sha256']})
    command = protocol['command']
    if (len(command) != 6 or command[:4] != [sys.executable, '-m', 'pytest', '-q']
            or not command[4].startswith('--basetemp=') or command[5] != '--disable-warnings'
            or protocol['root'] != str(ROOT)):
        raise ValueError('default CPU suite must run completely without test selection')
    for ref in value['verification_receipts']:
        record = read_json_reference(ref)
        if record.get('status') != 'passed':
            raise ValueError('public/historical integrity checks failed')
    return value


def wait_for_serial(contract, refs, *, deadline):
    """Finite wait for the exact accepted upstream endpoint, never a retry."""
    import time
    import math
    c = thaw(contract); gate = Path(c['serial_gate']['directory'])
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('finite serial wait deadline required')
    while True:
        if time.time() >= deadline:
            raise ValueError('finite upstream wait expired; no native study started')
        for ref in refs:
            if file_sha256(ref['path']) != ref['sha256']:
                raise ValueError('queued source or configuration changed')
        registry = read_json_reference({'path': str(gate / 'registry.json'),
                                        'sha256': c['serial_gate']['registry_sha256']})
        campaign = Path(registry['campaign']) if 'campaign' in registry else None
        if ((gate / 'watch-failed.json').exists()
                or campaign is not None and (campaign / 'controller-failed.json').exists()):
            raise ValueError('upstream science/analysis failed; no automatic continuation')
        if (gate / 'watch-result.json').exists():
            return require_serial_gate(c)
        time.sleep(min(10, max(.01, deadline - time.time())))


def freeze_queue(contract, *, config_path):
    c = thaw(contract); require_own_controls(c); require_current_qualification(c)
    inspected = inspect_sources(c)
    if Path(c['out']).exists():
        raise FileExistsError('serial campaign must be never started; no retries')
    refs = [{'path': p, 'sha256': h} for p, h in implementation_sources(c).items()]
    refs += [dict(c['qualified_source']), dict(c['worker']),
             {'path': str(Path(config_path).resolve()), 'sha256': file_sha256(config_path)},
             {'path': c['inputs']['path'], 'sha256': c['inputs']['sha256']},
             {'path': str(Path(c['serial_gate']['directory']) / 'registry.json'),
              'sha256': c['serial_gate']['registry_sha256']}]
    refs += [dict(v) for v in c['reference'].values() if isinstance(v, dict)]
    for name in ('controls.json', 'qualification.json', 'prospective-design.json'):
        path = ROOT / 'provenance' / c['study'] / name
        refs.append({'path': str(path), 'sha256': file_sha256(path)})
    for ref in refs:
        if file_sha256(ref['path']) != ref['sha256']:
            raise ValueError('cannot queue altered source/reference')
    freeze = {'schema_version': 1, 'study_contract_sha256': contract_digest(c), 'order': list(POINT_LABELS),
              'sources': implementation_sources(c), 'config': str(Path(config_path).resolve()), 'frozen_files': refs,
              'deadline_unix': c['resource']['evaluation']['deadline_unix'], 'run_work': inspected['run_work'],
              'scope': 'after exact accepted F endpoint; complete fixed RTN study; no automatic retry'}
    path = ROOT / 'provenance' / c['study'] / 'queue-freeze.json'
    write_exclusive(path, freeze)
    return {'path': str(path), 'sha256': file_sha256(path)}


def queue_campaign(contract, *, config_path):
    import os
    import time
    c = thaw(contract); freeze_ref = freeze_queue(c, config_path=config_path)
    freeze = read_json_reference(freeze_ref); provenance = ROOT / 'provenance' / c['study']
    start = time.monotonic()
    write_exclusive(provenance / 'queue-start.json', {'pid': os.getpid(), 'freeze': freeze_ref,
                     'status': 'waiting exact accepted upstream endpoint; no real native work yet'})
    try:
        refs = freeze['frozen_files'] + [freeze_ref]
        refs += wait_for_serial(c, refs, deadline=freeze['deadline_unix'])
        phases = {}
        for mode, wall in (('profile', c['resource']['preparation']['wall_seconds'] + 30 + 8 * (c['resource']['profile']['wall_seconds'] + 30) + 300),
                           ('run', c['resource']['evaluation']['campaign_wall_seconds'] + 300), ('analyze', 180)):
            stage = execute_stage([sys.executable, str(Path(__file__).resolve()), '--config', str(Path(config_path).resolve()),
                                   '--mode', mode], root=ROOT, output=Path(c['out']),
                                  stage=provenance / ('queue-' + mode),
                                  caps={'wall_seconds': wall, 'memory_bytes': c['resource']['evaluation']['memory_bytes'],
                                        'disk_bytes': c['resource']['evaluation']['family_disk_bytes']}, frozen_files=refs)
            require_stage_success(stage); phases[mode] = stage
        result_ref = {'path': str(Path(c['out']) / 'analysis.json'), 'sha256': file_sha256(Path(c['out']) / 'analysis.json')}
        result = {'status': 'complete', 'freeze': freeze_ref, 'result': result_ref, 'stages': phases,
                  'wall_seconds_including_wait': time.monotonic() - start, 'no_automatic_retry': True}
        write_exclusive(provenance / 'queue-result.json', result)
        return result
    except BaseException as exc:
        write_exclusive(provenance / 'queue-failed.json', {'status': 'failed', 'freeze': freeze_ref,
                         'error': repr(exc), 'wall_seconds_including_wait': time.monotonic() - start,
                         'no_automatic_retry': True})
        raise


from contextlib import contextmanager


@contextmanager
def retain_packing_failures(destination):
    """Keep failed serializer copies without changing frozen RTN arithmetic.

    Single synchronous packing call only. Ordinary runtime and unrelated cleanup
    still run; only newly created immediate packing stage roots are quarantined.
    """
    import os
    target = Path(destination).resolve(); parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    preexisting = {p.name for p in parent.iterdir()}
    original = shutil.rmtree; retained = []; error = None
    def cleanup(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and kwargs.get('dir_fd') is None:
            p = Path(os.fsdecode(path))
            if (p.name.startswith('.quantized-') and p.name not in preexisting
                    and p.parent.resolve() == parent and not p.is_symlink() and p.is_dir()):
                quarantine = parent / (target.name + '.failed-' + uuid4().hex)
                p.rename(quarantine); retained.append((str(p), quarantine))
                return None
        return original(path, *args, **kwargs)
    shutil.rmtree = cleanup
    try:
        yield
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        shutil.rmtree = original
        for original_path, quarantine in retained:
            write_exclusive(quarantine / 'packing-failure.json', {
                'status': 'failed', 'error': error or 'unpublished packing stage reached cleanup',
                'original_stage': original_path, 'intended_artifact': str(target),
                'scope': 'retained failed copy, never accepted as a complete artifact; bytes count in study family'})


def check_stage_binding(stage, *, command, root, output, caps, required_refs, execution_env=None):
    """A valid receipt must prove this exact consumer's declared operation."""
    require_stage_success(stage)
    protocol = read_json_reference({'path': stage['protocol_path'], 'sha256': stage['protocol_sha256']})
    if (protocol.get('command') != command or protocol.get('root') != str(Path(root).resolve())
            or protocol.get('output') != str(Path(output).resolve()) or protocol.get('caps') != caps
            or protocol.get('provenance') != str(Path(stage['protocol_path']).parent)
            or protocol.get('supervisor_sha256') != 'a5266e26bf3d735d667b990efa4400b128771ca612d07a1e82466a7cf7294cba'):
        raise ValueError('successful guard does not bind the admitted command/output/resource operation')
    bound = {ref['path']: ref['sha256'] for ref in protocol['frozen_files']}
    if any(bound.get(ref['path']) != ref['sha256'] for ref in required_refs):
        raise ValueError('guard did not execute with required exact protocol/config/source bindings')
    if execution_env is not None and protocol.get('execution_env') != {
            'HF_HUB_OFFLINE': None, 'PYTORCH_ENABLE_MPS_FALLBACK': None, **execution_env}:
        raise ValueError('guard execution environment differs from native isolation contract')


def accept_native_job(contract, prepared, item, label, rows, *, mode, config_path, expected_protocol_ref=None):
    c = thaw(contract); output = Path(c['out']); case = output / 'workers' / (mode + '-' + label)
    protocol_path = output / 'protocols' / (mode + '-' + label + '.json')
    if item['protocol']['path'] != str(protocol_path) or item['output']['path'] != str(case / 'result.json'):
        raise ValueError('native acceptance names another worker protocol/output')
    protocol = read_json_reference(item['protocol'])
    expected = native_protocol(c, prepared, label, rows, mode=mode, resource=protocol['resource'],
                 output=case / 'result.json', ledger=case / 'costs.jsonl', profile_binding=protocol['profile_binding'])
    if protocol != expected:
        raise ValueError('native protocol differs from this fixed artifact/input operation')
    if mode == 'run' and (expected_protocol_ref is None or read_json_reference(expected_protocol_ref) != protocol):
        raise ValueError('native result did not execute the frozen prospective evaluation')
    refs = execution_references(c, config_path) + [dict(item['protocol']),
            {'path': str(output / 'preparation.json'), 'sha256': file_sha256(output / 'preparation.json')}]
    refs += [{'path': str(Path(protocol[k]['path']) / 'manifest.json'), 'sha256': protocol[k]['manifest_sha256']}
             for k in ('teacher', 'artifact')]
    refs += [ref for ref in (protocol['profile_binding'], expected_protocol_ref) if ref is not None]
    resource = protocol['resource']
    check_stage_binding(item['stage'], command=[sys.executable, c['worker']['path'], '--protocol', str(protocol_path)],
        root=Path('/private/tmp'), output=case,
        caps={'wall_seconds': resource['wall_seconds'] + 30, 'memory_bytes': resource['memory_bytes'],
              'disk_bytes': resource['output_disk_bytes']}, required_refs=refs, execution_env={'PYTHONPATH': ''})
    value = read_json_reference(item['output'])
    check_native_result(protocol, value, rows=rows if mode == 'run' else None)
    return protocol, value


# CLI_ENTRY
if __name__ == '__main__':
    raise SystemExit(main())
