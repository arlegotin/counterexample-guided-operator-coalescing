"""Evidence-only terminal reporting. Never imports a model or opens corpus readers."""
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re

from poc.records import canonical_json


class Outcome(str, Enum):
    CONFIRMED = 'confirmed_scoped_improvement'
    MECHANISM = 'useful_mechanism_without_deployment_advantage'
    AGAINST = 'evidence_against_tested_local_hypothesis'
    INCONCLUSIVE = 'diagnosed_inconclusive'


@dataclass(frozen=True)
class GateRecord:
    gate: str
    decision: str
    reason: str
    evidence_refs: tuple[str, ...] = ()
    category: str = 'unknown'
    record_id: str | None = None
    supersedes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisResult:
    deployment_improvement: bool = False
    attribution_improvement: bool = False
    guardrails_passed: bool | None = None
    independent_replication: bool = False
    mechanism_benefit: bool = False
    evidence_against: bool = False
    sensitivity_adequate: bool = False
    unresolved_sv_comparison: bool = False
    budget_limited_curve: bool = False
    insufficient_sensitivity: bool = False
    validity_confirmed: bool | None = None


def _effective_gates(gates: Sequence[GateRecord]) -> tuple[GateRecord, ...]:
    """Resolve only explicit, evidenced, ordered replacements of the same gate.

    The complete sequence remains provenance. Mere later success, reason prose,
    or a different prerequisite gate never implicitly resolves an earlier stop.
    """
    identified = {}
    superseded = set()
    categories = {'unknown', 'assessment', 'scientific_negative', 'control',
                  'integrity', 'resource', 'sensitivity', 'proxy', 'budget'}
    for index, gate in enumerate(gates):
        if gate.decision not in {'pass', 'stop', 'invalid', 'amend', 'inconclusive'} or gate.category not in categories:
            raise ValueError('gate decision/category is unknown')
        if gate.record_id is not None:
            if not isinstance(gate.record_id, str) or not gate.record_id or gate.record_id in identified:
                raise ValueError('gate record IDs must be nonempty and unique')
        if gate.supersedes and (not gate.record_id or not gate.evidence_refs):
            raise ValueError('gate supersession requires a record ID and resolution evidence')
        for identity in gate.supersedes:
            prior = identified.get(identity)
            if prior is None or prior in superseded or gates[prior].gate != gate.gate:
                raise ValueError('gate supersession must name an effective earlier record of the same gate')
            superseded.add(prior)
        if gate.record_id is not None:
            identified[gate.record_id] = index
    return tuple(g for i, g in enumerate(gates) if i not in superseded)


def classify_outcome(gates: Sequence[GateRecord], analysis: AnalysisResult | None) -> Outcome:
    """Classify the effective assessment while preserving conservative stop guards."""
    effective = _effective_gates(gates)
    if analysis is None:
        return Outcome.INCONCLUSIVE
    a = analysis
    for gate in effective:
        if gate.decision in {'inconclusive', 'invalid', 'amend'}:
            return Outcome.INCONCLUSIVE
        if gate.decision == 'stop' and not (
                gate.category == 'scientific_negative' and gate.evidence_refs
                and a.evidence_against and a.validity_confirmed is True):
            return Outcome.INCONCLUSIVE
    if (a.validity_confirmed is False or not a.sensitivity_adequate or a.unresolved_sv_comparison or a.budget_limited_curve
            or a.insufficient_sensitivity):
        return Outcome.INCONCLUSIVE
    if a.guardrails_passed is False or a.evidence_against:
        return Outcome.AGAINST
    if (a.deployment_improvement and a.guardrails_passed is True and a.independent_replication):
        return Outcome.CONFIRMED
    if a.mechanism_benefit:
        return Outcome.MECHANISM
    return Outcome.INCONCLUSIVE


@dataclass(frozen=True)
class EvidenceRecord:
    path: str
    sha256: str


@dataclass(frozen=True)
class Claim:
    id: str
    value: object
    record: str
    pointer: str
    scope: str


@dataclass(frozen=True)
class EvidenceIndex:
    root: Path
    records: dict[str, EvidenceRecord]
    claims: tuple[Claim, ...]
    stages: dict[str, str]
    gates: tuple[GateRecord, ...]
    analysis: AnalysisResult | None
    sections: tuple = ()
    metrics: tuple[dict, ...] = ()
    failures: tuple[str, ...] = ()
    narrative: str = ''
    release_files: tuple[str, ...] = ()
    overview: tuple = ()


@dataclass(frozen=True)
class ReportBundle:
    markdown: str
    summary: dict
    claims: tuple[Claim, ...]
    metrics: tuple[dict, ...]


@dataclass(frozen=True)
class AuditResult:
    ok: bool
    errors: tuple[str, ...]
    claims_checked: int
    records_checked: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_pointer(value, pointer: str):
    if not pointer:
        return value
    if not pointer.startswith('/'):
        raise ValueError('evidence pointer must be an RFC6901 JSON pointer')
    for part in pointer[1:].split('/'):
        key = part.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _relative(root: Path, path: str) -> Path:
    full = (root / path).resolve()
    if not full.is_relative_to(root.resolve()):
        raise ValueError('evidence path escapes repository')
    return full


def _schema(valid: bool, description: str) -> None:
    if not valid:
        raise ValueError('evidence index: ' + description)


def _object(value, required: set[str], optional: set[str], location: str) -> None:
    _schema(isinstance(value, dict), location + ' must be an object')
    _schema(required <= value.keys(), location + ' missing required fields: ' + ', '.join(sorted(required - value.keys())))
    _schema(value.keys() <= required | optional, location + ' has unknown fields')


def _strings(value, location: str) -> None:
    _schema(isinstance(value, list) and all(isinstance(v, str) and v for v in value), location + ' must be a string list')


def _string(value, location: str) -> None:
    _schema(isinstance(value, str) and bool(value), location + ' must be a nonempty string')


def _digest(value, location: str) -> None:
    _schema(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, location + ' must be a SHA256')


def _metric_schema(rows, records: dict) -> None:
    _schema(isinstance(rows, list), 'metrics must be a list')
    identities = set()
    for row in rows:
        _schema(isinstance(row, dict), 'metrics entry must be an object')
        for name in ('metric_id', 'status', 'scope'):
            _string(row.get(name), 'metrics.' + name)
        _schema(row['metric_id'] not in identities, 'duplicate metric ID')
        identities.add(row['metric_id'])
        _schema(row['status'] in {'measured', 'unexecuted', 'unavailable', 'unsupported'}, 'metrics status is unknown')
        for key in ('kl_sum', 'nll_sum'):
            val = row.get(key)
            _schema(val is None or (type(val) in (int, float) and math.isfinite(val)), 'metrics.' + key + ' must be finite or null')
        targets = row.get('valid_targets')
        _schema(targets is None or (type(targets) is int and targets > 0), 'metrics denominator must be positive or null')
        for key in ('seed', 'depth'):
            _schema(row.get(key) is None or type(row[key]) is int, 'metrics.' + key + ' must be an integer or null')
        for key in ('study', 'model_revision', 'candidate_digest', 'component_id', 'document_id',
                    'pair', 'unavailable_reason', 'source_path', 'source_pointer', 'probe_id', 'probe_source'):
            _schema(row.get(key) is None or isinstance(row[key], str), 'metrics.' + key + ' must be a string or null')
        if row.get('record') is not None:
            _string(row['record'], 'metrics.record')
            _schema(row['record'] in records, 'metrics record reference is unknown')
        if row.get('pointer') is not None:
            _schema(isinstance(row['pointer'], str) and row['pointer'].startswith('/'), 'metrics.pointer must be a JSON pointer')
        if row.get('source_sha256') is not None:
            _digest(row['source_sha256'], 'metrics.source_sha256')
        if row['status'] == 'measured':
            _string(row.get('record'), 'metrics.record')
            _schema(row['record'] in records, 'metrics record reference is unknown')
            _string(row.get('pointer'), 'metrics.pointer')
            _schema(row['pointer'].startswith('/'), 'metrics pointer must be a JSON pointer')
            _digest(row.get('source_sha256'), 'metrics.source_sha256')
            _schema(row.get('kl_sum') is not None and targets is not None, 'measured metrics require sum and denominator')
        else:
            _schema(all(row.get(k) is None for k in ('kl_sum', 'nll_sum', 'valid_targets')), 'unavailable metrics require null outcomes')


def _index_schema(value) -> None:
    required = {'records', 'claims', 'stages', 'gates', 'analysis', 'sections', 'metrics',
                'failures', 'narrative', 'release_files'}
    _object(value, required, {'schema_version', 'metrics_record', 'overview'}, 'root')
    if 'schema_version' in value:
        _schema(type(value['schema_version']) is int and value['schema_version'] == 1, 'unsupported schema_version')
    records = value['records']
    _schema(isinstance(records, dict), 'records must be an object')
    for identity, row in records.items():
        _string(identity, 'records ID')
        _object(row, {'path', 'sha256'}, set(), 'records entry')
        _string(row['path'], 'records.path')
        _digest(row['sha256'], 'records.sha256')
    if 'metrics_record' in value:
        _string(value['metrics_record'], 'metrics_record')
        _schema(value['metrics_record'] in records, 'metrics_record reference is unknown')
    _schema(isinstance(value['claims'], list), 'claims must be a list')
    claims = {}
    for row in value['claims']:
        _object(row, {'id', 'value', 'record', 'pointer', 'scope'}, set(), 'claims entry')
        for key in ('id', 'record', 'scope'):
            _string(row[key], 'claims.' + key)
        _schema(isinstance(row['pointer'], str) and (not row['pointer'] or row['pointer'].startswith('/')), 'claims.pointer must be a JSON pointer')
        _schema(row['record'] in records, 'claim record reference is unknown')
        _schema(row['id'] not in claims, 'duplicate claim ID')
        # JSON numbers accepted by Python's decoder must still be finite.
        try:
            json.dumps(row['value'], allow_nan=False)
        except ValueError as exc:
            raise ValueError('evidence index: nonfinite claim value') from exc
        claims[row['id']] = row
    _schema(isinstance(value['stages'], dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value['stages'].items()), 'stages must map strings to statuses')
    _schema(isinstance(value['gates'], list), 'gates must be a list')
    for row in value['gates']:
        _object(row, {'gate', 'decision', 'reason'}, {'evidence_refs', 'category', 'record_id', 'supersedes'}, 'gates entry')
        for key in ('gate', 'decision', 'reason'):
            _string(row[key], 'gates.' + key)
        for key in ('evidence_refs', 'supersedes'):
            _strings(row.get(key, []), 'gates.' + key)
        _schema(all(ref in records for ref in row.get('evidence_refs', [])), 'gate evidence reference is unknown')
        if 'category' in row:
            _string(row['category'], 'gates.category')
        _schema(row.get('record_id') is None or isinstance(row['record_id'], str), 'gates.record_id must be a string or null')
    analysis = value['analysis']
    if analysis is not None:
        _object(analysis, set(), set(AnalysisResult.__dataclass_fields__), 'analysis')
        for key, val in analysis.items():
            _schema(type(val) is bool or (val is None and key in {'guardrails_passed', 'validity_confirmed'}), 'analysis.' + key + ' must be boolean' + (' or null' if key in {'guardrails_passed', 'validity_confirmed'} else ''))
    _schema(isinstance(value['sections'], list), 'sections must be a list')
    for section in value['sections']:
        _schema(isinstance(section, list) and len(section) == 2, 'section must contain title and rows')
        _string(section[0], 'section title')
        _schema(isinstance(section[1], list), 'section rows must be a list')
        for row in section[1]:
            _schema(isinstance(row, list) and len(row) == 2, 'section row must contain label and claim ID')
            _strings(row, 'section row')
            _schema(row[1] in claims, 'section claim reference is unknown')
    _schema(isinstance(value.get('overview', []), list), 'overview must be a list')
    for row in value.get('overview', []):
        _strings(row, 'overview row')
        _schema(len(row) == 5 and all(ref in claims for ref in row[1:]), 'overview must reference four claims')
        for ref in row[1:]:
            point = claims[ref]['value']
            _schema(isinstance(point, dict), 'overview claim must be a point object')
            pair = point.get('selected_pair')
            regret = point.get('regret')
            _schema(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair), 'overview selected pair must contain two integers')
            _schema(type(regret) in (int, float) and math.isfinite(regret), 'overview regret must be finite')
    _strings(value['failures'], 'failures')
    _strings(value['release_files'], 'release_files')
    _schema(isinstance(value['narrative'], str), 'narrative must be a string')
    _metric_schema(value['metrics'], records)


def load_evidence(root: Path) -> EvidenceIndex:
    """Validate user evidence before linked I/O; null analysis is an early unknown.

    Schema failures are ValueError at this boundary. Unrelated programming errors
    are not converted or broadly caught by the CLI.
    """
    path = root / 'provenance/report-evidence-2026-10-28.json'
    value = json.loads(path.read_text())
    _index_schema(value)
    gates = tuple(GateRecord(**{**row, 'evidence_refs': tuple(row.get('evidence_refs', ())),
                               'supersedes': tuple(row.get('supersedes', ()))}) for row in value['gates'])
    _effective_gates(gates)
    if 'metrics_record' in value:
        metric_record = value['records'][value['metrics_record']]
        metric_path = _relative(root, metric_record['path'])
        if sha256_file(metric_path) != metric_record['sha256']:
            raise ValueError('raw metric source hash mismatch')
        source = json.loads(metric_path.read_text())
        _schema(isinstance(source, dict) and 'metrics' in source, 'linked metrics source must contain metrics')
        _metric_schema(source['metrics'], value['records'])
        value['metrics'] = source['metrics']
    return EvidenceIndex(root=root, records={k: EvidenceRecord(**v) for k, v in value['records'].items()},
                         claims=tuple(Claim(**v) for v in value['claims']), stages=value['stages'], gates=gates,
                         analysis=AnalysisResult(**value['analysis']) if value['analysis'] is not None else None,
                         sections=tuple(value['sections']), metrics=tuple(value['metrics']), failures=tuple(value['failures']),
                         narrative=value['narrative'], release_files=tuple(value['release_files']),
                         overview=tuple(value.get('overview', ())))


def render_report(evidence: EvidenceIndex, *, include_failures: bool) -> ReportBundle:
    outcome = classify_outcome(evidence.gates, evidence.analysis)
    claims = {c.id: c for c in evidence.claims}
    if len(claims) != len(evidence.claims):
        raise ValueError('duplicate claim IDs')
    narrative = evidence.narrative.split('\n\n', 2) if evidence.overview else [evidence.narrative]
    opening = '\n\n'.join(narrative[:2])
    lines = ['# CGOC terminal report', '', f'**Outcome: {outcome.value.replace("_", " ")}.**', '',
             opening, '']
    used = []
    if evidence.overview:
        lines.extend(['## Main findings', '',
                      'Natural damage regret in nats per target, against the best sampled pair. '
                      'The table reports selection within each analysis; it does not compare quality across populations.', '',
                      '| Analysis | Ordinary natural KL | Random donor | Active λ1 | Active λ0.1 |',
                      '|---|---|---|---|---|'])
        for label, *identities in evidence.overview:
            cells = []
            for identity in identities:
                claim = claims[identity]
                pair = tuple(claim.value['selected_pair'])
                cells.append(f'{pair}; regret {claim.value["regret"]:.9g} [claim:{identity}]')
                used.append(claim)
            lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
        if len(narrative) > 2:
            lines.extend(['## Interpretation and scope', '', narrative[2], ''])
    for title, rows in evidence.sections:
        lines.extend([f'## {title}', '', '| Evidence item | Recorded value |', '|---|---|'])
        for label, identity in rows:
            claim = claims[identity]
            value = json.dumps(claim.value, ensure_ascii=False, allow_nan=False)
            lines.append(f'| {label} | {value} [claim:{identity}] |')
            used.append(claim)
        lines.append('')
    lines.extend(['## Stage availability', ''])
    lines.extend(f'- {stage}: {status}' for stage, status in evidence.stages.items())
    if include_failures:
        lines.extend(['', '## Failures and unresolved limits', ''])
        lines.extend(f'- {failure}' for failure in evidence.failures)
    lines.extend(['', '## Claim-to-evidence index', '',
                  '| Claim | Immutable receipt and JSON pointer | Population / units |', '|---|---|---|'])
    used = list({c.id: c for c in used}.values())
    for claim in used:
        record = evidence.records[claim.record]
        lines.append(f'| {claim.id} | [{record.path}]({record.path}) `{claim.pointer}`; SHA256 `{record.sha256}` | {claim.scope} |')
    lines.extend(['', 'Read [reproduction and release boundaries](docs/reproduction.md), '
                  '[decision appendix](docs/analysis/2026-10-28-decision-appendix.md), '
                  '[source-bound evidence index](provenance/report-evidence-2026-10-28.json), '
                  '[raw metric table](results/regressions.parquet), and '
                  '[release manifest](artifacts/release_manifest.json).', ''])
    summary = {'schema_version': 1, 'outcome': outcome.value,
               'gates': [asdict(g) for g in evidence.gates], 'analysis': asdict(evidence.analysis) if evidence.analysis else None,
               'stages': evidence.stages, 'claims': [asdict(c) for c in used],
               'records': {k: asdict(v) for k, v in evidence.records.items()},
               'failures': list(evidence.failures), 'include_failures': include_failures,
               'raw_metrics': 'results/regressions.parquet',
               'metric_scope': 'Recorded fidelity residuals and explicitly unavailable capability outcomes; no confirmed compressed-model capability regression.'}
    return ReportBundle('\n'.join(lines), summary, tuple(used), evidence.metrics)


def audit_claims(report: ReportBundle, evidence: EvidenceIndex) -> AuditResult:
    errors = []
    values = {}
    for identity, record in evidence.records.items():
        try:
            path = _relative(evidence.root, record.path)
            if sha256_file(path) != record.sha256:
                raise ValueError('hash mismatch')
            if path.suffix == '.json':
                values[identity] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f'{identity}: {exc}')
    expected = {c.id: c for c in evidence.claims}
    expected_report = render_report(evidence, include_failures=report.summary.get('include_failures', False))
    if report.markdown != expected_report.markdown:
        errors.append('report text differs from source-bound rendering')
    if report.summary != expected_report.summary:
        errors.append('report summary differs from source-bound rendering')
    markers = re.findall(r'\[claim:([^\]]+)\]', report.markdown)
    ids = [c.id for c in report.claims]
    if sorted(set(markers)) != sorted(ids) or len(set(ids)) != len(ids):
        errors.append('report claim markers are missing, duplicated or unregistered')
    for claim in report.claims:
        try:
            if expected.get(claim.id) != claim or not claim.scope:
                raise ValueError('claim differs from registered evidence')
            if resolve_pointer(values[claim.record], claim.pointer) != claim.value:
                raise ValueError('claimed value does not resolve to receipt')
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f'{claim.id}: {exc}')
    if report.metrics != evidence.metrics:
        errors.append('report metrics differ from evidence')
    for row in report.metrics:
        if row['status'] != 'measured':
            if any(row.get(k) is not None for k in ('kl_sum', 'nll_sum', 'valid_targets')):
                errors.append(f'{row["metric_id"]}: unavailable metric has numeric outcome')
            continue
        try:
            source = resolve_pointer(values[row['record']], row['pointer'])
            for key in ('kl_sum', 'nll_sum', 'valid_targets'):
                if row.get(key) != source.get(key):
                    raise ValueError(f'raw metric {key} differs from source')
            if row.get('source_path'):
                origin = resolve_pointer(values[row['record']], row['pointer'].rsplit('/rows/', 1)[0])
                if (origin['path'] != row['source_path'] or origin['sha256'] != row['source_sha256']
                        or source['source_pointer'] != row['source_pointer']):
                    raise ValueError('metric origin path/hash/pointer mismatch')
            elif row['source_sha256'] != evidence.records[row['record']].sha256:
                raise ValueError('metric receipt hash mismatch')
            if not row.get('scope') or row['valid_targets'] <= 0:
                raise ValueError('metric lacks scope/denominator')
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f'{row["metric_id"]}: {exc}')
    for gate in evidence.gates:
        if not gate.evidence_refs or any(ref not in evidence.records for ref in gate.evidence_refs):
            errors.append(f'{gate.gate}: gate evidence unavailable')
    if report.summary.get('outcome') != classify_outcome(evidence.gates, evidence.analysis).value:
        errors.append('outcome does not match gates')
    if report.summary.get('stages') != evidence.stages:
        errors.append('stage availability differs from evidence')
    return AuditResult(not errors, tuple(errors), len(report.claims), len(evidence.records))


def write_metrics(path: Path, rows: Sequence[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    string_fields = ['metric_id', 'status', 'study', 'scope', 'model_revision', 'candidate_digest',
                     'component_id', 'document_id', 'pair', 'record', 'pointer', 'source_sha256', 'unavailable_reason',
                     'source_path', 'source_pointer', 'probe_id', 'probe_source']
    schema = pa.schema([(name, pa.string()) for name in string_fields] +
                       [('seed', pa.int64()), ('depth', pa.int64()), ('kl_sum', pa.float64()), ('nll_sum', pa.float64()),
                        ('valid_targets', pa.int64())], metadata={b'cgoc_schema': b'regression-residuals-v1',
                        b'aggregation': b'FP64 sums / genuine target denominators; source components are sampling units'})
    for row in rows:
        if row['status'] == 'measured':
            if not isinstance(row.get('valid_targets'), int) or row['valid_targets'] <= 0:
                raise ValueError('measured metric requires a positive genuine denominator')
            if not math.isfinite(row.get('kl_sum', float('nan'))):
                raise ValueError('nonfinite raw KL sum')
        elif any(row.get(k) is not None for k in ('kl_sum', 'nll_sum', 'valid_targets')):
            raise ValueError('unavailable measurement must use null outcomes')
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), path, compression='zstd')


def inspect_existing_bundle(path: Path, expected_manifest_sha256: str) -> dict:
    """Verify an existing inventory only; this does not instantiate or requalify it."""
    path = Path(path)
    manifest_path = path / 'manifest.json'
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError('bundle manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['file_entries']
    if isinstance(entries, dict):
        entries = [{'path': name, **value} for name, value in entries.items()]
    names = set()
    total = manifest_path.stat().st_size
    for entry in entries:
        name = entry['path']
        item = _relative(path, name)
        if name in names or name == 'manifest.json' or (path/name).is_symlink() or not item.is_file():
            raise ValueError('bundle inventory duplicate, symlink or missing file')
        names.add(name)
        size = item.stat().st_size
        if size != entry['bytes'] or sha256_file(item) != entry['sha256']:
            raise ValueError(f'bundle file size/hash mismatch: {name}')
        total += size
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual != names | {'manifest.json'}:
        raise ValueError('bundle has undeclared files')
    return {'status': 'files_bytes_hashes_verified', 'manifest_sha256': expected_manifest_sha256,
            'payload_bytes': total, 'file_count': len(names)+1,
            'scope': 'metadata/bytes only; existing pinned offline proof required separately; no model loaded'}


def report_command(*, include_failures: bool, root: Path | None = None) -> int:
    root = Path(root or Path(__file__).resolve().parents[1])
    evidence = load_evidence(root)
    bundle = render_report(evidence, include_failures=include_failures)
    audit = audit_claims(bundle, evidence)
    if not audit.ok:
        raise ValueError('; '.join(audit.errors))
    (root/'REPORT.md').write_text(bundle.markdown)
    (root/'results').mkdir(exist_ok=True)
    write_metrics(root/'results/regressions.parquet', bundle.metrics)
    (root/'results/summary.json').write_bytes(canonical_json({**bundle.summary, 'audit': asdict(audit)}))
    files = sorted(set(evidence.release_files) | {'REPORT.md', 'results/summary.json', 'results/regressions.parquet',
                                                'provenance/report-evidence-2026-10-28.json'})
    manifest = {'schema_version': 1, 'scope': 'Public report evidence inventory; no deployable trained shared model; licensing scope in LICENSING.md',
                'outcome': bundle.summary['outcome'], 'files': [
                    {'path': name, 'bytes': _relative(root,name).stat().st_size, 'sha256': sha256_file(_relative(root,name)),
                     'distribution': 'owned project report/code or metadata only; third-party rights unchanged'} for name in files],
                'excluded': ['private corpus text', 'token IDs', 'large weights', 'restricted third-party code', 'credentials'],
                'retrieval_instructions': 'docs/reproduction.md',
                'self_hash_policy': 'manifest excludes itself to avoid a digest cycle'}
    (root/'artifacts').mkdir(exist_ok=True)
    (root/'artifacts/release_manifest.json').write_bytes(canonical_json(manifest))
    print(json.dumps({'outcome': bundle.summary['outcome'], 'audit': asdict(audit),
                      'outputs': ['REPORT.md','results/summary.json','results/regressions.parquet','artifacts/release_manifest.json']}, sort_keys=True))
    return 0
