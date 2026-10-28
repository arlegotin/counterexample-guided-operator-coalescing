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


def classify_outcome(gates: Sequence[GateRecord], analysis: AnalysisResult | None) -> Outcome:
    """A terminal diagnostic limits downstream claims even if a summary looks positive."""
    if any(g.decision in {'stop', 'inconclusive', 'invalid', 'amend'} for g in gates) or analysis is None:
        return Outcome.INCONCLUSIVE
    a = analysis
    if (not a.sensitivity_adequate or a.unresolved_sv_comparison or a.budget_limited_curve
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


def load_evidence(root: Path) -> EvidenceIndex:
    path = root / 'provenance/report-evidence-2026-10-28.json'
    value = json.loads(path.read_text())
    if 'metrics_record' in value:
        metric_record = value['records'][value['metrics_record']]
        metric_path = _relative(root, metric_record['path'])
        if sha256_file(metric_path) != metric_record['sha256']:
            raise ValueError('raw metric source hash mismatch')
        value['metrics'] = json.loads(metric_path.read_text())['metrics']
    return EvidenceIndex(root=root, records={k: EvidenceRecord(**v) for k, v in value['records'].items()},
                         claims=tuple(Claim(**v) for v in value['claims']), stages=value['stages'],
                         gates=tuple(GateRecord(**v) for v in value['gates']),
                         analysis=AnalysisResult(**value['analysis']), sections=tuple(value['sections']),
                         metrics=tuple(value['metrics']), failures=tuple(value['failures']),
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
    manifest = {'schema_version': 1, 'scope': 'Local report handoff; no deployable trained shared model, external publication or rights grant',
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
