"""Owned end-to-end selector sensitivity; never a pretrained selection result."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from poc.contracts import Batch, Boundary, CostRecord
from poc.data.access import open_records
from poc.data.manifest import DataManifest, DocumentRecord
from poc.diagnostics import fixed_scores
from poc.probes import (PathologyObservation, ProbePolicy, ProbePool, apply_probe,
                        filter_pathologies, fit_pathology_thresholds, generate_donors,
                        local_discrepancy, mine_probes)
from poc.records import canonical_json


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


class _Counter:
    def __init__(self, out, contract_sha):
        self.out, self.contract_sha, self.sequence = out, contract_sha, 0
        self.totals = {}
        self.started = time.monotonic()

    def check(self, cost):
        if time.monotonic()-self.started > 60:
            raise RuntimeError('owned control wall budget exhausted')
        for field, limit in (('search_proposals', 100), ('teacher_forward_calls', 512)):
            if self.totals.get(field, 0)+getattr(cost, field)>limit:
                raise RuntimeError('owned control query budget exhausted')

    def record(self, cost):
        self.check(cost)
        self.sequence += 1
        with (self.out/'costs.jsonl').open('ab') as stream:
            stream.write(canonical_json({'study_contract_sha256': self.contract_sha, 'sequence': self.sequence, 'cost': cost.to_dict()}))
        for field, value in cost.to_dict().items():
            if type(value) in (int, float):
                self.totals[field] = self.totals.get(field, 0)+value


class _CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text],
                'offset_mapping': [(i, i+1) for i in range(len(text))]}


class _OwnedTeacher(nn.Module):
    """One frozen identity FFN followed by a nonlinear suffix and 128-way head.

    Coordinate1 is invisible to the direct head but visible through the suffix.
    This deliberate blind spot is owned fixture information, not a learned fact.
    """
    digest = _hash({'fixture': 'owned_selector_teacher_v1', 'vocabulary': 128})

    def ff_residual(self, boundary, operator=None):
        return boundary.u if operator is None else boundary.u+operator(boundary.u)

    def suffix_hidden(self, z, boundary):
        return torch.stack((z[..., 0]+z[..., 1].square(), z[..., 1]), -1)

    def readout(self, hidden):
        v = hidden[..., :1]
        return torch.cat((v, -v, torch.zeros((*v.shape[:-1], 126), dtype=v.dtype)), -1)


def _boundary(tokens, doc, component, counter, teacher):
    counter.record(CostRecord(teacher_forward_calls=1, processed_inputs=1, processed_tokens=len(tokens)))
    ids = torch.tensor([tokens], dtype=torch.int64)
    batch = Batch(ids, torch.ones_like(ids, dtype=torch.bool),
                  torch.ones((1, len(tokens)-1), dtype=torch.bool),
                  torch.arange(len(tokens))[None], (doc,), (component,))
    u = torch.zeros((1, len(tokens), 4), dtype=torch.float64)
    u[..., 0] = 1.
    u[..., 1] = (ids == ord('C')).double()
    return Boundary(u, batch, 0, teacher.digest)


def _pathology(teacher, boundary, counter):
    counter.record(CostRecord(teacher_forward_calls=3))  # FFN, suffix, readout
    mask = boundary.batch.target_mask
    h = teacher.suffix_hidden(teacher.ff_residual(boundary), boundary)
    logp = teacher.readout(h[:, :-1][mask]).log_softmax(-1)
    entropy = -(logp.exp()*logp).sum(-1).mean().item()
    norm = boundary.u[:, :-1][mask].norm(dim=-1).mean().item()
    return norm, entropy


def _metric(teacher, operator, boundary, counter):
    counter.record(CostRecord(valid_target_presentations=int(boundary.batch.target_mask.sum())))
    return local_discrepancy(teacher, operator, boundary, 4, recorder=counter)


def _manifest(records, role, out):
    return DataManifest(tuple(records), {r.document_id: role for r in records},
                        {r.document_id: role+'-'+r.document_id for r in records},
                        'owned-selector-v1', exposure_path=str(out/'exposures.jsonl'))


def _episode(out, *, hidden_failure, contract_sha):
    out.mkdir()
    teacher = _OwnedTeacher()
    counter = _Counter(out, contract_sha)
    tokenizer = _CharacterTokenizer()
    fit = _manifest([DocumentRecord('fit-a', 'owned', 'v1', 'control', 0, text='Alice 3 dogs.'),
                     DocumentRecord('fit-c', 'owned', 'v1', 'control', 1, text='Clara 3 dogs.')], 'fit', out)
    observations = []
    for doc in open_records(fit, purpose='calibration'):
        b = _boundary(tuple(ord(c) for c in doc.text), doc.document_id,
                      fit.membership[doc.document_id], counter, teacher)
        norm, entropy = _pathology(teacher, b, counter)
        observations.append(PathologyObservation(doc.document_id, fit.membership[doc.document_id],
                                                'fit', norm, entropy, 0))
    thresholds = fit_pathology_thresholds(observations, lower_quantile=0., upper_quantile=1.)
    scoring = _manifest([DocumentRecord('score', 'owned', 'v1', 'control', 2, text='Alice 1 cats.')], 'probe_score', out)
    scoring_docs = tuple(open_records(scoring, purpose='ranking'))
    damage_manifest = _manifest([DocumentRecord('damage', 'owned', 'v1', 'control', 3, text='Clara 4 books.')], 'dev', out)
    damage_docs = tuple(open_records(damage_manifest, purpose='selection'))
    gain = 1. if hidden_failure else 0.

    def hidden_operator(x):
        return torch.stack((gain*x[..., 1].square(), torch.zeros_like(x[..., 0]),
                            torch.zeros_like(x[..., 0]), torch.zeros_like(x[..., 0])), -1)

    def uniform_operator(x):
        return torch.stack((torch.full_like(x[..., 0], .01), torch.zeros_like(x[..., 0]),
                            torch.zeros_like(x[..., 0]), torch.zeros_like(x[..., 0])), -1)

    natural, active, random, damage, raw_damage, raw_natural = {}, {}, {}, {}, {}, {}
    selected_edits, all_rows, all_scores, rejections = {}, {}, {}, []
    incompatible_rejected = False
    for label, operator in (('hidden_failure', hidden_operator), ('small_uniform_error', uniform_operator)):
        candidate_digest = _hash({'fixture': 'owned_operator_v1', 'label': label, 'gain': gain, 'uniform': .01})
        policy = ProbePolicy(selection_mode='active', seed=0, per_document_cap=1,
            generated_edits_per_document=3, random_component_fraction=0., depth_regions=((0,),),
            depth_region_balance=(1.,), selection_count=1, max_pool_records=8,
            max_cache_entries=8, max_cache_bytes=1024**2, thresholds=thresholds,
            roles=scoring.roles, components=scoring.membership, allowed_roles=('probe_score',), use='rank',
            teacher_digest=teacher.digest, candidate_digest=candidate_digest,
            tokenizer_digest=_hash({'tokenizer': 'owned_ascii_v1'}), dtype='float64', backend='torch_cpu',
            boundary_version='post_attention_v1', execution_profile='owned_selector_v1', context_tokens=512,
            recorder=counter)
        rows = tuple(generate_donors(scoring_docs, tokenizer, policy))
        scores, stats, raw_probe_metrics = {}, {}, {}
        natural_parts = []
        for doc in scoring_docs:
            b = _boundary(tuple(ord(c) for c in doc.text), doc.document_id,
                          scoring.membership[doc.document_id], counter, teacher)
            natural_parts.append(_metric(teacher, operator, b, counter))
        raw_natural[label] = [asdict(r) for r in natural_parts]
        natural[label] = sum(r.kl for r in natural_parts)/sum(r.valid_targets for r in natural_parts)
        for row in rows:
            if row.status != 'accepted':
                rejections.append(row.rejection_reason)
                continue
            b = _boundary(row.recipient_tokens, row.recipient_id, row.recipient_component, counter, teacher)
            d = _boundary(row.donor_tokens, row.donor_id, row.recipient_component, counter, teacher)
            patched = apply_probe(b, d, row)
            stats[row.probe_id] = _pathology(teacher, patched, counter)
            metric = _metric(teacher, operator, patched, counter)
            raw_probe_metrics[row.probe_id] = asdict(metric)
            scores[row.probe_id] = metric.kl/metric.valid_targets
            if not incompatible_rejected:
                wrong = replace(d, batch=replace(d.batch, input_ids=b.batch.input_ids))
                try:
                    apply_probe(b, wrong, row)
                except ValueError:
                    incompatible_rejected = True
                else:
                    raise AssertionError('incompatible donor provenance accepted')
        filtered = filter_pathologies(ProbePool(rows), stats, thresholds)
        mined = mine_probes(filtered, scores, policy)
        if len(mined.selected_ids) != 1:
            raise AssertionError('owned positive control did not retain one valid informative probe')
        random_pool = mine_probes(filtered, {}, replace(policy, selection_mode='seeded_random'))
        selected_id = mined.selected_ids[0]
        selected_edits[label] = next(r.edit_kind for r in rows if r.probe_id == selected_id)
        active[label] = scores[selected_id]
        random[label] = scores[random_pool.selected_ids[0]]
        all_rows[label] = {'records': [asdict(r) for r in rows], 'exclusions': dict(filtered.exclusions),
                           'active_ids': list(mined.selected_ids), 'random_ids': list(random_pool.selected_ids),
                           'scores': scores, 'raw_metrics': raw_probe_metrics, 'study_contract_sha256': contract_sha}
        all_scores[label] = fixed_scores(natural=natural[label], student=0.,
                                         random_donor=random[label], active_donor=active[label])
        parts = []
        for doc in damage_docs:
            b = _boundary(tuple(ord(c) for c in doc.text), doc.document_id,
                          damage_manifest.membership[doc.document_id], counter, teacher)
            parts.append(_metric(teacher, operator, b, counter))
        damage[label] = sum(r.kl for r in parts)/sum(r.valid_targets for r in parts)
        raw_damage[label] = [asdict(p) for p in parts]
    labels = sorted(natural)
    choose = lambda method: min(labels, key=lambda label: (all_scores[label][method], label))
    records = next(iter(all_rows.values()))['records']
    return {'study_contract_sha256': contract_sha, 'raw_natural': raw_natural, 'natural_choice': choose('natural_kl'), 'active_choice': choose('active_1'),
            'random_choice': choose('random_donor'), 'damage': damage, 'raw_damage': raw_damage,
            'natural_scores': natural, 'active_probe_scores': active, 'fixed_scores': all_scores,
            'selected_edit_kind': selected_edits, 'probes': all_rows,
            'accepted_proposals': sum(r['status'] == 'accepted' for r in records),
            'rejected_proposals': sum(r['status'] == 'rejected' for r in records),
            'rejection_reasons': sorted(set(rejections)), 'incompatible_boundary_rejected': incompatible_rejected,
            'score_components': sorted(set(scoring.membership.values())),
            'damage_components': sorted(set(damage_manifest.membership.values())),
            'thresholds_sha256': thresholds.digest, 'threshold_observations': [asdict(r) for r in observations],
            'manifest_object_sha256': {'fit': fit.digest(), 'score': scoring.digest(), 'damage': damage_manifest.digest()}, 'cost': counter.totals,
            'elapsed_seconds': time.monotonic()-counter.started}


def run_owned_controls(out):
    """Run positive/null cases with independent roles and the real probe pipeline."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    contract = {'schema_version': 1, 'study': 'owned-selector-pipeline-v1',
        'scope': 'owned sensitivity controls; no pretrained selection result',
        'teacher_identity': _OwnedTeacher.digest, 'candidate_definitions': {'hidden_failure_gain': [1., 0.], 'uniform_error': .01},
        'methods': 'existing fixed_scores: natural_kl, active_1, random_donor; student score zero in this single-operator fixture',
        'primary_control': 'known hidden failure reverses natural decision under active probe; independent owned damage prefers uniform-error operator',
        'null_control': 'remove hidden failure; natural and active both prefer exact operator',
        'generation': 'existing generate_donors; name, number and incompatible plural edit; seed0; context512',
        'validity': 'existing apply_probe, fit-only pathology thresholds and mine_probes; reject incompatible boundary identity',
        'resources': {'cpu_threads': 1, 'wall_seconds_per_case': 60, 'teacher_forward_calls_per_case': 512, 'search_proposals_per_case': 100},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_sha256': {n: hashlib.sha256((Path(__file__).resolve().parents[1]/n).read_bytes()).hexdigest()
            for n in ('poc/probes.py', 'poc/diagnostics.py', 'poc/losses.py', 'poc/contracts.py', 'poc/data/access.py', 'poc/data/manifest.py')}}
    contract_sha = _hash(contract)
    with (out/'contract.json').open('xb') as stream:
        stream.write(canonical_json(contract))
    result = {'schema_version': 1, 'study_contract_sha256': contract_sha, 'scope': 'owned sensitivity controls; no pretrained selection result',
              'ranking_reversal': _episode(out/'positive', hidden_failure=True, contract_sha=contract_sha),
              'no_added_information': _episode(out/'null', hidden_failure=False, contract_sha=contract_sha),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (out/'result.json').open('xb') as stream:
        stream.write(canonical_json(result))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = run_owned_controls(args.out)
    print(json.dumps({case: {k: value[case][k] for k in ('natural_choice', 'active_choice', 'damage')}
                      for case in ('ranking_reversal', 'no_added_information')}, sort_keys=True))
