"""Finite, auditable whole-template local fitting plans; no corpus readers."""
from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import wraps
import hashlib
import math

from poc.records import canonical_json
from poc.train import TrainingSources, iter_update_targets


def digest(value):
    return hashlib.sha256(canonical_json({'value':value})).hexdigest()


def window_metadata(batch):
    """Content hashes and positions only; original tensors stay private/in memory."""
    rows = []
    for row, (document, component) in enumerate(zip(batch.document_ids, batch.cluster_ids, strict=True)):
        arrays = {name:getattr(batch,name)[row].detach().cpu().tolist()
                  for name in ('input_ids','attention_mask','positions')}
        rows.append(dict(document_sha256=digest(document), component_sha256=digest(component),
            window_sha256=digest(arrays), target_positions=[int(batch.positions[row,i])+1
                for i, value in enumerate(batch.target_mask[row].tolist()) if value]))
    return rows


@dataclass(frozen=True)
class TargetPlan:
    sources: TrainingSources
    metadata: dict
    digest: str


def build_target_plan(items, depths, *, updates, weights, affected_depths):
    """Repeat/interleave complete ProbeTrainingSources, preserving full builders.

    The v1 policy requires equal eligible target counts dividing 8192. Incompatible
    metadata is rejected before predictions. Components rotate within source/depth;
    depth/source blocks follow exact rational mixture proportions. No epoch tails
    are used, and no ProbeRecord is sliced or fabricated by this planner.
    """
    from poc.train import ProbeTrainingSource, TrainingConfig
    TrainingConfig(max_updates=updates, phase='local', affected_depths=affected_depths,
                   mixture_weights=weights)
    if (not items or any(not isinstance(p,ProbeTrainingSource) for p in items)
            or len({p.identity for p in items}) != len(items)):
        raise ValueError('nonempty unique source templates required')
    counts = {int(p.template.target_mask.sum().item()) for p in items}
    if len(counts)!=1 or next(iter(counts))<1:
        raise ValueError('v1 plan requires equal positive eligible template counts')
    count = next(iter(counts))
    if 8192 % count:raise ValueError('template target count must divide 8192')
    fractions = {s:Fraction(str(w)) for s,w in sorted(weights.items())}
    denominator = math.lcm(*(w.denominator for w in fractions.values()))
    ratios = {s:int(w*denominator) for s,w in fractions.items()}
    divisor = math.gcd(*ratios.values())
    ratios = {s:n//divisor for s,n in ratios.items()}
    block = [(s,d) for d in affected_depths for s,n in ratios.items() for _ in range(n)]
    if (8192//count) % len(block):
        raise ValueError('source/depth balance cannot fill a complete update')
    groups = defaultdict(lambda:defaultdict(list))
    metadata = {p.identity:window_metadata(p.template) for p in items}
    for p in items:
        d = depths.get(p.identity)
        if p.source not in weights or d not in affected_depths:
            raise ValueError('unexpected source/depth coverage')
        components = tuple(sorted({r['component_sha256'] for r in metadata[p.identity]}))
        if len(components)!=1:raise ValueError('one component per template required for balance')
        groups[p.source,d][components[0]].append(p)
    if set(groups)!=set(block):raise ValueError('required source/depth coverage absent')
    queues = {}
    for key, components in groups.items():
        # Equal component rotation even with unequal windows/component.
        queues[key] = [sorted(components[c],key=lambda p:p.identity) for c in sorted(components)]
    cursors = defaultdict(int)
    selected, presentations = [], []
    visits = defaultdict(int)
    bound={}
    for original in items:
        expected_depth=depths[original.identity]
        @wraps(original.build)
        def build(student,teacher,recorder,original=original,expected_depth=expected_depth):
            boundary,record=original.build(student,teacher,recorder)
            if boundary.depth!=expected_depth or record.depth!=expected_depth:
                raise ValueError('probe builder differs from planned declared depth')
            return boundary,record
        bound[original.identity]=replace(original,build=build)
    for index in range(updates*8192//count):
        key = block[index % len(block)]
        queue = queues[key]
        cursor = cursors[key]; cursors[key]+=1
        p = queue[cursor % len(queue)][(cursor//len(queue)) % len(queue[cursor % len(queue)])]
        visits[p.identity]+=1;selected.append(bound[p.identity])
        presentations.append(dict(index=index, update=index*count//8192+1,
            template_sha256=digest(p.identity), source=p.source, depth=key[1],
            occurrence=visits[p.identity], targets=count, windows=metadata[p.identity]))
    sources = TrainingSources(probes=tuple(selected),manifest_hashes={'template_inventory':digest(metadata)})
    actual = tuple(iter_update_targets(tuple(p.template for p in sources.probes),8192))
    if len(actual)!=updates or any(u.valid_targets!=8192 for u in actual):
        raise ValueError('incomplete finite target plan')
    presented={k:v for k,v in metadata.items() if visits[k]}
    unique = {(r['document_sha256'],r['window_sha256'],pos) for rows in presented.values()
              for r in rows for pos in r['target_positions']}
    result = dict(schema_version=1, policy='whole_template_balanced_v1', updates=updates,
        logical_targets=updates*8192, presentations=presentations,
        template_inventory={digest(k):v for k,v in metadata.items()},
        unique_components=len({r['component_sha256'] for rows in presented.values() for r in rows}),
        unique_windows=len({r['window_sha256'] for rows in presented.values() for r in rows}),
        unique_target_positions=len(unique), unconsumed_tail_targets=0,
        unused_template_sha256=[digest(p.identity) for p in items if not visits[p.identity]],
        source_digest=sources.digest())
    return TargetPlan(sources,result,digest(result))
