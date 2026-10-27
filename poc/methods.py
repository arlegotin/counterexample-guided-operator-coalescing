"""Method-family readiness, independently implemented paper feasibility only."""
from dataclasses import dataclass,replace
from pathlib import Path
import json
from collections.abc import Mapping
from poc.contracts import freeze
import hashlib
from types import MappingProxyType


ELIGIBILITY_PATH=Path(__file__).resolve().parents[1]/'protocol/baseline-eligibility.json'


@dataclass(frozen=True)
class MethodDescriptor:
    method_id: str
    variant: str
    architecture: str
    trainable_owners: tuple[str,...]
    accepted_data_roles: tuple[str,...]
    supported_formats: tuple[str,...]
    status: str
    reason: str
    evidence_digest: str
    allowed_comparisons: tuple[str,...]
    cost_budget: Mapping
    implementation_label: str
    stage_readiness: str = 'future_task'

    def __post_init__(self):
        object.__setattr__(self,'cost_budget',freeze(self.cost_budget))


def method_registry():
    descriptions={
      'T':('original','dense original depth','eligible','Reviewed Task7/8 FP32 scientific and native floating identity evidence','qualified_floating'),
      'Q':('rtn_gptq_awq_dwq_dynamic','dense original depth','qualification_pending','Pinned official APIs available; per-artifact execution and fidelity evidence required','official_dense_adapters'),
      'R':('cycle_sequence_whole_block','FFN bank or separately labeled whole block','not_yet_implemented','SEQUENCE format anchor is not fitted R or whole-block comparator','pending'),
      'RC':('matched_shared_cases','FFN bank','not_yet_implemented','Common donor/student-state fitting and development selection pending','pending'),
      'CKA':('whole_ffn_outputs','FFN bank','not_yet_implemented','Matched-input similarity, degeneracy fallback and grouping pending','pending'),
      'BS':('output_shared_input_weighted','factorized projections','not_yet_implemented','Output-shared SVD and activation-weighted basis adapters pending','pending'),
      'SV':('independent_paper_feasibility','shared input basis/private output factors','feasible_adaptation_pending','Paper equations2–4 orientation/parameter calculation; pinned upstream PolyForm Noncommercial1.0.0 and conditional down-input mismatch; no code reused; Task15 owns adaptation','independent_paper_calculation'),
      'P':('ffn_block_adjacent_reconstruction','reduced FFN/depth','not_yet_implemented','Natural-KL selection and reconstruction pending','pending'),
      'H':('text_disagreement_mining','FFN bank','not_yet_implemented','Matched query/time mining and fitting pending','pending'),
      'QCE':('empirical_precision_allocation','dense and fixed bank','not_yet_implemented','Requires immutable measured B* and validated low-bit start before allocation','pending'),
      'M':('active_random_donor','FFN bank','not_yet_implemented','Single-merge diagnostic, fitting, compiler and matched controls pending','pending')}
    values={}
    for key,(variant,architecture,status,reason,label) in descriptions.items():
        trainable={'T':(),'Q':('DWQ affine scales','DWQ affine offsets'),'BS':('shared basis','private factors'),'SV':('shared input basis','private output factors','refinement vectors'),'P':('surviving adjacent block during reconstruction',),'QCE':()}.get(key,('canonical bank','private residual factors'))
        roles=('dev',) if key=='T' else ('fit','probe_fit','probe_score','dev') if key in ('RC','H','QCE','M') else ('fit','dev')
        values[key]=MethodDescriptor(key,variant,architecture,trainable,
            roles,('float32',) if key=='T' else (),status,reason,hashlib.sha256(reason.encode()).hexdigest(),
            ('format','identity') if key=='T' else ('feasibility',), {'stage':'Task9 format feasibility only','final_study':'unresolved'},label)
    if ELIGIBILITY_PATH.exists():
        measured=json.loads(ELIGIBILITY_PATH.read_text())
        if measured.get('schema_version')!=1 or not isinstance(measured.get('methods'),dict): raise ValueError('invalid eligibility evidence schema')
        for key,entry in measured['methods'].items():
            if key not in values: raise ValueError('unknown measured method family')
            overrides={name:entry[name] for name in ('status','reason','evidence_digest','stage_readiness','cost_budget') if name in entry}
            for name in ('allowed_comparisons','supported_formats'):
                if name in entry: overrides[name]=tuple(entry[name])
            values[key]=replace(values[key],**overrides)
    return MappingProxyType(values)


def sv_feasibility(shapes,rank):
    """Paper W:[input,output]; project W:[output,input], shared right basis.

    This tiny independent algebra check does not execute restricted source or
    establish reproduction, grouping, refinement, accuracy, or license rights.
    """
    import numpy as np
    if not shapes or len({s[1] for s in shapes})!=1 or rank<1 or rank>min(sum(s[0] for s in shapes),shapes[0][1]): raise ValueError('incompatible common input basis/rank')
    inputs=shapes[0][1];outputs=sum(s[0] for s in shapes)
    dense=sum(a*b for a,b in shapes);factor=rank*inputs+rank+rank*outputs
    # Bound feasibility proof independently of full teacher dimensions.
    a=np.arange(24,dtype=np.float64).reshape(6,4)/13
    b=np.cos(np.arange(24,dtype=np.float64)).reshape(6,4)
    joined=np.concatenate((a,b),axis=0);u,s,vh=np.linalg.svd(joined,full_matrices=False)
    error=float(np.max(np.abs((u*s)@vh-joined)))
    return {'orientation':'project [out,in]: stack vertically; shared right input basis, singular values, private left coefficients',
            'dense_parameters':dense,'factor_parameters':factor,'compression_feasible':factor<dense,
            'full_rank_reconstruction_max_abs_error':error,'scope':'independent tiny algebra only; Task15 owns full SV',
            'paper':'https://aclanthology.org/2026.acl-long.2213.pdf','terms':'PolyForm Noncommercial1.0.0 pinned upstream; no source execution/reuse'}
