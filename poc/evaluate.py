"""Common capability scoring and token-weighted, raw fidelity metrics.

All task inputs are already role-restricted by the caller. No asset/data loader
or protected permit is embedded in these routines. Failed tasks stay as rows.
"""
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any

from poc.contracts import Batch, ContinuationRequest, DecoderAdapter, MetricSums


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class PromptPolicy:
    tokenizer: Any
    context_tokens: int = 512
    tokenizer_digest: str = 'owned-fixture'
    # Native SmolLM2 adds no BOS. Sentence likelihood conditions on first token;
    # all remaining tokens and exactly one appended real EOS are scored.
    sentence_bos: bool = False
    sentence_eos: bool = True

    def __post_init__(self):
        if type(self.context_tokens) is not int or self.context_tokens < 2:
            raise ValueError('context must permit at least two tokens')
        if self.sentence_eos and self.tokenizer.eos_token_id is None:
            raise ValueError('sentence policy requires a real EOS')
        if self.sentence_bos and self.tokenizer.bos_token_id is None:
            raise ValueError('sentence policy requires a real BOS')

    @property
    def digest(self):
        return _digest(dict(version=1,tokenizer=self.tokenizer_digest,context=self.context_tokens,
            arc='Question: {question}\nAnswer: + leading space + complete answer',
            boundary='whole-concatenation-offsets-and-prefix-equality',arc_eos=False,
            sentence_bos=self.sentence_bos,sentence_eos=self.sentence_eos,
            arc_primary='mean_answer_token_log_likelihood',arc_secondary='sum',
            tie='lowest-original-index; BLiMP half-credit',temperature=1.0,
            reliability_bins=[i/10 for i in range(11)]))

    def request(self, prefix: str, answer: str, item_id: str) -> ContinuationRequest:
        if not prefix or not answer:
            raise ValueError('empty prompt or answer')
        encoded=self.tokenizer(prefix+answer,add_special_tokens=False,return_offsets_mapping=True)
        ids=tuple(encoded['input_ids']); offsets=encoded['offset_mapping']; boundary=len(prefix)
        if len(ids)>self.context_tokens:
            raise ValueError('context overflow; truncation forbidden')
        if any(start < boundary < end for start,end in offsets):
            raise ValueError('unresolved token crossing continuation boundary')
        cut=sum(end<=boundary for start,end in offsets)
        prefix_ids=tuple(self.tokenizer(prefix,add_special_tokens=False)['input_ids'])
        if not 0<cut<len(ids) or ids[:cut]!=prefix_ids:
            raise ValueError('unstable prompt/continuation boundary')
        return ContinuationRequest(ids[:cut],ids[cut:],self.digest,item_id)

    def sentence_request(self, sentence: str, item_id: str) -> ContinuationRequest:
        ids=list(self.tokenizer(sentence,add_special_tokens=False)['input_ids'])
        if not ids:
            raise ValueError('empty sentence')
        if self.sentence_bos: ids.insert(0,self.tokenizer.bos_token_id)
        if self.sentence_eos: ids.append(self.tokenizer.eos_token_id)
        if len(ids)>self.context_tokens:
            raise ValueError('context overflow; truncation forbidden')
        return ContinuationRequest(tuple(ids[:1]),tuple(ids[1:]),self.digest,item_id)


@dataclass(frozen=True)
class ARCItem:
    item_id: str
    question: str
    choices: tuple[str,...]
    label: int
    cluster_id: str


@dataclass(frozen=True)
class BlimpItem:
    item_id: str
    sentence_good: str
    sentence_bad: str
    paradigm: str


@dataclass(frozen=True)
class TaskPrediction:
    item_id: str
    task: str
    cluster_id: str
    scores: tuple[float,...]
    token_counts: tuple[int,...]
    label: int | str
    prediction: int | str | None
    tie: bool
    credit: float
    prompt_hash: str
    secondary_prediction: int | None = None
    probabilities: tuple[float,...] = ()
    log_loss: float | None = None
    brier: float | None = None
    status: str = 'ok'
    reason: str | None = None
    secondary_scores: tuple[float,...] = ()


def _scores(decoder, requests):
    scores=decoder.score_continuations(requests)
    if len(scores)!=len(requests) or any(s.item_id!=r.item_id or s.scored_tokens!=len(r.continuation_ids)
                                         or not math.isfinite(s.log_likelihood)
                                         for s,r in zip(scores,requests,strict=True)):
        raise ValueError('invalid or unmatched continuation scores')
    return scores


def score_arc(decoder: DecoderAdapter, items: Sequence[ARCItem], policy: PromptPolicy) -> list[TaskPrediction]:
    rows=[]
    for item in items:
        prompt=f'Question: {item.question}\nAnswer:'
        ph=_digest([policy.digest,prompt,item.choices])
        try:
            if len(item.choices)<2 or type(item.label) is not int or not 0<=item.label<len(item.choices):
                raise ValueError('invalid ARC choices/label')
            if any(not answer.strip() for answer in item.choices): raise ValueError('empty answer')
            scores=_scores(decoder,[policy.request(prompt,' '+answer,f'{item.item_id}:{i}')
                                   for i,answer in enumerate(item.choices)])
            counts=tuple(s.scored_tokens for s in scores)
            sums=tuple(s.log_likelihood for s in scores)
            means=tuple(s/n for s,n in zip(sums,counts,strict=True))
            maximum=max(means); prediction=means.index(maximum)
            log_z=maximum+math.log(math.fsum(math.exp(s-maximum) for s in means))
            probs=tuple(math.exp(s-log_z) for s in means)
            rows.append(TaskPrediction(item.item_id,'arc',item.cluster_id,means,counts,item.label,prediction,
                means.count(maximum)>1,float(prediction==item.label),ph,sums.index(max(sums)),probs,
                log_z-means[item.label],math.fsum((p-float(i==item.label))**2 for i,p in enumerate(probs)),secondary_scores=sums))
        except ValueError as exc:
            rows.append(TaskPrediction(item.item_id,'arc',item.cluster_id,(),(),item.label,None,False,0,ph,
                                       status='failed',reason=str(exc)))
    return rows


def score_blimp(decoder: DecoderAdapter, items: Sequence[BlimpItem], policy: PromptPolicy) -> list[TaskPrediction]:
    rows=[]
    for item in items:
        ph=_digest([policy.digest,item.sentence_good,item.sentence_bad])
        try:
            scores=_scores(decoder,[policy.sentence_request(s,f'{item.item_id}:{i}')
                for i,s in enumerate((item.sentence_good,item.sentence_bad))])
            values=tuple(s.log_likelihood for s in scores); tie=values[0]==values[1]
            prediction=0 if values[0]>=values[1] else 1
            rows.append(TaskPrediction(item.item_id,'blimp',item.paradigm,values,tuple(s.scored_tokens for s in scores),
                0,prediction,tie,.5 if tie else float(prediction==0),ph))
        except ValueError as exc:
            rows.append(TaskPrediction(item.item_id,'blimp',item.paradigm,(),(),0,None,False,0,ph,
                                       status='failed',reason=str(exc)))
    return rows


def aggregate_tasks(rows: Sequence[TaskPrediction]) -> dict:
    if not rows or len({r.item_id for r in rows})!=len(rows):
        raise ValueError('empty or duplicate task population')
    groups=defaultdict(list)
    for r in rows: groups[r.cluster_id].append(r.credit)
    # Failed rows remain in the denominator and counts; never silently filtered.
    out=dict(items=len(rows),failed=sum(r.status!='ok' for r in rows),
             accuracy=math.fsum(r.credit for r in rows)/len(rows),
             macro_accuracy=math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups),
             groups={k:dict(items=len(v),accuracy=math.fsum(v)/len(v)) for k,v in sorted(groups.items())})
    calibrated=[r for r in rows if r.probabilities]
    if calibrated:
        out.update(calibrated_items=len(calibrated),
            log_loss=math.fsum(r.log_loss for r in calibrated)/len(calibrated),
            brier=math.fsum(r.brier for r in calibrated)/len(calibrated),
            secondary_accuracy=sum(r.secondary_prediction==r.label for r in calibrated)/len(calibrated))
        bins=[]
        for i in range(10):
            members=[r for r in calibrated if min(int(max(r.probabilities)*10),9)==i]
            bins.append(dict(lower=i/10,upper=(i+1)/10,count=len(members),
                accuracy=math.fsum(r.credit for r in members)/len(members) if members else None,
                confidence=math.fsum(max(r.probabilities) for r in members)/len(members) if members else None))
        out['reliability_bins']=bins
    return out


@dataclass(frozen=True)
class DocumentMetric:
    document_id: str
    cluster_id: str
    metrics: MetricSums
    window_id: str | None = None
    source: str | None = None
    split: str = 'dev'
    context_policy: str = 'independent-position-reset'
    context_tokens: int | None = None
    backend: str | None = None
    dtype: str | None = None
    checkpoint_digest: str | None = None


def check_kl(metrics: MetricSums, *, negative_kl_floor: float) -> MetricSums:
    if not math.isfinite(negative_kl_floor) or negative_kl_floor<0:
        raise ValueError('calibrated negative KL floor must be finite and nonnegative')
    if metrics.kl < -negative_kl_floor*metrics.valid_targets:
        raise FloatingPointError('raw negative KL exceeds calibrated per-token floor')
    return metrics


def evaluate_documents(teacher: DecoderAdapter, candidate: DecoderAdapter, batches: Iterable[Batch], *,
                       negative_kl_floor: float, chunk_size: int = 32) -> Iterable[DocumentMetric]:
    from poc.losses import stream_metrics
    identity=candidate.manifest()
    for batch in batches:
        for i,(doc,cluster) in enumerate(zip(batch.document_ids,batch.cluster_ids,strict=True)):
            row=Batch(batch.input_ids[i:i+1],batch.attention_mask[i:i+1],batch.target_mask[i:i+1],
                      batch.positions[i:i+1],(doc,),(cluster,))
            metrics=stream_metrics(teacher,candidate,row,chunk_size)
            yield DocumentMetric(doc,cluster,check_kl(metrics,negative_kl_floor=negative_kl_floor),
                context_tokens=batch.input_ids.shape[1],backend=identity.backend,dtype=identity.dtype,
                checkpoint_digest=identity.checkpoint_digest)


def aggregate_documents(rows: Iterable[DocumentMetric]) -> dict:
    metrics=MetricSums(); count=0
    for row in rows: metrics=metrics+row.metrics; count+=1
    if not metrics.valid_targets: raise ValueError('zero valid targets in evaluation population')
    nll=metrics.nll/metrics.valid_targets
    return dict(windows=count,valid_targets=metrics.valid_targets,kl_sum=metrics.kl,nll_sum=metrics.nll,
        disagreement_count=metrics.disagreements,kl_nats_per_token=metrics.kl/metrics.valid_targets,
        nll_nats_per_token=nll,perplexity=math.exp(nll) if nll<709 else None,
        disagreement_rate=metrics.disagreements/metrics.valid_targets)


@dataclass(frozen=True)
class SyntheticCase:
    item_id: str
    family: str
    prompt: str
    answer: str
    template: str
    operands: tuple[int,...] = ()
    stratum: str = ''


def synthetic_cases(*, seed: int, split: str) -> list[SyntheticCase]:
    if split not in {'dev','confirm'}: raise ValueError('unknown synthetic split')
    rng=random.Random(seed+(0 if split=='dev' else 1000000)); rows=[]; shift=0 if split=='dev' else 50
    template='direct-v1' if split=='dev' else 'instruction-v2'
    for i in range(16):
        a,b=rng.randrange(10,40)+shift,rng.randrange(10,40)+shift
        payload=' '.join(str(rng.randrange(10)) for _ in range(8))
        lookup={f'key{j}':str(rng.randrange(100)) for j in range(4)}; key=f'key{i%4}'
        prompts=dict(copy=(f'Copy exactly: {payload}\nCopy:',payload),
            lookup=('Table: '+', '.join(f'{k}={v}' for k,v in lookup.items())+f'\nValue of {key}:',lookup[key]),
            arithmetic=(f'Compute {a} + {b}.\nAnswer:',str(a+b)),
            composition=(f'Let f(x)=x+3 and g(x)=2*x. Compute g(f({a})).\nAnswer:',str(2*(a+3))))
        for family,(prompt,answer) in prompts.items():
            if split=='confirm':
                if family=='composition':
                    prompt=f'Let f(x)=x+3 and g(x)=2*x. Compute f(g({a})).\nResult:'
                    answer=str(2*a+3)
                prompt='Follow the instruction.\n'+prompt.replace('Answer:','Result:')
            rows.append(SyntheticCase(f'{split}:{family}:{i}',family,prompt,answer,template,
                (a,b) if family=='arithmetic' else (a,) if family=='composition' else (),
                'carry' if family=='arithmetic' and a%10+b%10>=10 else 'no_carry' if family=='arithmetic' else ''))
    return rows


def copy_guard_cases(*, seed: int) -> list[SyntheticCase]:
    """Full unsliced labels; execution must explicitly guard prompt plus output."""
    rng=random.Random(seed)
    rows=[]
    for length in (128,512,2048,8192):
        answer=' '.join(rng.choice(('red','blue','green','gold')) for _ in range(length))
        rows.append(SyntheticCase(f'dev:copy_guard:{length}','copy_guard',
            f'Copy exactly: {answer}\nCopy:',answer,'copy-length-guard-v1',stratum=str(length)))
    return rows


@dataclass(frozen=True)
class GenerationPolicy:
    prompt: PromptPolicy
    max_new_tokens: int = 64
    seed: int = 17
    mode: str = 'greedy'

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens<1 or self.mode!='greedy':
            raise ValueError('only finite greedy generation is qualified')


def evaluate_synthetic(decoder: DecoderAdapter, cases: Sequence[SyntheticCase], policy: GenerationPolicy) -> list[TaskPrediction]:
    import torch
    rows=[]; tokenizer=policy.prompt.tokenizer
    for case in cases:
        ph=_digest([policy.prompt.digest,policy.max_new_tokens,policy.seed,policy.mode,case.prompt])
        try:
            ids=tuple(tokenizer(case.prompt,add_special_tokens=False)['input_ids'])
            if not ids or len(ids)+policy.max_new_tokens>policy.prompt.context_tokens:
                raise ValueError('generation context guard; truncation forbidden')
            cache=decoder.new_cache(); generated=[]
            with torch.no_grad():
                for step in range(policy.max_new_tokens):
                    current=ids if step==0 else (generated[-1],)
                    tensor=torch.tensor([current],device=decoder.device,dtype=torch.int64)
                    start=0 if step==0 else len(ids)+step-1
                    batch=Batch(tensor,torch.ones_like(tensor,dtype=torch.bool),
                        torch.ones((1,len(current)-1),device=decoder.device,dtype=torch.bool),
                        torch.arange(start,start+len(current),device=decoder.device)[None],(case.item_id,),(case.family,))
                    token=int(decoder.readout(decoder.final_hidden(batch,cache=cache)[:,-1:]).argmax(-1).item())
                    generated.append(token)
                    if token==tokenizer.eos_token_id: break
                    text=tokenizer.decode(generated,skip_special_tokens=True)
                    if '\n' in text.lstrip(): break
            prediction=tokenizer.decode(generated,skip_special_tokens=True).strip()
            rows.append(TaskPrediction(case.item_id,case.family,case.family,(),(len(generated),),
                case.answer,prediction,False,float(prediction==case.answer),ph))
        except ValueError as exc:
            rows.append(TaskPrediction(case.item_id,case.family,case.family,(),(),case.answer,None,False,0,ph,
                status='failed',reason=str(exc)))
    return rows
