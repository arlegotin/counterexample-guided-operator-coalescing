# Research and local proof-of-concept plan: counterexample-guided operator coalescing

**Research date:** 14 October 2026  
**Target:** one MacBook Pro with an M3 Max; installed memory, storage and macOS must be discovered.  
**Status:** researched proposal and executable experimental specification, not a demonstrated compression result.  
**Working method name:** Counterexample-Guided Operator Coalescing (**CGOC**). The name does not imply novelty.

## 1. Executive assessment

### Recommendation

Investigate whether **candidate-specific counterexamples identify which nonlinear computations can safely share a stored implementation better than existing sharing, distillation and quantization methods**. Start by compiling the feed-forward networks (FFNs) of a pretrained, natural-language decoder into a smaller bank of reusable FFNs, with a static depth-to-bank map and small depth-specific low-rank residuals. Keep attention, depth, tokenizer and context semantics unchanged initially.

This is a deliberate, restricted implementation of the proposed *Executable Causal Grammar*. It tests reusable computation without initially requiring a dynamic controller, a new tokenizer, a knowledge database, symbolic program synthesis or a new pretraining run. It is **not** a demonstration of a fundamentally minimal intelligence, an exact state quotient, or a general 10–100× compressor.

The strongest defensible research question is narrower than the original brainstorming:

> At a fixed complete model-payload budget, does actively finding counterexamples to proposed operator sharing improve held-out teacher fidelity and preserve independently measured capabilities more efficiently than strong, equally informed alternatives?

There is substantial prior art. Recursive Transformers and layer-wise LoRA already provide parameter reuse; cross-layer basis sharing and SharVeT already exploit cross-layer similarity; causal distillation already matches activation interventions; counterexample-guided quantization already exists in CEG4N, and neuron merge/split abstraction-refinement predates this proposal. Even a bank of complete FFNs conditioned on depth appears in the October 2026 reViT preprint. None of those broad concepts is an invention claim here. [R01] [R02] [R03] [R04] [R05] [R06] [R07] [R29] [R30]

**Proposed differentiation:** an explicit, byte-budgeted compilation procedure that (a) proposes sharing of whole nonlinear FFNs, (b) tests the proposed replacement through the actual downstream computation on natural, intervened and student-reached states, (c) uses counterexamples to refit or revise the sharing partition, and (d) revalidates the jointly compressed model. This is presently a plausible application-level refinement, not established methodological novelty. Contemporary depth-merging work further narrows any first-use claim. [R33] Whether the exact combination yields a substantive new contribution remains unresolved. Its scientific value must come from a measured advantage, or a useful negative finding about reuse, not from assembling familiar components.

### What to do first

Build a correct teacher clone, a native quantization frontier and a **single-merge diagnostic** before implementing a full compiler. Compare several ways of predicting the damage caused by the same candidate merges: weight similarity, activation similarity, ordinary downstream KL, random intervention KL, and actively selected counterexamples. Use a pretrained natural-language model, not only synthetic algorithms. Stop architectural expansion when the diagnostic is uninformative; distinguish insufficient sensitivity from genuine evidence that the extra machinery adds no value.

The primary asset is `HuggingFaceTB/SmolLM2-135M`, with `HuggingFaceTB/SmolLM2-360M` reserved for evidence-triggered scale confirmation. Their public configurations provide a small, tied-embedding Llama-style decoder suitable for a local experiment. The model cards declare Apache-2.0 licensing. Public configurations establish architecture, not measured M3 compatibility. [R08] [R09]

### Important headroom calculation

For the 135M configuration, FFNs contain 79,626,240 of 134,515,008 parameters. Keeping all other components unchanged means that replacing all 30 FFNs by even one shared FFN, with **no adapters**, reduces equal-precision weight storage by at most **2.338×**. With 15 operators and rank-8 residuals, the raw parameter reduction is only **1.398×**. These are calculations from the published configuration, not measured compression-quality results. Therefore:

- A positive initial result would establish a useful **component-level and possibly deployment-level frontier improvement**, not the original 10–100× ambition.
- A BF16 shared model can easily lose to a dense 4-bit model. Final comparisons must include learned and mixed-precision quantization, and apply compatible quantization to sharing methods too.
- Attention and embedding compression would be a separate, later contribution. A 10× ratio against raw 16-bit weights is not a 10× improvement over a strong quantized baseline.

### Deliverable contract for the coding agent

Implement the work packages below, preserve every experiment and decision, and finish with one of four outcomes: a confirmed scoped improvement; a useful mechanism without a deployment advantage; evidence against the tested local hypothesis; or an explicitly diagnosed inconclusive result. Do not manufacture positive results, silently alter the protected test set, or describe statistical non-significance as unchanged quality.

---

## 2. Problem, scope and hypotheses

### 2.1 Beneficiaries and bottleneck

The intended beneficiaries are developers deploying local language models under storage or resident-memory constraints. The present workflow is to select a pretrained decoder, quantize it, optionally prune or distill it, and serve it with a hardware-appropriate runtime. The proposed change is to replace redundant stored nonlinear transformations before, or alongside, numerical compression.

Stored independent matrices are an observable cost. The assumption that they contain reusable computation **beyond what existing methods already exploit** is unproven. FFN reuse does not remove the number of executed depths. It does not reduce the attention KV cache, and low-rank residuals add operations. Any latency benefit depends on actual weight reuse, caching, kernel launches and hardware utilization, not just parameter count.

### 2.2 Objects and accounting

Let the frozen teacher be an autoregressive distribution

\[
p_T(x_t\mid x_{<t}),
\]

with tokenizer \(\tau\), context limit \(C\), and checkpoint digest \(h_T\). A deployed candidate \(M\) must operate without teacher access, a network connection or uncounted auxiliary models.

Define:

| Quantity | Definition and unit |
|---|---|
| \(B_{payload}\) | Actual bytes of canonical weights, quantization parameters, tokenizer, bank map, configuration, custom inference code and any candidate-specific assets needed to execute the model. No symlinks to omitted weights. |
| \(B_{archive}\) | Bytes after the same deterministic lossless packaging procedure is applied to each complete payload. Report codec settings and unpack cost. |
| \(B_{install}\) | Payload plus the pinned runtime environment. Report separately because common dependency sizes can dominate small weight savings. OS and pre-existing system libraries are explicitly outside this boundary for every method. |
| \(B_{resident}\) | Unique resident model tensor bytes after materialization, including cached dequantized or reconstructed weights. |
| \(M_{peak}\) | Peak process and accelerator memory during load, prefill and decode, separately; includes activations, caches and temporary buffers. |
| \(D_{KL}\) | Teacher-to-student next-token KL, in nats per predicted token, on the stated evaluation population. |
| \(L_{NLL}\) | Ground-truth negative log likelihood, nats per token; report perplexity as \(\exp(L_{NLL})\), not as the sole aggregation. |
| \(A_j\) | Ground-truth performance for capability family \(j\), with its own metric and uncertainty. |
| \(C_{build}\) | Project-specific preparation, teacher queries, search, optimization and quantization expenditure: seconds, token presentations, forward/backward passes, and estimated operations. |
| \(t_{first},v_{decode}\) | Time to first token and decode tokens/second under explicitly defined prompt, batch and cache conditions. |

Pretrained assets are shared starting resources, not free evidence that a smaller architecture could have been trained equally cheaply. Record their provenance separately from project-specific costs.

The broad optimization is

\[
\min_M B_{payload}(M)\quad\text{subject to}\quad
A_j(M)\ge A_j(T)-\epsilon_j,\quad
M_{peak}(M)\le M_{available},\quad t(M)\le t_{allowed}.
\]

No application-specific tolerances or latency budget were supplied. **Do not invent them.** The initial study measures a frontier and makes a narrower fidelity claim. A claim of “no important capability loss” requires tolerances fixed before confirmation by an explicit deployment requirement; absent that requirement, report confidence bounds without certifying equivalence. With strict \(\epsilon_j=0\), a non-inferiority conclusion requires the relevant one-sided bounds, not a failure to reject equality.

### 2.3 Scope of a valid local conclusion

The core population is English next-token modeling, syntactic discrimination and limited factual/science-question answering supported by the selected teacher. Synthetic copying and composition probes test narrower mechanisms. Coding, multilingual competence, broad instruction following, alignment and frontier reasoning are **not established** by this base-model PoC. Context configuration is not evidence of effective long-context competence.

Measure the teacher on every declared task. Preserve all results, including near-chance performance. A task on which the teacher lacks measurable competence cannot support a claim that advanced competence was preserved. Do not select favorable teacher-success items using candidate outcomes. A prespecified teacher-success subset can be a secondary retention diagnostic, but the full task remains reported.

### 2.4 Facts, assumptions, hypotheses and extensions

**Established from inspected sources:** parameter sharing, shared low-rank bases, intervention distillation, dynamic recursive depth, low-bit native training, learned quantization and externalized knowledge all exist. Their reported results depend on particular models, training resources and evaluations. [R01] [R02] [R05] [R10] [R11] [R12] [R14] [R15] [R16] [R18] [R29] [R30]

**Mechanism assumptions:** some FFNs are interchangeable after modest residual adaptation; intervention-based discrepancy predicts natural deployment errors; the optimizer can discover useful reuse with local data and compute; and additional runtime overhead does not erase the storage benefit. None is taken as a fact.

| Hypothesis | Supporting motivation, not proof | Alternative explanation | Observation that weakens it | Discriminating experiment |
|---|---|---|---|---|
| H1: Behavioral probes identify useful sharing better than cheap similarity. | Causal distillation and similarity-aware compression suggest activations matter. | Ordinary natural-state downstream KL already contains all useful information. | Active probes have no reliable held-out ranking advantage after matching teacher-query cost. | Same candidate merges, same fitting, different scoring rules; compare selected-merge regret on untouched documents. |
| H2: Counterexamples improve the **partition**, not merely training data quantity. | Candidate-specific failures could expose unsafe sharing. | Extra hard-example distillation or optimization explains the gain. | A fixed partition with the same mined data, or a text-hard-example baseline, matches CGOC. | Factor grouping choice from training examples; reuse the exact probe pool and compute allowance. |
| H3: Useful reuse survives composition and student-state drift. | Whole-operator sharing preserves more nonlinear structure than independent matrix approximation. | Pairwise substitutions work only in teacher states. | Combined replacements fail despite low single-replacement error, or dictionary size approaches depth. | Evaluate the complete model after each merge; include student-reached states and an independent natural-text confirmation. |
| H4: Reuse improves quality per deployed byte after strong quantization. | Fewer unique operators might tolerate a higher bit width than a same-byte dense model. | BF16 savings disappear against GPTQ/AWQ/DWQ or mixed precision. | The best dense or shared-basis quantized baseline dominates the full frontier. | Native-runtime comparisons at actual payload budgets, with compatible quantization for every method. |
| H5: The effect is not confined to one small teacher. | A reusable computation could recur across depths and scales. | Small-model regularization or model-specific redundancy creates the result. | No transfer to the 360M teacher or a fresh composition family despite adequate sensitivity and adaptation. | Evidence-triggered second-scale experiment; do not infer a universal scaling law from two points. |

**Speculative extensions, excluded from the core:** token-dependent operator routing; attention sharing; latent-state quotienting; learned symbolic programs; repair atlases; knowledge databases; pretraining a new architecture; and claims about 70B or frontier models. Each requires a fresh hypothesis and accounting boundary.

### 2.5 Strongest case for and against

The strongest case is that many stored FFNs implement similar input-output transformations in different numeric coordinates. Measuring behavior through downstream computation could avoid preserving differences that do not matter, while counterexamples identify the rare differences that do. Static sharing may improve storage without requiring extra recursion depth.

The strongest case against is that the apparent redundancy is a property of the observed calibration distribution, not of the full function. Rare facts, unusual syntax or shifted states may require almost every distinction. A sufficiently powerful probe search may simply reconstruct the information that sharing removed. Even when compression is possible, ordinary learned quantization or recursive LoRA may use the same bytes better and be much cheaper to construct.

Approximate interchangeability is not generally transitive. Separate successful replacements are not guaranteed to compose. A finite collection of continuation tests cannot establish behavioral equivalence on all future strings. For an autoregressive process, the relevant state includes the prefix-dependent state of every attention layer, not merely one residual vector.

There is also an information constraint independent of neural architecture: specifying arbitrary labels for \(N\) independent binary facts requires \(N\) bits in the zero-error, no-side-information case. There are \(2^N\) possible assignments; a shorter uniform code cannot uniquely identify them all. A compact program can replace regularities, not arbitrary missing information. A database, code constant, lookup table or pseudorandom seed that carries task information belongs in the accounting.

---

## 3. Research landscape and novelty assessment

### 3.1 Closest prior-art comparison

“Local adaptation” below means implementing the published mechanism on the chosen small teacher with the local data/compute budget. It does not mean reproducing the paper's large-scale result.

| Prior art | Problem and mechanism | Evidence / resource setting / implementation | Exact overlap and consequence for this proposal |
|---|---|---|---|
| Universal Transformers; ALBERT; HyperNetworks [R10] | Recurrence, cross-layer parameter sharing and conditional weight generation. | Foundational architecture papers; their original training settings differ from this post-training experiment. Primary abstracts inspected. | Reusing layers or generating weights is not novel. These motivate simple fixed-sharing and depth-conditioned controls. |
| Relaxed Recursive Transformers, RRT [R01] | Reuse a block across depth with depth-specific low-rank residuals; SVD-based initialization. | Paper methods inspected. Important uptraining experiments use roughly 60B tokens, far beyond the intended local adaptation. | The proposed bank plus per-layer LoRA is close architecture prior art. RRT-style CYCLE and SEQUENCE mappings are mandatory controls. |
| Basis Sharing [R02] | Shared low-rank bases across matrices/layers; activation-aware reconstruction. | Paper method and public repository inspected; published large-model adaptation is not an M3 measurement. | Weight-coordinate sharing and activation-weighted low rank are established. Include a local shared-basis baseline. |
| SharVeT, ACL 2026 [R03] [R04] | Structural and functional similarity, spectral groups, adaptive group ranks, dual-vector refinement and distillation. | Paper, method-page screenshots, README and two implementation files inspected. Public code needs a SmolLM adapter; license is noncommercial. | “Use functional similarity instead of weights” is already known. This is a central contemporary baseline, not an optional footnote. |
| Causal Distillation / DIITO, NAACL 2022 [R05] | Interchange interventions at aligned teacher/student hidden layers and distill intervened predictions. | Full method inspected; BERT-style language tasks, not this autoregressive compiled-bank experiment. Code linked by authors. | Activation interventions as distillation supervision are not novel. Compare CGOC against fixed sharing plus DIITO-style supervision. |
| Dynamic Layer Tying [R11] | Learn which layers reuse earlier parameters. | ICLR 2024 paper inspected; training/architecture choices differ. | Learning a nonuniform depth-sharing assignment is not, by itself, a contribution. |
| Mixture-of-Recursions [R12] | Shared recurrence with adaptive token-level recursive depth. | Paper and public implementation inspected; reported pretraining and optimized execution are not reproduced here. | A learned recursion controller is existing work. Defer dynamic routing to avoid introducing compute and kernel confounds. |
| reViT, October 2026 [R06] | A recurrent vision block with depth-conditioned mixtures of complete FFNs. | Full method inspected; vision training/distillation, not decoder compression. Compact execution versus expanded materialization is explicitly distinguished. | A complete FFN bank or a depth-only controller is not new. The proposed distinction must be the compilation/test/refinement procedure. |
| CEG4N [R29] | Counterexample-guided bit allocation plus formal equivalence checking. | 2022 primary method and 2023/24 journal record inspected; verification on specified input regions of small networks. | Broad counterexample-guided neural compression is already known. Compare against an empirical counterexample-guided precision-allocation control; do not claim its formal guarantees. |
| Neural abstraction/refinement and robust repair [R30] | Merge/split neurons to simplify verification; search for counterexamples and refine or repair a network. | Primary methods inspected; verification abstractions can over-approximate rather than reproduce outputs. | Even neural merge-and-refine is prior art. CGOC tests deployment fidelity of shared FFNs, not a sound verification abstraction. |
| FIPE [R31] | Counterexample/separation-oracle pruning of tree ensembles while preserving classifications. | Primary definitions and algorithm inspected; tree-specific oracle and finite-termination result. | Class-label preservation is not full probability preservation; neither guarantee transfers to an LLM sampler. |
| XMerge [R33] | Reconstruct adjacent blocks into a surviving nonlinear block; causal patching informs direction. | September 2026 primary method and appendix inspected; different models and accelerator. | Direct modern depth-compression competitor and additional causal-merging overlap; local adaptation specified below. |
| CEGAR and active automata learning [R07] | Counterexamples refine abstractions or distinguish candidate states. | Primary bibliographic/abstract material; CEGAR record inspected. Formal oracle assumptions do not hold for an LLM probe sampler. | Counterexample-guided refinement is inherited. Do not claim formal verification or transfer automata guarantees to this experiment. |
| Activated Parameter Locating; Output-Space Projection [R13] | Causal parameter importance or output-space objectives for merging related checkpoints. | Primary method text inspected; checkpoint/task merging differs from tying depths within one decoder. | Causal selection and output-space merging are adjacent prior art. A first-ever “causal merging” claim would be false. |
| GPTQ, AWQ and native MLX learned quantization [R14] [R15] | Low-bit approximations, activation-aware calibration, mixed precision and distillation-based refinement. | Original papers and current official implementation/docs inspected. Local operation paths still need tests. | Strong practical competitors. Shared models must not be compared only with weak round-to-nearest quantization. |
| BitNet b1.58 2B4T [R16] | A natively low-bit trained model. | Technical report inspected; 4T-token training and different pretrained asset. | Useful reference, not a matched post-hoc compression baseline. Nominal bits do not establish total system or quality parity. |
| Lossless weight compression, June 2026 [R17] | Exploit numerical entropy; GPU ANS decoding alongside matrix multiplication. | Full report inspected, including implementation setting; CUDA/CUTLASS, not Apple Metal. | Archive entropy is a distinct axis. Apply common lossless packaging to all models; do not assume an M3 on-the-fly decoder exists. |
| Limited Memory Language Models and follow-on work [R18] | Teach factual lookups during pretraining; store knowledge outside weights. | Primary paper and follow-on paper inspected. Original pipeline uses substantial preparation and external stores. | Moving facts outside weights is known and changes the storage boundary. It is not part of the core PoC. |
| MIPS / algorithm distillation [R19] | Extract compact algorithms from small learned systems. | Primary indexed article summary inspected; later full-text access was blocked. Small algorithmic tasks, not LLM compression. | Supports testing program-like compression on controlled systems, not extrapolating it to a general LLM. |

### 3.2 Reconciling apparently favorable claims

The brainstorming's reasoning-versus-knowledge claim has a real source in *When Reasoning Meets Compression*. Its evidence concerns particular reasoning models, tasks, pruning/distillation and quantization settings; differences in source architecture and training complicate a universal parameter-count interpretation. It motivates separate knowledge-retention guardrails, not an assumption that reasoning is freely compressible. [R20]

The June 2026 entropy paper also exists. Its authors report substantial differences between nominal precision and estimated weight entropy in tested settings and propose GPU decompression. This is not evidence that arbitrary pretrained models can lose the same fraction of represented information. Archive size, compressed resident representation, decompression cost and lossy capability retention are separate measurements. [R17]

RRT's parameter efficiency does not promise a locally reproducible result after a tiny adaptation run. Conversely, failure with little local adaptation would not refute results obtained with vastly more uptraining. The experiment must measure learning curves and distinguish an optimization/data shortfall from a representational limitation. [R01]

The SharVeT webpage and PDF expose differing headline improvement numbers. This plan does not use either headline as a target or import it into a local comparison. The consequential evidence is its actual similarity/grouping/refinement mechanism. [R03]

### 3.3 Implementation evidence that changes the plan

**SharVeT is not plug-and-play for this teacher.** The inspected `model_factory.py` uses model-name and dimension lookup tables. The inspected `similarity.py` samples token embeddings and multiplies them by candidate projection weights. Applying that path unchanged to a Llama FFN down projection creates an input-dimension mismatch because the down projection expects intermediate width, not embedding width. This is a conditional code-path issue, not a claim that the authors' published experiments all failed. Confirm which projections their chosen configuration uses. [R04]

The local adapter must infer dimensions, preserve the paper's baseline, and make deviations explicit. Run the unmodified supported projection path as a reproduction check. For down projections, use matched actual module-input samples in a separately labeled paper-faithful adaptation, or keep that projection private in an upstream-compatible variant. Report both when feasible; never quietly repair the method and label its result the unmodified official implementation.

The README declares **PolyForm Noncommercial 1.0.0**, not an unrestricted permissive license. A noncommercial research execution can follow the declared conditions; a commercial project must resolve permission or use a separately implemented paper-based baseline without copying restricted code. Preserve the license text and attribution, and do not assume that downloadable code is unrestricted open source. This document is not a legal opinion. [R04]

**MLX has stronger native quantization tools than basic conversion alone.** Official current documentation covers GPTQ, AWQ, dynamic quantization and distilled weight quantization (DWQ). They belong in the local baseline search. However, the inspected default calibration loader downloads a separate text corpus and caches it. The project must replace that data-loading boundary with its own hashed calibration manifest; merely passing the same sample count is not equal information access. [R15]

**Generic weight saving and quantization may destroy sharing.** A serializer can write the same tensor under every depth, and per-depth AWQ transformations can create different versions of a supposedly shared matrix. The canonical storage and sharing-safe transformations in Section 4 are mandatory. Saving a compact archive that expands into all original matrices is a different storage/runtime tradeoff, not a resident compression result.

### 3.4 Explicit novelty answers

1. **What exists?** All architectural primitives and the broad causal-training/refinement ideas above.
2. **What can be reused?** Public model/tokenizer assets; official Llama execution semantics; PyTorch autograd; MLX kernels and quantization; existing benchmark data; published RRT, shared-basis and SharVeT mechanisms.
3. **What may differ?** A tightly specified candidate-specific counterexample compiler over shared nonlinear operators, with joint state-drift checking and exact-byte evaluation.
4. **Why could it matter?** It might allocate shared capacity according to downstream distinctions rather than superficial similarity, avoiding costly but behaviorally irrelevant reconstruction.
5. **What isolates it?** Same bank architecture and bytes, same teacher/probe access and fitting, but fixed/similarity grouping versus active partition refinement; plus equally funded hard-example and counterexample-guided precision-allocation controls.
6. **What invalidates novelty?** Prior work implementing materially the same candidate generation, intervention search and partition revision; or evidence that the claimed mechanism reduces exactly to an established objective/search procedure. In that case, rename the work as replication, local engineering or an empirical comparison. A better M3 implementation is useful but not automatically methodological novelty.

**Additional negative evidence and boundary:** robust neural repair does not have a general termination guarantee; the finite visited-state and expenditure limits here ensure operational termination only, not convergence or global equivalence. [R30] Work on looped-transformer length generalization also distinguishes answer success from faithful intermediate computation; semantic content and computational phase can both move in an activation patch. [R34] Accordingly, probe success alone is never the primary capability claim.

A targeted literature and patent search is not proof of absence of prior art, patentability or freedom to operate. Broad layer-sharing patent material was found, including US10628683B2; it does not establish coverage or noncoverage of this method. Search limitations are recorded in Section 12. [R21]

---

## 4. Selected mechanism: implementation specification

### 4.1 Decoder and shared representation

Use zero-based depth indices \(\ell=0,\ldots,L-1\). For a full, causally masked sequence, write a pre-normalized teacher block as

\[
u_\ell=h_\ell+\operatorname{Attn}_\ell(\operatorname{Norm}_{1,\ell}(h_\ell)),
\qquad
h_{\ell+1}=u_\ell+F_\ell(\operatorname{Norm}_{2,\ell}(u_\ell)).
\]

Here each state is the **entire sequence tensor**, not a single allegedly sufficient latent state. With column-vector notation,

\[
F_\ell(z)=W^{down}_\ell\left[\operatorname{SiLU}(W^{gate}_\ell z)\odot W^{up}_\ell z\right].
\]

The candidate stores \(K\) shared FFNs. A static integer map \(c:\{0,\ldots,L-1\}\to\{0,\ldots,K-1\}\) chooses the bank member at each original depth. For projection \(p\in\{gate,up,down\}\),

\[
\widetilde W^p_\ell=W^p_{c(\ell)}+B^p_\ell A^p_\ell,
\quad A^p_\ell\in\mathbb R^{r\times d_{in,p}},
\quad B^p_\ell\in\mathbb R^{d_{out,p}\times r}.
\]

Execute a projection as `shared_linear(x) + B(A(x))`. **Do not materialize a separate dense effective matrix for every depth during deployment.** The LoRA scale is exactly one; implementations using `alpha/r` must set `alpha=r`. Rank zero must take an explicit no-adapter code path.

All original attention modules, normalization parameters, embedding/head tying, causal masks, RoPE settings and depth positions remain unchanged in the controlled architecture study. Initially train only the bank and depth-specific adapters. A separate, labeled full-uptraining variant may unfreeze other weights for all matching sharing baselines; it is not the primary attribution comparison.

There is no token-dependent route, no extra computation depth and no external knowledge. The static map is a restricted “program,” but no human-readable semantic operators are claimed. The public MLX Llama implementation confirms the necessary gate/up/down structure, separate per-depth attention/cache semantics, and tied-output handling for this model family. [R22]

### 4.2 Parameter and compute invariants

With hidden width \(d\), FFN width \(m\), fixed rank \(r\) at all depths,

\[
P_{candidate}=P_{nonFFN}+3Kdm+3Lr(d+m).
\]

This formula excludes container overhead and assumes all residual factors are retained. Store the same residual rank at every depth in matched bank comparisons, including singleton groups, so group imbalance does not secretly change the byte budget. A completely unmodified teacher has no adapters.

Sharing reduces unique weights but not the number of FFN evaluations. With a multiply-add counted as two operations, FFN matrix operations remain approximately \(6Ldm\) per token, plus \(6Lr(d+m)\) for adapters. Attention, normalization, activations and the output head add costs beyond that expression. Do not infer end-to-end speed from this approximation.

The canonical bank owns each shared parameter exactly once. A depth wrapper stores only a bank index and its private factors. Optimizer parameter lists must contain each shared tensor once; gradients from every use must accumulate into that owner. Attention KV caches remain separate by original depth, even when the FFN is shared. Generic parameter-name deduplication is not a sufficient storage test.

### 4.3 Teacher oracle and boundary API

Implement these project interfaces; they are **not existing package APIs**:

```python
@dataclass
class Boundary:
    u: Tensor                 # [batch, sequence, hidden], post-attention/pre-FFN
    valid: Tensor             # boolean [batch, sequence]
    positions: Tensor         # absolute causal positions
    depth: int
    doc_ids: list[str]

class DecoderAdapter(Protocol):
    def logits(self, token_ids, valid, *, cache=None): ...
    def prefix_boundary(self, token_ids, valid, depth: int) -> Boundary: ...
    def ff_residual(self, boundary: Boundary, operator=None): ...
    def suffix_logits(self, after_ffn, boundary: Boundary): ...
    def score_continuations(self, requests): ...
    def save_bundle(self, directory): ...
    def manifest(self) -> dict: ...
```

`ff_residual` returns `u + operator(norm2(u))`. `suffix_logits` starts at the **next** original block and includes final normalization and the correct tied output head. It must retain gradients with respect to its input even when all suffix parameters are frozen. Teacher target branches use no-grad; candidate-through-teacher-suffix branches do not. Never detach the candidate residual before the suffix.

The prefix is recomputed or streamed by minibatch; a full-corpus activation dump is unnecessary. For student-reached states, run the current complete candidate prefix on the same token sequence, detach its boundary, and compare teacher and candidate FFNs acting on **that same** boundary. The teacher oracle on a student boundary is a local reference computation, not necessarily the teacher's original prediction for the text.

These full-sequence tests use teacher forcing. For a continuation/rollout challenge, use complete model recomputation with the designated hook on every growing prefix, or a separately verified intervention-aware cache. Do not interpret one patched residual vector as a complete autoregressive state. Normal deployment timing uses the unmodified, correctly cached candidate, not this expensive diagnostic path.

### 4.4 Natural and intervention losses

For teacher logits \(a\), candidate logits \(b\), and a boolean mask selecting predicted positions,

\[
\operatorname{KL}(a,b)=\frac{1}{N}\sum_{t\in mask}\sum_v
p_a(v,t)\,[\log p_a(v,t)-\log p_b(v,t)].
\]

Use temperature 1 for the reported metric. Higher-temperature distillation is a tunable training option shared with baselines, not a change in the reported metric. Compute log-softmax and reductions in FP32 during acceleration; use FP64 on CPU for numerical reference tests and final scalar aggregation. Keep the teacher distribution on the left. Do not use logit MSE as an unlabeled substitute.

The default global fitting objective is

\[
\mathcal L_{global}=D_{KL}(T,M;D_{fit})+\alpha L_{NLL}(M;D_{fit}).
\]

Start exploratory runs with \(\alpha=0.1\); include \(\alpha=0\) and \(1\) in common development tuning. Natural NLL discourages a student that matches teacher errors while becoming worse on actual text, but teacher fidelity remains the primary endpoint.

For a candidate operator \(G\) replacing \(F_\ell\), define

\[
\mathcal D_\ell(G;u)=D_{KL}\left(
Q^T_{>\ell}[u+F_\ell(N_{2,\ell}(u))],
Q^T_{>\ell}[u+G(N_{2,\ell}(u))]
\right),
\]

where \(Q^T_{>\ell}\) includes the complete teacher suffix and output head. The same boundary and mask enter both branches. This is an intervention on the computation graph; it is not proof of a semantic causal explanation of language.

Probe states come from three sources:

| Source | Construction | Interpretation |
|---|---|---|
| Natural teacher states | Original held-out-from-fitting training documents passed through the teacher. | Relevant local approximation at the teacher operating distribution. |
| Student-reached states | Current candidate boundaries on the same documents, before substitution. | Tests whether local sharing remains useful after upstream drift. |
| Matched donor interventions | At the same depth and absolute position, replace an explicitly recorded part of `u` using a token-aligned donor context. | Challenges under-observed distinctions; hybrid states need not lie on the natural manifold. |

Initial donor contexts use auditable text changes: entity/name substitution, singular/plural changes and changed numeric values. Accept only donors with verified alignment at the patched span; reject examples where BPE length changes invalidate the position match. Record the original and modified token IDs. Donor selection must depend only on allowed training/probe content, not on confirmatory answers.

The initial patch family is: one aligned token's full residual vector; a contiguous aligned span; and one of four fixed channel quarters at an aligned token. Record the exact mask. Masked prediction positions start at the patched position, which predicts the following token, and end at the last valid next-token target. No future activation may be copied into an earlier position. Equal-state and zero-mask interventions must be exact null controls.

Channel masks are coordinate-dependent stress tests, not discovered semantic variables. Gaussian perturbations and gradient-adversarial hidden states are **secondary robustness diagnostics**, not the default evidence of useful causal reuse. Use train-derived norm and entropy summaries to identify pathological off-manifold patches; report their exclusion rule before pilot comparison and keep an unfiltered stress-test result. Do not filter out patches merely because the candidate does badly.

A default local fitting mixture assigns weight 1/2 to natural states, 1/4 to student states and 1/4 to donor states. Explore donor weights 0, 1/4 and 1/2 with the same allowance for the fixed-partition causal-distillation baseline. These are initial engineering settings, not scientific success thresholds. Freeze the selected weights before confirmation.

### 4.5 Counterexample construction and independence

Separate the allowed development data into `fit`, `probe_fit`, `probe_score` and `dev` roles. Both probe roles belong to exploratory training/search. `probe_score` is never used for gradients within a candidate fitting round, but repeated selection can overfit it; only protected confirmation estimates final generalization.

For a proposed merge, evaluate a seeded pool of donor/state examples, rank them by the candidate-specific \(\mathcal D_\ell\), and retain high-loss examples **with a cap per source document and equal opportunity across depth regions**. Keep a random component rather than selecting only extremes. Refit on `probe_fit` examples, then rerank on disjoint `probe_score` documents. Record every teacher query and rejected patch. A text-only hard-example control receives equally many candidate-disagreement documents and the same compute allowance, without hidden-state patches.

Refresh a probe state when upstream bank assignments or weights change; a stale boundary is not a current student-reached state. Cache keys include teacher hash, current candidate hash, depth, tokenizer hash, token IDs and patch mask. Cache invalidation is a correctness requirement, not an optional optimization.

### 4.6 Candidate initialization and search

Start from one bank member per teacher FFN. Candidate groups partition the original depths. For a merge of groups \(A,B\):

1. Consider the existing bank member from each group as an initialization. Do not average independently permuted FFN neurons blindly.
2. For each member depth and projection, compute a rank-\(r\) approximation to `teacher_weight - proposed_bank_weight` on CPU. With \(\Delta W=U\Sigma V^\top\), initialize \(B=U_r\Sigma_r^{1/2}\), \(A=\Sigma_r^{1/2}V_r^\top\). A zero residual uses small seeded random `A` and zero `B`, avoiding a permanently zero gradient in both factors.
3. Choose the medoid initialization with lower local activation reconstruction loss on `fit`, using exactly the same initialization candidates for all scoring methods. Fit the new shared operator and affected residuals; leave other groups fixed during the local trial.
4. Evaluate the provisional **complete** candidate and its local probe discrepancy. Reject NaNs or invariant violations as implementation failures, not as low-quality candidates.
5. Rank legal merges by the prespecified combined score per byte saved, then validate the leading trial on fresh exploratory documents. Save its complete state before accepting it.

For fixed-rank, equal-shape banks each merge saves one shared FFN's encoded storage; exact bytes still come from the serializer. A useful exploratory ranking is

\[
score(A,B)=\frac{
\Delta D_{natural}(M_{A\cup B},M_{current})+
\lambda\,\overline{\mathcal D}_{probe}(A\cup B)
}{B_{current}-B_{A\cup B}}.
\]

The natural term is candidate KL to the teacher minus current-model KL to the teacher. Average the probe term over original depths, then documents, rather than letting large groups or long documents receive accidental extra weight. Start with \(\lambda\in\{0,0.1,1\}\); include the ordinary downstream-KL score \(\lambda=0\) as a particularly important control. Coefficients are selected only on development data.

The full pair set initially contains \(30\times29/2=435\) unordered possibilities. Evaluate all cheap local similarities once per refreshed state pool. To avoid fitting every possible merge at every step, use a common candidate shortlist: high-scoring cheap candidates, adjacent-depth candidates, and a seeded uniform sample from the rest. Start with eight from each category where available; deduplicate. This shortlist is a computational heuristic, not evidence that other pairs are irrelevant. On a stratified sample of excluded pairs, measure how often the screen misses better held-out candidates; expand the shortlist when its uncertainty could change the selected comparison. Match candidate pools across scoring controls.

The single-merge diagnostic must also sample directly from the full pair population, rather than only this favorable shortlist. Stratify by early/middle/late depth and separation. Report results conditional on the sampled candidate population; candidate pairs sharing a layer are not independent statistical samples.

Continue merges until the requested \(K\) or payload budget is reached, preserving intermediate frontier points. A development guardrail may prevent an unsafe merge, but then the result is **failure to reach that budget under that guardrail**, not a successful compressed model. If counterexamples reveal an earlier bad merge, restore that checkpoint and try the next unvisited alternative. For fixed-budget comparisons a split must be accompanied by another merge or increased compression elsewhere; increasing \(K\) is not a free repair.

A visited-partition digest prevents cycles. Record the finite search allowance chosen from pilot cost and candidate-ranking uncertainty. When exhausted, report search-limited uncertainty instead of implying optimal compilation. After the final partition is chosen, globally uptrain all bank and residual parameters using the frozen common schedule.

### 4.7 Reference pseudocode

```text
compile(teacher, fit, probe_fit, probe_score, config):
    current = initialize_singletons(teacher, rank=config.rank)
    history, visited_partitions = [], {canonical_partition(current)}
    while current.bank_count > config.target_K:
        states = stream_current_boundaries(teacher, current, allowed_probe_ids)
        pairs = common_shortlist(current, states, config.search_seed)
        best_cpu_delta, trial_records = None, []
        for pair in pairs:
            trial = initialize_merge_from_best_medoid(current, teacher, pair)
            trial = fit_local(trial, fit, fixed_local_update_allowance)
            counterexamples = mine(trial, states, probe_fit, query_allowance)
            trial = refit_local(trial, counterexamples, fixed_refit_allowance)
            score = complete_model_and_probe_score(trial, probe_score)
            record = record_metrics_cost_and_exact_bytes(trial, score)
            trial_records.append(record)  # never keep every live trial model
            if legal_and_better(record, best_cpu_delta):
                best_cpu_delta = save_changed_tensors_and_map_to_cpu(trial)
            release_trial_graph_and_tensors(trial)
        if best_cpu_delta is None:
            return budget_not_reached_with_diagnostics(history, trial_records)
        current = apply_delta(current, best_cpu_delta)
        assert_joint_forward_and_storage_invariants(current)
        visited_partitions.add(canonical_partition(current))
        history.append(trial_records)
        atomic_checkpoint(current, history, visited_partitions, all_rng_states)
    current = optional_budget_neutral_partition_repair(
        current, visited_partitions, probe_fit, probe_score, config)
    return globally_uptrain(current, common_fit_schedule)
```

**Exact split/repair extension:** implement a separately selectable `merge_only` versus `merge_then_repair` policy. At target bank count, score each member depth's local error on the allowed probe pool. Propose splitting the worst-scoring member from a non-singleton group, initializing its private operator from that depth's teacher FFN and its residual as zero. Immediately propose compensating merges elsewhere so the final bank count and byte ceiling do not increase. Refit and score complete resulting models; accept only a previously unvisited partition with a lower frozen development search objective than the current partition by more than the measured identity-run numerical floor. If no budget-neutral proposal improves, restore the current state. Repeat only within the prespecified remaining search expenditure. This finite visited-partition policy guarantees the search returns, not that a globally optimal or equivalent model is found. Temporary split models can be larger and their memory peak must be calibrated. Freeze the policy before confirmation; the extra repair pass needs its own same-cost ablation.

Controls replace only the scoring/mining/grouping decision under study, not the teacher, parameterization, probe access or bookkeeping. For matched-compute analyses, unused search/refit time is available to the competing method for ordinary fitting or a stronger search.

### 4.8 Memory-safe KL and numerical conventions

Compute the entire vocabulary distribution, but chunk **prediction positions**, for example 32 positions at a time, through the final head. Weight every chunk by its actual valid-token count; the last ragged chunk must not get equal weight to a full chunk. Accumulate gradients across head chunks without storing all vocabulary logits simultaneously. Suffix hidden activations can be recomputed with checkpointing; teacher targets are recomputed or streamed under no-grad.

Do not silently truncate teacher distributions to top-k, clamp probabilities, change the vocabulary or fit a new tokenizer. Those alter the objective or comparison. FP32 `log_softmax` avoids explicit `log(softmax())`. A tiny negative KL from floating-point reduction can be displayed as zero only after the raw value is retained and a numerical tolerance is justified by identity tests. Larger negative values are bugs.

Padding has no loss. A real EOS token remains a target even when the padding ID equals EOS; the mask comes from sequence length, not from equality to the token ID. The final input position without a next-token label is excluded. Context positions and special-token insertion are identical across methods.

### 4.9 Canonical export and quantization

Use a payload such as:

```text
bundle/
  manifest.json
  config.json
  tokenizer/                 # exact original assets and special-token settings
  weights/non_ffn.safetensors
  weights/bank.safetensors   # each bank tensor once
  weights/residuals.safetensors
  quantization.json          # per-canonical-tensor bits, grouping, scale dtype, padding
  runtime/                   # only candidate-specific implementation
  LICENSES/
```

`manifest.json` lists every relative path, size, SHA-256, tensor owner, shape, dtype, bank use count, original layer map and dependency-lock hash. Sum actual file sizes. Verify that no weight is omitted by an external reference. Load and run from a fresh, offline process with the training directory inaccessible.

For affine groupwise quantization, a first-order packed weight estimate is

\[
B_Q\approx Pb/8+(P/g)(b_s+b_z)+B_{padding}+B_{headers},
\]

where \(b\) is weight bits, \(g\) group size, and \(b_s,b_z\) are **bytes** per scale/offset. With two FP16 values per group of 64, overhead is 0.5 bits/weight. Actual kernel formats, unquantized tensors and tied embeddings must be measured. The documentation exposes quantized linear operations but does not certify every desired shape, precision or backward path. [R23]

Apply GPTQ calibration to each canonical shared matrix using input statistics aggregated over **all of its uses**; do not quantize separate per-depth copies. A deployment GPTQ representation must remain shared. For AWQ, shared groups need common algebraic scales. A safe input rescaling is

\[
x'=D_s^{-1}x,\quad W'=WD_s,\quad A'=AD_s,
\]

with the preceding per-depth normalization adjusted consistently. Gate and up projections receiving that normalized input use the same input transformation. A common intermediate-channel transformation can scale the **up output** by \(D_t^{-1}\) and the down input weights by \(D_t\), with corresponding changes to LoRA factors; scaling the SiLU gate itself is not generally equivalent. Verify the complete unquantized function before quantizing.

Use one common scale vector per shared group at these boundaries; independent per-depth scales that require different stored bank matrices invalidate the intended representation. Group-consistent AWQ is a labeled sharing-safe adaptation, not an unmodified upstream run. Retain a round-to-nearest shared export as a correctness baseline and an unmodified AWQ/GPTQ dense baseline as a strong competitor.

DWQ requires its specific training path. Do not infer trainability of packed weights from general autograd support: the current `QuantizedLinear` documentation says its parameters are frozen by default. Test the official DWQ path first. For the bank model, train only supported floating parameters—residuals, compatible scales or other explicitly allowed terms—and count them. If an equivalent shared DWQ implementation is unavailable, report that limitation and compare against the stronger dense DWQ result anyway; do not weaken the dense baseline. [R15] [R23]

Mixed precision allocates bits to canonical tensors under exact byte cost, including embeddings and residuals. Candidate selection uses development loss, not just independent tensor sensitivity; verify the jointly quantized model. Never fold adapters into per-depth dense weights to obtain a misleading fast benchmark and still report the smaller bank's bytes.

---

## 5. Baselines, assets and evaluation design

### 5.1 Claims and the two comparison tracks

**Mechanism track:** at the same \(K\), adapter rank, numerical precision, original depth, optimization budget and information access, determine whether counterexample-guided partition selection adds value. This isolates a scientific mechanism but is not sufficient for a deployment claim.

**Deployment track:** at a fixed actual payload ceiling \(B^*\), compare the complete quantized candidate with the best credible locally executable alternative selected on development data. Also report the full quality/bytes/latency frontier; a slower but smaller model is not an unconditional improvement.

The default first complete-bank point is \(K=\lceil L/2\rceil\), rank 8: roughly two uses per operator, enough reuse to test the thesis without immediately collapsing all depth diversity. This is an experimental operating point, **not a minimum percentage required for success**. Explore \(K=L\), \(\lceil L/2\rceil\), \(\lceil L/3\rceil\), and 1 where the earlier diagnostics justify the latter; preserve all results. The one-merge study comes first.

Set the initial deployment ceiling to the actual serialized size of the \(K=\lceil L/2\rceil\), rank-8, 4-bit group-64 bank payload, including tokenizer and code, after format validation. Thus the ceiling is determined by a concrete architecture, not tuned to a favorable test result. Freeze it before confirmatory evaluation. Baselines may use any valid configuration at or below that ceiling; they do not need identical parameter counts. Keep their best smaller point rather than padding unused bytes with nonsense.

### 5.2 Mandatory local comparisons

| ID | Method | Exact local specification and purpose |
|---|---|---|
| T | Uncompressed teacher | Original checkpoint and tokenizer; FP32 reference for scientific tests, native deployment precision for final runtime comparisons. Neither is claimed bit-identical to the other without testing. |
| Q | Dense learned quantization frontier | Native MLX affine quantization plus GPTQ, AWQ, DWQ and dynamic/mixed precision where critical kernels pass. Test supported low-bit formats that bracket \(B^*\), including 2/3/4 bits when actually supported, and an 8-bit fidelity anchor. Group size 64 initially; include 32/128 when supported and helpful under exact bytes. Select on development data. |
| R | RRT-style FFN sharing, plus whole-block deployment variant | Same bank/residual implementation. CYCLE map `c(l)=l % K`; SEQUENCE map `c(l)=min(K-1, floor(l*K/L))`. SVD residual initialization, common training data and fair tuning. The matched mechanism control shares FFNs only. The practical variant additionally shares q/k/v/o, retains distinct per-depth KV caches and uses correctly shaped LoRA; see WP05. Both are local adaptations, not full published training reproductions. |
| RC | Fixed sharing plus causal distillation | Best development R map with DIITO-style donor interventions, student-state exposure and exactly the same mined examples available to CGOC. Isolates whether topology search contributes beyond intervention training. |
| CKA | Similarity-based nonlinear FFN grouping | Cluster actual FFN output similarity on common source documents, then use the same whole-FFN bank and residuals. Compare natural KD and the same intervention training. Separates the representation from active grouping. |
| BS/SV | Shared-basis baseline and SharVeT adaptation | Implement Section 5.3. Preserve their low-rank parameterization and allocate ranks under \(B^*\). Include prescribed vector-only refinement and a separately labeled stronger common-KD/full-factor variant where affordable. Quantize compatible factors and count padding. |
| P | Depth reduction plus distillation | Natural-KL-guided FFN/block removal, plus the XMerge-derived nonlinear reconstruction control in Section 5.3. Quantize under \(B^*\), preserve positional semantics and use a correct reduced-depth cache. |
| H | Generic hard-example control | Best matching sharing architecture with ordinary text disagreement mining, the same teacher-query and wall-time allowance, and no hidden-state interventions. Exposes an advantage caused merely by extra useful examples. |
| QCE | Counterexample-guided empirical precision allocation | Unshared dense model; same allowed counterexamples and search-cost accounting, but use bytes to refine precision instead of a sharing partition. Section 5.3 specifies the procedure. This is not a formal CEG4N reproduction. |
| M | CGOC | Section 4, followed by the same compatible quantization/refinement search and exact accounting. |

A freshly initialized smaller decoder trained on a short local corpus is **not** a strong main baseline against a pretrained compressed model. It is optional as an information-access control. At the 360M stage, the already-pretrained SmolLM2-135M is a useful practical smaller-model reference, but it has different historical pretraining; disclose that rather than calling it matched distillation. [R08] [R09]

RRT full uptraining, Mixture-of-Recursions pretraining, BitNet pretraining and CUDA-specific entropy kernels are nonlocal reference evidence. Their published numbers never enter a controlled local significance test. Exclude a baseline implementation only for a documented compatibility, license or resource reason; keep the resulting comparison limitation in the headline conclusion. [R01] [R12] [R16] [R17]

### 5.3 Implementable shared-basis and SharVeT specification

Use `W` shaped `[out, in]` internally. A simple output-shared basis baseline for a same-shape group concatenates `[W_1, ..., W_n]` horizontally, computes truncated SVD, and stores a common left factor and one right block per layer. Also implement the activation-weighted input-basis alternative below; select the stronger on development data rather than asserting a paper reproduction from a superficial SVD.

For an input-basis, activation-weighted fit, let \(X_i\) be actual module inputs and use a group-pooled covariance \(C=\sum_i X_iX_i^\top/N+\eta I\), with \(LL^\top=C\). Vertically stack the matrices \(W_iL\), take their rank-\(r\) SVD \(U\Sigma V^\top\), and split `UΣ` by layer. The shared input transform is \(V_r^\top L^{-1}\); each layer has its own output factor. This minimizes the corresponding pooled quadratic approximation, not the full language-model loss. Solve triangular systems rather than explicitly inverting \(L\). Choose ridge relative to the covariance trace, test sensitivity, and record it. Label this precise local baseline rather than claiming every detail is the official Basis Sharing implementation. [R02]

For SharVeT, retain projection-type-specific grouping, not just whole-FFN grouping. In the paper's `[in, out]` convention, concatenate group weights horizontally and approximate

\[
W_i\simeq \Lambda_i U_g\Sigma_i V_i^\top.
\]

`U_g` is shared, `V_i` private, and the trainable diagonal vectors in \(\Lambda_i\) and \(\Sigma_i\) allow layer-specific refinement. Initialize \(\Lambda_i=I\); absorb \(\Sigma_i\) into the private factor at inference when valid. Unrefined group rank costs approximately \(r(d_{in}+1+n_gd_{out})\) scalar parameters before the extra layer-specific vectors and format overhead. Recompute the exact deployed cost after absorption; do not double-count a singular-value vector that is no longer separately stored. [R03]

Structural similarity uses normalized weight cosine and inverse Frobenius distance. The inspected code uses `N(x)=1-exp(-x)`, combines `N(cos+1)` and `N(inverse_distance/10)` equally, and combines structural and functional scores as `1-(1-S_struct)*(1-S_func)`. Functional similarity is centered linear CKA of matched sample outputs. Use fixed seeds for spectral clustering and handle zero CKA denominators explicitly: identical nonzero outputs score one, otherwise mark the degenerate pair for a defined fallback rather than producing NaNs. [R04]

Allocate a minimum group budget and distribute the remainder inversely to within-group similarity, then round ranks down to legal integer/kernel shapes. Optimize any remaining byte allowance by measured development benefit per added rank increment. Tune the minimum-budget fraction on development data. Report the upstream-compatible input-embedding version and the actual-module-input version distinctly when their paths differ. Vector-only refinement uses language-model loss, teacher KL, hidden-feature consistency and vector regularization; give it a fair coefficient search rather than forcing the CGOC loss unchanged.

Implementation acceptance requires a full-rank reconstruction test, an identity-vector test, exact group/rank accounting, and a small supported-shape comparison against upstream functions where licensing permits. If unresolved code discrepancies remain, the final report must say “local SharVeT adaptation,” not “faithful reproduction.” An inability to establish this comparator blocks a claim of superiority over the closest contemporary sharing method, but not the simpler mechanistic experiment.

#### Contemporary depth-reconstruction control and empirical precision control

For the XMerge-derived control, compute original-model per-block `RM=mean(||h_out-h_in||/(||h_in||+eps))` and `BI=mean(1-cos(h_in,h_out))`; standardize each across eligible blocks and order by their maximum. Retain this original ordering. Collapse an eligible adjacent pair into its surviving neighbor, initialized from that neighbor, minimizing masked two-block output MSE with all survivor parameters trainable. Refresh reconstruction targets after each collapse. The published starting recipe is Adam, 300 steps, learning rate `1e-5`, effective batch 16, and 128 sequences of length 512; it uses patch-recovery correlation to select the neighbor. [R33]

**Local conventions and deviations:** use this project's allowed calibration pool rather than a hidden second corpus, and microbatch/accumulate to preserve the effective batch. Set numerical `eps=1e-8` in FP32 after checking the identity cases. For direction scoring, construct the same token-aligned corrupted/clean pairs already available in `probe_fit`, patch clean attention and FFN outputs separately, and record true-next-token log-probability recovery over common positions. Correlate adjacent per-document recovery profiles. Choose the higher-correlation side; a zero-variance profile is uninformative, so evaluate both sides on the separate development slice and log that fallback. Also retain a stronger development-selected variant that refits both neighbor choices and selects the lower natural KL at counted extra cost. These corruption/readout and tuning conventions are explicit adaptations, not claims of byte-for-byte official reproduction. Keep original layer IDs and a separate surviving-position map; skip nonexistent neighbors. Count all calibration/refit work; apply the common end-to-end distillation and quantization allowance. Include plain removal as a control for reconstruction. Do not extrapolate published behavior at a few removed blocks to the much deeper reductions that this budget may demand.

For **QCE**, use the same frozen teacher and validated native quantizer, but do not share FFNs. Enumerate supported bit widths per canonical matrix; include format-specific scale/padding cost. Start from a feasible lowest-precision payload. On allowed natural and counterexample inputs, measure each single-matrix precision upgrade's *actual* complete-model KL reduction and bytes. Greedily accept the best positive reduction per additional byte that fits the ceiling, remeasure after each accepted change, and retain every considered allocation. Then permit budget-preserving up/down swaps on disjoint probe-scoring records. Refresh the counterexample pool against the current quantized model using the same donor/text procedures and teacher-query allowance. Stop at the prespecified search expenditure or when the finite unvisited neighborhood has no improving move. Select the final allocation only on development data. A global dynamic-quantization search supplied by MLX remains a separate strong Q baseline, not replaced by this greedy control.

If the lowest validated precision cannot meet the budget, QCE is infeasible at that operating point; report that rather than inventing a kernel. This control tests whether the useful ingredient is generic hard-case precision allocation. It is inspired by the already-established CEG4N direction but replaces formal verification with sampled KL, so it offers **no equivalence certificate**. [R29] For mechanism-only same-bank comparisons, also apply this policy to a fixed bank so partition selection cannot win merely by receiving more informed precision allocation.

### 5.4 Common tuning and expenditure policy

All methods see the same allowed natural text, tokenizer and teacher. Supply the common activation/donor pool to any baseline that can use it; the proposed method is not allowed private extra labels or benchmark answers. Probe generation is counted as construction cost even when cached.

Start bank/residual optimization with AdamW, bank learning rate `1e-4`, residual learning rate `3e-4`, betas `(0.9, 0.95)`, gradient-norm clipping at 1, and a 5% warmup followed by cosine decay. Treat these as exploratory starting points, not asserted optimal settings. Weight-decay choices `0` and `0.01`, and learning rates a factor of three above/below the starting values, define the first common local sweep. Check gradient scales and training stability before expanding it. Baselines with published, justified recipes also receive those settings as candidates.

Use FP32 scientific fitting first, microbatch one and sequence length 512 where feasible; accumulate to a common target of 8,192 valid predicted tokens per optimizer update. Maintain the same valid-token count when memory changes force smaller microbatches. A shorter context is a distinct workload, not automatically an equivalent memory optimization.

Choose the total adaptation budget from a pilot ladder rather than an arbitrary final token cap. Initial resource/learning-curve probes can use approximately 0.25M, 1M and 4M token presentations, with continuation by doubling when marginal development improvement remains distinguishable at the planned evaluation precision. This ladder is not an upper bound. Plot every method's learning curve, including the strongest baseline. If all plausible local budgets leave a method clearly improving, call the comparison budget-conditioned rather than representationally conclusive.

Before confirmation, freeze a common token/update budget for the causal attribution comparison and a common **total construction-time ceiling** for a second practical comparison. Search, data generation and quantization fit within the latter. A baseline may spend time not used for active search on additional optimization. Report both comparisons; one matches optimization exposure and the other addresses actual cost. Select configurations using development results only. Use multi-fidelity exploration and retain rejected configurations; do not tune CGOC extensively while using one arbitrary baseline setting.

### 5.5 Data, provenance and splits

No training corpus or checkpoint was downloaded into the research environment. The coding agent must resolve immutable revisions and hashes before execution. The following are real asset identifiers, not invented project scripts.

| Asset | Role and exact starting choice | Provenance / terms / limitations |
|---|---|---|
| `HuggingFaceTB/SmolLM2-135M` | Primary base teacher; original tokenizer; no instruction template. | Model card and config inspected; Apache-2.0 declared. Resolve the complete model-repository revision, not merely the short config-file commit. [R08] |
| `HuggingFaceTB/SmolLM2-360M` | Conditional second scale. | Same provenance procedure and declared license; different training history remains a confound for scaling interpretations. [R09] |
| `HuggingFaceFW/fineweb-edu`, `sample-10BT` | Stream a reproducible local pool for fitting, probes, development and protected natural-text evaluation. Do **not** download the entire 10B-token subset. | Config is documented. ODC-By 1.0 plus Common Crawl terms; underlying web-text rights and sensitive content require care. Pretraining overlap is possible. [R24] |
| `Salesforce/wikitext`, `wikitext-103-raw-v1` | Independent-source transfer evaluation, using the official test partition only after freeze. Use validation for harness smoke checks, never test for tuning. | Card states CC BY-SA 4.0. Preserve raw article structure, not normalized text from a different configuration. [R25] |
| `allenai/ai2_arc`, `ARC-Easy` and `ARC-Challenge` | Science/factual multiple-choice guardrails. Use official training/validation only for prompt/harness checks; protect test. | Card declares CC BY-SA 4.0. Ground-truth labels, not teacher answers, determine correctness. Public benchmark contamination is possible. [R26] |
| BLiMP, original `alexwarstadt/blimp` repository | English syntactic minimal-pair guardrail, all declared paradigms. | Original README specifies 67 paradigms and a CC-BY license; JSONL fields and errata inspected. Resolve its linked license and commit. Use the original distribution if the HF mirror fails. [R27] |
| Project-generated tasks | Copy/lookup, bounded arithmetic, finite-function composition and planted-reuse controls. | Generate locally from explicit algorithms; store seeds, generator revision and exact labels. Synthetic evidence is not real-world quality parity. |

HellaSwag was considered as a complementary commonsense test, but access/licensing inspection was incomplete in this research pass: the original repository returned an access error and later mirror reads failed. It is not a hidden core dependency. The agent may add it only after its provenance and terms are resolved **before** freeze, with a recorded protocol amendment and no removal of unfavorable existing tasks. The core ARC/BLiMP/natural-text evaluation does not require it.

#### Building the local natural-text pool

Resolve dataset revision, enumerate source shards, choose a seeded sample of shards across available provenance strata, and stream records into a local manifest. Avoid “the first N records” of an unshuffled stream as the sole population. Retain source identifiers, URL/domain where available, record IDs, text SHA-256 and the exact shard path/revision. Measure the text-to-token storage ratio on the initial sample before allocating the full pool.

Normalize only for duplicate detection; retain original text for tokenization. Exact duplicate hashes and near-duplicate connected components must be computed before splitting the pool. A practical near-duplicate screen uses token/word 5-gram MinHash signatures, followed by exact Jaccard verification on proposed matches; begin with a conservative 0.8 near-duplicate threshold and publish sensitivity to 0.7/0.9. These are leakage-screen settings, not performance targets. Shared URL/article IDs are grouped even when text revisions differ.

Freeze the pool and component IDs before assigning roles. Stable salted hashes of component IDs assign approximately 70% to `fit`, 10% to `probe`, 10% to `dev`, and 5% each to protected `confirm_A` and `confirm_B`. Split `probe` by component into `probe_fit` and `probe_score`. Ratios are initial storage-planning choices; grow the pool before freeze to meet pilot-derived sample needs. Never repartition after seeing protected results. If additional source records would change duplicate components, rebuild all affected exploratory splits and record an amendment before confirmation.

Use tokenizer-native IDs without introducing a new vocabulary. Train on within-document windows initially; avoid cross-document attention as a confound. Record how BOS/EOS are inserted after testing the original tokenizer/model convention. Never concatenate protected data into calibration text through a generic loader.

#### Prediction-window population

Primary natural-text evaluation samples document components and deterministically selects windows with a saved hash-derived offset. Use up to four 512-token windows per sufficiently long document, scored with identical context and masks for every model; shorter documents remain included with valid-token masking. The estimand is the token-weighted KL/NLL of this declared window population, not an undisclosed estimate of all web text.

For WikiText, parse article boundaries from top-level headings and preserve all original content. Score using a fixed 512-token rolling window and stride 256, counting each newly introduced target once; the initial window scores its available targets. Publish this convention because absolute perplexity is not interchangeable with a different context/stride setup. Article identity is the statistical cluster. Evaluate 2,048-token context as a prespecified secondary condition after memory testing; long-context stress tests use their own population and reported context lengths.

### 5.6 Task and quality metrics

**Primary metric:** complete-model teacher-to-candidate KL on `confirm_A`, in nats per predicted token. Compute \(\sum_d KLsum_d/\sum_d n_d\); do not average document perplexities. The primary effect is

\[
\Delta_{deploy}=D_{KL}(p_T\Vert p_{baseline^*})-D_{KL}(p_T\Vert p_{CGOC}),
\]

where `baseline*` is the strongest eligible development-selected model under \(B^*\). In implementation, name this explicitly `kl_teacher_to_baseline - kl_teacher_to_candidate` to avoid reversing arguments.

**Attribution contrast:** the corresponding KL difference against the strongest matched-bank control selected on development data. Both comparisons are frozen before test access. A deployment win without the attribution win does not demonstrate that counterexample-guided partition selection was responsible.

Report NLL and perplexity, next-token top-1 disagreement, mean and upper-tail document KL, and performance by document length and prespecified source strata. Ground-truth NLL can improve even when fidelity worsens; report that honestly as a different benefit.

For ARC, score each complete answer continuation under an identical zero-shot prompt such as `Question: {question}\nAnswer:` followed by the answer text. Freeze whitespace, tokenizer boundary handling and answer normalization on validation data. Report both summed and token-length-normalized continuation likelihood accuracy; predeclare normalized accuracy as the primary ARC measure. Ties use the lowest original choice index and are counted. Never score just `A/B/C/D` unless that separately frozen prompt explicitly presents the labeled alternatives.

For BLiMP, compare whole-sentence log probabilities of `sentence_good` and `sentence_bad` under the same BOS/EOS convention; do not length-normalize one and not the other. Exact ties count as half a correct pair, with tie frequency reported. Report the macro-average over paradigms and individual paradigm results; sentences from a common generation paradigm are not 67,000 independent tests of grammatical generalization.

For multiple-choice tasks also report the normalized option distribution's log loss and Brier score, and a reliability plot using development-fixed bins. This tests confidence on the task, not open-ended truthfulness. Keep calibration temperature fixed or fit it only on validation for every method and label calibrated results separately.

Synthetic natural-language probes are generated from these exact specifications:

- **Lookup/copy:** randomly permuted key–value pairs, an explicit query key, and an exact value label; vary the number of distractors and placement. No answer label is present outside the context facts.
- **Arithmetic:** addition of bounded nonnegative integers, with carry/no-carry strata; exact integer labels from the generator. Hold out operand combinations, not merely reordered duplicate strings.
- **Function composition:** explicitly defined finite maps over a small symbol alphabet; ask for compositions such as `f(g(a))`. Hold out composition trees and surface templates, not just random examples from the same templates.

Store separate template families for exploration and confirmation. Increase lengths and distractors only where teacher competence and local memory permit an interpretable test. Report failures at the teacher level rather than deleting hard families. The project should not teach the teacher these tasks merely to manufacture a compression benchmark; a separately trained planted teacher is a labeled component control.

Context guardrails use 128, 512, 2,048 and, when feasible, 8,192 tokens for the lookup/copy tasks, with exact prompt construction and no hidden truncation. Evaluate the final prediction only when appropriate. Record teacher accuracy and effective context use at every length. A nominal 8,192-position configuration alone supports no long-context claim. [R08] [R09]

Generate a small, fixed set of short continuations for qualitative error categorization. Use deterministic greedy decoding and a saved-seed stochastic condition, fixed maximum new tokens and EOS handling. These examples are illustrative; they cannot substitute for the protected metrics or certify safety.

### 5.7 Protected confirmation and sample-size procedure

Before accessing `confirm_A`, create `protocol/freeze.json` with hashes of the candidate and all finalist baselines, data manifests, source/dependency revisions, numerical formats, payload ceiling, hypotheses, metrics, prompts, sample sizes, seeds, timing conditions and analysis code. Protected data must be accessible only to an evaluation command that verifies this freeze. Operational separation and access logs reduce accidental leakage; they do not create an independent external evaluator.

A pilot uses **development** documents and separately fitted exploratory models. Estimate paired per-component KL differences and their variance. For a fixed pair of frozen artifacts, a useful prospective approximation for the number of independent clusters is

\[
N\approx ((z_{1-\alpha/2}+z_{1-\beta})s/\delta)^2,
\]

where \(s\) is the pilot standard deviation of the appropriate paired cluster influence values, and \(\delta\) is the effect size to be resolved. With unequal document lengths, use ratio-estimator influence values or bootstrap planning, not raw unweighted token counts. Confirm the approximation by resampling the pilot at candidate cluster counts.

There is no invented required percentage gain. Produce a table of achievable CI half-width, sample count and measured evaluation time. Select the finest useful precision that the local execution can sustain, and freeze it. If a practical decision margin exists, power the experiment for that margin; otherwise report the achieved detectable effect and avoid claiming that an unresolved small effect is absent.

Use a **fixed-sample** confirmatory design. Additional intermediate inspection is exploratory and consumes that holdout. Do not extend the sample until significance. A later changed claim requires the still-unopened `confirm_B` or a fresh source pool, a new freeze and a clear distinction from the first test.

Training randomness and document sampling are different uncertainty sources. Begin pilot resource estimates with three independent paired training seeds; this is not a guarantee of adequate replication. Estimate seed variability and choose the confirmatory seed count before testing. Report:

1. The document-cluster uncertainty for each frozen model pair, conditional on that training run.
2. The distribution of effects across independently trained seeds, without treating `seeds × tokens` as independent observations.
3. A crossed seed/document bootstrap when there are enough seeds to make it informative: resample seed indices and document-component indices independently, use the same sampled indices for every compared method, and recompute token-weighted ratios.

With very few seeds, show the individual effects and a seed-level interval/sensitivity analysis; do not claim a precise population-over-training-runs effect from a bootstrap over three values. If additional adequately powered training replicates are unaffordable, the strongest conclusion is about the fixed artifacts plus observed replication, not universal reliability of the training procedure.

For primary and attribution contrasts, construct simultaneous 97.5% two-sided intervals for each of the two prespecified effects (Bonferroni family coverage at least 95%). Bootstrap complete source components; use a second sensitivity analysis grouped by source domain when available. Choose enough bootstrap replicates that reported interval endpoints are stable relative to the requested precision, starting at 10,000 inexpensive scalar resamples. Positive lower bounds support the corresponding superiority claims. An interval overlapping zero is inconclusive at that precision, not equivalence.

For the capability guardrail family, use prespecified simultaneous one-sided bounds, with Bonferroni or Holm-adjusted inference across the declared tasks. Freeze the method before test access. With loss-oriented measures, upper bounds must be below a prespecified allowable degradation; with accuracy, lower bounds must be above minus that allowance. Without defensible allowances, publish the bounds and withhold the “no meaningful quality loss” label. Report every important subgroup regression even if an aggregate passes.

Finite public tasks can be evaluated in full. For such a fixed set, the observed aggregate is exact for those items; resampling reflects an explicitly modeled broader item/template population, not measurement noise in deterministic predictions. Explain that distinction. Do not overstate what a bootstrap can say about new kinds of tasks.

### 5.8 Harness validity and leakage checks

Before comparing new methods, the harness must demonstrate all of the following:

- An identical checkpoint compared with itself yields zero KL to justified numerical tolerance and exactly matching task predictions.
- A deliberately corrupted checkpoint produces worse fidelity; the test must not hide the difference through an incorrect mask or reversed target.
- A dense expansion of the shared operator bank matches the compact implementation before quantization; expansion is diagnostic only and its large bytes are counted if it is deployed.
- Full-sequence and incremental cached logits agree at every tested continuation position, including padding-free mixed prompt lengths, RoPE offsets and repeated bank uses.
- Position-chunked KL agrees with an unchunked small CPU reference, including a ragged final chunk and masked padding.
- Duplicate components never cross splits; tokenizer/config/model hashes match; quantization calibration imports no protected text or default external corpus.

All models face the same known pretraining-contamination limitations. The experiment measures compression relative to fixed pretrained assets, not uncontaminated general intelligence. Synthetic held-out templates test a different kind of generalization and cannot erase benchmark contamination retrospectively.

### 5.9 Runtime measurement protocol

Use the same final native runtime for all directly compared deployment artifacts. PyTorch/MPS is primarily the scientific implementation; MLX is the intended deployment comparator. Cross-runtime timings are supplementary, not evidence of an architectural speedup.

Measure cold process load, warm load, prompt tokenization, prefill, first-token latency and steady decode separately. Include a full end-to-end request boundary as well. Use batch one as the primary local workload, fixed prompt lengths 128/512/2,048 where supported, and fixed generated lengths such as 128 new tokens. Timing-only runs may ignore EOS to standardize work, but real-request timings with normal stopping must be separate and clearly labeled.

Materialize lazy MLX arrays and synchronize around timed regions. For MPS call the backend synchronizer. Exclude compilation/warmup from warm throughput but include it in cold-start results. Warm until the fixed-shape kernel path has executed and timing stabilizes on pilot runs; freeze the warmup procedure before comparison. Do not include asynchronous enqueue time as inference time.

Randomize model order in paired timing blocks and alternate methods across repeated blocks on the same prompt bank. Record power connection, low-power mode, background workload, memory pressure, operating-system version and signs of thermal slowdown. Use pilot block variance to choose the number of repeated blocks for the desired timing precision. Tokens within one decode and repeated timings of one prompt are not independent user requests.

Report median and tail request latency, tokens/second, variation across blocks, cold-load bytes/seconds and peak memory. Do not sum process RSS and GPU allocation counters as though unified-memory pages were disjoint. Preserve the raw counters and their definitions. Report any benefit from memory mapping, dequantization caching, compilation or specialized kernels for all methods that can use the optimization.

---

## 6. M3 Max feasibility and hardware adaptation

### 6.1 Discovery before dependent work

Run these **existing operating-system commands**, recording stdout/stderr and exit status:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
system_profiler SPHardwareDataType SPDisplaysDataType
df -h .
xcode-select -p
```

Record available memory/pressure and swap using supported OS tools, not just physical capacity. Discover native Python architecture, version and available disk before installing dependencies. Do not run under Rosetta. Do not assume a maximum-memory M3 Max or that the machine is otherwise idle.

The inspected Apple page specifies Apple Silicon, macOS 14+, Python 3.10+ and Xcode command-line tools for the PyTorch release it describes. Its stated version and the versioned PyTorch documentation reached during research were not identical. Therefore this plan does not invent a uniquely “latest” validated combination: select a compatible stable native wheel, record the exact resolution and pass local kernel tests before locking it. Do not automatically choose a nightly or upgrade the user's OS. [R28]

### 6.2 Software stack and dependency policy

Use a native virtual environment for the PyTorch reference/scientific path. Python 3.11 or 3.12 is a reasonable compatibility starting choice if available and supported by resolved wheels; another supported native version is acceptable after tests. Required roles are:

| Component | Role | Required verification |
|---|---|---|
| PyTorch with MPS | Autograd, teacher hooks, shared FFNs, training and suffix tests. | FP32 forward/backward, causal attention, RMSNorm/SiLU, indexing/patching, checkpoint recomputation and shared-parameter gradients. |
| Transformers + tokenizer dependencies | Load original model/config and provide a reference forward. | Model type/weights/tokenizer support; no `trust_remote_code=True`; no automatic CUDA device map. |
| MLX and `mlx-lm[train]` | Native inference, quantized baselines and supported learned quantization. | Exact Llama-family compatibility, desired bits/groups/shapes, tied embeddings, quantized inference and intended refinement path. |
| NumPy, SciPy, scikit-learn | CPU SVD, Cholesky, clustering and analysis. | Float64 linear algebra stays on CPU; deterministic seeds; bounded workspace. |
| datasets / safetensors / YAML / pytest / process metrics | Streaming assets, safe serialization, configuration, tests and telemetry. | Immutable revisions, explicit schema, no unsafe pickle loading of unknown code, correct units. |

An existing documented installation command is `pip install "mlx-lm[train]"`. Current official documentation also lists `mlx_lm.gptq`, `mlx_lm.awq`, `mlx_lm.dwq` and `mlx_lm.dynamic_quant`. Run their `--help` and inspect the resolved entry points before constructing wrappers; these commands were researched, **not executed on an M3 here**. [R15]

Use a project data adapter at the quantization calibration boundary. The inspected loader signature is `load_data(tokenizer, num_samples, sequence_length)` and its default path downloads/caches separate text. Override or replace that function in a versioned local adapter that supplies exactly the project manifest. Record the patch and test that the default download cannot occur. Do not mutate an unrelated user-wide cache to trick the loader. [R15]

Resolve versions first, then save a complete dependency lock with package hashes where available, interpreter version, platform, source commits and local patches. Reproduce from that lock in a second environment before confirmation. No automatic dependency updates during a study. MPS fallback to CPU must be explicit and logged; silent fallback is unacceptable for a GPU performance result.

Custom Metal kernels, CUDA, bitsandbytes, FlashAttention and remote services are not core dependencies. Eager attention is the faithful fallback when a fused path fails, subject to memory/time recalibration. A changed numerical algorithm or shortened context requires a recorded comparison amendment; a transparent scheduling change with identical semantics does not.

### 6.3 Mandatory local compatibility gates

Before large asset downloads, test a tiny model with the same operations and head/KV structure. Then test the actual 135M checkpoint on short sequences. Required checks include:

1. FP32 CPU versus MPS logits/loss/gradients; determine tolerances from identity, finite-difference and repeated numerical tests, not a generous arbitrary allowance that hides errors.
2. Frozen-teacher suffix propagation: candidate gradients are nonzero and correct; teacher parameter gradients remain absent.
3. Reused operator gradients equal the sum from explicit independent uses at identical weights.
4. FP32 versus candidate FP16/BF16 behavior. Reduced precision is optional only after measured fidelity and stability checks, and must be applied comparably to competitors.
5. MLX import/export, weight orientation, RMSNorm epsilon, RoPE, GQA head repetition, tied output and cache parity.
6. Supported quantization bit widths, group sizes, matrix dimensions, padding and any needed gradients. Fail a specific unsupported configuration; do not assume support from the package name.
7. Clean restart and offline load from the canonical bundle without expanding per-depth bank copies.

If MPS lacks a critical backward operation, first use an equivalent eager primitive and validate it. Otherwise run the small scientific test on CPU, retaining a clear CPU label; use MLX for the final execution path if verified. If neither offers a faithful local operation at a useful scale, stop that dependent claim as a local compatibility limitation. Do not secretly replace the intervention objective with a different surrogate and call the same experiment complete.

### 6.4 Calculated model sizes

The inspected configs yield the following counts. They must be checked against loaded tensor ownership, including tied embeddings, before execution. Counts are calculations, not observed package memory usage. [R08] [R09]

| Quantity | SmolLM2-135M | SmolLM2-360M |
|---|---:|---:|
| Depth \(L\) | 30 | 32 |
| Hidden / intermediate width | 576 / 1,536 | 960 / 2,560 |
| Query heads / KV heads / head width | 9 / 3 / 64 | 15 / 5 / 64 |
| Vocabulary | 49,152 | 49,152 |
| Unique teacher parameters | 134,515,008 | 361,821,120 |
| All FFN parameters | 79,626,240 | 235,929,600 |
| Non-FFN parameters | 54,888,768 | 125,891,520 |
| Rank-8 FFN residual parameters | 1,520,640 | 2,703,360 |
| Half-depth bank, rank 8: unique parameters | 96,222,528 | 246,559,680 |
| Teacher raw BF16 weight bytes | 269,030,016 | 723,642,240 |
| Half-depth bank raw BF16 bytes | 192,445,056 | 493,119,360 |
| Maximum FFN-only ratio, one operator, rank zero | 2.338× | 2.715× |

For the 135M teacher, an idealized all-quantizable 4-bit group-64 representation with two FP16 calibration values per group is approximately `134,515,008 × 4.5/8 ≈ 75.66 MB`, before real unquantized tensors, headers and tokenizer assets. The half-depth rank-8 bank would have approximately 94.70M base parameters at that precision plus 1.52M BF16 residual parameters, roughly 56.31 MB before those overheads. These are planning approximations; actual bundles determine \(B^*\). A dense 3-bit/mixed-precision competitor is therefore especially relevant.

The raw BF16 KV-cache estimate is

\[
B_{KV}=2LB C H_{KV}d_h\cdot2\text{ bytes}.
\]

At batch one and context 8,192, this is 188,743,680 bytes for 135M and 335,544,320 bytes for 360M. FFN sharing leaves it unchanged. Quantized KV caches, when available, are a separate common optimization and must not be awarded only to CGOC.

### 6.5 Memory model

Budget peak memory by contributors, not only weight size:

\[
M_{peak}\approx b_TP_T+b_fP_{frozen}
 +(b_p+b_g+b_{m1}+b_{m2})P_{train}
 +A_{student}+A_{suffix}+L_{logits}+W_{workspace}
 +C_{allocator}+C_{CPUcopies}.
\]

For FP32 Adam, the trainable-weight term is typically 16 bytes/parameter: weights, gradients and two moments. A separate master copy would add to it. The teacher is frozen but differentiating a suffix still needs intermediate activations or recomputation. Count copied teacher weights, candidate trials and CPU SVD workspaces when simultaneously alive.

For 135M with a half-depth bank and rank 8, training only the bank/residuals gives 41,333,760 trainable parameters. FP32 teacher weights, frozen candidate non-FFN weights and FP32 Adam state total approximately **1.42 GB** before activations, logits, allocator and conversion overhead. Unfreezing the entire candidate raises that persistent subtotal to about **2.08 GB**. The corresponding bank-only 360M subtotal is about **3.88 GB**. These calculations suggest local feasibility but are not peak-memory measurements.

One `[1,512,49152]` FP32 logits tensor occupies 100,663,296 bytes; at sequence 2,048 it occupies four times as much. Several logits/probability tensors and backward graphs multiply this cost. Position chunking and recomputation are important even for a small model. Eager attention adds approximately `batch × query_heads × sequence² × bytes` per materialized score tensor, potentially at many depths.

Never retain every candidate model or every document's activations. Trial merges are evaluated sequentially; retain compact changed tensors for the best candidates, metrics for all candidates, and atomic checkpoints for accepted states. Stream covariances and compute one projection's factorization at a time on CPU. Clear dead accelerator graphs explicitly and verify counters fall as expected.

### 6.6 Measured calibration, not promised throughput

The first actual hardware calibration must measure, separately: teacher forward; candidate forward/backward; early/middle/late teacher-suffix backward; prefix extraction; candidate initialization/SVD; quantization; export/load; prefill and decode. Use real representative sequence lengths and synchronized timing.

If \(v\) is the measured end-to-end optimization throughput including the chosen teacher/probe schedule, a training pass of \(N\) token presentations costs approximately \(N/v\) seconds. For **illustration only**, 10M presentations at 100, 300 or 1,000 tokens/second take about 27.8, 9.3 or 2.8 hours. None of these speeds was measured on an M3 in this work. Search, evaluation, quantization and rejected runs are additional costs.

Forecast candidate search using measured per-prefix and per-suffix costs weighted by depth, not a single optimistic full-forward number. Publish a cost ledger for each method and the uncertainty of extrapolated runtime. Long runs are allowed when they answer a sufficiently sensitive question; the plan does not promise completion overnight.

| Stage | Principal resources | How to size it |
|---|---|---|
| Metadata, licenses, tiny tests | Small CPU arrays and package installation. | Measure free disk; do not download large corpora. |
| Teacher clone and one-merge study | One teacher, streamed states, one trial and suffix graph. | Sequence 128 for smoke, then 512; fit peak-memory model before increasing probes. |
| Full-bank fitting | Teacher, current student, optimizer and recomputation buffers. | Persistent subtotals above plus measured activation peak; use common gradient accumulation. |
| Shared-basis/SharVeT preparation | CPU covariance/SVD workspace plus one projection group. | Stream inputs; factor groups sequentially; measure BLAS workspace and thread count. |
| Native quantization and export | Original/quantized models and calibration buffers can coexist. | Measure cold conversion peak independently of steady inference. |
| Confirmation | Frozen artifacts, raw per-document metrics, possibly one model at a time. | Evaluate finalists sequentially and join by immutable item IDs. |
| Scale confirmation | Larger versions of the above. | Recalibrate; do not multiply 135M time by a parameter ratio and present it as measured. |

### 6.7 Hardware-dependent rules and recovery

Use a conservative initial process allocation target around 60% of physical unified memory, additionally limited by current free/available memory and observed OS pressure. This is a starting safety margin, not a claim that the task needs that much memory. Never disable OS memory protections or push allocator high-water settings to unlimited to make a benchmark run.

When headroom is low, in order: evaluate trials sequentially; remove retained graphs; reduce head-position chunk size; checkpoint/recompute suffixes; reduce microbatch and increase accumulation; move offline factorization to CPU; shorten **exploratory** contexts with a separately labeled configuration. Keep the primary 512-context comparison unchanged unless a recorded amendment is made before confirmation. If even the 135M natural-model diagnostic cannot run faithfully, report the local resource limitation; a synthetic-only fallback cannot establish the original LLM claim.

On roomier hardware, increase minibatch/probe throughput before increasing model size. Escalate to 360M only after the earlier mechanism and baseline gates pass and the calibrated costs support useful precision. Do not purchase or require a remote accelerator.

Disk planning includes original weights, conversions, candidate bundles, retained optimizer checkpoints, token arrays, source manifests and failed-run records. A uint32 token array costs four bytes/token before indexing; text and metadata add measured overhead. Keep best/final and last-plus-previous resumable checkpoints, while retaining configurations and raw metrics for every rejected run. Store bulky regenerable trial weights only when needed for a diagnosed failure. Forecast retained disk from the actual run plan; no blanket assumption of hundreds of free gigabytes.

Checkpoint after each accepted merge and at a wall-time interval chosen so losing the interval is affordable, initially about 15 minutes. Save optimizer, scheduler, global step, consumed sample order, RNG states, partition, canonical tensor ownership, probe-pool cursor and source hashes. Write to a temporary directory, fsync where supported, verify hashes, then atomically rename. A resumed run must reproduce the next unbroken reference step within the established numerical tolerance. A changed map, tokenizer, data order or dependency lock requires a new run lineage, not an in-place resume.

---

## 7. Risk-ordered work packages and explicit gates

All `python -m poc ...` commands below are **proposed project commands that the agent must implement**. They are not claims that scripts already exist. Existing upstream commands are identified separately in Section 6. The sequence is dependency-aware; do not launch all experiments as an undifferentiated sweep.

### WP00 — Establish the environment, source ledger and project skeleton

**Objective / risk:** prevent incompatible software, hidden downloads, licensing errors and unbounded resource assumptions. **Entry:** this document and a fresh target laptop; no pretrained model is needed yet.

**Actions:** create the repository skeleton in Section 9; implement the CLI dispatcher, configuration validation and `doctor`; run OS discovery; resolve stable native dependencies in isolated environments; save exact versions and source revisions; snapshot the inspected model/data terms and SharVeT README license statement. Inspect the actual quantization entry points and calibration import path. Generate a tiny Llama-shaped model for operation tests. Infer dimensions dynamically in adapters instead of relying on hard-coded model-name tables.

**Files / command:** `poc/cli.py`, `poc/doctor.py`, `poc/config.py`, `protocol/assumptions.md`, `provenance/sources.jsonl`, dependency locks; `python -m poc doctor --out artifacts/doctor.json`.

**Output / tests:** machine/OS/architecture, RAM/disk bytes, dependency versions, backend availability, operation forward/backward outcomes, license/access status, and forecast limits in machine-readable JSON. A missing or unsupported field is `null` plus a reason, never a fictitious zero.

**Resources:** package installation and tiny arrays only. **Gate:** proceed when a faithful local FP32 scientific path and a plausible native deployment path exist. Resolve a compatible version or eager equivalent when a kernel fails. A licensed paper-based comparator can replace restricted code, with attribution and an implementation label. Missing paid/remote services are irrelevant; they are not permitted dependencies. If no faithful operation path exists, record a compatibility-limited outcome before downloading larger assets.

### WP01 — Verify mathematical and implementation limiting cases

**Objective / risk:** distinguish a meaningful probe from activation similarity, broken gradients or false equivalence. **Entry:** WP00's CPU path passes; repeat on MPS where supported.

**Actions:** implement the reference tests in Section 11. Add a planted-reuse decoder with eight depths, hidden width 32, FFN width 64 and two randomly initialized FFNs reused by the map `[0,1,0,0,1,1,0,1]`. Independently permute gate/up rows and matching down columns at each depth to preserve the FFN function while making raw weights look different. Keep an unpermuted canonical bank as an exact oracle. Construct a no-reuse control with independently drawn FFNs and a nondegenerate readout. Use fresh full-support inputs for validation; do not claim a language-model result from either control.

**Files / command:** `poc/synthetic.py`, `tests/test_kl.py`, `tests/test_gradients.py`, `tests/test_bank.py`, `tests/test_patching.py`; `python -m poc validate --suite component`.

**Output / tests:** numerical tolerances; identity and permutation-invariance errors; known-counterexample detection; accumulated shared gradients; no future-to-past patch leakage; exact parameter counts; zero-rank behavior; full-rank SVD reconstruction. Serialize and reload the bank, confirming owner/use counts.

**Resources:** seconds-to-minutes-scale arrays are the intended size, but measure elapsed time rather than asserting it. **Gate:** a failure to recover the *known exact* bank at identity settings is an implementation or search failure. A no-reuse control must not be falsely labeled equivalent because the probe set is empty or too narrow. Passing permits a natural-model study; it does not support general intelligence compression.

### WP02 — Load the real teacher and establish a trustworthy evaluation harness

**Objective / risk:** unsupported checkpoint semantics, teacher floor effects and invalid metrics. **Entry:** WP01 passes and model/data terms are recorded.

**Actions:** fetch only the selected 135M checkpoint/tokenizer at a resolved immutable revision. Build the native reference adapter and compare it against the original model implementation on short, then 512-token inputs. Build the local data pool and duplicate components; create exploratory/protected roles. Implement natural-text metrics, ARC continuation scoring, BLiMP pair scoring and synthetic generators. Evaluate teacher competence on development data. Check full-sequence/cached generation equivalence and save example-level diagnostics.

**Files / command:** `poc/assets.py`, `poc/data.py`, `poc/teacher.py`, `poc/eval.py`, `data/manifests/`, `provenance/assets.jsonl`; `python -m poc prepare --config configs/135m.yaml`; `python -m poc validate --suite teacher_and_harness`.

**Output / tests:** tensor census agreeing with Section 6, immutable asset hashes, split-leakage report, teacher metrics with explicit task floors, tokenizer/position test fixtures, a corrupted-model positive control and identity null control.

**Resources:** teacher weights plus streamed minibatches; measure memory and teacher throughput. **Gate:** fix semantic mismatches before compression. If the teacher cannot perform a claimed task, retain the measured task but narrow the preservation claim. If a required core dataset is unavailable, repair access or make a pre-confirmation population amendment; do not silently replace protected data with training data.

### WP03 — Establish native storage/quantization baselines and baseline feasibility

**Objective / risk:** the idea is dominated by simple quantization, or the proposed compact storage does not execute compactly. **Entry:** WP02 reference parity passes.

**Actions:** produce dense native-precision, 8-bit and representative lower-bit MLX models. Exercise GPTQ/AWQ/DWQ/dynamic-quant paths on the shared calibration manifest; no external default corpus may be loaded. Measure all bundle and load costs. Implement an identity shared bank and its canonical exporter. Run SharVeT's supported projection calculations on small dimensions, and test the dynamic-dimension/down-projection adapter described in Section 3. Record whether upstream-compatible and paper-based paths agree where they should.

**Files / command:** `poc/bundle.py`, `poc/quantize.py`, `poc/baselines/sharvet.py`, `tests/test_export.py`, `tests/test_quantization.py`; `python -m poc baselines --stage format_and_quant --config configs/135m.yaml`.

**Output / tests:** preliminary development frontier, exact bits/overhead census, clean offline load, no duplicated bank weights, quantized numerical checks, baseline compatibility/license report. Confirm that changes to bit width actually alter stored bytes and predictions as expected.

**Resources:** one conversion/calibration run at a time; measure conversion peaks. **Gate:** proceed to the mechanism test even if dense quantization is strong, but use that frontier as the practical hurdle. If compact execution cannot be preserved, the surviving claim is archive compression only. Unresolved SharVeT fidelity limits the final comparative claim and must be resolved before claiming contemporary sharing superiority.

### WP04 — Test the indispensable mechanism with single merges

**Objective / hypotheses:** H1 and the first part of H3. **Entry:** WP03's harness and actual byte accounting work; teacher has usable natural-text behavior.

**Actions:** draw a depth/separation-stratified sample from all FFN pairs, plus planted/null controls. Fit the same medoid-plus-residual trial for each. Rank each pair using weight distance, activation MSE, CKA, natural downstream KL, random donor probes and actively mined donor/student-state probes. Use identical candidate sets and separately compare equal query counts and equal total compute. Measure held-out natural damage and, separately, fresh intervention damage on development documents not used in candidate scoring.

**Files / command:** `poc/probes.py`, `poc/merge.py`, `poc/diagnostics.py`; `python -m poc diagnose --kind single_merge --config configs/pilot.yaml`.

**Output / tests:** per-pair fitting/scoring costs, observed damage, predicted rank, selected-merge regret and uncertainty under document resampling; stratified candidate coverage; false-negative rate of the cheap shortlist; qualitative failure classes. Pairwise rank correlation is a secondary descriptive metric, not a significance test treating overlapping pairs as independent samples.

**Resources:** streamed prefixes and a single trial; expand sampled pairs/probes according to uncertainty and measured cost. **Gate:** continue when active information could improve decisions at meaningful precision and cost, or when a well-defined ambiguity warrants a larger diagnostic. If ordinary downstream KL performs equally well within a narrow CI, carry it forward as the leading simpler method; do not claim causal novelty. If one-merge effects are below sensitivity, use a prespecified small joint-merge stress test or more independent documents before declaring no effect. If gradients or controls fail, return to WP01/02.

### WP05 — Build the minimum complete compiler and matched sharing baselines

**Objective / hypotheses:** H2 and H3, without a controller or knowledge store. **Entry:** WP04 gives either a plausible signal or a specific, justified joint-composition question.

**Actions:** implement Section 4's partition search, state refreshing, counterexample refitting, finite visited-partition history and restoration. Implement RRT-style CYCLE/SEQUENCE, CKA grouping, fixed sharing plus the same causal data, text-only hard examples, shared-basis/SharVeT, XMerge-derived reconstruction, counterexample-guided precision allocation and depth-pruning controls. Add a **whole-block RRT-style practical baseline** sharing attention as well as FFNs, with separate caches at each original depth and appropriately shaped depth-specific LoRA. This stronger baseline may exploit a larger storage opportunity than the FFN-only proposed method; that is allowed in the deployment track.

For the whole-block baseline, apply the same `shared_linear + B(A(x))` representation to q/k/v/o and gate/up/down; use original GQA dimensions and apply q/k transformations before RoPE. Keep small depth-specific normalization vectors and count them. Tune its bank count and ranks under the exact payload ceiling. Label it a local RRT adaptation rather than a reproduction of the original large-scale training run.

**Files / command:** `poc/compiler.py`, `poc/bank.py`, `poc/train.py`, `poc/baselines/`, `artifacts/partitions/`; `python -m poc compile --config configs/pilot.yaml`; `python -m poc fit --method METHOD --config configs/pilot.yaml`.

**Output / tests:** complete forward-capable compressed models at requested bank counts, every merge/rejection record, joint-error versus pairwise-error plots, convergence/resource traces and matched-ablation results. For SharVeT's local refinement, specify feature consistency at the final normalized hidden representation; use mean squared feature error over valid tokens, and regularize scaling vectors toward their initialization. Record this convention and compare upstream behavior on a supported fixture rather than implying an unspecified feature loss was replicated exactly.

**Resources:** measured training budget ladder; sequential trials; one active training run at a time. **Gate:** advance when a complete bank can be trained and its result is interpretable relative to tuned controls. If only local substitutions work, diagnose composition drift before increasing scale. If a conventional mapping performs as well, retain the simpler method as the practical result and classify CGOC's extra search as unsupported. Optimization still improving at the available budget means budget-conditioned evidence, not a capacity impossibility.

### WP06 — Export the complete native frontier and test end-to-end overhead

**Objective / hypothesis:** H4 and storage/runtime honesty. **Entry:** WP05 yields valid candidates and tuned baseline artifacts.

**Actions:** export every finalist to the same native runtime. Quantize canonical shared tensors with aggregated calibration; validate sharing-safe AWQ/GPTQ where implemented. Compare supported refinement paths, including dense DWQ. Allow baseline-specific ranks, pruning depth and mixed precision to use available bytes sensibly. Measure actual payload ceilings, persistent reconstruction caches, cold/warm load, prefill, decode and peak memory. Apply the same lossless packer to all payloads.

**Files / command:** `poc/export.py`, `poc/quantize.py`, `poc/bench.py`, `artifacts/frontier_dev.parquet`; `python -m poc export --run RUN_ID`; `python -m poc bench --split dev --config configs/runtime.yaml`.

**Output / tests:** feasible Pareto frontier with no extrapolated impossible byte points; logits/choice parity after export; per-canonical-tensor bit allocation; resource traces; an offline-only deployment test; development-selected finalist and best comparator under the frozen candidate operating budget.

**Resources:** conversion peaks and sequential timing blocks, calibrated in WP03. **Gate:** a quantized bank must beat the relevant development frontier enough to justify confirmation, or provide an interpretable close comparison at the feasible resolution. Do not suppress a confirmation merely because the anticipated effect is small; choose by information value and achievable precision. A slower/smaller candidate remains a tradeoff, not a failure by definition. A fully dominated candidate does not warrant architectural escalation without a diagnosed recoverable cause.

### WP07 — Freeze confirmation, sample sizes and analysis

**Objective / risk:** prevent adaptive benchmark selection and underpowered “parity” claims. **Entry:** correctness, export and baseline tuning gates pass; protected outcomes have not been accessed.

**Actions:** use pilot variance/costs to choose document counts, training replicates, timing blocks and attainable precision. Fix \(B^*\), candidate, best practical baseline and best matched-bank comparator. Freeze prompts, task family, guardrail interpretation, numerical settings, data windows, bootstrap/multiplicity analysis and allowed exclusions. Generate a signed/hash-verified freeze record. Test the analysis pipeline on artificial equal and known-difference arrays.

**Files / command:** `protocol/freeze.json`, `protocol/analysis.md`, `poc/stats.py`, `tests/test_statistics.py`; `python -m poc freeze --config configs/confirm.yaml`.

**Output / tests:** a complete machine-readable protocol with no unresolved primary metric, baseline, data-population or numerical-format field. Protected evaluation refuses to run if a dependency/hash is missing or changed.

**Resources:** pilot analysis and any prespecified fresh training replicates; no protected tuning. **Gate:** execute confirmation only after all consequential choices are frozen. An underpowered feasible design is allowed only with its limited claim stated in advance. A change to a primary endpoint after seeing test results creates a new study and requires untouched data.

### WP08 — Execute protected comparison and failure analysis

**Objective:** estimate the prespecified effects without favorable selection. **Entry:** WP07 freeze verifies successfully.

**Actions:** evaluate all frozen finalists on the same protected item IDs, sequentially if necessary. Run prespecified timing blocks. Compute exact payload costs, token-weighted natural metrics, paired intervals, capability guardrails and seed-specific results. Apply only prespecified exclusions such as corrupted input records, logging every exclusion equally for all models. Generate a regression ledger before writing any success narrative.

**Files / command:** `results/raw/`, `results/confirm_A/`, `results/regressions.parquet`; `python -m poc evaluate --split confirm_A --freeze protocol/freeze.json`; `python -m poc analyze --freeze protocol/freeze.json`.

**Output / tests:** machine-readable and Markdown result tables with absolute/relative changes, uncertainty, sample/cluster/seed counts, bytes, construction cost and runtime. All negative and inconclusive results remain visible. A test failure or export mismatch invalidates the affected run and triggers a repair lineage, not selective deletion.

**Resources:** pilot-derived fixed sample and repeat count. **Gate:** classify according to Section 10. No further tuning on `confirm_A`. An unexpected regression can motivate a new hypothesis tested on `confirm_B`; it cannot be repaired and re-reported as the same confirmation.

### WP09 — Independent confirmation and evidence-triggered scale test

**Objective / hypothesis:** H5 and reproducibility beyond one favorable checkpoint. **Entry:** an interpretable result worth confirming, or a precise new hypothesis arising from a negative/inconclusive result.

**Actions:** reproduce from the dependency lock in a clean environment; reload final payloads without training assets; repeat a selected training seed/resume test. Use `confirm_B` for a predeclared independent-source/template or method-revision check. Escalate to SmolLM2-360M only after recalibrating memory and throughput and freezing a second-scale design. Include the pretrained 135M model as a practical reference at that stage. No uncontrolled frontier-model extrapolation.

**Files / command:** `protocol/confirmation_B.json`, `configs/360m.yaml`, `results/replication/`, `provenance/rebuild.json`; `python -m poc reproduce --run RUN_ID`; conditional `python -m poc run --config configs/360m.yaml`.

**Output / tests:** reproducibility discrepancies, independent effects, scale-dependent bank/rank requirements, construction cost and factual-retention changes. Publish model-specific failures.

**Resources:** measured, not assumed from the first model's size. **Gate:** consistent results strengthen the scoped claim. Failure to transfer narrows it. An effect requiring greater teacher competence can justify a targeted stronger local teacher only after verifying that the missing competence, rather than uncontrolled tuning, is the bottleneck. Lack of larger hardware cannot be presented as evidence that a benefit exists there.

### WP10 — Synthesize, audit and package a reproducible result

**Objective:** deliver an interpretable scientific outcome, not just a checkpoint. **Entry:** any terminal gate, including failure or inconclusive outcomes.

**Actions:** assemble the source/novelty update, complete result frontier, regression ledger, methods and cost report. Trace every claim to a frozen comparison or clearly labeled exploration. Include failed trials and exclusion reasons. Rebuild the deployable bundle and verify its independence from the teacher/network. Perform the three audits in Section 11 against the actual implementation and results.

**Files / command:** `REPORT.md`, `DECISIONS.md`, `results/summary.json`, `artifacts/release_manifest.json`; `python -m poc report --include-failures`.

**Output / acceptance:** another agent can reconstruct the selected runs from source revisions, configs, manifests and locks; claimed bytes equal real bundles; scope and uncertainty are explicit. The report states either a supported improvement, a narrower finding or a diagnosed absence of evidence. It must never claim global equivalence, absence of all prior art, or successful M3 execution for a run that was not actually performed there.

---

## 8. Attribution, robustness and adversarial failure program

Run the cheap attribution tests during exploration; reserve their confirmatory versions only when the associated claim is retained. This is a dependency-driven program, not a requirement to train an exhaustive factorial grid before learning anything.

| Threat or question | Required control / measurement | Interpretation |
|---|---|---|
| Ordinary sharing explains the result. | CYCLE, SEQUENCE and CKA mappings with the same bank/rank and natural fitting. | A tie supports sharing, not the active compiler. |
| Causal distillation alone explains it. | Fixed partition receives the exact CGOC donor pool, student states and fitting allowance. | A tie removes evidence for partition refinement. |
| Counterexample refinement is useful, but sharing is not. | QCE uses the same evidence to allocate native precision under the same payload and search-cost budget. | A tie favors the simpler empirical quantization mechanism; CEG4N already establishes the broad idea. |
| Extra examples or teacher compute explain it. | Text-hard-example and random-donor baselines at matched teacher-query and construction-time budgets. | Extra information or compute must be credited rather than hidden. |
| Larger residuals carry all the capacity. | Rank-zero, rank-8 and development-selected larger-rank controls; allocate extra bytes to a simpler bank or dense precision. | A benefit that disappears after fair allocation is not improved bit efficiency. |
| Low activation MSE is mistaken for functional similarity. | Readout-nullspace and permuted-neuron controls; compare downstream KL. | Large internal differences can be irrelevant; small ones can matter. |
| Off-manifold patches drive an artificial win. | Natural edits, teacher-reachable donor vectors, filtered/unfiltered hybrids and natural-only confirmation. | Improvement only on hybrid states is a probe-specific robustness result. |
| Pairwise success does not compose. | Natural full-model error after every merge; multiple simultaneous replacements; refreshed student-state probes. | Non-additive damage requires a different partition or narrower scope. |
| The teacher's rare knowledge is lost. | ARC errors, rare-token/frequency strata, teacher-success retention and fresh entity/number variants. | Aggregate KL is not sufficient to claim quality preservation. Frequency bins come only from allowed training text. |
| Teacher errors are copied faithfully. | Ground-truth NLL and labeled-task correctness in addition to KL. | More teacher-like is not always more useful. |
| Quantization removes or reverses the advantage. | Dense learned/mixed precision and compatible shared quantization under real bytes. | BF16-only findings remain component evidence. |
| Export silently expands the model. | Inspect canonical owners, payload files and resident reconstructed tensors; offline clean-process load. | Expanded deployment cannot inherit the compact bank's byte claim. |
| Caches are incorrectly shared. | Per-depth cache identity checks; full-forward versus decode parity at several lengths. | Incorrect cache reuse is a bug, not a speedup. |
| Attention to future tokens leaks labels. | Alter future tokens and prove earlier logits unchanged; prohibit future-position donor patches. | Any failure invalidates the affected evaluation. |
| Numerical precision creates a false effect. | CPU FP64 references, FP32 scientific runs, matched export dtype, per-format null tests. | Numerical artifacts cannot establish a representational gain. |
| Search overfits its scoring pool. | Disjoint fit/score probes, untouched natural documents, independent `confirm_B`; candidate-search cost included. | More search can worsen generalization; report it. |
| One initialization or easy family creates the win. | Paired independent seeds, held-out composition templates, all declared natural domains. | A fragile result gets a narrow claim and wide uncertainty. |
| A favorable average hides a severe regression. | Per-family bounds and a regression ledger with worst observed important strata. | No aggregate-only “same abilities” claim. |
| The effect is too small to matter practically. | Absolute nats/accuracy changes, actual bytes saved, total install savings, timing and construction-cost tradeoffs. | Statistical detection alone is not a breakthrough. |
| A named architecture is already known. | Revisit precise mechanism searches before publication; compare algorithm steps, not names. | Reclassify as replication/engineering/empirical finding when appropriate. |

For a latency advantage, construction-time break-even can be reported as

\[
N_{requests}^{break-even}=\frac{C_{build,M}-C_{build,B}}{t_B-t_M},
\]

only when the denominator is positive and timing boundaries match. If the candidate is slower, there is no latency-based payback; storage feasibility or another explicitly valued dimension must justify it. This calculation is a deployment scenario, not a substitute for a real user's utility requirements.

Do not hide overhead by invoking an exact arithmetic tool or an external fact store for the candidate alone. Such changes implement the brainstorm's repair/knowledge directions and require counting code, routing, data, false activations and latency, plus an equally equipped baseline.

---

## 9. Repository, interfaces and reproducibility contract

### 9.1 Minimal repository

The following is the repository the coding agent must create. These are **proposed project files and commands**, not a claim that this implementation already exists.

```text
cgoc/
  README.md                      # installation, entry points, first unfinished gate
  pyproject.toml
  poc/
    __init__.py
    __main__.py                  # dispatch the project commands used in Section 7
    config.py                    # validated configuration and immutable run identity
    contracts.py                 # batches, boundaries, methods, metrics, manifests
    doctor.py                    # hardware/operation probes and throughput calibration
    assets.py                    # pinned downloads, hashes and license manifests
    data.py                      # article parsing, deduplication, tokenization, splits
    teacher.py                   # exact original decoder and boundary hooks
    bank.py                      # canonical shared FFNs and depth-specific residuals
    probes.py                    # donor generation, boundary replacement, hard cases
    merge.py                     # initialization and candidate scoring
    diagnostics.py               # single-merge ranking and planted controls
    compiler.py                  # partition search, repair, finite history
    train.py                     # common fitting, loss chunking, optimizer and resume
    quantize.py                  # controlled calibration, native quantization adapters
    bundle.py                    # canonical ownership, serialization and byte census
    export.py                    # PyTorch-to-native conversion and numerical checks
    evaluate.py                  # teacher fidelity, NLL and task scoring
    bench.py                     # synchronized cold/warm timing and memory records
    stats.py                     # paired cluster analysis and prespecified intervals
    report.py                    # traceable tables, frontiers and failure summaries
    baselines/
      rrt.py
      cka.py
      basis.py
      sharvet.py
      prune.py
      reconstruct.py
      precision_search.py
  configs/
    135m.yaml
    pilot.yaml
    runtime.yaml
    confirm.yaml
    360m.yaml
  protocol/
    assumptions.md
    development.md
    freeze.json
    amendments.jsonl
  provenance/
    sources.jsonl
    assets.jsonl
    hardware.json
    operations.json
    dependency-locks/
  data/
    manifests/                  # document IDs, source revisions, cluster assignments
    tokens/                     # allowed local arrays, not redistributable by default
    protected/                  # immutable manifests; no tuning access
  artifacts/
    runs/<run_id>/
      config.resolved.json
      run.json
      events.jsonl
      checkpoint-last/
      checkpoint-previous/
      bundle/
      metrics/
    partitions/
    releases/
  results/
    raw/
    exploratory/
    confirm_A/
    confirm_B/
    regressions.parquet
    summary.json
  tests/
    test_model_parity.py
    test_attention_causality.py
    test_suffix_gradients.py
    test_aliasing.py
    test_chunked_loss.py
    test_probe_isolation.py
    test_data_splits.py
    test_export.py
    test_quantization.py
    test_stats.py
    test_resume.py
  PROGRESS.md
  DECISIONS.md
  REPORT.md
```

Keep implementation small: command handlers can share modules and data classes rather than becoming a distributed experiment platform. No service, database server, cluster scheduler or external teacher API is required. A plain filesystem and one active process suffice. Do not place copyrighted datasets, credentials or whole third-party checkpoints in Git.

### 9.2 Common method contract

Use one experiment harness for every method. A method may have different architecture-specific preparation but cannot select its own test tokenizer, data, scoring or byte boundary.

```python
# Proposed interfaces; Tensor and protocol details follow Section 4.
@dataclass(frozen=True)
class Batch:
    input_ids: Tensor        # integer [B, S]
    attention_mask: Tensor   # boolean [B, S], true for real input positions
    target_mask: Tensor      # boolean [B, S-1], true for scored next-token targets
    document_ids: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    positions: Tensor       # same absolute-position convention for all methods

@dataclass(frozen=True)
class CostRecord:
    wall_seconds: float
    input_tokens: int        # token presentations, not necessarily unique tokens
    valid_target_tokens: int
    teacher_forward_calls: int
    teacher_suffix_backward_calls: int
    student_backward_calls: int
    peak_process_bytes: int | None
    peak_accelerator_bytes: int | None

class Method(Protocol):
    def prepare(self, teacher, config, calibration_manifest): ...
    def fit(self, batches, teacher, budget, recorder): ...
    def decoder(self) -> DecoderAdapter: ...
    def export(self, destination, runtime_config) -> dict: ...
    def cost_record(self) -> CostRecord: ...
```

For memory-safe training, expose final hidden states and a readout operation in addition to the convenient full-logits interface. `logits` can remain a test convenience; training and evaluation must be able to stream vocabulary projections over positions. The head, embedding and all shared operators retain a single canonical parameter owner. A readout transpose is a view, not a second learned tensor.

A `ProbeRecord` contains recipient/donor document IDs and hashes, source split, original depth, absolute positions, patch mask specification, teacher/candidate checkpoint hashes, generation seed, scorer definition, original/intervened scores, and whether the probe was used for fitting, ranking or confirmation. Store compact token/mask references rather than enormous activation dumps. A cached activation is valid only for its exact checkpoint, numerical backend and boundary convention.

### 9.3 Configuration and validation

A starting exploratory configuration is below. `null` values are **deliberately unresolved blocking fields**, not implicit defaults. WP00 resolves source revisions and operations; WP07 resolves final sample sizes, seeds and budgets. The confirmation command must reject any unresolved field that affects its claim.

```yaml
schema_version: 1
study: cgoc_smol135
phase: exploratory
teacher:
  repository: HuggingFaceTB/SmolLM2-135M
  revision: null                 # full repository revision resolved by WP00
  weights_sha256: null          # digest of actual canonical asset manifest
  tokenizer_revision: null
  trust_remote_code: false
numerics:
  scientific_backend: torch_mps
  scientific_dtype: float32
  reference_backend: cpu
  reference_dtype: float64
  deployment_backend: mlx
  loss_temperature: 1.0
  head_position_chunk: 32
model:
  preserve_original_depth: true
  share_attention: false
  initial_target_bank_count: 15
  ff_residual_rank: 8
  residual_scale: 1.0
  tied_embedding: true
training:
  context_tokens: 512
  microbatch_sequences: 1
  target_valid_tokens_per_update: 8192
  bank_learning_rate: 0.0001
  residual_learning_rate: 0.0003
  adam_betas: [0.9, 0.95]
  nll_weight: 0.1
  gradient_clip_norm: 1.0
  final_target_presentations: null
  final_training_seeds: null
search:
  seed: 17
  natural_state_weight: 0.5
  student_state_weight: 0.25
  donor_state_weight: 0.25
  teacher_query_budget: null
  total_search_seconds: null
  final_topology_policy: null
quantization:
  initial_bits: 4
  initial_group_size: 64
  validated_formats: null
  calibration_manifest: null
  preserve_canonical_sharing: true
evaluation:
  primary_context_tokens: 512
  primary_metric: kl_teacher_to_model_nats_per_token
  payload_budget_bytes: null
  protected_manifest: null
  independent_cluster_count: null
  timing_pair_count: null
  analysis_freeze_sha256: null
```

This is a reproducible starting point, not a assertion that these hyperparameters are optimal. Shared development tuning can alter them before freeze. A new field must have a type, unit, allowed range and provenance. Fail on unknown/misspelled fields rather than silently ignoring them. Resolve relative paths against the project root; never infer an asset revision from a filename.

Distinguish three clocks: unique corpus tokens, presented optimization tokens, and teacher computation. For example, replaying a hard example counts as another presentation and another teacher query when recomputed. Checkpoint recomputation adds execution cost even when it does not add a new unique example. Count padding in processed-token throughput but not in loss denominators; report both.

### 9.4 Run records and raw measurement schemas

Each run gets a deterministic identity derived from code revision, resolved configuration, source manifests and seeds, plus a unique attempt ID for retries. Never overwrite the previous attempt.

`run.json` must contain:

```json
{
  "schema_version": 1,
  "run_id": "content-derived ID",
  "attempt_id": "unique retry ID",
  "parent_run_ids": [],
  "phase": "exploratory | confirmatory | replication",
  "status": "planned | running | complete | failed | interrupted | invalidated",
  "method_id": "T | Q | R | RC | CKA | BS | SV | P | H | QCE | M",
  "method_variant": "explicit configuration label",
  "source_commit": "resolved Git commit",
  "dirty_patch_sha256": "hash, or null only for a clean tree",
  "dependency_lock_sha256": "resolved hash",
  "hardware_manifest_sha256": "resolved hash",
  "teacher_manifest_sha256": "resolved hash",
  "data_manifest_sha256": "resolved hash",
  "protocol_freeze_sha256": "required for protected evaluation",
  "seeds": {"data": 17, "initialization": 17, "search": 17, "timing_order": 17},
  "costs": {"wall_seconds": 0, "valid_target_presentations": 0},
  "output_manifest_sha256": "filled only after verified output creation",
  "failure_category": "null or a category from Section 10",
  "failure_details": "null or an actionable diagnostic"
}
```

The illustrative zeros and text labels above are schema examples, not measured results. Running code writes actual numbers and validated identifiers. Timestamped `events.jsonl` records start/end, each candidate trial, selected/rejected moves, exact bytes, optimizer state transitions, memory readings, warnings and exceptions.

Minimum raw result tables:

| Table | Required columns and units |
|---|---|
| `document_metrics` | run/artifact ID, split, source and duplicate-cluster ID, document ID, valid-target count, sum KL in nats, sum NLL in nats, top-1 disagreement count, context/window policy, dtype/backend, error status. |
| `task_predictions` | run ID, task/version, item and dependency-cluster ID, frozen prompt hash, choice or sentence scores, predicted choice, exact label, correctness, relevant denominator and tie flag. |
| `merge_trials` | parent partition hash, trial partition hash, candidate groups/depths, initialization seed, probe manifests, training presentations, teacher calls/seconds, natural/probe score, byte change, selected flag and rejection reason. |
| `timing` | artifact ID, workload/prompt hash, randomized pair/block ID, cold/warm flag, prefill/decode token counts, synchronized seconds, load boundary, process/accelerator counters, power/thermal/context notes and exclusion reason. |
| `tensor_census` | owner ID, all use sites, shape, actual element/storage dtype, quantizer/version, bits/groups, scale/offset bytes, padding bytes, tensor-file hash, reconstructed-cache bytes. |
| `analysis` | frozen contrast, effect estimate, units, interval endpoints/method, independent clusters, training seeds, timing blocks where relevant, multiplicity rule and exclusions. |

Do not retain full-vocabulary logits for every token. Per-document sufficient sums plus immutable artifacts permit the planned paired analysis. Save a small, explicitly selected debugging fixture with logits and mask details to detect implementation drift. The evaluator must retain failed items and reasons, never silently remove hard examples or replace a NaN with a good score.

### 9.5 Progress, recovery and agent authority

`PROGRESS.md` is the restart index: last successful gate; current unresolved blocking risk; active run/attempt; checkpoint hash; next allowed command; and links to the relevant decisions. `DECISIONS.md` records a concise decision, evidence, rejected alternatives, status and downstream consequences. An interrupted agent starts by reading these files and validating artifacts rather than reconstructing intent from chat memory.

The agent may choose routine non-destructive details such as module organization, logging format, batch scheduling, BLAS thread counts and equivalent memory-saving implementations after parity tests. It may not silently change the teacher, task population, primary endpoint, data split, numerical objective, baseline access, resource boundary or inference semantics. Such changes require a recorded methodological amendment, an updated dependency graph and, after protected results are seen, fresh independent confirmation.

All downloads are public permitted assets, pinned and logged. No paid API, remote accelerator or second machine may become a core dependency. Do not upload source text or model data to another service. Keep a documented list of what can be redistributed: original code/configs/manifests may be shareable, while raw web text, third-party checkpoints or restricted baseline code may need retrieval instructions rather than inclusion. License review remains asset-specific.

The final report must contain all feasible frontier points, all frozen comparisons, training curves, cost and byte ledgers, important regressions and failed-run counts. A main figure may highlight a selected point but must link to the full raw comparison. Prefer measured tables over an unexplained composite “intelligence per parameter” score.

---

## 10. Outcome interpretation and evidence-triggered continuation

### 10.1 What each result supports

| Observed outcome | Supported conclusion | Required next action |
|---|---|---|
| Export, gradients, causality, source isolation or null controls fail. | The implementation or measurement is invalid. | Repair the failing gate; invalidate descendants; reproduce the control before interpreting model quality. |
| Hardware cannot execute the required faithful operations or workload. | A local resource/compatibility limitation, not evidence against reusable computation. | Use an equivalent tested local path; otherwise stop the affected claim and preserve the diagnostic. |
| Effects are smaller than the attainable interval width, or the teacher cannot perform the claimed task. | Insufficient sensitivity or an unsuitable proxy. | Report the interval/competence limitation. Use more independent data, a justified stress test or a more capable local teacher only under a new calibrated protocol. |
| Active probes improve their own loss but not independent natural text or capability measures. | Probe-specific robustness, possibly off-manifold overfitting. | Narrow the claim; prefer natural-state scoring for the practical compressor. Do not call the probe improvement general capability preservation. |
| Sharing beats an untied size-matched model, but fixed sharing or SharVeT matches CGOC. | Reuse can help; the active compiler has no demonstrated additional value. | Report a replication/empirical result and retain the cheaper method. Do not relabel the baseline as an invention. |
| QCE or dense learned quantization wins at the same bytes. | The tested sharing strategy is not the best local deployment choice. | Keep the better conventional result; diagnose whether unchanged non-FFN weights, residual cost or kernel overhead caused the loss. |
| CGOC improves the matched-bank contrast but not the full deployment frontier. | A scoped algorithmic/empirical mechanism result, not an end-to-end advantage. | Publish component evidence with costs. Escalate only with a concrete mechanism for overcoming the identified deployment cost. |
| CGOC improves the frozen deployment contrast, but attribution does not favor active selection. | A useful artifact may exist; the proposed causal explanation is unsupported. | Credit architecture, data or optimization as supported by controls. |
| CGOC improves both frozen contrasts, with reproducible export and no disallowed guardrail regression. | A scoped new local quality–storage point with evidence for the selected mechanism. | Confirm on independent data/seeds, then another teacher scale. State observed uncertainty and finite evaluation scope. |
| Aggregate fidelity improves while a major capability degrades. | A tradeoff, not unchanged model quality. | Report the affected family prominently; do not average away the regression. |
| Gains are precise but tiny relative to byte/install/runtime or construction costs. | A detectable effect that may have little practical value. | Report absolute effects and cost; avoid “breakthrough” language. |
| The bank or residual budget grows nearly as quickly as the original depth under adequate fitting and diverse probes. | Evidence against strong reuse in the tested regime. | Quantify the scaling and retention curve. Do not infer that all architectures or scales share the same limitation. |

“No disallowed guardrail regression” requires a declared application tolerance or the zero-loss standard in Section 5; absence of statistical significance is not enough. When no useful tolerance was supplied, report simultaneous bounds and avoid categorical “same qualities” claims. Even excellent finite-benchmark retention cannot establish equality on every prompt, language, rare fact, context length or future distribution.

There is no arbitrary percentage required to recognize a genuine improvement. However, a small statistically detected gain, a conditional fixed-checkpoint gain, a Pareto tradeoff and a practically important deployment gain are different conclusions. State which one the evidence supports.

### 10.2 Continue only for a reason tied to the measured failure mode

**If ordinary natural downstream KL is sufficient:** retain the same reusable-operator architecture but remove active donor mining and split repair. This is an evidence-backed simplification, not a failure to pursue novelty. The useful finding may be that expensive causal probing offers no extra information at this scale.

**If parameter sharing works but residual storage dominates:** compare fixed versus adaptive residual ranks, and allocate the same extra bytes to additional bank operators or native precision. A learned adaptive allocation is a fresh controlled factor, not automatically a new invention. Do not discard the strong fixed-rank baseline.

**If whole-block RRT or depth reconstruction is superior:** choose the better practical alternative and report why. A subsequent attention-sharing study must handle per-depth KV state, RoPE and attention nonlinearity directly. FFN evidence alone does not justify merging attention caches or assuming that all computational primitives are reusable.

**If a small number of repeatable failure mechanisms accounts for the residual gap:** a separately registered *Executable Disagreement Atlas* experiment becomes justified. Use a fixed byte budget for guarded corrections, compare against spending those bytes on ordinary residual weights/precision, measure harmful false activations, and test new error instances. Do not hand-code benchmark answers, use an uncounted tool, or train on confirmation failures and reuse the same confirmation.

**If factual retention, rather than compositional computation, limits compression:** investigate an external-memory model only as a new whole-system comparison against LMLM and ordinary retrieval baselines. Count the index, database, embedding model, lookup code and latency. Existing external-knowledge work prevents a novelty claim for simply moving facts outside weights. [R18]

**If two natural-language scales support the mechanism:** the next justified question is whether bank size, residual rank and probe cost grow sublinearly in independently stored teacher computation at matched retention. Two points cannot establish an asymptotic law. Add scales or architectures only as measured resources allow, recording pretrained-data differences. A local success can motivate larger external research, but the completed laptop PoC remains explicitly limited to what it ran.

**If only planted synthetic systems work:** report recovery of a known compact program as a sanity check. A synthetic architecture can be designed to have low description length by construction; that does not demonstrate hidden redundancy in a pretrained natural-language teacher. Preserve the natural-model negative result rather than replacing it with the easier task.

### 10.3 Stopping without overinterpreting failure

A terminal decision must identify its category: invalid implementation, unsuitable proxy, inadequate sensitivity, local resource limitation, evidence against the tested mechanism, or useful mechanism without competitive benefit. “No win” is not a sufficiently informative label.

Stop expanding the architecture when additional complexity lacks a specific prediction that distinguishes it from the strongest simpler explanation. Stop a search when its prespecified budget or finite neighborhood is exhausted; do not keep inspecting protected results until an attractive variant appears. Longer training remains legitimate when development curves and resource calibration justify it, but any conclusion is conditional on the adaptation and data actually provided.

The original 10–100× ambition is not a required milestone for this component study. It remains unverified and would require a different storage floor, additional mechanisms and broader quality evidence. It must never be reported by multiplying nominal bit-width savings by theoretical layer-sharing ratios without evaluating the actual combined model.

---

## 11. Audits, checks performed and residual uncertainty

These are internal technical audits of this plan, not independent peer review. They identify specific defects and revisions. They do not certify novelty, guarantee a win, or imply that the proposed experiments have run.

### 11.1 Scientific and prior-art audit

| Defect or challenge found | Revision made | Residual risk / acceptance criterion |
|---|---|---|
| “Reuse computations rather than weights” obscured extensive recursive, basis-sharing and conditional-parameter prior art. | Selected a specific post-training FFN-bank compiler; added matched RRT, CKA, shared-basis and SharVeT controls. | A positive artifact still needs the attribution comparison; architecture renaming is not novelty. |
| Activation interventions as distillation supervision already exist. | Fixed sharing receives the same causal examples and student-state exposure. | Any advantage could still arise from optimization; common data and counted expenditure remain necessary. |
| Counterexample-guided compression and neural merge/refine existed before this proposal. | Added CEG4N, neural abstraction/refinement, robust repair and FIPE; downgraded methodological novelty to unresolved; added QCE. | The exact compiler may be a new application rather than a new method. Publication must compare algorithm steps directly. |
| A current depth-compression alternative was missing. | Added the contemporary reconstruction control and explicitly recorded its local deviations. | Its original reported result is not a locally reproduced result; an implementation fidelity audit remains required. |
| A complete depth-conditioned nonlinear FFN bank was not uniquely new. | Included reViT in the novelty boundary and excluded claims based solely on the bank architecture. | Vision-to-language adaptation can be useful but does not establish a new computational primitive. |
| The unchanged non-FFN portion makes extreme equal-precision compression impossible in the initial representation. | Calculated the actual storage floor and separated component headroom from the 10–100× ambition. | Compression of attention/embeddings needs additional hypotheses and cannot be inferred from this PoC. |
| Finite probes were being described too much like an equivalence oracle. | Used sampled fidelity language, preserved population limits and distinguished classifications from probability distributions. | No formal guarantee over all prompts or continuations exists here. |
| Successful final answers can hide shortcut computation. | Added complete-model fidelity, composition controls and intermediate-state diagnostics without treating them as universal circuit proof. | The learned FFNs may remain entangled and non-interpretable. |

### 11.2 Experimental-validity audit

| Defect or alternative explanation | Revision made | Required implementation check |
|---|---|---|
| Comparing a shared BF16 candidate only with a dense BF16 model would overstate value. | Added native learned/mixed quantization, whole-block sharing and depth reduction at actual payload budgets. | Development selects the strongest eligible baseline, not the easiest one to beat. |
| The extra probe-generation and fitting work could explain any gain. | Added same-data, same-query and matched-construction-time contrasts; retained a separate practical unequal-cost frontier. | Log every teacher call, rejected trial, conversion and fitting presentation. |
| A reversed KL argument could silently change the objective. | Wrote the teacher distribution explicitly on the left in equations and schema names. | Asymmetric two-distribution fixture distinguishes both orientations. |
| Document/token dependence or seed pseudo-replication could create misleading significance. | Clustered document analysis, paired item joins, seed-specific results and restricted procedure-level claims. | Synthetic null simulation and known-offset tests cover the exact estimator, not only a generic t-test. |
| A zero-difference significance test cannot establish unchanged quality. | Prespecified guardrail bounds; no invented tolerances; finite-population versus broader-population interpretation. | Report interval endpoints and sample/seed counts even when a p-value is not significant. |
| Search and evaluation could share documents, near duplicates or benchmark-derived repairs. | Immutable deduplicated cluster splits, fit/score separation, confirmation freeze and independent second confirmation. | Protected evaluator refuses missing or changed hashes. |
| Patch artifacts might reward impossible states rather than language performance. | Natural edits and natural-text confirmation; phase/context-aware controls; no claim of an on-manifold guarantee. | Report intervention-only gains separately from real text retention. |
| Scalar activation similarity could conceal harmful joint replacements. | Full-model scoring after each merge and state refreshing; split/repair is budget-neutral and separately ablated. | Planted nontransitivity and combined-replacement fixtures must expose failure. |
| Benchmark contamination cannot be ruled out for a pretrained model. | Disclosed it; synthetic held-out templates supplement, not replace, natural evaluation. | Do not call public benchmark scores contamination-free or evidence of newly acquired knowledge. |
| Fixed near-chance teacher performance could create apparently perfect retention. | Report complete teacher competence and permit only prespecified secondary teacher-success subsets. | A high-level task is not evidence for a capability the teacher cannot demonstrate. |

### 11.3 Implementation and feasibility audit

| Defect found during the executable walk-through | Revision made | Blocking gate or residual risk |
|---|---|---|
| M3 Max does not imply a known memory configuration or compatible operation path. | Hardware discovery and real forward/backward/export probes precede dependent runs. | No M3 measurement was possible in this planning environment. |
| Freezing a suffix could accidentally disable the input gradient. | Distinguished frozen parameters from no-grad execution; included a nonzero-gradient fixture. | The actual attention/RMSNorm/MLP suffix must pass on MPS. |
| Chunking vocabulary projections could give wrong ragged weighting or lose tied-embedding gradients. | Exact full-vocabulary position chunks and the two-stage gradient pattern below. | Actual MPS implementation must match the unchunked fixture within measured tolerance. |
| Reference pseudocode retained every trial model, contradicting the memory plan. | Replaced live trial lists by metric records and one best CPU delta; release graphs between trials. | Measure memory returning to baseline across many rejected trials. |
| The initial high-level compiler omitted an executable split rule. | Specified worst-member split plus compensating merge, finite visited partitions and a separately frozen repair policy. | It is heuristic and can miss better partitions; record the miss rather than asserting optimality. |
| Checkpoint history could lag the accepted state. | Append trial/history records before atomic checkpoint; include visited partitions and RNG/data cursors. | Interrupted and uninterrupted next-step results must agree under the established numerical convention. |
| Generic serialization or per-depth quantization could expand or change shared weights. | Canonical owners, actual byte census, sharing-safe transforms and clean-process load tests. | Some optimized formats may remain unsupported for shared execution. Report this as a limitation. |
| Default quantization calibration could introduce hidden external data. | Replace the loader boundary with a pinned project manifest and isolate caches. | Verify that no second corpus is fetched in the actual installed revision. |
| The strongest named sharing code has restrictive terms and model-specific assumptions. | Explicit license gate, shape adapter and disclosed upstream-compatible versus adapted variants. | Do not claim unmodified official replication or commercial permission without evidence. |
| Runtime estimates could be mistaken for M3 benchmarks. | Published only analytical sizes and illustrative throughput arithmetic; all heavy stages depend on local calibration. | Thermal limits, implementation efficiency and adaptation cost remain unknown. |

### 11.4 Small checks actually performed during planning

**Environment:** Linux `6.18.44`, x86-64, glibc 2.41; Python 3.13.5; PyTorch `2.10.0+cpu`; NumPy 2.3.5; one CPU thread. These are the planning environment's versions, **not recommended or validated M3 dependency pins**. No pretrained checkpoint or corpus was downloaded, no LLM inference/training was performed, and no Apple accelerator was available.

The model counts in Section 6 were calculated from the inspected configuration fields. In column notation, with hidden width `d`, intermediate width `m`, depth `L`, vocabulary `V`, query heads `Hq`, KV heads `Hkv`, and head width `dh=d/Hq`:

```text
FFN_per_depth = 3*d*m
attention_per_depth = 2*d*d + 2*d*Hkv*dh
tied_embedding = V*d
normalizations = (2*L + 1)*d
teacher_parameters = L*(FFN_per_depth + attention_per_depth)
                     + tied_embedding + normalizations
bank_parameters = non_FFN_parameters + K*FFN_per_depth + 3*L*r*(d+m)
```

The original teacher count has no residual adapters. An identity shared-bank fixture with `K=L` and explicitly allocated rank-`r` residuals has additional stored adapter parameters even if their output is initially zero; do not use the original-teacher byte count for that fixture. Adapter-free identity and parameter-matched rank-`r` controls are different artifacts.

Measured toy checks, in FP64:

| Check | Observed result | What it does and does not show |
|---|---:|---|
| Teacher and candidate on a restricted observational support | KL `0.0` nats | Ordinary observations can miss a relevant coordinate in this constructed example. |
| Set the omitted coordinate to one | KL `0.27813327305224816` nats | An intervention can reveal a designed discrepancy; not evidence that it helps an LLM. |
| Differentiate that discrepancy through a frozen readout | Candidate-gain gradient `-0.5752072453438606` | Frozen suffix weights can preserve an input gradient in this CPU example. |
| Large change in a readout-null coordinate | State MSE `5000.0`, output KL `0.0` | Internal reconstruction error need not equal behavioral error. |
| Reuse one scalar parameter twice | Gradient `8.0` from coefficients 3 and 5 | Shared uses accumulate gradients. This does not test a full model's serializer. |
| Approximate pairwise relation | Distances `0.75`, `0.75`, `1.5` | At tolerance one, pairwise closeness is not transitive. |
| Full versus ragged position-chunked KL | Absolute difference `0.0` | The weighted reduction matches on the constructed random logits. |
| Two-stage chunking with a tied input/output embedding | Maximum embedding-gradient difference `6.938893903907228e-18`; upstream difference `0.0` | The gradient pattern below matches the full loss in a small CPU fixture, including tied-weight use. |

A self-contained reference for the first set of checks follows. Save it as a proposed `tests/reference_math_checks.py`; it requires PyTorch and no data download. These numerical values may differ in their last digits on another build; tolerance assertions, not exact decimal strings, define correctness.

```python
import torch

torch.manual_seed(17)
torch.set_num_threads(1)
dtype = torch.float64

def kl(a, b):
    la = torch.log_softmax(a, dim=-1)
    lb = torch.log_softmax(b, dim=-1)
    return (la.exp() * (la - lb)).sum(dim=-1).mean()

x = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]], dtype=dtype)
gain = torch.tensor(0., dtype=dtype, requires_grad=True)
frozen_readout = torch.tensor([[1., 1.], [-1., -1.]], dtype=dtype)

def output(z, g):
    state = torch.stack((z[:, 0], g * z[:, 1]), dim=1)
    return state @ frozen_readout.T

observed = kl(output(x, 1.), output(x, gain))
patched = x.clone()
patched[:, 1] = 1.
intervened = kl(output(patched, 1.), output(patched, gain))
intervened.backward()
assert abs(observed.item()) < 1e-12
assert intervened.item() > 0 and abs(gain.grad.item()) > 0

y = torch.tensor([[.25, 0.]], dtype=dtype)
y2 = y + torch.tensor([[0., 100.]], dtype=dtype)
readout_null = torch.tensor([[1., 0.], [-1., 0.]], dtype=dtype)
assert abs(kl(y @ readout_null.T, y2 @ readout_null.T).item()) < 1e-12
assert ((y-y2)**2).mean().item() == 5000.

w = torch.nn.Parameter(torch.tensor(2., dtype=dtype))
(3*w + 5*w).backward()
assert w.grad.item() == 8.
assert abs(0.-.75) <= 1 and abs(.75-1.5) <= 1 and abs(0.-1.5) > 1

a = torch.randn(11, 7, dtype=dtype)
b = torch.randn(11, 7, dtype=dtype)
chunked = sum(kl(a[i:i+4], b[i:i+4]) * len(a[i:i+4])
              for i in range(0, 11, 4)) / 11
assert abs(kl(a, b).item() - chunked.item()) < 1e-12
print({"observed_kl": observed.item(), "intervened_kl": intervened.item(),
       "frozen_suffix_input_gradient": gain.grad.item()})
```

For exact memory-safe gradient accumulation through a trainable tied readout, the separately checked pattern is:

```text
H = student.final_hidden(batch)          # keep the upstream graph once
Z = H.detach().requires_grad_(True)      # a leaf for accumulating readout VJPs
N = total_valid_target_count(batch)
for positions in disjoint_position_chunks:
    teacher_logits = teacher.readout(teacher_hidden[positions]) under no_grad
    student_logits = student.readout(Z[positions])
    loss = masked_KL_sum(teacher_logits, student_logits) / N
    loss += alpha * masked_NLL_sum(student_logits, targets[positions]) / N
    loss.backward()                     # accumulates Z.grad and head parameter grads
H.backward(Z.grad)                      # adds the upstream parameter contributions
```

Call `optimizer.zero_grad()` once before this whole sequence, not between chunks. Use the same canonical embedding parameter in the readout and the upstream model; its contributions sum. Skip a batch with zero valid targets with an explicit record rather than dividing by zero. This reference establishes the computational pattern; actual dropout, recomputation, mixed precision, multiple losses and MPS operation paths still require parity tests. The CPU test used a small nonlinear hidden transformation, ragged chunks, a nontrivial target mask and a tied embedding, not the full Transformer.

### 11.5 Remaining blind spots and blocking unknowns

The central empirical unknown is whether there is any useful operator reuse beyond existing methods at the chosen budgets. There is no compression-quality result yet. The method could improve probe sensitivity without improving decisions, or improve decisions while losing to quantization. Whole-model capability equality is not established by this plan and cannot be inferred from its toy calculations.

The actual laptop's RAM, free disk, macOS, compatible package revisions, critical kernel paths, long-run thermal behavior and training throughput remain unknown. These are early gates, not assumptions hidden in the resource estimates. Downloaded model/data revisions and full license texts also need to be recorded on the execution machine; web inspection alone does not install or validate them.

Exact methodological novelty remains unresolved after substantial overlap was found. A future equivalent prior method would change the novelty classification, not invalidate an honestly controlled empirical comparison. The patent search was limited and is not a legal opinion. A commercial deployment must independently resolve licenses and permissions.

Uncertainty about rare knowledge, multilingual abilities, instruction behavior, safety behavior and much larger models remains outside the core evidence. Preserving a base teacher on the selected tasks is a deliberately scoped test of the original goal, not a silent replacement for that goal.

---

## 12. References and search-coverage record

### 12.1 Evidence conventions

References below are ordinary Markdown links and remain usable without this conversation. Research was conducted on **14 October 2026**. A paper's publication date is not an asset revision: the agent must record the exact checkpoint, repository and dataset revisions actually used. “Inspected” means the stated text, method, documentation or code was read through the research tools; it does not mean its experiments or implementation were executed.

The preceding sections distinguish source findings, analytical deductions, the small checks actually run, proposed experiments and unresolved local prerequisites. Contemporary preprints are relevant competitors, not automatically validated results. The source-specific scope and limitations in Section 3 remain part of each reference's interpretation.

### 12.2 Primary references

**R01 — Recursive sharing.** Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim and Tal Schuster. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA*. First posted 28 October 2024; ICLR 2025. [Stable record](https://arxiv.org/abs/2410.20672); [inspected HTML version](https://arxiv.org/html/2410.20672v3). Inspection: initialization, sharing mappings, residual adaptation and experimental training setting. Informs R and the architectural novelty boundary; no local reproduction was performed.

**R02 — Shared bases.** Jingcun Wang, Yu-Guang Chen, Ing-Chao Lin, Bing Li and Grace Li Zhang. *Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression*. First posted 2 October 2024; ICLR 2025. [Record](https://arxiv.org/abs/2410.03765); [method text](https://arxiv.org/html/2410.03765v1); [author repository](https://github.com/TUDa-HWAI/Basis_Sharing). Inspection: paper method and repository documentation. Informs the local shared-basis control; the local whitening implementation is explicitly specified rather than assumed identical to an unchecked code path.

**R03 — SharVeT paper.** Jeongin Yun et al. *SharVeT: Similarity-aware Parameter Sharing with Vector-based Tuning for Efficient LLM Compression*. ACL 2026. DOI: [10.18653/v1/2026.acl-long.2213](https://doi.org/10.18653/v1/2026.acl-long.2213). [ACL record](https://aclanthology.org/2026.acl-long.2213/); [paper](https://aclanthology.org/2026.acl-long.2213.pdf). Inspection: methods, equations and rendered screenshots of PDF pages 3–5. Informs the closest similarity-aware baseline; numerical headlines were not used as local targets.

**R04 — SharVeT implementation and declared terms.** SNU Data Mining Laboratory / SharVeT authors. [Repository](https://github.com/snudatalab/SharVeT), [similarity implementation](https://raw.githubusercontent.com/snudatalab/SharVeT/main/src/similarity.py), [model factory](https://raw.githubusercontent.com/snudatalab/SharVeT/main/src/model_factory.py), accessed 14 October 2026. Inspection: README and both source files. README declares [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). Direct guessed raw license-file paths were unavailable; the execution gate must preserve the actual license text and applicable permission. No repository code was run.

**R05 — Causal distillation.** Zhengxuan Wu et al. *Causal Distillation for Language Models*. NAACL 2022. DOI: [10.18653/v1/2022.naacl-main.318](https://doi.org/10.18653/v1/2022.naacl-main.318). [ACL record](https://aclanthology.org/2022.naacl-main.318/); [preprint/method](https://arxiv.org/html/2112.02505); [author code](https://github.com/frankaging/Causal-Distill). Inspection: intervention-distillation method. Informs RC and invalidates novelty for intervention supervision alone; no BERT-to-decoder equivalence is presumed.

**R06 — Depth-conditioned complete FFNs.** Adrian Bulat, Yassine Ouali and Georgios Tzimiropoulos. *One Block, Multiple Depths: Recurrent Vision Transformers with Depth-Programmed Experts*. 8 October 2026. [arXiv:2610.12448](https://arxiv.org/abs/2610.12448); [inspected version](https://arxiv.org/html/2610.12448v1). Inspection: architecture and compact versus materialized execution. This is a vision-model preprint; it is architectural prior art, not an LLM result or an M3 benchmark.

**R07 — Historical abstraction and counterexample oracles.** Edmund Clarke, Orna Grumberg, Somesh Jha, Yuan Lu and Helmut Veith. *Counterexample-Guided Abstraction Refinement*. CAV 2000. DOI: [10.1007/10722167_15](https://doi.org/10.1007/10722167_15). Dana Angluin. *Learning Regular Sets from Queries and Counterexamples*. 1987. DOI: [10.1016/0890-5401(87)90052-6](https://doi.org/10.1016/0890-5401(87)90052-6). Inspection: primary bibliographic/abstract material; the Angluin publisher full text was not accessible. Used for conceptual lineage and oracle limitations, not an uninspected proof or an LLM guarantee.

**R08 — Primary teacher asset.** Hugging Face SmolLM team. *SmolLM2-135M*, model card and configuration, accessed 14 October 2026. [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M); [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json). Inspection: architecture fields and declared Apache-2.0 license. The exact full repository revision and file hashes were not downloaded or resolved into a local asset manifest during planning.

**R09 — Scale-confirmation asset.** Hugging Face SmolLM team. *SmolLM2-360M*, model card and configuration, accessed 14 October 2026. [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M); [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json). Inspection and limitations as for R08. Supports architectural arithmetic, not measured throughput or retained quality.

**R10 — Foundational architectural alternatives.** Mostafa Dehghani et al., *Universal Transformers*, 2018 / ICLR 2019, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819). Zhenzhong Lan et al., *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*, 2019 / ICLR 2020, [arXiv:1909.11942](https://arxiv.org/abs/1909.11942). David Ha, Andrew Dai and Quoc V. Le, *HyperNetworks*, 2016 / ICLR 2017, [arXiv:1609.09106](https://arxiv.org/abs/1609.09106). Inspection: primary abstracts. Establish recurrence/sharing/weight-generation lineage; detailed original implementations are not claimed reproduced.

**R11 — Learned tying.** Tamir David-Hay and Lior Wolf. *Dynamic Layer Tying for Parameter-Efficient Transformers*. ICLR 2024. [arXiv:2401.12819](https://arxiv.org/abs/2401.12819); [inspected method](https://arxiv.org/html/2401.12819). Informs the novelty boundary for learning the sharing assignment rather than fixing it manually.

**R12 — Dynamic recursive depth.** Sangmin Bae et al. *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*. 2025. [arXiv:2507.10524](https://arxiv.org/abs/2507.10524); [method](https://arxiv.org/html/2507.10524); [author repository](https://github.com/raymin0223/mixture_of_recursions). Inspection: paper and repository documentation. Informs the deferred controller alternative; its pretraining/runtime results are nonlocal reference evidence.

**R13 — Causal and output-space checkpoint merging.** Fanshuang Kong, Richong Zhang and Ziqiao Wang. *Activated Parameter Locating via Causal Intervention for Model Merging*, August 2024, [arXiv:2408.09485](https://arxiv.org/abs/2408.09485), [method](https://arxiv.org/html/2408.09485v1). Bethan Evans, Benjamin Etheridge, Stephen Roberts and Jared Tanner. *Model Merging by Output-Space Projection*, first posted May 2026, [arXiv:2605.29101](https://arxiv.org/abs/2605.29101), [inspected HTML](https://arxiv.org/html/2605.29101v1). Inspection: primary methods. These address related-checkpoint merging, not the same depth-sharing problem, but prevent a broad first-ever causal/output-space merging claim.

**R14 — Strong numerical compression.** Elias Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022 / ICLR 2023, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323). Ji Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, first posted 2023, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978). Inspection: primary records; operational details additionally checked in R15. The actual native implementation, calibration and serialized bytes define the local baseline.

**R15 — Native learned quantization and calibration implementation.** Apple / MLX contributors. [MLX-LM repository](https://github.com/ml-explore/mlx-lm), [learned-quantization documentation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/LEARNED_QUANTS.md), [AWQ implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/awq.py), [GPTQ implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/gptq.py), [calibration loader](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/utils.py), accessed 14 October 2026. Inspection: documentation and source. Informs native baseline inclusion, CLI discovery, sharing-safe adaptation and data isolation. Main-branch content is not a dependency lock; actual operations were not executed on Apple hardware.

**R16 — Native low-bit training reference.** Microsoft Research / BitNet team. *The BitNet b1.58 2B4T Technical Report*. April 2025. [arXiv:2504.12285](https://arxiv.org/abs/2504.12285); [technical report](https://arxiv.org/html/2504.12285); [model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T). Inspection: primary report/card. Different training and asset; not a matched post-hoc compression control.

**R17 — Lossless numerical entropy.** Hongshi Tan, Yao Chen, Gustavo Alonso, Weng-Fai Wong and Bingsheng He. *Approaching Shannon Bound with Lossless LLM Weight Compression*. 14 June 2026; record states accepted to ISCA 2026. [arXiv:2606.15789](https://arxiv.org/abs/2606.15789); [inspected report](https://arxiv.org/html/2606.15789v1). Inspection: report and implementation setting. Informs separate archive/resident accounting and the absence of a verified Apple implementation; no reported speedup is imported into this PoC.

**R18 — Externalized knowledge.** Linxi Zhao et al. *Pre-training Limited Memory Language Models with Internal and External Knowledge*. First posted May 2025; revised October 2025; ICLR 2026 context. [arXiv:2505.15962](https://arxiv.org/abs/2505.15962); [method](https://arxiv.org/html/2505.15962); [author repository](https://github.com/kilian-group/LMLM). Follow-on: Feldman et al., *Co-LMLM: Continuous-Query Limited Memory Language Models*, July 2026, [arXiv:2607.07707](https://arxiv.org/abs/2607.07707). Inspection: original method/repository and follow-on primary text. Informs a conditional pivot, not a dependency of the initial experiment.

**R19 — Algorithm extraction.** Eric J. Michaud et al. *Opening the AI Black Box: Distilling Machine-Learned Algorithms into Code*. Entropy 26(12), 1046, December 2024. DOI: [10.3390/e26121046](https://doi.org/10.3390/e26121046). Earlier title: *Opening the AI black box: program synthesis via mechanistic interpretability*, [arXiv:2402.05110](https://arxiv.org/abs/2402.05110). [Primary indexed record](https://pubmed.ncbi.nlm.nih.gov/39766675/). Inspection: primary indexed summary/preprint abstract; later PMC access required a browser challenge and the publisher full-text fetch failed. Supports a limited small-algorithm precedent only.

**R20 — Compression and reasoning/knowledge retention.** Nan Zhang, Eugene Kwek, Yusen Zhang, Ngoc-Hieu Nguyen, Prasenjit Mitra and Rui Zhang. *When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models*. First posted April 2025; inspected revision 2 March 2026; ICLR 2026. [arXiv:2504.02010](https://arxiv.org/abs/2504.02010); [method/report](https://arxiv.org/html/2504.02010v3). Inspection: primary text and model/task scope. Informs separate knowledge and reasoning diagnostics, not a universal law of what parameters store.

**R21 — Limited patent search.** United States patent record **US10628683B2**, *System and method for CNN layer sharing*, priority 5 December 2016; issued 21 April 2020. [Patent record](https://patents.google.com/patent/US10628683B2/en). Inspection: bibliographic/abstract and broad sharing material. Used only to document that patent prior art exists; no claim chart, legal status certification or freedom-to-operate opinion was performed.

**R22 — Native decoder semantics.** Apple / MLX-LM contributors. [Llama implementation](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/llama.py), accessed 14 October 2026. Inspection: gated MLP, attention dimensions, rotary/cache handling and tied output path. Informs the native adapter; compatibility still requires actual asset/operation tests.

**R23 — Quantized module contract.** Apple / MLX contributors. [`mlx.nn.QuantizedLinear` documentation](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html), accessed 14 October 2026. Inspection: constructor, conversion and frozen-parameter semantics. Informs the warning that default packed quantized weights are not automatically a trainable DWQ path.

**R24 — Natural-text corpus.** Hugging Face FineWeb team. [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), accessed 14 October 2026. Inspection: available sampled configurations, streaming guidance, provenance and declared ODC-By-1.0/Common Crawl terms. The plan uses a pinned permitted stream subset, not a full-corpus download; exact source records must be retained locally.

**R25 — Language-modeling test corpus.** Salesforce / WikiText maintainers. [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext), accessed 14 October 2026. Inspection: raw configurations, official splits and declared CC-BY-SA-4.0 metadata. Informs the article-based test and provenance gate; do not substitute a differently processed dataset silently.

**R26 — Science-question benchmark.** Allen Institute for AI. [AI2 ARC dataset card](https://huggingface.co/datasets/allenai/ai2_arc), accessed 14 October 2026. Inspection: Easy/Challenge configurations, splits, fields and declared CC-BY-SA-4.0 metadata. Informs the answer-text evaluation; model pretraining contamination is not ruled out.

**R27 — Syntactic discrimination.** Alex Warstadt et al. *BLiMP: The Benchmark of Linguistic Minimal Pairs for English*. TACL 2020. DOI: [10.1162/tacl_a_00321](https://doi.org/10.1162/tacl_a_00321). [Original repository](https://github.com/alexwarstadt/blimp), accessed 14 October 2026. Inspection: original README, benchmark organization and linked CC-BY license statement. Resolve the complete linked license version and data revision before redistribution.

**R28 — Apple acceleration prerequisites.** Apple. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/). PyTorch contributors. [MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html). Accessed 14 October 2026. Inspection: prerequisites and MPS availability checks. The retrieved Apple page and redirected PyTorch documentation exposed different version labels; these are not proof of a mutually tested dependency set. The actual installed combination must pass WP00–WP02.

**R29 — Counterexample-guided neural quantization.** João Batista P. Matos Jr., Iury Bessa, Edoardo Manino, Xidan Song and Lucas C. Cordeiro. *CEG4N: Counter-Example Guided Neural Network Quantization Refinement*. July 2022. [arXiv:2207.04231](https://arxiv.org/abs/2207.04231); [inspected method](https://arxiv.org/html/2207.04231); [authors' archived artifacts](https://zenodo.org/record/6791964). Extended journal article also includes Eddie B. de Lima Filho: *Counterexample Guided Neural Network Quantization Refinement*, online November 2023 / IEEE TCAD 43(4), 2024, DOI [10.1109/TCAD.2023.3335313](https://doi.org/10.1109/TCAD.2023.3335313). [University author record](https://research.manchester.ac.uk/en/publications/counterexample-guided-neural-network-quantization-refinement/). Inspection: 2022 definitions/method and journal metadata/abstract. IEEE full PDF retrieval failed. The precedent includes top-1 equivalence on specified domains, not a demonstrated LLM probability-equivalence certificate.

**R30 — Neural abstraction/refinement and repair.** Yizhak Yisrael Elboher, Justin Gottschlich and Guy Katz. *An Abstraction-Based Framework for Neural Network Verification*. CAV 2020; preprint October 2019. DOI [10.1007/978-3-030-53288-8_3](https://doi.org/10.1007/978-3-030-53288-8_3); [inspected method](https://arxiv.org/html/1910.14574). David Boetius, Stefan Leue and Tobias Sutter. *A Robust Optimisation Perspective on Counterexample-Guided Repair of Neural Networks*. 2023. [arXiv:2301.11342](https://arxiv.org/abs/2301.11342); [inspected version](https://arxiv.org/html/2301.11342v2). Inspection: merge/refine and repair/termination discussions. Informs the inherited algorithmic pattern and the absence of a general convergence claim here.

**R31 — Faithful tree pruning.** Youssouf Emine, Alexandre Forel, Idriss Malek and Thibaut Vidal. *Free Lunch in the Forest: Functionally-Identical Pruning of Boosted Tree Ensembles*. First posted 2024; AAAI 2025. [arXiv:2408.16167](https://arxiv.org/abs/2408.16167); [inspected method](https://arxiv.org/html/2408.16167). Inspection: classification-equivalence definition, pruner/separation-oracle procedure and tree-specific guarantee. Prevents treating the general “smallest functionally equivalent model” framing as unprecedented.

**R32 — Distribution-limited faithful pruning.** Haruki Yajima and Yusuke Matsui. *PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence*. 2026. [arXiv:2605.28068](https://arxiv.org/abs/2605.28068); [inspected primary text](https://arxiv.org/html/2605.28068v2). Inspection: framing and method organization. Adjacent evidence that distribution-limited fidelity is itself an established target; its conformal/tree assumptions are not implemented or claimed for CGOC.

**R33 — Contemporary depth reconstruction.** Jundong Hu and Shekar Ramachandran. *XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression*. September 2026. [arXiv:2609.02083](https://arxiv.org/abs/2609.02083); [inspected method and appendix](https://arxiv.org/html/2609.02083v1). Informs P; local adaptation and unverified reproduction are explicit in Section 5.3.

**R34 — Reuse versus faithful execution.** Jia Liang, Xi Jin and Liangming Pan. *Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers*. 27 September 2026. [arXiv:2609.33144](https://arxiv.org/abs/2609.33144); [inspected method/analysis](https://arxiv.org/html/2609.33144v1). Inspection: abstract and causal/length-generalization analysis. Informs phase-sensitive interventions and the warning against inferring faithful algorithms from final-answer success.

### 12.3 Search coverage and what changed

The search followed mechanisms rather than only the brainstorm's names. The most consequential searches and traces were:

| Search family / representative terms | Backward or forward trace | Consequence |
|---|---|---|
| `recursive transformer layer tying LoRA`, `shared nonlinear FFN bank`, `depth-programmed experts` | Universal Transformers/ALBERT/HyperNetworks → RRT/dynamic tying → Mixture-of-Recursions/reViT. | Removed architecture-only novelty claims; retained strong fixed and whole-block sharing controls. |
| `cross-layer basis sharing`, `functional similarity compression CKA`, `vector tuning` | Basis Sharing → SharVeT paper → similarity/model-factory code. | Added contemporary structural/functional grouping and discovered local adapter/license risks. |
| `causal distillation interchange intervention`, `causal model merging`, `activation patching layer compression` | DIITO → checkpoint-merging methods → XMerge method/appendix. | Separated intervention training from topology selection and added a current depth alternative. |
| `counterexample guided neural compression quantization`, `functionally identical pruning`, `neuron merge split abstraction` | CEGAR/active learning → neural abstraction/repair → CEG4N/FIPE/PINE. | Established much closer prior art than the brainstorm named; narrowed the contribution and added QCE. |
| `reasoning knowledge compression`, `limited memory language model`, `program extraction RNN` | Reasoning-compression study, LMLM/follow-on and MIPS. | Preserved knowledge guardrails, whole-system accounting and limits on synthetic evidence. |
| `lossless LLM weights entropy 2026`, `MLX GPTQ AWQ DWQ dynamic quantization` | Entropy paper → implementation setting; official MLX docs → actual calibration and quantizer source. | Distinguished archive compression from native execution; strengthened local quantization controls and data isolation. |
| `SmolLM2 135M 360M config license`, `PyTorch MPS macOS requirements`, `MLX quantized linear` | Official model/config cards and current platform documentation. | Derived storage/memory arithmetic and made all real operation support an early measured gate. |
| Dataset provenance, split and license queries | Original or maintainer cards for FineWeb-Edu, WikiText, ARC and BLiMP. | Chose a public local workload with reproducible manifests; retained contamination and license limitations. |
| Layer-sharing patent terminology | Broad CNN sharing record. | Recorded prior patent material without making a legal novelty or freedom-to-operate conclusion. |

Follow-up searches were used to challenge the selected refinement, not merely support it. The final material additions were counterexample-guided neural compression/abstraction, current nonlinear depth reconstruction and phase-sensitive looped-state analysis. Subsequent source/metadata checks resolved attribution and access details rather than changing the selected core experiment. This is substantial but not exhaustive coverage of every language, unpublished work, patent family or proprietary deployment. There was no fixed paper-count quota.

Additional alternatives such as SliceGPT and SVD-LLM were located through their primary records ([arXiv:2401.15024](https://arxiv.org/abs/2401.15024), [arXiv:2403.07378](https://arxiv.org/abs/2403.07378)); their full published pipelines were not reproduced. The direct local claim is against the specified, tuned executable comparison set, not against every published compressor. If a newly verified native implementation materially improves that set before protocol freeze, add it with equal data/cost accounting. After freeze, record an amendment and use fresh confirmation rather than silently replacing a comparator.

### 12.4 Access limitations and integrity ledger

Public web research succeeded. Direct network requests from the execution container failed DNS resolution, so no checkpoint, corpus or public repository was downloaded or executed there. Public GitHub files could still be inspected through the web tool. The small numerical checks used only locally installed Python packages and generated arrays.

The IEEE journal PDF for CEG4N was inaccessible; its earlier primary preprint supplied the method and the university record supplied journal metadata. The MIPS full-text paths encountered access errors, so its limited role rests on primary indexed/preprint summaries. Angluin's publisher full text was unavailable. SharVeT's README declared terms, but attempted raw license paths did not provide the license file. These limitations are not silently upgraded into full-text inspection.

HellaSwag was considered, but an original repository fetch returned a legal-access error and attempted mirror reads were unreliable in this session. It is not a core dependency. Adding it later requires verified access, license/provenance, frozen scoring and a pre-confirmation amendment; a missing benchmark must not be replaced after observing an unfavorable result.

No independent reproduction of any paper's quantitative results was performed. No M3 benchmark, training run, learned operator bank, precision-search result or compression-quality improvement exists from this planning work. The deliverable is the researched specification, analytical headroom calculations and explicitly limited CPU validation above.

### 12.5 First execution action and completion condition

Start at WP00. Create the repository and provenance files, discover the actual laptop, validate the critical operations and resolve the public assets. Then run the CPU/accelerator fixtures and establish the teacher and native quantization baseline before committing to a full compiler.

The final experimental result is complete when every claim traces to preserved source/config/data/artifact records, every direct comparison uses the same declared measurement boundary, failed and unfavorable results remain visible, and the report selects an outcome from Section 10. A defensible negative or inconclusive finding satisfies this research program; an unsupported novelty or “no quality loss” assertion does not.

[R01]: https://arxiv.org/abs/2410.20672
[R02]: https://arxiv.org/abs/2410.03765
[R03]: https://aclanthology.org/2026.acl-long.2213/
[R04]: https://github.com/snudatalab/SharVeT
[R05]: https://aclanthology.org/2022.naacl-main.318/
[R06]: https://arxiv.org/abs/2610.12448
[R07]: https://doi.org/10.1007/10722167_15
[R08]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json
[R09]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json
[R10]: https://arxiv.org/abs/1807.03819
[R11]: https://arxiv.org/abs/2401.12819
[R12]: https://arxiv.org/abs/2507.10524
[R13]: https://arxiv.org/abs/2408.09485
[R14]: https://arxiv.org/abs/2210.17323
[R15]: https://github.com/ml-explore/mlx-lm
[R16]: https://arxiv.org/abs/2504.12285
[R17]: https://arxiv.org/abs/2606.15789
[R18]: https://arxiv.org/abs/2505.15962
[R19]: https://doi.org/10.3390/e26121046
[R20]: https://arxiv.org/abs/2504.02010
[R21]: https://patents.google.com/patent/US10628683B2/en
[R22]: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/llama.py
[R23]: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html
[R24]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
[R25]: https://huggingface.co/datasets/Salesforce/wikitext
[R26]: https://huggingface.co/datasets/allenai/ai2_arc
[R27]: https://github.com/alexwarstadt/blimp
[R28]: https://developer.apple.com/metal/pytorch/
[R29]: https://arxiv.org/abs/2207.04231
[R30]: https://arxiv.org/abs/1910.14574
[R31]: https://arxiv.org/abs/2408.16167
[R32]: https://arxiv.org/abs/2605.28068
[R33]: https://arxiv.org/abs/2609.02083
[R34]: https://arxiv.org/abs/2609.33144
