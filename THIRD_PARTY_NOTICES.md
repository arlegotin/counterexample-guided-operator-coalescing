# Third-party attribution and boundaries

The repository publishes original research software, reports and nontext metric metadata. Model weights, downloaded corpora, token arrays, optimizer checkpoints and restricted upstream source are excluded. Historical receipts contain paths, hashes and measurements of local assets; they do not supply those assets or grant rights to them.

## Model and evaluation sources

The immutable [source qualification record](provenance/assets-135m-sources.json) retains exact revisions, file digests, retrieval URLs and reviewed terms. The [preparation record](provenance/preparation-135m-2026-10-27.json) records the actual processing and exposure lineage. The table summarizes those records; upstream terms continue to apply.

| Source | Attribution and recorded terms | Use in this study |
| --- | --- | --- |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), Hugging Face | Apache-2.0; revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | Frozen teacher and tokenizer, acquired separately. No weights or tokenizer files are shipped. |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), Hugging Face | ODC-By-1.0 database terms and [Common Crawl terms](https://commoncrawl.org/terms-of-use); underlying web rights remain separate | Local calibration and evaluation. No source documents or token arrays are shipped. |
| [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), Salesforce Research | Pinned card metadata says CC-BY-SA-3.0 and GFDL; its prose links CC-BY-SA-4.0. This conflict remains unresolved. | Local source preparation. No redistribution interpretation or raw-text release is asserted. |
| [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc), Allen Institute for AI | CC-BY-SA-4.0 in the recorded source card | Teacher development competence screen. No questions or answers are shipped. |
| [BLiMP](https://huggingface.co/datasets/alexwarstadt/blimp), Alex Warstadt and collaborators | CC-BY-4.0 in the recorded source card | Teacher development competence screen. No sentence pairs are shipped. |

The local development screens do not establish preserved capability in a compressed model. Protected evaluations remain unexecuted.

## Dependencies and related implementations

Runtime libraries are installed separately. Their license notices remain in their distributions. The [wheel provenance](provenance/dependency-locks/wheel-provenance.json) records declared licenses, public upstream URLs and verified archive hashes for all 67 packages in the qualified native environment. Core dependencies include PyTorch (BSD-3-Clause), Transformers and Datasets (Apache-2.0), MLX and MLX-LM (MIT), NumPy and SciPy (BSD-3-Clause), PyYAML (MIT) and Apache Arrow/PyArrow (Apache-2.0). The project grant does not replace their terms.

Frozen executed-source archives in `provenance/dense-baseline-expansion-v2/` include the installed MLX-LM Python source used by the retained qualification attempts. Those source copies retain their MIT license; [the complete upstream notice](licenses/mlx-lm-MIT.txt) is included in the repository and inside each archive. Their recorded distribution is MLX-LM 0.32.2. These archives contain source and configuration, not installed binaries, model weights or corpus contents. The project's licenses do not replace MIT terms on these copies.

SharVeT's pinned PolyForm Noncommercial source was privately inspected for the planned baseline. It was neither copied into this repository nor executed; the full baseline remains unexecuted. The [blueprint review](docs/analysis/2026-10-27-blueprint-review.md) and [source attribution record](provenance/quantization-source-attribution.json) retain the prior-art discussion and source constraints.

Development, experiments and reporting used AI assistance under the owner's direction. README editing used [write-like-me](https://github.com/arlegotin/write-like-me); the skill is not bundled here. Standard license documents retain their own copyright notices.
