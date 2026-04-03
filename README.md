# SVG Kaggle Competition

This repository is intentionally docs-first at the root. Active notes, handoff material, and score provenance stay top-level; all notebooks and historical submission artifacts live under [`archive/`](archive/).

## Root Contract

The root is reserved for:

- execution and handoff docs
- report and planning material
- tracked scripts under `scripts/`
- tracked datasets under `datasets/`
- archived notebooks and submissions under `archive/`
- lightweight model metadata under `svg-model-merged/`, `svg-lora-adapter/`, and `svg-lora-checkpoints/`

No notebooks should remain at the repository root.

## Where Things Live

- [`PROCESS.md`](PROCESS.md): consolidated process notes for data cleaning, training, inference, and artifact locations.
- [`TESTING_AND_PUSH.md`](TESTING_AND_PUSH.md): validation commands and the pre-push checklist for your friend.
- [`SCORECARD.md`](SCORECARD.md): score provenance, submission artifact reference, commit SHA, and Kaggle submission metadata.
- [`archive/README.md`](archive/README.md): inventory of every archived notebook and retired experiment path.
- [`archive/notebooks/inference/submission.ipynb`](archive/notebooks/inference/submission.ipynb): archived one-pass baseline submission notebook.
- [`archive/notebooks/evaluation/DL_Midterm_Eval.ipynb`](archive/notebooks/evaluation/DL_Midterm_Eval.ipynb): archived earlier midterm evaluation notebook mirrored by the baseline inference flow.

## Model Artifact Policy

The repository no longer stores merged model weights in git. The path `svg-model-merged/model.safetensors` is intentionally ignored and must be supplied from external storage when needed.

The lightweight metadata that remains tracked in `svg-model-merged/` is:

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`

LoRA adapter and checkpoint directories are still preserved in the repo history and working tree for now, but they are treated as historical artifacts rather than files that should grow further.

## Baseline Workflow Reference

The archived baseline inference path is documented in [`PROCESS.md`](PROCESS.md) and preserved in [`archive/notebooks/inference/submission.ipynb`](archive/notebooks/inference/submission.ipynb).

Baseline inference settings:

- `max_new_tokens = 1536`
- batch size `512`
- deterministic decoding with `do_sample = False`

Supported runtime contract:

- Input: `test.csv`
- Outputs: `submission.csv`, `submission_debug.csv`

## Archive Policy

Everything notebook-shaped or submission-output-shaped that is not root documentation belongs under `archive/`.

- Historical experiment notebooks live under `archive/notebooks/`.
- Historical submission CSVs live under `archive/submissions/`.
- The retry-aware variant remains preserved at [`archive/2026-03-30-retry-experiment/`](archive/2026-03-30-retry-experiment/).

If new experiments are added later, archive them by category rather than putting them back at the root.
