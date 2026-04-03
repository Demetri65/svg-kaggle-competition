# SVG Kaggle Competition

This repository is the midterm-compliant surface for the DL Spring 2026 text-to-SVG Kaggle project. The only official experiment is the original raw-data one-pass baseline. Everything else is preserved for history under [`archive/`](/Users/demetrilopez/dev/svg-kaggle-comptetition/archive).

## Public Links

- GitHub repository: `https://github.com/Demetri65/svg-kaggle-competition.git`
- Public raw-baseline adapter bundle: `PENDING_PUBLIC_GOOGLE_DRIVE_URL`

## Canonical Raw Baseline

- Training data: raw `train.csv`
- Training format:

```text
Prompt: {prompt}
SVG:
{svg}
```

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Training hyperparameters: `1` epoch, batch size `4`, gradient accumulation `4`, learning rate `2e-4`, cosine schedule, `warmup_ratio=0.05`, BF16, `paged_adamw_8bit`, `max_length=1024`, seed `42`
- Inference hyperparameters: deterministic decoding, `max_new_tokens=1536`, seed `1337`
- Submission guardrails: strict `256x256` SVG wrapper, renderability check, allowed-tag whitelist, max length `16000`, max path count `256`

The checked-in [`artifacts/raw_baseline_manifest.json`](/Users/demetrilopez/dev/svg-kaggle-comptetition/artifacts/raw_baseline_manifest.json) is the source of truth for the canonical lineage, expected files, seeds, and required Kaggle inputs. It also marks the checked-in [`svg-lora-adapter/`](/Users/demetrilopez/dev/svg-kaggle-comptetition/svg-lora-adapter) and [`svg-model-merged/`](/Users/demetrilopez/dev/svg-kaggle-comptetition/svg-model-merged) directories as preserved legacy artifacts rather than verified canonical weights.

## Reproduction

Install the pinned environment:

```bash
python3 -m pip install -r requirements.txt
```

Train the canonical raw baseline with an offline base-model snapshot:

```bash
python scripts/train_raw_baseline.py \
  --train-csv train.csv \
  --base-model-dir /path/to/qwen25-coder-1p5b-instruct \
  --output-root runs/raw_baseline
```

This writes:

- `runs/raw_baseline/checkpoints/`
- `runs/raw_baseline/adapter/`
- `runs/raw_baseline/metadata/baseline_reference.json`
- `runs/raw_baseline/metadata/run_summary.json`

For Kaggle code submission, upload three offline inputs:

- `/kaggle/input/svg-kaggle-data/`
- `/kaggle/input/qwen25-coder-1p5b-instruct/`
- `/kaggle/input/svg-raw-baseline-adapter/`

The adapter dataset should contain the files produced in `runs/raw_baseline/adapter/`:

- `adapter_config.json`
- `adapter_model.safetensors`
- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`

Run the official notebook [`notebooks/kaggle_submit_raw_baseline.ipynb`](/Users/demetrilopez/dev/svg-kaggle-comptetition/notebooks/kaggle_submit_raw_baseline.ipynb). It runs fully offline, loads the base model plus adapter, and writes only:

- `/kaggle/working/submission.csv`
- `/kaggle/working/submission_debug.csv`

If you need a final contract-only rewrite pass on an existing submission CSV, run:

```bash
python scripts/normalize_submission_svg_wrappers.py archive/submissions/submission\ \(4\).csv
```

## Compliance Check

Run the repo audit before submission:

```bash
python scripts/audit_midterm_compliance.py
```

The audit checks the required files, the official notebook constraints, the required public links in the docs, and unresolved placeholders on the canonical surface.

## Archive Policy

Everything under [`archive/`](/Users/demetrilopez/dev/svg-kaggle-comptetition/archive) is historical and non-canonical. That includes the former root final notebook, the cleaned-data Kaggle workflow, retry experiments, batch-history notebooks, and exploratory evaluation variants. They are preserved for chronology and report evidence, not for official reproduction.
