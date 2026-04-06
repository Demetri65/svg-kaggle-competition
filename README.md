# SVG Kaggle Competition

This repository is the midterm-compliant surface for the DL Spring 2026 text-to-SVG Kaggle project. The only official experiment is the original raw-data one-pass baseline in `DL_Midterm_Final.ipynb`. Everything else is preserved for history under [`archive/`](/Users/demetrilopez/dev/svg-kaggle-comptetition/archive).

## Public Links

- GitHub repository: `https://github.com/Demetri65/svg-kaggle-competition.git`
- Public raw-baseline adapter bundle: `https://drive.google.com/drive/folders/1UCJATHdn5yBFJJzH_TNXpmjuZBHDgyhX?usp=drive_link`

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
- Inference hyperparameters: deterministic decoding, `max_new_tokens=1536`
- Submission guardrails: strict `256x256` SVG wrapper, renderability check, allowed-tag whitelist, max length `16000`, max path count `256`

## Reproduction

Run the `DL_Midterm_Final.ipynb` (Evaluation Code Section) notebook with the appropriate paths changed for inputs and outputs along with model weights. You can add a shortcut to the google drive link shared above and then execute everything.

## Archive Policy

Everything under [`archive/`](/Users/demetrilopez/dev/svg-kaggle-comptetition/archive) is historical and non-canonical. That includes the former root final notebook, the cleaned-data Kaggle workflow, retry experiments, batch-history notebooks, and exploratory evaluation variants. They are preserved for chronology and report evidence, not for official reproduction.
