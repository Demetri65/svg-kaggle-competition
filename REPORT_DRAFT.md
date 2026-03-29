# Initial Draft Report: Text-to-SVG Generation for the DL Spring 2026 Kaggle Contest

This markdown draft mirrors the required ACL report structure and is intentionally organized around the revised course priorities: methodology, code and reproducibility, ablations, and leaderboard performance. It treats the current midterm-evaluation and submission workflow as the completed baseline and labels all future work explicitly as planned.

Code repository: `TODO: insert public GitHub repository URL`

Model weights: `TODO: insert public model weights URL`

Public leaderboard score: `TODO: insert current public leaderboard score`

Private leaderboard score: `TODO: insert after private leaderboard release`

AI tooling disclosure: LLM assistance was used for coding/debugging and for preparing this report draft. The final submission should replace this line with the exact tools used and the scope of that assistance.

## Introduction

This project targets the DL Spring 2026 Kaggle contest on text-to-SVG generation. The task is to map a natural-language prompt to a valid SVG string in `id,svg` format. The competition is evaluator-constrained: every submission first passes through a hard validity gate, and any SVG that fails parsing, violates the allowed-tag policy, exceeds the complexity limits, or fails to render receives a zero for that sample. Only valid SVGs proceed to the quality-scoring stage. As a result, methodology for this task must optimize not only for visual quality but also for structural validity and renderability.

Our completed baseline uses `Qwen/Qwen2.5-Coder-1.5B-Instruct` fine-tuned with LoRA on prompt-to-SVG pairs, then performs deterministic inference with a merged model and a post-generation cleanup pipeline. The core methodological choice is evaluator alignment: instead of treating SVG generation as unconstrained free-form text generation, the pipeline explicitly optimizes for producing parser-valid, renderable SVGs that satisfy the contest contract. In practice, this means preferring a full-SVG prompt format, using deterministic decoding for the baseline submission path, and applying layered output cleanup consisting of extraction, regex repair, XML recovery, validation, and fallback behavior.

This baseline is not yet the final optimized competition system. In particular, broader score optimization, expanded ablations, Kaggle no-internet packaging, and leaderboard tracking are planned next steps and are not claimed as completed here.

## Dataset

The dataset contains 50,000 training prompt-SVG pairs and a hidden evaluation set of 1,000 prompts. The provided `test.csv` and `sample_submission.csv` files each contain 1,000 rows. Evaluation uses a hard validity gate followed by a quality score computed from rendered-image similarity, structural similarity, and a compactness term.

The contest contract places concrete constraints on every SVG:

- Canvas size: `256x256`
- Maximum SVG length: `16,000` characters
- Maximum path count: `256`
- Allowed tags include `svg`, `g`, `path`, `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`, `defs`, `use`, `symbol`, `clipPath`, `mask`, `linearGradient`, `radialGradient`, `stop`, `text`, `tspan`, `title`, `desc`, `style`, `pattern`, `marker`, and `filter`
- Disallowed categories include scripts, event handlers, animation, `foreignObject`, and external references

Training filters empty rows, converts each example to a plain-text instruction format, and uses a `90/10` train/eval split with seed `42`. The formatting function used for supervised fine-tuning is:

```text
Prompt: {prompt}
SVG:
{svg}
```

One important finding is that the raw training set is poorly aligned with the competition gate. Only `9 / 50,000` reference SVGs pass the current gate as-is. The dominant issue is a non-canonical `viewBox`, followed by missing `xmlns` and width/height mismatches. This result strongly suggests that evaluator alignment is a first-order problem: a model can learn from visually plausible SVGs that nevertheless violate the final scoring contract.

Completed preprocessing is minimal: row filtering, text formatting, and train/eval splitting. Planned preprocessing, not yet completed, includes canonicalizing training SVGs to the competition contract before fine-tuning.

## Model

The base model is `Qwen/Qwen2.5-Coder-1.5B-Instruct`, chosen because SVG generation is fundamentally structured text generation, and coder-oriented language models are strong at producing nested, syntax-constrained outputs. The system fine-tunes this model using LoRA and then supports two inference paths:

- Preferred path: merged model weights from `svg-model-merged`
- Temporary fallback path: base model plus LoRA adapter from `svg-lora-adapter`

The LoRA adapter configuration uses:

- Rank `r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

The training setup reports:

- Trainable parameters: `18,464,768`
- Total parameters: `1,562,179,072`
- Trainable fraction: `1.1820%`

This parameter-efficient setup keeps the base model fixed while adapting a small subset of attention and MLP projections. Inference then uses the merged model to avoid LoRA overhead at generation time.

The completed baseline methodology also includes a nontrivial post-generation pipeline:

1. Extract the SVG block from model text.
2. Apply lightweight regex repair for truncation or malformed suffixes.
3. Attempt XML recovery with `lxml`.
4. Validate against a whitelist-based gate and renderability check.
5. Fall back to a simple valid SVG when recovery fails.

This repair-and-validate layer is central to the project’s current methodology. Given the competition’s zero-score validity gate, this systems-level design choice is at least as important as the model choice itself.

## Experimentation

### Training Setup

The current fine-tuning configuration uses:

- One training epoch
- Per-device batch size `4`
- Gradient accumulation steps `4`
- Learning rate `2e-4`
- Cosine learning-rate schedule
- Warmup ratio `0.05`
- BF16 enabled
- Optimizer `paged_adamw_8bit`
- No evaluation during training
- `max_length = 1024`
- `packing = False`

Checkpoint logs show that training progressed in a stable direction over one epoch:

- Loss decreased from approximately `0.679` to `0.338`
- Mean token accuracy increased from approximately `0.793` to `0.885`

However, sampled formatted examples have an average token length of about `2358` and reach a sampled maximum of `7439`, while the configured training `max_length` is only `1024`. This makes truncation a likely baseline limitation and motivates future work on context length or better preprocessing.

### Inference Setup

The current supported workflow is a one-pass submission path that packages the baseline logic for Colab and Drive execution.

The completed baseline inference settings are:

- Merged-model-first loading
- Deterministic decoding with `do_sample = False`
- `max_new_tokens = 1536`
- Batch size `512`
- Fixed seed `1337`
- Output artifacts: `submission.csv` and `submission_debug.csv`

The baseline is therefore reproducible at the level of seeds and execution logic, but it still has important reproducibility gaps:

- The current supported path is Colab-oriented rather than Kaggle no-internet compliant.
- There is not yet a pinned `requirements.txt` or equivalent environment file.
- The public code URL is not yet available and must be inserted later.
- A public weights URL is not yet provided.
- The inference path still contains a merged-model fallback, which is practical during development but ambiguous for final reproducibility.

### Ablation Study

The strongest current ablation is a smoke-test comparison between `body_only` and `full_svg` generation modes on 10 training rows. This comparison selected `full_svg` as the better prompt/output framing.

| Generation mode | Rows | Gate-valid rows | Zero-score rows | Mean surrogate score | Mean runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `body_only` | 10 | 5 | 5 | 0.110638 | 84.51 |
| `full_svg` | 10 | 7 | 3 | 0.281280 | 64.35 |

This is a small ablation, not a final benchmark, but it is informative. It directly justifies the choice to prefer full-SVG generation for the submission path.

Planned ablations, not yet completed, include larger-scale prompt-format comparisons, repair-on/off experiments, merged-model versus base-plus-LoRA comparisons, and controlled decoding experiments.

## Results

### Completed Baseline Evidence

A full 1,000-row baseline run produced the following final reason counts:

- `xml`: `550`
- `valid`: `449`
- `repaired`: `1`

These counts indicate that the model-plus-cleanup pipeline successfully produced a full submission-sized output set, and that XML recovery was heavily used in the baseline. In other words, the cleanup layer is not incidental; it is a substantial contributor to the baseline submission path.

A second useful result is that the reference SVG audit shows the competition gate is much stricter than the raw training corpus format. Only `9 / 50,000` reference SVGs passed the current gate. The most common failures were:

- `viewBox must be exactly '0 0 256 256'`: `49,523`
- Missing `xmlns`: `324`
- Width/height not both `256`: `144`

This finding helps explain why evaluator-aware post-processing is necessary. A model trained naively on the raw corpus may reproduce the dataset distribution but still underperform on the actual contest metric if its outputs violate the final contract.

### Leaderboard Performance

An authoritative Kaggle leaderboard value for the baseline submission is not yet recorded here, so no score is claimed.

Public leaderboard score: `TODO: insert exact score from Kaggle`

Private leaderboard score: `TODO: insert after private leaderboard release`

Kaggle submission identifier: `TODO: insert exact submission reference`

This omission is deliberate. The report should not fabricate a leaderboard result. Once the baseline is rerun and logged on Kaggle, this section should be updated with the exact public score and the submission metadata needed for reproducibility.

### Planned Evaluations

The following evaluations are planned and are not claimed as completed:

- Re-run the current baseline end-to-end and log the exact public leaderboard score.
- Expand the `full_svg` versus `body_only` ablation beyond the 10-row smoke test.
- Measure the effect of repair logic on validity rate and score.
- Evaluate methods for reducing truncation and increasing gate pass rate.

## Conclusion

The completed baseline demonstrates a workable evaluator-aware text-to-SVG pipeline built on `Qwen/Qwen2.5-Coder-1.5B-Instruct`, LoRA fine-tuning, merged-model deterministic inference, and layered SVG cleanup. The strongest current evidence supports three conclusions.

First, validity is a first-class challenge in this competition. Because invalid SVGs score zero, the project must optimize for parser-valid and renderable outputs, not just plausible-looking markup. Second, `full_svg` is a better generation framing than `body_only` in the current archived smoke test. Third, the cleanup pipeline is operationally important: more than half of the recorded baseline outputs required XML recovery rather than being valid immediately after decoding.

At the same time, the system still has clear limitations. The current supported pipeline is Colab-oriented rather than packaged as a Kaggle no-internet notebook, the final public links required by the report are not yet ready, the leaderboard score is not yet recorded, and the training configuration likely truncates many examples. These limitations do not invalidate the current baseline, but they do define the highest-priority next steps.

Planned next work, not yet completed, is to lock the baseline with an exact leaderboard score, close reproducibility gaps, port the workflow into a competition-compliant Kaggle notebook, broaden the ablation suite, and improve training-data alignment with the evaluation contract.
