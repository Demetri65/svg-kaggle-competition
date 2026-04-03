# Process

This file consolidates the data cleaning, training, inference, and artifact notes that used to be split across separate root docs.

## Archived Notebook Entry Points

- Cleaning notebook: `archive/notebooks/training/clean_training_data.ipynb`
- Training notebook: `archive/notebooks/training/main.ipynb`
- Baseline inference notebook: `archive/notebooks/inference/submission.ipynb`
- Midterm evaluation notebook: `archive/notebooks/evaluation/DL_Midterm_Eval.ipynb`

These notebooks are preserved for reference. The repository root is documentation-first by design.

## Data Cleaning Summary

All token measurements below use the `Qwen/Qwen2.5-Coder-1.5B-Instruct` tokenizer with a target training `max_length = 2048`.

### Raw Baseline

- Starting file: `train.csv`
- Rows: `50,000`
- Exact duplicate `(prompt, svg)` rows: `0`
- Conflicting prompts: `1,071`
- Rows inside conflicting prompts: `5,140` (`10.28%`)
- Parse failures: `0`
- Non-`<svg>` roots: `0`
- Missing `viewBox`: `83`
- `currentColor` rows: `6,507`
- Raw SVG length: `p50=2110`, `p95=6078`, `p99=7514`, `max=15937`

Main conclusion: the raw file parses cleanly, but it contains real prompt-to-SVG label conflicts and is too long for a `2048` token training budget without cleaning or filtering.

### Cleaning Contract

The cleaning pass aligns training outputs to the competition contract:

- Root normalized to `viewBox="0 0 256 256"`, `width="256"`, `height="256"`
- Allowed-tag whitelist enforced
- `MAX_SVG_LENGTH = 16000`
- `MAX_PATH_COUNT = 256`
- Conflicting prompts resolved with `keep_most_frequent`
- Exact duplicate cleaned pairs removed

### Cleaning Result

The current selected pipeline is:

1. Conservative canonicalization
2. Midterm-style root normalization and gate
3. Conflict resolution with `keep_most_frequent`
4. Selective `svgo` rescue only for rows over `2048`
5. Drop the remaining rows still over `2048`

Final strict `2048` dataset:

- Rows evaluated after cleaning/conflict resolution: `45,878`
- Rows dropped for exceeding `2048`: `6,412`
- Final training rows kept: `39,466`
- Final over-budget rate after filtering: `0%`
- Final token stats: `p50=917`, `p95=1864`, `p99=2027`, `max=2048`
- Final dataset conflicts: `0`
- Final midterm root exact matches: `39,466 / 39,466`
- Final gate failures: `0`

Current tracked outputs:

- Candidate CSV: `datasets/train_canonicalized_candidate.csv`
- Cleaning report: `datasets/train_canonicalized_candidate_report.json`

## Training Baseline

Archived source notebook: `archive/notebooks/training/main.ipynb`

Frozen baseline artifacts are still documented by reference:

- `/content/drive/MyDrive/svg-kaggle-comptetition/svg-lora-checkpoints`
- `/content/drive/MyDrive/svg-kaggle-comptetition/svg-lora-adapter`
- `/content/drive/MyDrive/svg-kaggle-comptetition/svg-model-merged`

Baseline training settings:

- `epochs = 1`
- `eval_strategy = "no"`
- `max_length = 1024`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 4`

Known baseline limitation:

- no canonicalization
- `max_length = 1024` likely truncates long SVG samples

Pending run namespace retained for reference:

- `RUN_TAG = qwen25coder15b_canon_len2048_singlepass`
- Artifact prefix: `qwen25coder15b_canon_len2048`
- Drive output root: `/content/drive/MyDrive/svg-kaggle-comptetition/runs/{RUN_ID}/`
- Local output root: `/content/{RUN_ID}/`

## Inference Baseline

Archived source notebook: `archive/notebooks/inference/submission.ipynb`

Baseline inference behavior:

- merged-model-first loading
- temporary base-model-plus-LoRA fallback when merged weights are not yet available externally
- deterministic decoding with `do_sample = False`
- `max_new_tokens = 1536`
- batch size `512`

Supported runtime contract:

- Input: `test.csv`
- Outputs: `submission.csv`, `submission_debug.csv`

Preferred model path:

- lightweight tracked metadata: `svg-model-merged/`
- external merged weight file: `svg-model-merged/model.safetensors` supplied outside git

## Artifact Handling

- Root notebooks are not allowed.
- Historical notebook runs belong under `archive/notebooks/`.
- Historical submission CSVs belong under `archive/submissions/`.
- Score provenance belongs in `SCORECARD.md`.
- Validation and push steps belong in `TESTING_AND_PUSH.md`.
