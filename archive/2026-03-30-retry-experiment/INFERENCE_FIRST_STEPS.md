# Inference-First Improvement Runbook

This runbook keeps the repo on the current inference-only path. It does not require retraining.

## What Changed

The submission flow now uses `submission_inference_utils.py` to add four inference-level improvements:

- Longer first-pass decoding: `PASS1_MAX_NEW_TOKENS = 2048`
- Selective second pass: retries only rows that land in `xml` or `fallback`, or rows that look token-capped
- Richer debug output: the debug CSV now records pass-1 vs retry behavior, token-cap flags, strict-contract diagnostics, and per-stage failure reasons
- OOM backoff: if a generation batch runs out of GPU memory, the helper halves the batch size and retries automatically

The notebook still writes the same final artifacts:

- `submission.csv`
- `submission_debug.csv`

## Recommended First Run

1. Open `archive/notebooks/inference/submission.ipynb` in Colab.
2. Run the notebook top to bottom without changing any settings.
3. Wait for the final submission build to finish.
4. Download or inspect `submission_debug.csv`.
5. Check these columns first:
   - `reason`
   - `gate_reason`
   - `raw_gate_reason`
   - `repaired_gate_reason`
   - `xml_gate_reason`
   - `failure_reasons`
   - `retry_attempted`
   - `retry_used`
   - `final_hit_token_cap`
   - `raw_text`
   - `extracted_svg`
   - `final_svg`

## Safe Runtime Defaults

Use these notebook defaults for the next run:

- `PASS1_BATCH_SIZE = 250`
- `PASS2_BATCH_SIZE = 128`
- `PASS1_MAX_NEW_TOKENS = 2048`
- `PASS2_MAX_NEW_TOKENS = 3072`
- `RETRY_MODE = "long_greedy"`

The helper now shrinks each pass independently on CUDA OOM, so the run should continue even if either starting batch size is too large for the current Colab runtime.

## How To Read The Debug File

- `reason`: the final stage that produced the submitted SVG
- `gate_reason`: the final selected candidate's real validation reason; fallback rows now keep the underlying failure reason instead of just saying `fallback`
- `raw_gate_reason`: the validation result from the direct extracted SVG
- `repaired_gate_reason`: the validation result after the lightweight repair step
- `xml_gate_reason`: the validation result after XML recovery
- `failure_reasons`: compact summary of the failed stages for the selected row
- `source_gate_reason`: the validation result before root-attribute normalization for the selected candidate
- `normalized_gate_reason`: the validation result after root-attribute normalization for the selected candidate
- `normalization_status`: whether normalization was applied, unchanged, reverted, failed, or fell back
- `pass1_reason`: the stage from the first pass before any retry
- `retry_attempted`: whether the row was retried
- `retry_used`: whether the retry beat the first pass and replaced it
- `final_hit_token_cap`: whether the selected output consumed the full token budget
- `strict_contract`: whether the selected SVG matches the stricter local contract check
- `strict_issues`: comma-separated reasons when `strict_contract` is false

## Safe Experiment Order

Run these one at a time so you can compare Kaggle scores cleanly.

1. Baseline improved inference
   - Keep:
     - `PASS1_MAX_NEW_TOKENS = 2048`
     - `PASS2_MAX_NEW_TOKENS = 3072`
     - `RETRY_MODE = "long_greedy"`
2. If many rows still show `final_hit_token_cap = True`
   - Raise `PASS2_MAX_NEW_TOKENS` to `3584`
3. If you still hit CUDA OOM at `batch_size = 1`
   - Lower `PASS2_MAX_NEW_TOKENS` to `2560`
   - If needed, lower `PASS1_MAX_NEW_TOKENS` to `1792`
4. If retries rarely help
   - Set `RETRY_MODE = "sample"`
   - Keep the current low-temperature defaults
5. If sampled retry hurts stability
   - Switch back to `RETRY_MODE = "long_greedy"`

## What To Avoid For Now

- Do not change the prompt format in inference only. The current training format is still plain `Prompt: ...` plus `SVG:` in the repo.
- Do not tighten the acceptance gate so much that good rendered SVGs get replaced by the blank fallback.
- Do not mix multiple inference changes into one Kaggle submission. You only get a few submissions per day.

## If This Helps

If the improved inference path moves the score up, the next step is retraining on canonicalized SVGs with a larger training context window. That should be treated as a separate experiment, not bundled into the inference tests.
