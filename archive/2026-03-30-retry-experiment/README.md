# 2026-03-30 Retry Experiment Snapshot

This snapshot preserves the retry-aware inference experiment exactly as it existed before the repo was rolled back to the simpler one-pass baseline.

- Purpose: retry-aware inference and XML-debug workflow for `archive/notebooks/inference/submission.ipynb`
- Helper version: `2026-03-29-debug-fix-v1`
- Main config:
  - pass 1 `2048`
  - pass 2 `3072`
  - retry mode `long_greedy`
  - pass 1 batch size `250`
  - pass 2 batch size `128`
- Observed Kaggle score: `15`
- Observed final reason counts:
  - `valid 620`
  - `xml 229`
  - `fallback 151`
- Baseline comparison note: earlier one-pass baseline scored `18.22`

Files in this snapshot:

- `../notebooks/inference/submission.ipynb`: archived one-pass baseline notebook that this retry path was compared against
- `submission_inference_utils.py`: retry/helper implementation used by the notebook
- `INFERENCE_FIRST_STEPS.md`: runbook and debug notes for the retry experiment

The notebook was copied exactly as it existed in the repo working tree at archive time. That copy did not contain embedded cell outputs, so the snapshot preserves the executed configuration and code state rather than a rendered output history.
