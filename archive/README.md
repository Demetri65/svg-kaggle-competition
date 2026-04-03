# Archive Index

This directory holds all notebooks, retired submission artifacts, and archived experiments. Nothing here is the root entrypoint anymore.

## Notebook Groups

- `notebooks/evaluation/`
  - `DL_Midterm_Eval.ipynb`: earlier midterm evaluation notebook kept as the reference baseline source.
  - `DL_Midterm_Eval_best_of_n_bad_rows.ipynb`: archived best-of-N retry experiment notebook.
  - `DL_Midterm_Eval_hybrid_retry.ipynb`: archived hybrid retry experiment notebook.
  - `DL_Midterm_Eval_adaptive_token_retry.ipynb`: archived adaptive token retry experiment notebook.
- `notebooks/training/`
  - `main.ipynb`: archived LoRA training notebook.
  - `clean_training_data.ipynb`: archived dataset cleaning and canonicalization notebook.
- `notebooks/inference/`
  - `submission.ipynb`: archived one-pass baseline inference notebook.
  - `kaggle_train_submit_offline.ipynb`: archived offline Kaggle-oriented notebook.
- `notebooks/batch-history/`
  - `submission_analysis.ipynb`: archived analysis notebook from the multi-batch workflow.
  - `submission_batch_1.ipynb` through `submission_batch_4.ipynb`: archived split-run inference notebooks.
  - `submission_batch_3a_temp.ipynb` and `submission_batch_3b_temp.ipynb`: archived temporary supporting split notebooks.
  - `submission_merge_and_repair.ipynb`: archived merge-and-repair notebook from the split workflow.

## Archived Submissions

- `submissions/submission (3).csv`
- `submissions/submission (4).csv`

Keep the score-linked artifact recorded in `../SCORECARD.md`.

## Preserved Experiment Snapshot

- `2026-03-30-retry-experiment/`: self-contained retry-aware inference experiment with helper code, run notes, and its own archived notebook.

## Policy

- New notebooks should be archived by function, not left at the repository root.
- Historical outputs belong under `archive/submissions/`.
- Root docs should point here instead of duplicating notebook inventories.
