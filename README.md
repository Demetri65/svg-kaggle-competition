# SVG Kaggle Competition

This repo now standardizes on a single supported submission path: [submission.ipynb](submission.ipynb).

The current notebook mirrors the simple one-pass batched flow from [DL_Midterm_Eval.ipynb](DL_Midterm_Eval.ipynb), but adapts it for Google Colab with Google Drive mounting, repo checkout, persistent output folders, and merged-model discovery with a temporary base-plus-LoRA fallback.

## Supported Workflow

Use [submission.ipynb](submission.ipynb) for all current runs.

It is the only supported execution path and is responsible for:

- Mounting Google Drive and checking out the repo in the Colab runtime.
- Locating the required CSV inputs: `test.csv`, `train.csv`, and `sample_submission.csv`.
- Loading the merged inference model when available.
- Falling back to base model plus LoRA adapter only when merged weights are not available yet.
- Running midterm-style batched SVG generation and cleanup.
- Writing the two supported outputs: `submission.csv` and `submission_debug.csv`.

Supported runtime contract:

- Input: `test.csv`
- Outputs: `submission.csv`, `submission_debug.csv`

## Model Loading Rule

The preferred inference path is the merged model directory, `svg-model-merged`.

The current notebook first checks for merged weights inside the runtime checkout. If they are missing, it checks the Drive-backed project folder and copies them into the runtime checkout. If merged weights still are not available, the notebook temporarily falls back to the current base-model-plus-LoRA load path.

That fallback exists only to keep the Colab workflow usable until the merged model is uploaded to Drive. Once merged weights are always available, the fallback should be removed.

## Repo Layout

- [submission.ipynb](submission.ipynb): the only supported Colab submission notebook.
- [DL_Midterm_Eval.ipynb](DL_Midterm_Eval.ipynb): the earlier midterm evaluation notebook that the current supported notebook intentionally mirrors.
- [`archive/`](./archive): historical experiment notebooks preserved for documentation only.

`submission_pipeline.py` is intentionally no longer part of the supported or archived submission flow.

## Historical Strategies

The notebooks in [`archive/`](./archive) are preserved to document strategies that were tried before the repo was simplified into the current single-notebook flow. They are historical references, not supported entrypoints.

- [archive/submission_analysis.ipynb](archive/submission_analysis.ipynb): analyzed `train.csv`, compared `body_only` versus `full_svg`, and produced mode-selection guidance for the older batch workflow. This was superseded because the repo no longer needs a separate analysis notebook to drive multiple execution notebooks.
- [archive/submission_batch_1.ipynb](archive/submission_batch_1.ipynb): ran deterministic first-pass inference for batch 1 of 4 with no retry stage. This was superseded by a single batched notebook that handles the full test set directly.
- [archive/submission_batch_2.ipynb](archive/submission_batch_2.ipynb): ran deterministic first-pass inference for batch 2 of 4 with no retry stage. This was superseded for the same reason as batch 1.
- [archive/submission_batch_3.ipynb](archive/submission_batch_3.ipynb): handled the primary H100-backed split for batch 3 and wrote partial outputs until the coordinated split notebooks were complete. This was superseded because the supported flow no longer depends on split coordination across multiple notebooks.
- [archive/submission_batch_3a_temp.ipynb](archive/submission_batch_3a_temp.ipynb): temporary supporting split for batch 3 used to produce one partial shard of the coordinated multi-notebook run. This was superseded by the single-pass Colab workflow.
- [archive/submission_batch_3b_temp.ipynb](archive/submission_batch_3b_temp.ipynb): final temporary supporting split for batch 3 used to complete the coordinated multi-notebook run. This was superseded by the single-pass Colab workflow.
- [archive/submission_batch_4.ipynb](archive/submission_batch_4.ipynb): ran deterministic first-pass inference for batch 4 of 4 with no retry stage. This was superseded by full-dataset batching in the supported notebook.
- [archive/submission_merge_and_repair.ipynb](archive/submission_merge_and_repair.ipynb): merged batch outputs, repaired only failed rows, and wrote the final `submission.csv`. This was superseded because the repo now prioritizes the simpler midterm-style one-pass generation flow in Colab.

## Migration Note

The repo previously explored a more operational workflow built around split-batch inference, merge-and-repair, and separate train-set analysis. That structure documented useful experiments, but it also made submission runs harder to follow and slower to reproduce in Google Colab.

The current repo keeps those experiments under `archive/` for documentation while standardizing on [submission.ipynb](submission.ipynb) as the canonical path because it is simpler, faster to run end-to-end in Colab, and closer to the successful `DL_Midterm_Eval.ipynb` flow.
