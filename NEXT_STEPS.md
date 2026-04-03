# NEXT_STEPS

This file is the prioritized completion plan for turning the current repo baseline into a fully reproducible, competition-compliant, and report-ready project. The ordering reflects the revised course rubric:

- Code and Reproducibility: `30%`
- Methodology: `25%`
- Ablations: `25%`
- Leaderboard Performance: `20%`

The current assumption is that the archived baseline preserved in `archive/notebooks/evaluation/DL_Midterm_Eval.ipynb` and `archive/notebooks/inference/submission.ipynb` is the completed baseline. Everything below is structured to preserve that baseline first, then improve rigor and score. The retry-aware inference variant archived under `archive/2026-03-30-retry-experiment` is exploratory only and should not be treated as the canonical path.

## 1. Lock the Baseline

Rubric impact: leaderboard performance, reproducibility, report accuracy.

- Re-run the baseline submission path end-to-end with the current merged model and fixed seed.
- Record the exact public leaderboard score from Kaggle.
- Save the exact `submission.csv` and `submission_debug.csv` used for that score.
- Record the exact git commit SHA used for the baseline run.
- Create a lightweight git tag for the baseline report state, for example `baseline-report-v1`.
- Capture the Kaggle notebook version or submission identifier so the report can cite the exact run.

Exit criteria:

- One baseline submission is tied to one commit, one pair of output files, and one public leaderboard score.

## 2. Fix Reproducibility Gaps

Rubric impact: code and reproducibility.

- Publish the final public GitHub repository URL and replace the placeholder in `REPORT_DRAFT.md`.
- Publish a public model-weights URL and replace the placeholder in `REPORT_DRAFT.md`.
- Add a pinned environment file, preferably `requirements.txt`, with the exact package versions needed for training and inference.
- Add explicit reproduction instructions for:
  - training the LoRA adapter
  - merging the adapter
  - running inference
  - producing `submission.csv`
- Make the canonical execution path unambiguous by documenting whether the final submission must use merged weights only.
- Remove or clearly fence off temporary development-only behavior, especially the merged-model fallback logic once the final packaging is stable.

Exit criteria:

- An instructor can clone the repo, install the environment, fetch the published weights, and reproduce the baseline artifacts using written instructions only.

## 3. Make the Pipeline Competition-Compliant

Rubric impact: code and reproducibility, leaderboard performance.

- Port the final inference workflow into a Kaggle notebook that runs end-to-end without internet access.
- Package all required assets as Kaggle-compatible inputs, including model files and any auxiliary resources.
- Remove Colab-only assumptions such as Drive mounting, runtime repo cloning, or ad hoc file copying.
- Verify that the final notebook writes the expected `id,svg` CSV directly in the Kaggle environment.
- Document the exact difference between the local/Colab development path and the final Kaggle submission path if both are kept.

Exit criteria:

- The competition submission notebook runs offline on Kaggle and produces a valid submission artifact without manual intervention.

## 4. Strengthen Ablations

Rubric impact: ablations, methodology.

- Expand the existing `full_svg` versus `body_only` comparison beyond the current 10-row smoke test.
- Compare merged-model inference against base-model-plus-LoRA inference to confirm parity or justify one path.
- Compare the repair pipeline on versus off to quantify how much validity and quality it contributes.
- Compare deterministic decoding against a controlled sampled-repair variant that preserves the baseline as the reference point.
- Compare multiple `max_new_tokens` values and measure their effect on truncation, validity rate, and downstream score.
- Compare structured multi-pass inference ideas such as layout planning, SVG generation, and targeted revision only after the one-pass baseline is locked again.
- Store ablation outputs in a format that can be summarized directly in the final report.

Exit criteria:

- The report can include at least two nontrivial ablations beyond the current smoke-test result, each with enough rows to support a defensible conclusion.

## 5. Improve Methodology

Rubric impact: methodology, leaderboard performance.

- Canonicalize training SVGs to the competition contract before fine-tuning, especially `viewBox`, `xmlns`, and canvas-size normalization.
- Revisit the training sequence-length setting because sampled training examples average roughly `2358` tokens and reach at least `7439`, while the baseline uses `max_length = 1024`.
- Test context-length increases or alternative preprocessing that preserve more of each SVG target.
- Consider validity-aware reranking or multi-candidate generation if the baseline validity bottleneck persists after canonicalization.
- Measure how each methodology change affects:
  - validity gate pass rate
  - fallback frequency
  - surrogate quality metrics
  - leaderboard score

Exit criteria:

- At least one methodology improvement is validated with quantitative evidence rather than described only as an idea.

## 6. Finish the Report

Rubric impact: methodology, code and reproducibility, ablations, leaderboard performance.

- Replace all report placeholders for:
  - public code URL
  - public weights URL
  - public leaderboard score
  - private leaderboard score, when available
- Keep completed work and planned work clearly separated in the final report.
- Convert the markdown draft into the final ACL-format paper.
- Ensure the final report explicitly includes the AI-tooling disclosure required by the course instructions.
- Cross-check that every quantitative statement in the report is backed by either the competition handout, repo artifacts, or logged experiment outputs.

Exit criteria:

- The final ACL-format report is complete, evidence-backed, and aligned with the revised grading priorities.

## Validation Checklist

- `REPORT_DRAFT.md` includes all required handout sections:
  - Introduction
  - Dataset
  - Model
  - Experimentation
  - Results
  - Conclusion
- `REPORT_DRAFT.md` contains explicit placeholders for:
  - public code URL
  - public weights URL
  - leaderboard score
- `REPORT_DRAFT.md` includes an AI-tooling disclosure.
- Every future-looking claim in the report is labeled as planned or not yet completed.
- Every quantitative claim comes from the competition handout or repo evidence.
- `NEXT_STEPS.md` remains ordered by execution priority rather than by brainstorm category.
- The final workflow decision between Colab and Kaggle is documented unambiguously before the report is finalized.

## Assumptions

- The archived `DL_Midterm_Eval.ipynb` / `submission.ipynb` path under `archive/notebooks/` is the completed baseline state.
- No public GitHub URL, public model-weights URL, or leaderboard score should be fabricated.
- The initial draft is markdown-first rather than LaTeX-first.
- Forward-looking content can appear in the report only when it is clearly labeled as planned.
