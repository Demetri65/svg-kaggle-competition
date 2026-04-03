# Testing And Push

Use this checklist before your friend pushes code or records a score.

## Quick Validation

Run these commands from the repo root:

```bash
find . -maxdepth 1 -name '*.ipynb'
python3 -m py_compile scripts/*.py
rg -n 'submission\.ipynb|DL_Midterm_Eval\.ipynb|main\.ipynb|clean_training_data\.ipynb|kaggle_train_submit_offline\.ipynb' README.md PROCESS.md NEXT_STEPS.md REPORT_DRAFT.md TESTING_AND_PUSH.md SCORECARD.md scripts archive/README.md
git status --short
```

Expected results:

- `find` returns nothing.
- `py_compile` exits cleanly.
- `rg` only returns archive paths, not root notebook paths.
- `git status --short` only shows intended changes.

## Manual Review

Before pushing:

- Confirm `SCORECARD.md` has the correct Kaggle score, submission ID, and commit SHA.
- Confirm the archived submission artifact path in `SCORECARD.md` matches the file tied to that score.
- Confirm no new large model weights or local outputs were staged.
- Confirm the root still looks docs-first.

## Git Hygiene

Check ignored/generated files:

```bash
git check-ignore -v .DS_Store
git check-ignore -v submission.csv
git check-ignore -v submission_debug.csv
git check-ignore -v svg-model-merged/model.safetensors
```

Each command should show a matching `.gitignore` rule.

## Pre-Push Checklist

1. Run the quick validation commands.
2. Review `git diff --stat` and `git diff --cached --stat`.
3. Update `SCORECARD.md` if the score, artifact path, or commit SHA changed.
4. Make sure the archived notebook/script paths referenced in docs still exist.
5. Push only after the repo is free of accidental outputs and weight files.

## History Rewrite Verification

After a history purge of the merged weights, re-run:

```bash
git log --stat -- svg-model-merged/model.safetensors
git rev-list --objects --all | rg 'svg-model-merged/model.safetensors'
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$1=="blob" {print}' | sort -k3 -n | tail -20
git count-objects -vH
```

Expected results:

- no commit history for `svg-model-merged/model.safetensors`
- no object listing for `svg-model-merged/model.safetensors`
- largest remaining blob under GitHub's 100 MB hard limit
- git pack size materially reduced relative to the pre-cleanup baseline
