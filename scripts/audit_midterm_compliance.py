#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
REPORT_PATH = REPO_ROOT / "REPORT_DRAFT.md"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "kaggle_submit_raw_baseline.ipynb"
TRAIN_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_raw_baseline.py"
MANIFEST_PATH = REPO_ROOT / "artifacts" / "raw_baseline_manifest.json"
ARCHIVE_README_PATH = REPO_ROOT / "archive" / "README.md"

REQUIRED_PATHS = (
    README_PATH,
    REPORT_PATH,
    REQUIREMENTS_PATH,
    NOTEBOOK_PATH,
    TRAIN_SCRIPT_PATH,
    MANIFEST_PATH,
    ARCHIVE_README_PATH,
)
OFFICIAL_DOCS = (README_PATH, REPORT_PATH)
PLACEHOLDER_PATTERNS = (
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"PENDING_PUBLIC_GOOGLE_DRIVE_URL"),
)
FORBIDDEN_NOTEBOOK_PATTERNS = {
    "drive mount": re.compile(r"drive\.mount|/content/drive|MyDrive"),
    "git clone": re.compile(r"git clone"),
    "runtime pip install": re.compile(r"!pip\b|pip install|subprocess\.check_call\([^)]*pip"),
    "network checkout": re.compile(r"https?://(?!www\.w3\.org/2000/svg)"),
}
REQUIRED_NOTEBOOK_SNIPPETS = (
    "/kaggle/input/svg-kaggle-data/",
    "/kaggle/input/qwen25-coder-1p5b-instruct/",
    "/kaggle/input/svg-raw-baseline-adapter/",
    "/kaggle/working/submission.csv",
    "/kaggle/working/submission_debug.csv",
)
REQUIRED_LINKS = (
    "https://github.com/Demetri65/svg-kaggle-competition.git",
    "https://drive.google.com/",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def notebook_source_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def main() -> int:
    failures: list[str] = []

    for path in REQUIRED_PATHS:
        if not path.exists():
            failures.append(f"Missing required file: {path.relative_to(REPO_ROOT)}")

    if failures:
        print("\n".join(failures))
        return 1

    for path in (*OFFICIAL_DOCS, TRAIN_SCRIPT_PATH, MANIFEST_PATH):
        text = read_text(path)
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                failures.append(
                    f"Placeholder pattern '{pattern.pattern}' found in {path.relative_to(REPO_ROOT)}"
                )

    notebook_text = notebook_source_text(NOTEBOOK_PATH)
    for label, pattern in FORBIDDEN_NOTEBOOK_PATTERNS.items():
        if pattern.search(notebook_text):
            failures.append(f"Official Kaggle notebook contains forbidden {label} usage.")
    for snippet in REQUIRED_NOTEBOOK_SNIPPETS:
        if snippet not in notebook_text:
            failures.append(f"Official Kaggle notebook is missing required snippet: {snippet}")

    for path in OFFICIAL_DOCS:
        text = read_text(path)
        for link in REQUIRED_LINKS:
            if link not in text:
                failures.append(f"Required link '{link}' not found in {path.relative_to(REPO_ROOT)}")

    requirements_text = read_text(REQUIREMENTS_PATH)
    for package_name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "trl"):
        if not re.search(rf"^{re.escape(package_name)}==", requirements_text, flags=re.MULTILINE):
            failures.append(f"requirements.txt is missing a pinned entry for {package_name}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("canonical_experiment", {}).get("name") != "raw_one_pass_baseline":
        failures.append("Artifact manifest does not declare the raw one-pass baseline as canonical.")
    if manifest.get("public_repo_url") != "https://github.com/Demetri65/svg-kaggle-competition.git":
        failures.append("Artifact manifest public_repo_url does not match the canonical GitHub URL.")

    if failures:
        print("Midterm compliance audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Midterm compliance audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
