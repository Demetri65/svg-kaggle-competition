from __future__ import annotations

import gc
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import textwrap
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import cairosvg
import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from skimage.feature import canny
from skimage.metrics import structural_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SVG_NS = "http://www.w3.org/2000/svg"
ZERO_SCORE_SVG = f'<svg xmlns="{SVG_NS}" width="256" height="256"></svg>'

ALLOWED_TAGS = {
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "defs",
    "use",
    "symbol",
    "clipPath",
    "mask",
    "linearGradient",
    "radialGradient",
    "stop",
    "text",
    "tspan",
    "title",
    "desc",
    "style",
    "pattern",
    "marker",
    "filter",
}
DISALLOWED_TAGS = {"script", "foreignObject", "animate", "animateTransform", "animateMotion", "set"}
CANONICAL_TAGS = {tag.lower(): tag for tag in ALLOWED_TAGS}
URL_PATTERN = re.compile(r"(?:https?:)?//|javascript:|data:", re.IGNORECASE)
SVG_BLOCK_PATTERN = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)

REQUIRED_INPUT_FILES = ("test.csv", "train.csv", "sample_submission.csv")
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "cairosvg": "cairosvg",
    "skimage": "scikit-image",
    "PIL": "pillow",
}

RESULT_COLUMNS = [
    "id",
    "prompt",
    "batch_id",
    "row_index",
    "generation_mode",
    "reference_gate_valid",
    "repair_strategy",
    "first_pass_success",
    "raw_text",
    "svg_body",
    "wrapped_svg",
    "clean_svg",
    "attempt_count",
    "selected_strategy",
    "valid_final",
    "gate_error",
    "error_reason",
    "char_len",
    "path_count",
    "runtime_seconds",
    "surrogate_score",
]
TEXT_RESULT_COLUMNS = {
    "id",
    "prompt",
    "batch_id",
    "generation_mode",
    "repair_strategy",
    "raw_text",
    "svg_body",
    "wrapped_svg",
    "clean_svg",
    "selected_strategy",
    "gate_error",
    "error_reason",
}
BOOL_RESULT_COLUMNS = {"reference_gate_valid", "first_pass_success", "valid_final"}
INT_RESULT_COLUMNS = {"row_index", "attempt_count", "char_len", "path_count"}
FLOAT_RESULT_COLUMNS = {"runtime_seconds", "surrogate_score"}


@dataclass(frozen=True)
class PipelinePaths:
    repo_root: Path
    output_root: Path
    analysis_dir: Path
    batches_dir: Path
    merge_dir: Path
    adapter_dir: Path
    test_csv: Path
    train_csv: Path
    sample_submission_csv: Path
    analysis_recommendation_path: Path


@dataclass
class RuntimeBundle:
    tokenizer: Any
    model: Any
    runtime_device: str
    compute_dtype: Any


@dataclass(frozen=True)
class GenerationConfig:
    model_id: str = MODEL_ID
    max_new_tokens: int = 640
    retry_temperature: float = 0.2
    retry_top_p: float = 0.9
    max_repair_context_chars: int = 2500
    verbose_progress: bool = True


class RowInferenceError(RuntimeError):
    """Raised when a row cannot be converted into a valid SVG."""

    def __init__(self, message: str, *, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


def run_command(cmd: Sequence[str]) -> None:
    """Run a subprocess command and echo it for notebook logs."""

    print("+", " ".join(cmd))
    subprocess.check_call(list(cmd))


def install_missing_packages() -> list[str]:
    """Install runtime packages needed by the Colab notebooks."""

    import importlib.util
    import sys

    missing = [
        package_name
        for module_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        run_command([sys.executable, "-m", "pip", "install", *missing])
    return missing


def clone_repo_checkout(repo_url: str, checkout_path: Path, *, reset_existing: bool = True) -> Path:
    """Clone the repo into the Colab runtime."""

    if reset_existing and checkout_path.exists():
        shutil.rmtree(checkout_path)
    if not checkout_path.exists():
        run_command(["git", "clone", repo_url, str(checkout_path)])
    return checkout_path


def copy_required_file_from_drive(name: str, checkout_path: Path, drive_root: Path) -> Path:
    """Copy a required CSV from Drive into the runtime checkout."""

    destination = checkout_path / name
    if destination.exists():
        return destination

    preferred_sources = [
        drive_root / "svg-kaggle-comptetition" / name,
        drive_root / "Colab Notebooks" / "svg-kaggle-comptetition" / name,
    ]
    for candidate in preferred_sources:
        if candidate.exists():
            shutil.copy2(candidate, destination)
            return destination

    for candidate in drive_root.rglob(name):
        if candidate.is_file():
            shutil.copy2(candidate, destination)
            return destination

    raise FileNotFoundError(
        f"Could not find {name} in Google Drive. Copy it into {checkout_path} manually and rerun this cell."
    )


def prepare_colab_checkout(repo_url: str, checkout_path: Path, drive_root: Path) -> Path:
    """Clone the repo and ensure the required CSV inputs are present."""

    clone_repo_checkout(repo_url, checkout_path, reset_existing=True)
    for required_name in REQUIRED_INPUT_FILES:
        copy_required_file_from_drive(required_name, checkout_path, drive_root)
    os.environ["SVG_KAGGLE_REPO_ROOT"] = str(checkout_path)
    return checkout_path


def resolve_pipeline_paths(repo_root: Path, output_root: Path) -> PipelinePaths:
    """Resolve all repo and output paths used by the notebooks."""

    repo_root = repo_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for subdir in ["analysis", "batches", "merge"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    required_repo_items = ["svg-lora-adapter", "test.csv", "sample_submission.csv", "train.csv"]
    missing_repo_items = [name for name in required_repo_items if not (repo_root / name).exists()]
    if missing_repo_items:
        raise RuntimeError(
            "Repo root is missing required files: "
            + ", ".join(missing_repo_items)
            + f" at {repo_root}"
        )

    return PipelinePaths(
        repo_root=repo_root,
        output_root=output_root,
        analysis_dir=output_root / "analysis",
        batches_dir=output_root / "batches",
        merge_dir=output_root / "merge",
        adapter_dir=repo_root / "svg-lora-adapter",
        test_csv=repo_root / "test.csv",
        train_csv=repo_root / "train.csv",
        sample_submission_csv=repo_root / "sample_submission.csv",
        analysis_recommendation_path=output_root / "analysis" / "analysis_recommendation.json",
    )


def batch_output_dir(paths: PipelinePaths, batch_id: int) -> Path:
    """Return the output directory for one batch notebook."""

    batch_dir = paths.batches_dir / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def partial_batch_output_dir(paths: PipelinePaths, batch_id: int) -> Path:
    """Return the partial-output directory for one split batch."""

    partial_dir = batch_output_dir(paths, batch_id) / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    return partial_dir


def prepare_split_batch_outputs(
    batch_dir: Path,
    expected_parts: Sequence[str],
    *,
    force_reset: bool = False,
    run_label: str = "",
) -> dict[str, Any]:
    """Initialize one split-batch run and clear stale canonical files once."""

    batch_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = batch_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    marker_path = partial_dir / "split_run_state.json"

    marker_payload: dict[str, Any] = {}
    if marker_path.exists():
        try:
            marker_payload = json.loads(marker_path.read_text())
        except Exception:
            marker_payload = {}

    existing_label = str(marker_payload.get("run_label", ""))
    did_reset = force_reset or not marker_path.exists() or (bool(run_label) and existing_label != run_label)
    if did_reset:
        for filename in ["batch_results.csv", "batch_success_submission.csv", "batch_failed_rows.csv"]:
            path = batch_dir / filename
            if path.exists():
                path.unlink()
        for path in partial_dir.glob("*.csv"):
            path.unlink()
        marker_payload = {
            "expected_parts": list(expected_parts),
            "run_label": run_label,
            "initialized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        marker_path.write_text(json.dumps(marker_payload, indent=2))

    return {
        "did_reset": did_reset,
        "marker_path": marker_path,
        "expected_parts": list(expected_parts),
        "partial_dir": partial_dir,
        "run_label": run_label or existing_label,
    }


def artifact_status_table(paths: PipelinePaths, *, batch_count: int = 4) -> pd.DataFrame:
    """Summarize the current artifact files present in Drive."""

    records: list[dict[str, Any]] = []
    artifact_map = {
        "analysis_recommendation": paths.analysis_recommendation_path,
        "train_reference_gate_audit": paths.analysis_dir / "train_reference_gate_audit.csv",
        "train_smoke_test_results": paths.analysis_dir / "train_smoke_test_results.csv",
        "final_submission": paths.merge_dir / "submission.csv",
        "repaired_rows": paths.merge_dir / "repaired_rows.csv",
        "remaining_failures": paths.merge_dir / "remaining_failures.csv",
    }
    for name, path in artifact_map.items():
        records.append({"artifact": name, "path": str(path), "exists": path.exists()})
    for batch_id in range(1, batch_count + 1):
        batch_dir = batch_output_dir(paths, batch_id)
        for filename in ["batch_results.csv", "batch_success_submission.csv", "batch_failed_rows.csv"]:
            path = batch_dir / filename
            records.append(
                {
                    "artifact": f"batch_{batch_id}:{filename}",
                    "path": str(path),
                    "exists": path.exists(),
                }
            )
    return pd.DataFrame(records)


def preferred_runtime_device() -> str:
    """Return the preferred torch runtime device."""

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def release_runtime_memory() -> None:
    """Release Python and CUDA memory between notebook phases."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_lora_runtime(
    adapter_dir: Path,
    *,
    model_id: str = MODEL_ID,
    allow_4bit: bool | None = None,
) -> RuntimeBundle:
    """Load the base model and LoRA adapter into a CUDA runtime."""

    runtime_device = preferred_runtime_device()
    if runtime_device != "cuda":
        raise RuntimeError(
            "CUDA/GPU is required for these notebooks. In Colab, switch to Runtime > Change runtime type > GPU."
        )

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json at {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text())
    base_model_name = adapter_config.get("base_model_name_or_path")
    if base_model_name != model_id:
        raise RuntimeError(f"Adapter expects base model {base_model_name}, not {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if allow_4bit is None:
        import importlib.util

        allow_4bit = importlib.util.find_spec("bitsandbytes") is not None

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
        "device_map": "auto",
    }
    if allow_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return RuntimeBundle(
        tokenizer=tokenizer,
        model=model,
        runtime_device=runtime_device,
        compute_dtype=compute_dtype,
    )


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for the notebook runtime."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_competition_frames(paths: PipelinePaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load test, sample submission, and train tables from the repo checkout."""

    test_df = pd.read_csv(paths.test_csv, keep_default_na=False)
    sample_submission_df = pd.read_csv(paths.sample_submission_csv, keep_default_na=False)
    train_df = pd.read_csv(paths.train_csv, keep_default_na=False)

    assert list(test_df.columns) == ["id", "prompt"], f"Unexpected test.csv columns: {list(test_df.columns)}"
    assert len(test_df) == 1000, f"Expected 1000 test rows, found {len(test_df)}"
    assert list(sample_submission_df.columns) == ["id", "svg"], (
        f"Unexpected sample_submission.csv columns: {list(sample_submission_df.columns)}"
    )
    assert len(sample_submission_df) == len(test_df), (
        f"Expected sample_submission.csv to have {len(test_df)} rows, found {len(sample_submission_df)}"
    )
    assert sample_submission_df["id"].tolist() == test_df["id"].tolist(), (
        "sample_submission.csv IDs do not match test.csv order."
    )
    assert {"prompt", "svg"}.issubset(train_df.columns), (
        f"train.csv must contain prompt and svg columns, found {list(train_df.columns)}"
    )
    if "id" not in train_df.columns:
        train_df = train_df.copy()
        train_df.insert(0, "id", [f"train_{idx:06d}" for idx in range(len(train_df))])

    return test_df, sample_submission_df, train_df


def local_name(name: str) -> str:
    """Strip an XML namespace prefix from a tag or attribute name."""

    return name.split("}", 1)[-1] if "}" in name else name


def contains_external_reference(value: str) -> bool:
    """Detect an external URL or resource reference inside an SVG value."""

    value = (value or "").strip()
    if not value:
        return False
    if URL_PATTERN.search(value):
        return True
    if value.lower().startswith("url(") and value.endswith(")"):
        inner = value[4:-1].strip().strip("'\"")
        return bool(inner) and not inner.startswith("#")
    return False


def extract_svg(text: str) -> str:
    """Extract the first SVG block from model output or return the raw text."""

    text = (text or "").strip()
    text = re.sub(r"^```(?:xml|svg)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^<\?xml[^>]*>\s*", "", text, flags=re.IGNORECASE)
    match = SVG_BLOCK_PATTERN.search(text)
    return match.group(0).strip() if match else text.strip()


def serialize_children(element: ET.Element) -> str:
    """Serialize an element's children without the element wrapper."""

    parts: list[str] = []
    if element.text and element.text.strip():
        parts.append(element.text)
    for child in list(element):
        parts.append(ET.tostring(child, encoding="unicode", short_empty_elements=True))
        if child.tail and child.tail.strip():
            parts.append(child.tail)
    return "".join(parts).strip()


def wrap_svg_body(svg_body: str) -> str:
    """Wrap an inner SVG fragment in the canonical competition root."""

    body = (svg_body or "").strip()
    return (
        f'<svg xmlns="{SVG_NS}" width="256" height="256" viewBox="0 0 256 256">'
        f"{body}"
        "</svg>"
    )


def unwrap_if_full_svg(raw_text: str) -> tuple[str, str]:
    """If a model returned a full SVG, strip the wrapper and keep the body."""

    extracted = extract_svg(raw_text)
    if not extracted.lower().startswith("<svg"):
        return extracted, extracted

    try:
        root = ET.fromstring(extracted)
    except Exception:
        return extracted, extracted
    if local_name(root.tag) != "svg":
        return extracted, extracted

    preserved_attrs = {
        key: value
        for key, value in root.attrib.items()
        if local_name(key) not in {"xmlns", "width", "height", "viewBox"}
    }
    body = serialize_children(root)
    if preserved_attrs:
        attr_string = " ".join(f'{key}="{value}"' for key, value in preserved_attrs.items())
        body = f"<g {attr_string}>{body}</g>" if body else f"<g {attr_string}></g>"
    return body.strip(), extracted


def extract_svg_body(raw_text: str) -> tuple[str, str]:
    """Extract a body fragment, tolerating accidental full-SVG responses."""

    raw_text = "" if raw_text is None else str(raw_text)
    body, normalized_source = unwrap_if_full_svg(raw_text)
    return body.strip(), normalized_source.strip()


def sanitize_attributes(raw_attributes: dict[str, Any]) -> dict[str, str]:
    """Normalize and validate SVG attributes."""

    clean_attributes: dict[str, str] = {}
    for key, value in raw_attributes.items():
        attr_name = local_name(key)
        attr_value = str(value)
        lowered = attr_name.lower()
        if lowered.startswith("on"):
            raise ValueError(f"Event handler attributes are not allowed: {attr_name}")
        if lowered in {"href", "xlink:href"} and attr_value.strip() and not attr_value.strip().startswith("#"):
            raise ValueError(f"External href is not allowed: {attr_value}")
        if contains_external_reference(attr_value):
            raise ValueError(f"External reference is not allowed: {attr_value}")
        clean_attributes[attr_name] = attr_value
    return clean_attributes


def sanitize_element(raw_element: ET.Element, *, is_root: bool = False) -> ET.Element:
    """Recursively sanitize an SVG element tree."""

    raw_tag = local_name(raw_element.tag)
    if raw_tag in DISALLOWED_TAGS:
        raise ValueError(f"Disallowed SVG tag: {raw_tag}")
    canonical_tag = CANONICAL_TAGS.get(raw_tag.lower())
    if canonical_tag is None:
        raise ValueError(f"Disallowed tag: {raw_tag}")

    clean_attributes = sanitize_attributes(raw_element.attrib)
    if is_root:
        preserved_root_attributes = {
            key: value
            for key, value in clean_attributes.items()
            if key not in {"xmlns", "width", "height", "viewBox"}
        }
        preserved_root_attributes.update(
            {
                "xmlns": SVG_NS,
                "width": "256",
                "height": "256",
                "viewBox": "0 0 256 256",
            }
        )
        clean_element = ET.Element("svg", preserved_root_attributes)
    else:
        clean_element = ET.Element(canonical_tag, clean_attributes)

    if raw_element.text:
        if canonical_tag == "style" and contains_external_reference(raw_element.text):
            raise ValueError("External references inside <style> are not allowed.")
        clean_element.text = raw_element.text

    for child in list(raw_element):
        clean_child = sanitize_element(child, is_root=False)
        clean_element.append(clean_child)
        if child.tail:
            clean_child.tail = child.tail

    return clean_element


def serialize_svg(root: ET.Element) -> str:
    """Serialize a cleaned SVG tree into a compact XML string."""

    svg = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return re.sub(r">\s+<", "><", svg).strip()


def render_svg(svg: str) -> np.ndarray:
    """Render an SVG into a 256x256 RGBA image using CairoSVG."""

    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=256, output_height=256)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return np.asarray(image)


def validate_svg(svg: str) -> dict[str, Any]:
    """Validate SVG structure, constraints, and renderability."""

    result = {
        "valid": False,
        "render_valid": False,
        "error_reason": "",
        "gate_error": "",
        "char_len": len(svg or ""),
        "path_count": 0,
    }

    if not svg:
        result["error_reason"] = "Empty SVG string."
        result["gate_error"] = result["error_reason"]
        return result
    if not svg.startswith("<svg"):
        result["error_reason"] = "SVG must begin with <svg."
        result["gate_error"] = result["error_reason"]
        return result
    if len(svg) > 16000:
        result["error_reason"] = "SVG exceeds 16000 characters."
        result["gate_error"] = result["error_reason"]
        return result
    if not re.search(r"<svg\b[^>]*\bxmlns=['\"]http://www\.w3\.org/2000/svg['\"]", svg):
        result["error_reason"] = "SVG root is missing the xmlns attribute."
        result["gate_error"] = result["error_reason"]
        return result

    try:
        root = ET.fromstring(svg)
    except Exception as exc:
        result["error_reason"] = f"Malformed XML: {exc}"
        result["gate_error"] = result["error_reason"]
        return result

    if local_name(root.tag) != "svg":
        result["error_reason"] = "Root tag is not <svg>."
        result["gate_error"] = result["error_reason"]
        return result
    if root.attrib.get("viewBox") != "0 0 256 256":
        result["error_reason"] = "viewBox must be exactly '0 0 256 256'."
        result["gate_error"] = result["error_reason"]
        return result
    if root.attrib.get("width") != "256" or root.attrib.get("height") != "256":
        result["error_reason"] = "width and height must both be '256'."
        result["gate_error"] = result["error_reason"]
        return result

    path_count = 0
    for element in root.iter():
        tag = local_name(element.tag)
        if tag in DISALLOWED_TAGS:
            result["error_reason"] = f"Disallowed SVG tag found during validation: {tag}"
            result["gate_error"] = result["error_reason"]
            return result
        if CANONICAL_TAGS.get(tag.lower()) is None:
            result["error_reason"] = f"Disallowed tag found during validation: {tag}"
            result["gate_error"] = result["error_reason"]
            return result
        if tag == "path":
            path_count += 1
        for key, value in element.attrib.items():
            attr_name = local_name(key)
            attr_value = str(value)
            lowered = attr_name.lower()
            if lowered.startswith("on"):
                result["error_reason"] = f"Event handler attribute found: {attr_name}"
                result["gate_error"] = result["error_reason"]
                return result
            if lowered in {"href", "xlink:href"} and attr_value.strip() and not attr_value.strip().startswith("#"):
                result["error_reason"] = f"External href found: {attr_value}"
                result["gate_error"] = result["error_reason"]
                return result
            if contains_external_reference(attr_value):
                result["error_reason"] = f"External reference found: {attr_value}"
                result["gate_error"] = result["error_reason"]
                return result
        if tag == "style" and contains_external_reference(element.text or ""):
            result["error_reason"] = "External references inside <style> are not allowed."
            result["gate_error"] = result["error_reason"]
            return result

    result["path_count"] = path_count
    if path_count > 256:
        result["error_reason"] = "SVG exceeds 256 path elements."
        result["gate_error"] = result["error_reason"]
        return result

    try:
        render_svg(svg)
    except Exception as exc:
        result["error_reason"] = f"Render failure: {exc}"
        result["gate_error"] = result["error_reason"]
        return result

    result["valid"] = True
    result["render_valid"] = True
    return result


def normalize_generation_mode(generation_mode: str) -> str:
    """Normalize a generation mode string."""

    mode = str(generation_mode or "body_only").strip().lower()
    if mode not in {"body_only", "full_svg"}:
        raise ValueError(f"Unsupported generation mode: {generation_mode}")
    return mode


def prepare_candidate(raw_text: str, *, generation_mode: str) -> dict[str, Any]:
    """Convert model output into a cleaned and validated SVG candidate."""

    raw_text = "" if raw_text is None else str(raw_text)
    generation_mode = normalize_generation_mode(generation_mode)

    if generation_mode == "body_only":
        svg_body, normalized_source = extract_svg_body(raw_text)
        wrapped_svg = wrap_svg_body(svg_body)
        parsed_root = ET.fromstring(wrapped_svg)
    else:
        normalized_source = extract_svg(raw_text)
        wrapped_svg = normalized_source.strip()
        parsed_root = ET.fromstring(wrapped_svg)
        svg_body = serialize_children(parsed_root)

    clean_root = sanitize_element(parsed_root, is_root=True)
    clean_svg = serialize_svg(clean_root)
    if generation_mode == "full_svg":
        svg_body = serialize_children(clean_root)

    validation = validate_svg(clean_svg)
    if not validation["valid"]:
        raise ValueError(validation["error_reason"])

    return {
        "raw_text": raw_text,
        "svg_body": svg_body,
        "wrapped_svg": wrapped_svg,
        "clean_svg": clean_svg,
        "cleanup_applied": clean_svg.strip() != normalized_source.strip(),
        "render_valid": validation["render_valid"],
        "char_len": validation["char_len"],
        "path_count": validation["path_count"],
        "gate_error": "",
        "error_reason": "",
        "valid_final": True,
    }


def final_svg_constraints_text(*, generation_mode: str) -> str:
    """Return the prompt constraint block for a generation mode."""

    base = textwrap.dedent(
        """
        SVG constraints:
        - Output must be valid XML
        - width="256" height="256" viewBox="0 0 256 256"
        - Maximum 16000 characters
        - Maximum 256 <path> elements
        - Use only allowed SVG tags and no external references, scripts, animation, or foreignObject
        """
    ).strip()
    mode = normalize_generation_mode(generation_mode)
    if mode == "body_only":
        return base + "\n- Return only the inner SVG content, not the outer <svg> wrapper"
    return base + "\n- Return exactly one complete <svg>...</svg> document and nothing else"


def system_prompt(*, generation_mode: str) -> str:
    """Return the system prompt for one generation mode."""

    mode = normalize_generation_mode(generation_mode)
    if mode == "body_only":
        return (
            "You are an expert SVG body generator for a Kaggle competition. "
            "Return only the inner SVG content, not the outer <svg> wrapper. "
            "Do not include markdown, explanations, backticks, or any extra text. "
            "The notebook will wrap your content in a canonical 256x256 SVG root. "
            "Keep the final wrapped SVG under 16000 characters and at or under 256 path elements. "
            "Prefer compact SVGs with as few paths as possible."
        )
    return (
        "You are an expert SVG generator for a Kaggle competition. "
        "Return exactly one complete SVG document and nothing else. "
        "Do not include markdown, explanations, backticks, or any extra text. "
        "The SVG must begin with <svg and satisfy the 256x256 competition constraints. "
        "Prefer compact SVGs with as few paths as possible."
    )


def build_user_prompt(
    prompt: str,
    *,
    generation_mode: str,
    config: GenerationConfig,
    previous_output: str = "",
    previous_error: str = "",
) -> str:
    """Build the user prompt for either first-pass generation or repair."""

    prompt_text = str(prompt).strip()
    constraints = final_svg_constraints_text(generation_mode=generation_mode)
    mode = normalize_generation_mode(generation_mode)

    if not previous_output:
        if mode == "body_only":
            return textwrap.dedent(
                f"""
                Create the inner SVG content for this prompt:
                {prompt_text}

                {constraints}
                Do not return the outer <svg> wrapper.
                """
            ).strip()
        return textwrap.dedent(
            f"""
            Create a complete SVG for this prompt:
            {prompt_text}

            {constraints}
            Return exactly one complete SVG document.
            """
        ).strip()

    repair_output = str(previous_output).strip()[: config.max_repair_context_chars]
    repair_error = str(previous_error).strip()
    if mode == "body_only":
        return textwrap.dedent(
            f"""
            The previous SVG body attempt was invalid. Rewrite only the inner SVG content so the wrapped SVG satisfies every constraint.

            Original prompt:
            {prompt_text}

            Validation error:
            {repair_error}

            Previous invalid output:
            {repair_output}

            {constraints}
            Do not return the outer <svg> wrapper.
            """
        ).strip()
    return textwrap.dedent(
        f"""
        The previous full SVG attempt was invalid. Rewrite a complete SVG document so it satisfies every constraint.

        Original prompt:
        {prompt_text}

        Validation error:
        {repair_error}

        Previous invalid output:
        {repair_output}

        {constraints}
        Return exactly one complete SVG document.
        """
    ).strip()


def build_model_input(
    tokenizer: Any,
    prompt: str,
    *,
    generation_mode: str,
    config: GenerationConfig,
    previous_output: str = "",
    previous_error: str = "",
) -> str:
    """Build the chat template input string for Qwen."""

    messages = [
        {"role": "system", "content": system_prompt(generation_mode=generation_mode)},
        {
            "role": "user",
            "content": build_user_prompt(
                prompt,
                generation_mode=generation_mode,
                config=config,
                previous_output=previous_output,
                previous_error=previous_error,
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def active_model_device(runtime: RuntimeBundle) -> torch.device:
    """Return the current device hosting the model weights."""

    return next(runtime.model.parameters()).device


def generate_raw_text(
    runtime: RuntimeBundle,
    prompt: str,
    *,
    generation_mode: str,
    config: GenerationConfig,
    previous_output: str = "",
    previous_error: str = "",
    use_sampling: bool = False,
) -> str:
    """Run one model generation pass for a prompt."""

    if runtime.runtime_device != "cuda":
        raise RuntimeError("LoRA-backed CUDA model is not loaded.")

    input_text = build_model_input(
        runtime.tokenizer,
        prompt,
        generation_mode=generation_mode,
        config=config,
        previous_output=previous_output,
        previous_error=previous_error,
    )
    tokenized = runtime.tokenizer(input_text, return_tensors="pt", truncation=True)
    model_device = active_model_device(runtime)
    inputs = {name: tensor.to(model_device) for name, tensor in tokenized.items()}
    input_length = inputs["input_ids"].shape[1]
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": use_sampling,
        "pad_token_id": runtime.tokenizer.eos_token_id,
        "eos_token_id": runtime.tokenizer.eos_token_id,
        "use_cache": True,
    }
    if use_sampling:
        generation_kwargs["temperature"] = config.retry_temperature
        generation_kwargs["top_p"] = config.retry_top_p

    with torch.no_grad():
        outputs = runtime.model.generate(**inputs, **generation_kwargs)

    generated_tokens = outputs[0][input_length:]
    return runtime.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def infer_svg_candidate(
    prompt: str,
    runtime: RuntimeBundle,
    *,
    generation_mode: str,
    config: GenerationConfig,
    allow_sampled_repair: bool,
    progress_label: str = "",
    row_id: str = "",
) -> dict[str, Any]:
    """Generate a candidate SVG with an optional sampled repair attempt."""

    generation_mode = normalize_generation_mode(generation_mode)
    last_raw_text = ""
    attempt_errors: list[str] = []
    max_attempts = 1 + int(bool(allow_sampled_repair))
    label_prefix = f"[{progress_label}] " if progress_label else ""

    for attempt in range(max_attempts):
        raw_text = ""
        try:
            if config.verbose_progress:
                attempt_strategy = "sampled_repair" if attempt > 0 else "deterministic"
                print(f"{label_prefix}row {row_id}: attempt {attempt + 1}/{max_attempts} ({attempt_strategy})")
            raw_text = generate_raw_text(
                runtime,
                prompt,
                generation_mode=generation_mode,
                config=config,
                previous_output=last_raw_text,
                previous_error=attempt_errors[-1] if attempt_errors else "",
                use_sampling=allow_sampled_repair and attempt > 0,
            )
            prepared = prepare_candidate(raw_text, generation_mode=generation_mode)
            prepared["attempt_count"] = attempt + 1
            prepared["selected_strategy"] = "sampled_repair" if attempt > 0 else "deterministic"
            if config.verbose_progress:
                print(f"{label_prefix}row {row_id}: success on attempt {attempt + 1}/{max_attempts}")
            return prepared
        except Exception as exc:
            if raw_text:
                last_raw_text = raw_text
            attempt_errors.append(f"attempt {attempt + 1}: {exc}")
            if config.verbose_progress:
                print(f"{label_prefix}row {row_id}: attempt {attempt + 1}/{max_attempts} failed: {exc}")
            release_runtime_memory()

    raise RowInferenceError(
        f"Failed after {max_attempts} attempts. {' | '.join(attempt_errors)}",
        raw_text=last_raw_text,
    )


def _base_result_record(
    row_id: str,
    prompt: str,
    *,
    batch_id: str = "",
    row_index: int = 0,
    generation_mode: str,
    reference_gate_valid: bool = False,
    repair_strategy: str = "",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "prompt": prompt,
        "batch_id": batch_id,
        "row_index": row_index,
        "generation_mode": normalize_generation_mode(generation_mode),
        "reference_gate_valid": reference_gate_valid,
        "repair_strategy": repair_strategy,
        "first_pass_success": False,
        "raw_text": "",
        "svg_body": "",
        "wrapped_svg": "",
        "clean_svg": "",
        "attempt_count": 0,
        "selected_strategy": "",
        "valid_final": False,
        "gate_error": "",
        "error_reason": "",
        "char_len": 0,
        "path_count": 0,
        "runtime_seconds": 0.0,
        "surrogate_score": 0.0,
    }


def build_success_record(
    row_id: str,
    prompt: str,
    prepared: dict[str, Any],
    *,
    batch_id: str = "",
    row_index: int = 0,
    generation_mode: str,
    reference_gate_valid: bool = False,
    repair_strategy: str = "",
    first_pass_success: bool,
    runtime_seconds: float,
    surrogate_score: float = 0.0,
) -> dict[str, Any]:
    """Build a normalized record for a successful inference."""

    record = _base_result_record(
        row_id,
        prompt,
        batch_id=batch_id,
        row_index=row_index,
        generation_mode=generation_mode,
        reference_gate_valid=reference_gate_valid,
        repair_strategy=repair_strategy,
    )
    record.update(prepared)
    record["first_pass_success"] = first_pass_success
    record["runtime_seconds"] = float(runtime_seconds)
    record["surrogate_score"] = float(surrogate_score)
    return normalize_results_df(pd.DataFrame([record])).to_dict(orient="records")[0]


def build_failed_record(
    row_id: str,
    prompt: str,
    exc: RowInferenceError,
    *,
    batch_id: str = "",
    row_index: int = 0,
    generation_mode: str,
    reference_gate_valid: bool = False,
    repair_strategy: str = "",
    first_pass_success: bool,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build a normalized record for an invalid first-pass or repair attempt."""

    record = _base_result_record(
        row_id,
        prompt,
        batch_id=batch_id,
        row_index=row_index,
        generation_mode=generation_mode,
        reference_gate_valid=reference_gate_valid,
        repair_strategy=repair_strategy,
    )
    raw_text = exc.raw_text or ""
    try:
        svg_body, wrapped_svg = extract_svg_body(raw_text)
    except Exception:
        svg_body = extract_svg(raw_text)
        wrapped_svg = raw_text
    record.update(
        {
            "first_pass_success": first_pass_success,
            "raw_text": raw_text,
            "svg_body": svg_body,
            "wrapped_svg": wrapped_svg,
            "clean_svg": "",
            "attempt_count": max(1, 1 + int(repair_strategy.endswith("sampled_repair"))),
            "selected_strategy": "failed_first_pass" if not repair_strategy else repair_strategy,
            "valid_final": False,
            "gate_error": str(exc),
            "error_reason": str(exc),
            "runtime_seconds": float(runtime_seconds),
        }
    )
    return normalize_results_df(pd.DataFrame([record])).to_dict(orient="records")[0]


def build_zero_score_record(
    row_id: str,
    prompt: str,
    *,
    batch_id: str = "",
    row_index: int = 0,
    generation_mode: str,
    reference_gate_valid: bool = False,
    repair_strategy: str,
    runtime_seconds: float,
    gate_error: str,
) -> dict[str, Any]:
    """Build the final fallback record that intentionally scores zero."""

    record = _base_result_record(
        row_id,
        prompt,
        batch_id=batch_id,
        row_index=row_index,
        generation_mode=generation_mode,
        reference_gate_valid=reference_gate_valid,
        repair_strategy=repair_strategy,
    )
    record.update(
        {
            "raw_text": "",
            "svg_body": "",
            "wrapped_svg": ZERO_SCORE_SVG,
            "clean_svg": ZERO_SCORE_SVG,
            "attempt_count": 0,
            "selected_strategy": "zero_score",
            "valid_final": True,
            "gate_error": gate_error,
            "error_reason": gate_error,
            "char_len": len(ZERO_SCORE_SVG),
            "runtime_seconds": float(runtime_seconds),
        }
    )
    return normalize_results_df(pd.DataFrame([record])).to_dict(orient="records")[0]


def normalize_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize result DataFrame column types and order."""

    normalized = results_df.copy()
    for column in RESULT_COLUMNS:
        if column not in normalized.columns:
            if column in TEXT_RESULT_COLUMNS:
                normalized[column] = ""
            elif column in BOOL_RESULT_COLUMNS:
                normalized[column] = False
            else:
                normalized[column] = 0
    for column in TEXT_RESULT_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str)
    for column in BOOL_RESULT_COLUMNS:
        normalized[column] = normalized[column].fillna(False).astype(bool)
    for column in INT_RESULT_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype(int)
    for column in FLOAT_RESULT_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0).astype(float)
    return normalized[RESULT_COLUMNS].copy()


def select_batch_df(test_df: pd.DataFrame, *, batch_id: int, batch_count: int = 4) -> pd.DataFrame:
    """Select a fixed contiguous slice of test rows for one notebook."""

    if batch_id < 1 or batch_id > batch_count:
        raise ValueError(f"batch_id must be in [1, {batch_count}]")
    total_rows = len(test_df)
    batch_size = total_rows // batch_count
    start = (batch_id - 1) * batch_size
    end = total_rows if batch_id == batch_count else start + batch_size
    batch_df = test_df.iloc[start:end].copy().reset_index(drop=True)
    batch_df["row_index"] = range(start, end)
    return batch_df


def select_row_slice(test_df: pd.DataFrame, start_row: int, end_row: int) -> pd.DataFrame:
    """Select an exact contiguous row slice from the test frame."""

    total_rows = len(test_df)
    if start_row < 0 or end_row <= start_row or end_row > total_rows:
        raise ValueError(f"Row slice must satisfy 0 <= start < end <= {total_rows}")
    slice_df = test_df.iloc[start_row:end_row].copy().reset_index(drop=True)
    slice_df["row_index"] = range(start_row, end_row)
    return slice_df


def run_first_pass_batch(
    batch_df: pd.DataFrame,
    runtime: RuntimeBundle,
    *,
    batch_id: int,
    generation_mode: str,
    config: GenerationConfig,
) -> pd.DataFrame:
    """Run the deterministic first pass for one test batch."""

    records: list[dict[str, Any]] = []
    generation_mode = normalize_generation_mode(generation_mode)
    for local_index, row in enumerate(batch_df.to_dict(orient="records"), start=1):
        row_id = row["id"]
        prompt = row["prompt"]
        row_index = int(row["row_index"])
        started_at = time.perf_counter()
        try:
            prepared = infer_svg_candidate(
                prompt,
                runtime,
                generation_mode=generation_mode,
                config=config,
                allow_sampled_repair=False,
                progress_label=f"batch_{batch_id} {local_index}/{len(batch_df)}",
                row_id=row_id,
            )
            runtime_seconds = time.perf_counter() - started_at
            records.append(
                build_success_record(
                    row_id,
                    prompt,
                    prepared,
                    batch_id=str(batch_id),
                    row_index=row_index,
                    generation_mode=generation_mode,
                    first_pass_success=True,
                    runtime_seconds=runtime_seconds,
                )
            )
        except RowInferenceError as exc:
            runtime_seconds = time.perf_counter() - started_at
            records.append(
                build_failed_record(
                    row_id,
                    prompt,
                    exc,
                    batch_id=str(batch_id),
                    row_index=row_index,
                    generation_mode=generation_mode,
                    first_pass_success=False,
                    runtime_seconds=runtime_seconds,
                )
            )
        release_runtime_memory()
    return normalize_results_df(pd.DataFrame(records))


def _write_result_artifacts(results_df: pd.DataFrame, results_path: Path, success_path: Path, failed_path: Path) -> None:
    """Write full, success-only, and failed-only result artifacts."""

    results_df = normalize_results_df(results_df).sort_values("row_index").drop_duplicates(subset=["id"], keep="last")
    results_df.to_csv(results_path, index=False)

    success_df = results_df[results_df["first_pass_success"]].copy()
    success_submission_df = success_df[["id", "clean_svg"]].rename(columns={"clean_svg": "svg"})
    success_submission_df.to_csv(success_path, index=False)

    failed_df = results_df[~results_df["first_pass_success"]].copy()
    failed_df.to_csv(failed_path, index=False)


def save_batch_outputs(results_df: pd.DataFrame, batch_dir: Path) -> None:
    """Write the full, success-only, and failed-only batch CSV artifacts."""

    batch_dir.mkdir(parents=True, exist_ok=True)
    _write_result_artifacts(
        results_df,
        batch_dir / "batch_results.csv",
        batch_dir / "batch_success_submission.csv",
        batch_dir / "batch_failed_rows.csv",
    )


def save_partial_batch_outputs(results_df: pd.DataFrame, batch_dir: Path, part_name: str) -> None:
    """Write split-batch partial artifacts for one half of a batch."""

    partial_dir = batch_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    _write_result_artifacts(
        results_df,
        partial_dir / f"{part_name}_results.csv",
        partial_dir / f"{part_name}_success_submission.csv",
        partial_dir / f"{part_name}_failed_rows.csv",
    )


def load_partial_batch_outputs(batch_dir: Path, expected_parts: Sequence[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load available partial batch outputs for the requested split parts."""

    partial_dir = batch_dir / "partials"
    frames: dict[str, pd.DataFrame] = {}
    missing_parts: list[str] = []
    for part_name in expected_parts:
        path = partial_dir / f"{part_name}_results.csv"
        if not path.exists():
            missing_parts.append(part_name)
            continue
        frames[part_name] = normalize_results_df(pd.read_csv(path, keep_default_na=False))
    return frames, missing_parts


def finalize_split_batch_outputs(batch_dir: Path, expected_row_count: int = 250) -> dict[str, Any]:
    """Combine split-batch partial outputs into canonical batch artifacts when ready."""

    partial_dir = batch_dir / "partials"
    marker_path = partial_dir / "split_run_state.json"
    expected_parts: list[str]
    if marker_path.exists():
        marker_payload = json.loads(marker_path.read_text())
        expected_parts = [str(part) for part in marker_payload.get("expected_parts", [])]
    else:
        expected_parts = sorted(path.name[: -len("_results.csv")] for path in partial_dir.glob("*_results.csv"))

    if not expected_parts:
        return {"ready": False, "missing_parts": [], "rows": 0, "batch_dir": str(batch_dir)}

    partial_frames, missing_parts = load_partial_batch_outputs(batch_dir, expected_parts)
    if missing_parts:
        return {
            "ready": False,
            "missing_parts": missing_parts,
            "rows": sum(len(frame) for frame in partial_frames.values()),
            "batch_dir": str(batch_dir),
        }

    combined_df = normalize_results_df(pd.concat(partial_frames.values(), ignore_index=True))
    combined_df = combined_df.sort_values("row_index").drop_duplicates(subset=["id"], keep="last")

    if len(combined_df) != expected_row_count:
        raise RuntimeError(
            f"Expected {expected_row_count} combined rows for split batch, found {len(combined_df)}."
        )

    row_indexes = combined_df["row_index"].astype(int).tolist()
    if len(set(row_indexes)) != expected_row_count:
        raise RuntimeError("Combined split batch contains duplicate row_index values.")
    if row_indexes and max(row_indexes) - min(row_indexes) + 1 != expected_row_count:
        raise RuntimeError("Combined split batch does not cover one contiguous row range.")

    save_batch_outputs(combined_df, batch_dir)
    return {
        "ready": True,
        "missing_parts": [],
        "rows": len(combined_df),
        "batch_dir": str(batch_dir),
        "first_row_index": int(min(row_indexes)) if row_indexes else 0,
        "last_row_index": int(max(row_indexes)) if row_indexes else 0,
        "expected_parts": expected_parts,
    }


def summarize_batch_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize successes and failures for one batch run."""

    results_df = normalize_results_df(results_df)
    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "first_pass_successes",
                "first_pass_failures",
                "mean_runtime_seconds",
                "mean_char_len_successes",
            ],
            "value": [
                int(len(results_df)),
                int(results_df["first_pass_success"].sum()),
                int((~results_df["first_pass_success"]).sum()),
                float(results_df["runtime_seconds"].mean() if not results_df.empty else 0.0),
                float(
                    results_df.loc[results_df["first_pass_success"], "char_len"].mean()
                    if results_df["first_pass_success"].any()
                    else 0.0
                ),
            ],
        }
    )


def audit_reference_gate(train_df: pd.DataFrame) -> pd.DataFrame:
    """Run the current SVG gate against every reference row in train.csv."""

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(train_df.to_dict(orient="records")):
        gate = validate_svg(row["svg"])
        records.append(
            {
                "id": row["id"],
                "row_index": row_index,
                "prompt": row["prompt"],
                "reference_gate_valid": bool(gate["valid"]),
                "gate_error": gate["gate_error"],
                "char_len": int(gate["char_len"]),
                "path_count": int(gate["path_count"]),
            }
        )
    return pd.DataFrame(records)


def summarize_reference_gate_audit(audit_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize overall gate pass rates and top failure reasons."""

    summary = pd.DataFrame(
        {
            "metric": ["rows", "reference_gate_valid", "reference_gate_invalid"],
            "value": [
                int(len(audit_df)),
                int(audit_df["reference_gate_valid"].sum()),
                int((~audit_df["reference_gate_valid"]).sum()),
            ],
        }
    )
    top_errors = (
        audit_df.loc[~audit_df["reference_gate_valid"], "gate_error"]
        .fillna("")
        .replace("", "<empty>")
        .value_counts()
        .rename_axis("gate_error")
        .reset_index(name="count")
        .head(10)
    )
    return summary, top_errors


def select_smoke_test_rows(
    train_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    *,
    rows: int,
    seed: int,
) -> pd.DataFrame:
    """Select smoke-test rows, prioritizing reference SVGs that pass the gate."""

    rng = random.Random(seed)
    merged = train_df.merge(audit_df[["id", "reference_gate_valid"]], on="id", how="left")
    valid_rows = merged[merged["reference_gate_valid"]].copy()
    invalid_rows = merged[~merged["reference_gate_valid"]].copy()

    valid_records = valid_rows.to_dict(orient="records")
    invalid_records = invalid_rows.to_dict(orient="records")
    rng.shuffle(valid_records)
    rng.shuffle(invalid_records)

    selected = valid_records[:rows]
    if len(selected) < rows:
        selected.extend(invalid_records[: rows - len(selected)])
    return pd.DataFrame(selected).reset_index(drop=True)


def svg_to_grayscale(svg: str) -> np.ndarray:
    """Render and flatten an SVG into grayscale for surrogate scoring."""

    rgba = render_svg(svg).astype(np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def edge_f1_score(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Compute an edge-based F1 proxy between two rendered images."""

    edges_a = canny(image_a, sigma=1.0)
    edges_b = canny(image_b, sigma=1.0)
    count_a = int(edges_a.sum())
    count_b = int(edges_b.sum())
    if count_a == 0 and count_b == 0:
        return 1.0
    if count_a == 0 or count_b == 0:
        return 0.0
    true_positive = int(np.logical_and(edges_a, edges_b).sum())
    precision = true_positive / max(count_a, 1)
    recall = true_positive / max(count_b, 1)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def structural_similarity_proxy(candidate_svg: str, reference_svg: str) -> float:
    """Compute a simple tree-structure similarity proxy between two SVGs."""

    try:
        candidate_root = ET.fromstring(candidate_svg)
        reference_root = ET.fromstring(reference_svg)
    except Exception:
        return 0.0
    candidate_signature = " ".join(local_name(element.tag) for element in candidate_root.iter())
    reference_signature = " ".join(local_name(element.tag) for element in reference_root.iter())
    return float(SequenceMatcher(None, candidate_signature, reference_signature).ratio())


def surrogate_quality_score(candidate_svg: str, reference_svg: str) -> float:
    """Compute the local train-set proxy score used by the analysis notebook."""

    candidate_gate = validate_svg(candidate_svg)
    reference_gate = validate_svg(reference_svg)
    if not candidate_gate["valid"] or not reference_gate["valid"]:
        return 0.0

    candidate_gray = svg_to_grayscale(candidate_svg)
    reference_gray = svg_to_grayscale(reference_svg)
    visual_score = 0.7 * structural_similarity(candidate_gray, reference_gray, data_range=1.0) + 0.3 * edge_f1_score(candidate_gray, reference_gray)
    structural_score = max(structural_similarity_proxy(candidate_svg, reference_svg), 1e-6)
    compactness_score = math.exp(-abs(math.log((len(candidate_svg) + 50) / (len(reference_svg) + 50))))
    return float((max(visual_score, 1e-6) ** 0.85) * (structural_score ** 0.12) * (max(compactness_score, 1e-6) ** 0.03))


def run_smoke_test(
    smoke_df: pd.DataFrame,
    runtime: RuntimeBundle,
    *,
    generation_mode: str,
    allow_sampled_repair: bool,
    config: GenerationConfig,
) -> pd.DataFrame:
    """Run one generation mode across the selected train smoke-test rows."""

    results: list[dict[str, Any]] = []
    generation_mode = normalize_generation_mode(generation_mode)
    for sample_index, row in enumerate(smoke_df.to_dict(orient="records"), start=1):
        row_id = row["id"]
        prompt = row["prompt"]
        reference_gate_valid = bool(row.get("reference_gate_valid", False))
        started_at = time.perf_counter()
        try:
            prepared = infer_svg_candidate(
                prompt,
                runtime,
                generation_mode=generation_mode,
                config=config,
                allow_sampled_repair=allow_sampled_repair,
                progress_label=f"smoke:{generation_mode} {sample_index}/{len(smoke_df)}",
                row_id=row_id,
            )
            runtime_seconds = time.perf_counter() - started_at
            surrogate_score = (
                surrogate_quality_score(prepared["clean_svg"], row["svg"]) if reference_gate_valid else 0.0
            )
            results.append(
                build_success_record(
                    row_id,
                    prompt,
                    prepared,
                    row_index=sample_index - 1,
                    generation_mode=generation_mode,
                    reference_gate_valid=reference_gate_valid,
                    repair_strategy="smoke_test",
                    first_pass_success=prepared["valid_final"],
                    runtime_seconds=runtime_seconds,
                    surrogate_score=surrogate_score,
                )
            )
        except RowInferenceError as exc:
            runtime_seconds = time.perf_counter() - started_at
            results.append(
                build_failed_record(
                    row_id,
                    prompt,
                    exc,
                    row_index=sample_index - 1,
                    generation_mode=generation_mode,
                    reference_gate_valid=reference_gate_valid,
                    repair_strategy="smoke_test",
                    first_pass_success=False,
                    runtime_seconds=runtime_seconds,
                )
            )
        release_runtime_memory()
    return normalize_results_df(pd.DataFrame(results))


def summarize_smoke_test(smoke_results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate smoke-test metrics by generation mode."""

    smoke_results_df = normalize_results_df(smoke_results_df)
    grouped = []
    for generation_mode, group in smoke_results_df.groupby("generation_mode", sort=True):
        comparable = group[group["reference_gate_valid"]]
        grouped.append(
            {
                "generation_mode": generation_mode,
                "rows": int(len(group)),
                "gate_valid_count": int(group["valid_final"].sum()),
                "zero_score_count": int((~group["valid_final"]).sum()),
                "mean_surrogate_score": float(comparable["surrogate_score"].mean() if not comparable.empty else 0.0),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean() if not group.empty else 0.0),
                "mean_char_len": float(group["char_len"].mean() if not group.empty else 0.0),
                "mean_path_count": float(group["path_count"].mean() if not group.empty else 0.0),
            }
        )
    return pd.DataFrame(grouped)


def choose_generation_mode(smoke_summary_df: pd.DataFrame) -> str:
    """Choose the winning generation mode using the agreed tie-break rules."""

    if smoke_summary_df.empty:
        raise RuntimeError("Smoke summary is empty.")
    ranking = smoke_summary_df.sort_values(
        by=["gate_valid_count", "mean_surrogate_score", "mean_runtime_seconds"],
        ascending=[False, False, True],
    )
    return str(ranking.iloc[0]["generation_mode"])


def save_analysis_artifacts(
    paths: PipelinePaths,
    *,
    audit_df: pd.DataFrame,
    smoke_results_df: pd.DataFrame,
    smoke_summary_df: pd.DataFrame,
    recommended_generation_mode: str,
    fixed_strategy: str,
) -> None:
    """Persist the analysis notebook outputs into the analysis directory."""

    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(paths.analysis_dir / "train_reference_gate_audit.csv", index=False)
    smoke_results_df.to_csv(paths.analysis_dir / "train_smoke_test_results.csv", index=False)
    recommendation = {
        "recommended_generation_mode": recommended_generation_mode,
        "fixed_strategy": fixed_strategy,
        "smoke_summary": smoke_summary_df.to_dict(orient="records"),
        "smoke_test_rows": smoke_results_df["id"].drop_duplicates().tolist(),
    }
    paths.analysis_recommendation_path.write_text(json.dumps(recommendation, indent=2))


def load_analysis_recommendation(paths: PipelinePaths) -> dict[str, Any]:
    """Load the analysis recommendation JSON if present."""

    if not paths.analysis_recommendation_path.exists():
        raise FileNotFoundError(
            f"Missing analysis recommendation at {paths.analysis_recommendation_path}. Run submission_analysis.ipynb first."
        )
    return json.loads(paths.analysis_recommendation_path.read_text())


def load_batch_results(paths: PipelinePaths, *, batch_id: int) -> pd.DataFrame:
    """Load one batch_results.csv artifact."""

    path = batch_output_dir(paths, batch_id) / "batch_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing batch results for batch {batch_id}: {path}")
    return normalize_results_df(pd.read_csv(path, keep_default_na=False))


def load_all_batch_results(paths: PipelinePaths, *, batch_count: int = 4) -> tuple[pd.DataFrame, list[int]]:
    """Load all available batch results and report missing batches."""

    frames: list[pd.DataFrame] = []
    missing_batches: list[int] = []
    for batch_id in range(1, batch_count + 1):
        path = batch_output_dir(paths, batch_id) / "batch_results.csv"
        if not path.exists():
            missing_batches.append(batch_id)
            continue
        frames.append(normalize_results_df(pd.read_csv(path, keep_default_na=False)))
    if not frames:
        return normalize_results_df(pd.DataFrame(columns=RESULT_COLUMNS)), missing_batches
    return normalize_results_df(pd.concat(frames, ignore_index=True)), missing_batches


def split_success_and_failures(batch_results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split merged batch results into successful and failed first-pass rows."""

    batch_results_df = normalize_results_df(batch_results_df)
    successes = batch_results_df[batch_results_df["first_pass_success"]].copy()
    failures = batch_results_df[~batch_results_df["first_pass_success"]].copy()
    return successes, failures


def repair_failed_rows(
    failed_df: pd.DataFrame,
    runtime: RuntimeBundle,
    *,
    recommended_generation_mode: str,
    config: GenerationConfig,
) -> pd.DataFrame:
    """Repair failed rows using the agreed fallback order."""

    recommended_generation_mode = normalize_generation_mode(recommended_generation_mode)
    alternate_generation_mode = "full_svg" if recommended_generation_mode == "body_only" else "body_only"
    repair_plan = [
        ("recommended_sampled_repair", recommended_generation_mode, True),
        ("alternate_deterministic", alternate_generation_mode, False),
        ("alternate_sampled_repair", alternate_generation_mode, True),
    ]

    repaired_records: list[dict[str, Any]] = []
    for repair_index, row in enumerate(failed_df.to_dict(orient="records"), start=1):
        row_id = row["id"]
        prompt = row["prompt"]
        final_record: dict[str, Any] | None = None
        accumulated_runtime = 0.0

        for repair_strategy, generation_mode, allow_sampled_repair in repair_plan:
            started_at = time.perf_counter()
            try:
                prepared = infer_svg_candidate(
                    prompt,
                    runtime,
                    generation_mode=generation_mode,
                    config=config,
                    allow_sampled_repair=allow_sampled_repair,
                    progress_label=f"repair {repair_index}/{len(failed_df)}:{repair_strategy}",
                    row_id=row_id,
                )
                accumulated_runtime += time.perf_counter() - started_at
                final_record = build_success_record(
                    row_id,
                    prompt,
                    prepared,
                    batch_id=str(row.get("batch_id", "")),
                    row_index=int(row.get("row_index", 0)),
                    generation_mode=generation_mode,
                    repair_strategy=repair_strategy,
                    first_pass_success=False,
                    runtime_seconds=accumulated_runtime,
                )
                break
            except RowInferenceError:
                accumulated_runtime += time.perf_counter() - started_at
                release_runtime_memory()

        if final_record is None:
            final_record = build_zero_score_record(
                row_id,
                prompt,
                batch_id=str(row.get("batch_id", "")),
                row_index=int(row.get("row_index", 0)),
                generation_mode=recommended_generation_mode,
                repair_strategy="zero_score_fallback",
                runtime_seconds=accumulated_runtime,
                gate_error="All repair strategies failed.",
            )
        repaired_records.append(final_record)
        release_runtime_memory()

    return normalize_results_df(pd.DataFrame(repaired_records))


def summarize_repair_results(repaired_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize repair outcomes after the master repair pass."""

    repaired_df = normalize_results_df(repaired_df)
    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "gate_valid_after_repair",
                "zero_score_fallbacks",
                "mean_runtime_seconds",
            ],
            "value": [
                int(len(repaired_df)),
                int(repaired_df["selected_strategy"].ne("zero_score").sum()),
                int(repaired_df["selected_strategy"].eq("zero_score").sum()),
                float(repaired_df["runtime_seconds"].mean() if not repaired_df.empty else 0.0),
            ],
        }
    )


def write_repair_outputs(paths: PipelinePaths, repaired_df: pd.DataFrame) -> None:
    """Write repaired-row artifacts into the merge directory."""

    paths.merge_dir.mkdir(parents=True, exist_ok=True)
    repaired_df = normalize_results_df(repaired_df)
    repaired_df.to_csv(paths.merge_dir / "repaired_rows.csv", index=False)
    zero_score_rows = repaired_df[repaired_df["selected_strategy"].eq("zero_score")].copy()
    zero_score_rows.to_csv(paths.merge_dir / "remaining_failures.csv", index=False)


def assemble_final_submission(
    sample_submission_df: pd.DataFrame,
    first_pass_success_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge successful first-pass rows and repaired rows into final submission order."""

    successes = normalize_results_df(first_pass_success_df)
    repairs = normalize_results_df(repaired_df)
    combined = pd.concat([successes, repairs], ignore_index=True)
    combined = combined.drop_duplicates(subset=["id"], keep="last")

    submission_df = sample_submission_df[["id"]].merge(
        combined[["id", "clean_svg"]],
        on="id",
        how="left",
    )
    if submission_df["clean_svg"].isna().any():
        missing_ids = submission_df.loc[submission_df["clean_svg"].isna(), "id"].tolist()
        raise RuntimeError(f"Final submission is missing rows: {missing_ids[:10]}")
    submission_df = submission_df.rename(columns={"clean_svg": "svg"})
    return submission_df


def validate_submission_frame(
    submission_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the final submission frame against sample order and the SVG gate."""

    assert list(submission_df.columns) == ["id", "svg"], f"Unexpected submission columns: {list(submission_df.columns)}"
    assert list(submission_df.columns) == list(sample_submission_df.columns), (
        "Submission columns do not match sample_submission.csv."
    )
    assert len(submission_df) == len(sample_submission_df) == 1000, f"Submission row count mismatch: {len(submission_df)}"
    assert submission_df["id"].tolist() == sample_submission_df["id"].tolist(), (
        "Submission IDs do not match sample_submission.csv order."
    )
    assert not submission_df.isna().any().any(), "Submission contains missing values."
    assert submission_df["svg"].astype(str).str.strip().str.len().gt(0).all(), "Submission contains empty SVG values."
    validation_df = pd.DataFrame(submission_df["svg"].map(validate_svg).tolist())
    return validation_df


def write_final_submission(paths: PipelinePaths, submission_df: pd.DataFrame) -> Path:
    """Write the final submission.csv into the merge directory."""

    paths.merge_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.merge_dir / "submission.csv"
    submission_df.to_csv(output_path, index=False)
    return output_path
