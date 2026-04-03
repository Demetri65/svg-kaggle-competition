#!/usr/bin/env python3
from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_NOTEBOOKS_DIR = ROOT / "archive" / "notebooks" / "evaluation"
BASE_NOTEBOOK = EVALUATION_NOTEBOOKS_DIR / "DL_Midterm_Eval.ipynb"


CELL4_TEMPLATE = textwrap.dedent(
    r'''
    import io
    import gc
    import json
    import re
    import time
    import torch
    try:
        import cairosvg
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing Python package 'cairosvg'. Run the package-install cell in this notebook with the active VS Code kernel."
        ) from exc
    except OSError as exc:
        raise OSError(
            "cairosvg is installed, but the native Cairo library is missing. On macOS run `brew install cairo`, then restart the kernel."
        ) from exc
    import numpy as np
    import pandas as pd
    import xml.etree.ElementTree as ET

    from pathlib import Path
    from PIL import Image
    from tqdm import tqdm
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lxml import etree

    # =========================
    # EXPERIMENT CONFIG
    # =========================
    EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
    EXPERIMENT_KIND = "__EXPERIMENT_KIND__"

    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    ARTIFACT_PREFIX = "qwen25coder15b_canon_nosvgo_len2048"
    PROJECT_ROOT = Path.cwd()
    PROJECT_ROOT_CANDIDATES = [
        PROJECT_ROOT,
        Path("/content/drive/MyDrive/svg-kaggle-comptetition"),
        Path("/content/drive/MyDrive/Colab Notebooks/svg-kaggle-comptetition"),
        Path("/content/drive/MyDrive/DL Midterm"),
    ]
    MERGED_WEIGHT_FILES = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )

    SUBMISSION_CSV = "__SUBMISSION_CSV__"
    DEBUG_CSV = "__DEBUG_CSV__"
    SUMMARY_JSON = "__SUMMARY_JSON__"
    SMOKE_SUBMISSION_CSV = "__SMOKE_SUBMISSION_CSV__"
    SMOKE_DEBUG_CSV = "__SMOKE_DEBUG_CSV__"
    SMOKE_SUMMARY_JSON = "__SMOKE_SUMMARY_JSON__"
    SMOKE_PASS1ONLY_SUBMISSION_CSV = "__SMOKE_PASS1ONLY_SUBMISSION_CSV__"
    SMOKE_PASS1ONLY_DEBUG_CSV = "__SMOKE_PASS1ONLY_DEBUG_CSV__"
    SMOKE_PASS1ONLY_SUMMARY_JSON = "__SMOKE_PASS1ONLY_SUMMARY_JSON__"

    PASS1_MAX_NEW_TOKENS = 1536
    PASS1_BATCH_SIZE = 128
    MAX_NEW_TOKENS = PASS1_MAX_NEW_TOKENS
    BATCH_SIZE = PASS1_BATCH_SIZE

    BEST_OF_N_PASS2_MAX_NEW_TOKENS = 2048
    BEST_OF_N_PASS2_BATCH_SIZE = 32
    NUM_RESCUE_CANDIDATES = 4

    HYBRID_GREEDY_MAX_NEW_TOKENS = 2048
    HYBRID_GREEDY_BATCH_SIZE = 64
    HYBRID_SAMPLE_MAX_NEW_TOKENS = 2048
    HYBRID_SAMPLE_BATCH_SIZE = 64

    ADAPTIVE_PASS2_MAX_NEW_TOKENS = 3072
    ADAPTIVE_PASS2_BATCH_SIZE = 32

    SAMPLE_TEMPERATURE = 0.25
    SAMPLE_TOP_P = 0.95
    SAMPLE_TOP_K = 20

    MAX_SVG_LENGTH = 16000
    MAX_PATH_COUNT = 256
    RENDER_SIZE = 256
    SVG_NS = "http://www.w3.org/2000/svg"
    STRICT_VIEWBOX = f"0 0 {RENDER_SIZE} {RENDER_SIZE}"
    STRICT_BOX = (0.0, 0.0, float(RENDER_SIZE), float(RENDER_SIZE))
    DIRECT_ROOT_TAGS = {"defs", "title", "desc", "style"}
    DRAWABLE_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "use", "text", "image"}
    ALLOWED_TAGS = {
        "svg", "g", "path", "rect", "circle", "ellipse",
        "line", "polyline", "polygon", "defs", "use",
        "symbol", "clipPath", "mask", "linearGradient",
        "radialGradient", "stop", "text", "tspan",
        "title", "desc", "style", "pattern", "marker", "filter"
    }
    EVENT_HANDLER_RE = re.compile(r"\son[a-zA-Z]+\s*=", re.IGNORECASE)
    EXTERNAL_HREF_RE = re.compile(r"\s(?:href|xlink:href)\s*=\s*[\"']\s*(?:https?:|//)", re.IGNORECASE)
    REMOTE_REF_RE = re.compile(r"@import\b|url\(\s*[\"']?(?:https?:|//)", re.IGNORECASE)
    ROOT_TAG_RE = re.compile(r"<svg\b[^>]*>", flags=re.IGNORECASE | re.DOTALL)


    def has_model_weights(path: Path) -> bool:
        return path.exists() and any((path / name).exists() for name in MERGED_WEIGHT_FILES)


    def first_existing_path(paths, predicate):
        for candidate in paths:
            if predicate(candidate):
                return candidate
        return None


    def latest_run_artifact(run_roots, artifact_dir_name, predicate):
        for runs_root in run_roots:
            if not runs_root.exists():
                continue
            for run_dir in sorted([path for path in runs_root.iterdir() if path.is_dir()], reverse=True):
                candidate = run_dir / artifact_dir_name
                if predicate(candidate):
                    return candidate
        return None


    RUNS_ROOT_CANDIDATES = [root / "runs" for root in PROJECT_ROOT_CANDIDATES]
    MERGED_PATH_CANDIDATES = [
        PROJECT_ROOT / f"{ARTIFACT_PREFIX}_merged",
        *(root / f"{ARTIFACT_PREFIX}_merged" for root in PROJECT_ROOT_CANDIDATES),
        PROJECT_ROOT / "svg-model-merged",
        *(root / "svg-model-merged" for root in PROJECT_ROOT_CANDIDATES),
    ]
    MERGED_PATH = first_existing_path(MERGED_PATH_CANDIDATES, has_model_weights)
    if MERGED_PATH is None:
        MERGED_PATH = latest_run_artifact(RUNS_ROOT_CANDIDATES, f"{ARTIFACT_PREFIX}_merged", has_model_weights)
    if MERGED_PATH is None:
        raise FileNotFoundError("Could not find a merged model. Checked current canon artifact first, then legacy svg-model-merged.")

    TEST_CSV_CANDIDATES = [
        PROJECT_ROOT / "test.csv",
        *(root / "test.csv" for root in PROJECT_ROOT_CANDIDATES),
    ]
    TEST_CSV = str(first_existing_path(TEST_CSV_CANDIDATES, lambda path: path.exists()))
    if TEST_CSV == "None":
        raise FileNotFoundError("Could not find test.csv in the expected project roots.")


    def make_fallback_svg() -> str:
        return (
            f'<svg xmlns="{SVG_NS}" viewBox="{STRICT_VIEWBOX}" width="{RENDER_SIZE}" height="{RENDER_SIZE}">'
            f'<rect width="{RENDER_SIZE}" height="{RENDER_SIZE}" fill="white"/>'
            "</svg>"
        )


    FALLBACK_SVG = make_fallback_svg()

    if EXPERIMENT_KIND == "best_of_n_bad_rows":
        DEBUG_SOURCE_PREFIXES = ["pass1", "sample1", "sample2", "sample3", "sample4"]
        RESCUE_SOURCE_NAMES = ["sample1", "sample2", "sample3", "sample4"]
    elif EXPERIMENT_KIND == "hybrid_retry":
        DEBUG_SOURCE_PREFIXES = ["pass1", "retry_greedy", "retry_sample"]
        RESCUE_SOURCE_NAMES = ["retry_greedy", "retry_sample"]
    elif EXPERIMENT_KIND == "adaptive_token_retry":
        DEBUG_SOURCE_PREFIXES = ["pass1", "retry_cap"]
        RESCUE_SOURCE_NAMES = ["retry_cap"]
    else:
        raise ValueError(f"Unsupported experiment kind: {EXPERIMENT_KIND}")

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Merged model path: {MERGED_PATH}")
    print(f"Test CSV path: {TEST_CSV}")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()

    print("pad:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("bos:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("eos:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("model pad:", model.config.pad_token_id)


    # =========================
    # SVG HELPERS
    # =========================
    def strip_namespace(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag


    def extract_svg(text: str) -> str:
        text = str(text or "").strip()

        match = re.search(r"<svg\b[^>]*>.*?</svg>", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0).strip()

        if "SVG:" in text:
            text = text.split("SVG:", 1)[1].strip()

        start = text.find("<svg")
        if start != -1:
            return text[start:].strip()

        return text


    def render_svg(svg: str, size: int = RENDER_SIZE):
        try:
            png = cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                output_width=size,
                output_height=size,
            )
            img = Image.open(io.BytesIO(png)).convert("RGB")
            return np.array(img)
        except Exception:
            return None


    def get_attr_value(opening_tag: str, attr_name: str):
        pattern = rf"(\s{re.escape(attr_name)}\s*=\s*)([\"'])(.*?)\2"
        match = re.search(pattern, opening_tag, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return None
        return match.group(3)


    def format_number(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        text = f"{value:.12g}"
        if "e" not in text and "." in text:
            text = text.rstrip("0").rstrip(".")
        return text


    def parse_numeric_attr(value):
        if value is None:
            return None
        match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(px)?\s*", str(value))
        if match is None:
            return None
        numeric = float(match.group(1))
        if numeric <= 0:
            return None
        return numeric


    def parse_viewbox(value):
        if value is None:
            return None
        pieces = [piece for piece in re.split(r"[\s,]+", str(value).strip()) if piece]
        if len(pieces) != 4:
            return None
        try:
            x, y, width, height = [float(piece) for piece in pieces]
        except ValueError:
            return None
        if width <= 0 or height <= 0:
            return None
        return (x, y, width, height)


    def derive_source_box(root: ET.Element):
        viewbox = parse_viewbox(root.attrib.get("viewBox"))
        if viewbox is not None:
            return viewbox, "viewBox"

        width = parse_numeric_attr(root.attrib.get("width"))
        height = parse_numeric_attr(root.attrib.get("height"))
        if width is not None and height is not None:
            return (0.0, 0.0, width, height), "width_height"

        return STRICT_BOX, "default"


    def clone_element(element: ET.Element) -> ET.Element:
        return ET.fromstring(ET.tostring(element, encoding="unicode"))


    def serialize_svg_element(root: ET.Element) -> str:
        return ET.tostring(root, encoding="unicode")


    def rebuild_svg_root(root: ET.Element, source_box, source_kind: str):
        new_root = ET.Element(f"{{{SVG_NS}}}svg")
        new_root.set("viewBox", STRICT_VIEWBOX)
        new_root.set("width", str(RENDER_SIZE))
        new_root.set("height", str(RENDER_SIZE))

        for attr_name, attr_value in root.attrib.items():
            if attr_name in {"viewBox", "width", "height"}:
                continue
            new_root.set(attr_name, attr_value)

        source_x, source_y, source_width, source_height = source_box
        needs_transform = source_box != STRICT_BOX
        transform_group = None

        if needs_transform:
            sx = RENDER_SIZE / source_width
            sy = RENDER_SIZE / source_height
            tx = -source_x * sx
            ty = -source_y * sy
            transform_group = ET.Element(f"{{{SVG_NS}}}g")
            transform_group.set(
                "transform",
                f"matrix({format_number(sx)} 0 0 {format_number(sy)} {format_number(tx)} {format_number(ty)})",
            )

        group_inserted = False
        for child in list(root):
            child_copy = clone_element(child)
            child_tag = strip_namespace(child.tag)
            if child_tag in DIRECT_ROOT_TAGS or transform_group is None:
                new_root.append(child_copy)
                continue
            if not group_inserted:
                new_root.append(transform_group)
                group_inserted = True
            transform_group.append(child_copy)

        status = "unchanged" if not needs_transform else f"scaled_from_{source_kind}"
        return serialize_svg_element(new_root), status


    def canonicalize_to_strict_256(svg: str):
        try:
            root = ET.fromstring(svg)
        except Exception:
            return svg, "parse_failed"

        if strip_namespace(root.tag) != "svg":
            return svg, "root_not_svg"

        source_box, source_kind = derive_source_box(root)
        return rebuild_svg_root(root, source_box, source_kind)


    def strict_contract_issues(svg: str) -> list[str]:
        issues: list[str] = []
        opening_match = ROOT_TAG_RE.search(svg)
        if opening_match is None:
            issues.append("strict_parse_failed")
            return issues

        opening_tag = opening_match.group(0)
        if get_attr_value(opening_tag, "xmlns") != SVG_NS:
            issues.append("missing_xmlns")

        try:
            root = ET.fromstring(svg)
        except Exception:
            issues.append("strict_parse_failed")
            return issues

        if strip_namespace(root.tag) != "svg":
            issues.append("root_not_svg")
            return issues

        if root.attrib.get("viewBox") != STRICT_VIEWBOX:
            issues.append("viewbox_not_exact")

        if root.attrib.get("width") != str(RENDER_SIZE) or root.attrib.get("height") != str(RENDER_SIZE):
            issues.append("width_height_not_exact")

        return issues


    def validity_gate(svg: str):
        if not isinstance(svg, str) or not svg.strip():
            return 0, "svg_too_long_or_empty"

        svg = svg.strip()
        if len(svg) > MAX_SVG_LENGTH:
            return 0, "svg_too_long_or_empty"
        if EVENT_HANDLER_RE.search(svg):
            return 0, "disallowed_attr:event_handler"
        if EXTERNAL_HREF_RE.search(svg):
            return 0, "disallowed_ref:external_href"
        if REMOTE_REF_RE.search(svg):
            return 0, "disallowed_ref:remote_url"

        try:
            root = ET.fromstring(svg)
        except Exception:
            return 0, "parse_failed"

        if strip_namespace(root.tag) != "svg":
            return 0, "root_not_svg"

        path_count = 0
        for elem in root.iter():
            tag = strip_namespace(elem.tag)
            if tag not in ALLOWED_TAGS:
                return 0, f"disallowed_tag:{tag}"
            if tag == "path":
                path_count += 1

        if path_count > MAX_PATH_COUNT:
            return 0, "too_many_paths"

        if render_svg(svg) is None:
            return 0, "render_failed"

        return 1, "valid"


    def repair_svg(svg: str) -> str:
        if not svg:
            return svg

        svg = svg.strip()
        start = svg.find("<svg")
        if start != -1:
            svg = svg[start:]
        if "SVG:" in svg:
            svg = svg.split("SVG:", 1)[-1].strip()
        if "</svg>" in svg:
            end = svg.rfind("</svg>")
            svg = svg[: end + len("</svg>")]
        if "<svg" in svg and "</svg>" not in svg:
            svg += "</svg>"
        svg = re.sub(r"[A-Za-z0-9.\-]+$", "", svg)
        svg = re.sub(r"<[^>]*$", "", svg)
        return svg


    def recover_svg_with_lxml(svg: str) -> str:
        if not svg or "<svg" not in svg:
            return svg
        try:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(svg.encode("utf-8"), parser=parser)
            if root is None:
                return svg
            return etree.tostring(root, encoding="unicode")
        except Exception:
            return svg


    def looks_collapsed(svg: str) -> bool:
        try:
            root = ET.fromstring(svg)
        except Exception:
            return True

        drawables = [elem for elem in root.iter() if strip_namespace(elem.tag) in DRAWABLE_TAGS]
        if not drawables:
            return True

        path_drawables = [elem for elem in drawables if strip_namespace(elem.tag) == "path"]
        if path_drawables and all(not elem.attrib.get("d", "").strip() for elem in path_drawables):
            non_path_drawables = [elem for elem in drawables if strip_namespace(elem.tag) != "path"]
            if not non_path_drawables:
                return True

        total_elems = sum(1 for _ in root.iter())
        return total_elems <= 1


    def summarize_stage_failure(source_gate_reason: str, normalized_gate_reason: str) -> str:
        parts = []
        if source_gate_reason and source_gate_reason != "valid":
            parts.append(f"source:{source_gate_reason}")
        if normalized_gate_reason and normalized_gate_reason != "valid" and normalized_gate_reason != source_gate_reason:
            parts.append(f"normalized:{normalized_gate_reason}")
        if parts:
            return ";".join(parts)
        return source_gate_reason or normalized_gate_reason or "unknown_failure"


    def candidate_from_svg(
        raw_text: str,
        extracted_svg: str,
        stage_svg: str,
        reason: str,
        *,
        generated_tokens: int,
        hit_token_cap: bool,
        normalized_model_score: float,
        source: str,
    ):
        source_valid, source_gate_reason = validity_gate(stage_svg)
        if source_valid == 0:
            return None, summarize_stage_failure(source_gate_reason, "")

        final_svg, normalization_status = canonicalize_to_strict_256(stage_svg)
        normalized_valid, normalized_gate_reason = validity_gate(final_svg)
        if normalized_valid == 0:
            return None, summarize_stage_failure(source_gate_reason, normalized_gate_reason)

        strict_issues = strict_contract_issues(final_svg)
        collapsed = looks_collapsed(final_svg)

        candidate = {
            "source": source,
            "reason": reason,
            "gate_reason": normalized_gate_reason,
            "source_gate_reason": source_gate_reason,
            "normalized_gate_reason": normalized_gate_reason,
            "normalization_status": normalization_status,
            "strict_contract": not strict_issues,
            "strict_issues": ",".join(strict_issues),
            "collapsed": collapsed,
            "render_success": normalized_gate_reason == "valid",
            "generated_tokens": int(generated_tokens),
            "hit_token_cap": bool(hit_token_cap),
            "raw_text": raw_text,
            "extracted_svg": extracted_svg,
            "final_svg": final_svg,
            "normalized_model_score": float(normalized_model_score),
            "failure_reasons": "",
        }
        return candidate, normalized_gate_reason


    def clean_svg_output(
        raw_text: str,
        *,
        generated_tokens: int,
        hit_token_cap: bool,
        normalized_model_score: float,
        source: str,
    ):
        cleaned_raw_text = str(raw_text or "")
        extracted_svg = extract_svg(cleaned_raw_text)
        failures = []

        raw_candidate, raw_failure = candidate_from_svg(
            cleaned_raw_text,
            extracted_svg,
            extracted_svg,
            "valid",
            generated_tokens=generated_tokens,
            hit_token_cap=hit_token_cap,
            normalized_model_score=normalized_model_score,
            source=source,
        )
        if raw_candidate is not None:
            return raw_candidate
        failures.append(f"raw={raw_failure}")

        repaired_svg = repair_svg(extracted_svg)
        repaired_candidate, repaired_failure = candidate_from_svg(
            cleaned_raw_text,
            extracted_svg,
            repaired_svg,
            "repaired",
            generated_tokens=generated_tokens,
            hit_token_cap=hit_token_cap,
            normalized_model_score=normalized_model_score,
            source=source,
        )
        if repaired_candidate is not None:
            repaired_candidate["failure_reasons"] = "|".join(failures)
            return repaired_candidate
        failures.append(f"repaired={repaired_failure}")

        xml_svg = recover_svg_with_lxml(repaired_svg)
        xml_candidate, xml_failure = candidate_from_svg(
            cleaned_raw_text,
            extracted_svg,
            xml_svg,
            "xml",
            generated_tokens=generated_tokens,
            hit_token_cap=hit_token_cap,
            normalized_model_score=normalized_model_score,
            source=source,
        )
        if xml_candidate is not None:
            xml_candidate["failure_reasons"] = "|".join(failures)
            return xml_candidate
        failures.append(f"xml={xml_failure}")

        fallback_valid, fallback_gate_reason = validity_gate(FALLBACK_SVG)
        if fallback_valid == 0:
            raise ValueError(f"Fallback SVG must be valid, got: {fallback_gate_reason}")
        fallback_issues = strict_contract_issues(FALLBACK_SVG)
        return {
            "source": source,
            "reason": "fallback",
            "gate_reason": failures[-1].split("=", 1)[-1] if failures else "fallback",
            "source_gate_reason": "",
            "normalized_gate_reason": "",
            "normalization_status": "fallback",
            "strict_contract": not fallback_issues,
            "strict_issues": ",".join(fallback_issues),
            "collapsed": False,
            "render_success": True,
            "generated_tokens": int(generated_tokens),
            "hit_token_cap": bool(hit_token_cap),
            "raw_text": cleaned_raw_text,
            "extracted_svg": extracted_svg,
            "final_svg": FALLBACK_SVG,
            "normalized_model_score": float(normalized_model_score),
            "failure_reasons": "|".join(failures),
        }


    FALLBACK_STRICT_ISSUES = strict_contract_issues(FALLBACK_SVG)
    if FALLBACK_STRICT_ISSUES:
        raise ValueError(f"Fallback SVG is not strict-contract compliant: {FALLBACK_STRICT_ISSUES}")


    # =========================
    # GENERATION
    # =========================
    def build_model_input(prompt: str) -> str:
        return f"Prompt: {prompt}\nSVG:\n"


    def clear_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def is_cuda_oom(error: BaseException) -> bool:
        message = str(error).lower()
        return isinstance(error, torch.OutOfMemoryError) or "out of memory" in message


    def generated_token_metrics(token_ids, transition_score_row, *, eos_token_id, pad_token_id, max_new_tokens: int):
        effective_scores = []
        effective_count = 0
        saw_eos = False
        for token_id, token_score in zip(token_ids, transition_score_row):
            token_id = int(token_id)
            if pad_token_id is not None and token_id == int(pad_token_id):
                continue
            if eos_token_id is not None and token_id == int(eos_token_id):
                saw_eos = True
                break
            effective_count += 1
            effective_scores.append(float(token_score))

        if effective_scores:
            normalized_model_score = float(sum(effective_scores) / len(effective_scores))
        else:
            normalized_model_score = float("-inf")

        hit_token_cap = (not saw_eos) and effective_count >= int(max_new_tokens)
        return effective_count, hit_token_cap, normalized_model_score


    def generate_batch_candidate_groups(
        batch_prompts,
        *,
        max_new_tokens: int,
        do_sample: bool,
        batch_size: int,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_return_sequences: int = 1,
        source_names: list[str] | None = None,
    ):
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        inputs_text = [build_model_input(prompt) for prompt in batch_prompts]
        inputs = None
        outputs = None
        try:
            inputs = tokenizer(
                inputs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                gen_kwargs["top_k"] = top_k

            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)
                transition_scores = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True,
                )

            sequences = outputs.sequences.detach().cpu()
            transition_scores = transition_scores.detach().cpu()
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        finally:
            del inputs
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grouped_candidates = []
        for prompt_index, prompt in enumerate(batch_prompts):
            prompt_length = int(prompt_lengths[prompt_index])
            prompt_candidates = []
            for candidate_index in range(num_return_sequences):
                row_index = prompt_index * num_return_sequences + candidate_index
                generated_ids = sequences[row_index, prompt_length:].tolist()
                score_row = transition_scores[row_index].tolist()
                generated_tokens, hit_token_cap, normalized_model_score = generated_token_metrics(
                    generated_ids,
                    score_row,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                )
                decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                source = source_names[candidate_index] if source_names else f"candidate{candidate_index + 1}"
                candidate = clean_svg_output(
                    decoded_text,
                    generated_tokens=generated_tokens,
                    hit_token_cap=hit_token_cap,
                    normalized_model_score=normalized_model_score,
                    source=source,
                )
                prompt_candidates.append(candidate)
            grouped_candidates.append(prompt_candidates)

        return grouped_candidates


    def run_generation_pass(
        prompts,
        *,
        batch_size: int,
        max_new_tokens: int,
        do_sample: bool,
        progress_label: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_return_sequences: int = 1,
        source_names: list[str] | None = None,
    ):
        grouped_results = []
        next_index = 0
        current_batch_size = max(1, int(batch_size))
        progress_bar = tqdm(total=len(prompts), desc=progress_label)

        while next_index < len(prompts):
            batch_prompts = prompts[next_index:next_index + current_batch_size]
            try:
                batch_groups = generate_batch_candidate_groups(
                    batch_prompts,
                    batch_size=current_batch_size,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    source_names=source_names,
                )
                grouped_results.extend(batch_groups)
                next_index += len(batch_prompts)
                progress_bar.update(len(batch_prompts))
            except RuntimeError as exc:
                if not is_cuda_oom(exc) or current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                print(f"CUDA OOM during {progress_label}. Retrying with batch size {current_batch_size}.")
                clear_cuda_memory()
            except torch.cuda.OutOfMemoryError:
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                print(f"CUDA OOM during {progress_label}. Retrying with batch size {current_batch_size}.")
                clear_cuda_memory()

        progress_bar.close()
        return grouped_results


    def candidate_sort_key(candidate) -> tuple[int, int, int, int, int, float]:
        return (
            int(candidate["reason"] != "fallback"),
            int(candidate["strict_contract"]),
            int(candidate["render_success"]),
            int(not candidate["hit_token_cap"]),
            int(not candidate["collapsed"]),
            float(candidate["normalized_model_score"]),
        )


    def choose_best_candidate(candidates):
        best_candidate = candidates[0]
        best_key = candidate_sort_key(best_candidate)
        for candidate in candidates[1:]:
            current_key = candidate_sort_key(candidate)
            if current_key > best_key:
                best_candidate = candidate
                best_key = current_key
        return best_candidate


    def candidate_to_row(prefix: str, candidate):
        if candidate is None:
            return {
                f"{prefix}_source": "",
                f"{prefix}_reason": "",
                f"{prefix}_gate_reason": "",
                f"{prefix}_source_gate_reason": "",
                f"{prefix}_normalized_gate_reason": "",
                f"{prefix}_normalization_status": "",
                f"{prefix}_strict_contract": False,
                f"{prefix}_strict_issues": "",
                f"{prefix}_collapsed": False,
                f"{prefix}_render_success": False,
                f"{prefix}_generated_tokens": 0,
                f"{prefix}_hit_token_cap": False,
                f"{prefix}_normalized_model_score": float("-inf"),
                f"{prefix}_raw_text": "",
                f"{prefix}_extracted_svg": "",
                f"{prefix}_final_svg": "",
                f"{prefix}_failure_reasons": "",
            }

        return {
            f"{prefix}_source": candidate["source"],
            f"{prefix}_reason": candidate["reason"],
            f"{prefix}_gate_reason": candidate["gate_reason"],
            f"{prefix}_source_gate_reason": candidate["source_gate_reason"],
            f"{prefix}_normalized_gate_reason": candidate["normalized_gate_reason"],
            f"{prefix}_normalization_status": candidate["normalization_status"],
            f"{prefix}_strict_contract": candidate["strict_contract"],
            f"{prefix}_strict_issues": candidate["strict_issues"],
            f"{prefix}_collapsed": candidate["collapsed"],
            f"{prefix}_render_success": candidate["render_success"],
            f"{prefix}_generated_tokens": candidate["generated_tokens"],
            f"{prefix}_hit_token_cap": candidate["hit_token_cap"],
            f"{prefix}_normalized_model_score": candidate["normalized_model_score"],
            f"{prefix}_raw_text": candidate["raw_text"],
            f"{prefix}_extracted_svg": candidate["extracted_svg"],
            f"{prefix}_final_svg": candidate["final_svg"],
            f"{prefix}_failure_reasons": candidate["failure_reasons"],
        }


    def should_rescue_candidate(candidate) -> bool:
        if EXPERIMENT_KIND in {"best_of_n_bad_rows", "hybrid_retry"}:
            return (
                candidate["reason"] in {"xml", "fallback"}
                or not candidate["strict_contract"]
                or candidate["hit_token_cap"]
            )
        if EXPERIMENT_KIND == "adaptive_token_retry":
            return candidate["hit_token_cap"]
        return False


    def run_rescue_candidates(prompts):
        if not prompts:
            return []

        if EXPERIMENT_KIND == "best_of_n_bad_rows":
            return run_generation_pass(
                prompts,
                batch_size=BEST_OF_N_PASS2_BATCH_SIZE,
                max_new_tokens=BEST_OF_N_PASS2_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                top_k=SAMPLE_TOP_K,
                num_return_sequences=NUM_RESCUE_CANDIDATES,
                source_names=RESCUE_SOURCE_NAMES,
                progress_label="Rescue sample",
            )

        if EXPERIMENT_KIND == "hybrid_retry":
            greedy_groups = run_generation_pass(
                prompts,
                batch_size=HYBRID_GREEDY_BATCH_SIZE,
                max_new_tokens=HYBRID_GREEDY_MAX_NEW_TOKENS,
                do_sample=False,
                num_return_sequences=1,
                source_names=["retry_greedy"],
                progress_label="Retry greedy",
            )
            sample_groups = run_generation_pass(
                prompts,
                batch_size=HYBRID_SAMPLE_BATCH_SIZE,
                max_new_tokens=HYBRID_SAMPLE_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                top_k=SAMPLE_TOP_K,
                num_return_sequences=1,
                source_names=["retry_sample"],
                progress_label="Retry sample",
            )
            return [[greedy_group[0], sample_group[0]] for greedy_group, sample_group in zip(greedy_groups, sample_groups)]

        if EXPERIMENT_KIND == "adaptive_token_retry":
            retry_groups = run_generation_pass(
                prompts,
                batch_size=ADAPTIVE_PASS2_BATCH_SIZE,
                max_new_tokens=ADAPTIVE_PASS2_MAX_NEW_TOKENS,
                do_sample=False,
                num_return_sequences=1,
                source_names=["retry_cap"],
                progress_label="Retry capped",
            )
            return [[retry_group[0]] for retry_group in retry_groups]

        raise ValueError(f"Unsupported experiment kind: {EXPERIMENT_KIND}")


    def build_summary_payload(*, ids, debug_df, summary_json_path, pass1_wall_time, rescue_wall_time, total_wall_time, rescue_attempt_count, rescue_replace_count, limit, rescue_enabled):
        strict_pass_rate = float(debug_df["strict_contract"].mean()) if len(debug_df) else 0.0
        token_cap_count = int(debug_df["hit_token_cap"].sum()) if len(debug_df) else 0
        payload = {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_kind": EXPERIMENT_KIND,
            "test_csv_path": TEST_CSV,
            "summary_json_path": summary_json_path,
            "rows": len(ids),
            "limit": limit,
            "rescue_enabled": rescue_enabled,
            "pass1_wall_time_seconds": pass1_wall_time,
            "rescue_wall_time_seconds": rescue_wall_time,
            "total_wall_time_seconds": total_wall_time,
            "rows_rescued": rescue_attempt_count,
            "rows_replaced": rescue_replace_count,
            "rescue_attempt_count": rescue_attempt_count,
            "rescue_replace_count": rescue_replace_count,
            "final_strict_contract_pass_rate": strict_pass_rate,
            "final_token_cap_count": token_cap_count,
            "pass1_reason_counts": debug_df["pass1_reason"].value_counts().to_dict() if "pass1_reason" in debug_df else {},
            "selected_reason_counts": debug_df["reason"].value_counts().to_dict() if "reason" in debug_df else {},
            "selected_source_counts": debug_df["selected_source"].value_counts().to_dict() if "selected_source" in debug_df else {},
        }
        return payload


    def build_submission_csv(
        *,
        test_csv_path: str = TEST_CSV,
        output_csv_path: str = SUBMISSION_CSV,
        debug_csv_path: str = DEBUG_CSV,
        summary_json_path: str = SUMMARY_JSON,
        batch_size: int = BATCH_SIZE,
        limit: int | None = None,
        enable_rescue: bool = True,
    ):
        total_started_at = time.perf_counter()
        test_df = pd.read_csv(test_csv_path)
        test_df = test_df.dropna(subset=["id", "prompt"]).copy()
        test_df["prompt"] = test_df["prompt"].astype(str)

        if limit is not None:
            test_df = test_df.head(limit).copy()

        prompts = test_df["prompt"].tolist()
        ids = test_df["id"].tolist()

        pass1_started_at = time.perf_counter()
        pass1_groups = run_generation_pass(
            prompts,
            batch_size=batch_size,
            max_new_tokens=PASS1_MAX_NEW_TOKENS,
            do_sample=False,
            num_return_sequences=1,
            source_names=["pass1"],
            progress_label="Pass 1 greedy",
        )
        pass1_wall_time = time.perf_counter() - pass1_started_at
        pass1_candidates = [group[0] for group in pass1_groups]

        pass1_reason_counts = pd.Series([candidate["reason"] for candidate in pass1_candidates]).value_counts()
        print("Pass 1 reason counts:")
        print(pass1_reason_counts)

        rescue_indices = []
        rescue_groups = []
        rescue_wall_time = 0.0

        if enable_rescue:
            rescue_indices = [index for index, candidate in enumerate(pass1_candidates) if should_rescue_candidate(candidate)]
            if rescue_indices:
                rescue_prompts = [prompts[index] for index in rescue_indices]
                rescue_started_at = time.perf_counter()
                rescue_groups = run_rescue_candidates(rescue_prompts)
                rescue_wall_time = time.perf_counter() - rescue_started_at

        rescue_candidates_by_index = {}
        for offset, index in enumerate(rescue_indices):
            rescue_candidates_by_index[index] = rescue_groups[offset]

        debug_rows = []
        final_candidates = []
        rescue_replace_count = 0
        for index, (row_id, prompt, pass1_candidate) in enumerate(zip(ids, prompts, pass1_candidates)):
            ordered_candidates = [pass1_candidate]
            rescue_attempted = index in rescue_candidates_by_index
            if rescue_attempted:
                ordered_candidates.extend(rescue_candidates_by_index[index])

            selected_candidate = choose_best_candidate(ordered_candidates)
            rescue_replaced = selected_candidate["source"] != "pass1"
            if rescue_replaced:
                rescue_replace_count += 1

            final_candidates.append(selected_candidate)

            debug_row = {
                "id": row_id,
                "prompt": prompt,
                "selected_source": selected_candidate["source"],
                "rescue_attempted": rescue_attempted,
                "rescue_replaced": rescue_replaced,
                "reason": selected_candidate["reason"],
                "gate_reason": selected_candidate["gate_reason"],
                "source_gate_reason": selected_candidate["source_gate_reason"],
                "normalized_gate_reason": selected_candidate["normalized_gate_reason"],
                "normalization_status": selected_candidate["normalization_status"],
                "strict_contract": selected_candidate["strict_contract"],
                "strict_issues": selected_candidate["strict_issues"],
                "collapsed": selected_candidate["collapsed"],
                "render_success": selected_candidate["render_success"],
                "generated_tokens": selected_candidate["generated_tokens"],
                "hit_token_cap": selected_candidate["hit_token_cap"],
                "normalized_model_score": selected_candidate["normalized_model_score"],
                "raw_text": selected_candidate["raw_text"],
                "extracted_svg": selected_candidate["extracted_svg"],
                "final_svg": selected_candidate["final_svg"],
                "failure_reasons": selected_candidate["failure_reasons"],
            }

            evidence_map = {candidate["source"]: candidate for candidate in ordered_candidates}
            for prefix in DEBUG_SOURCE_PREFIXES:
                debug_row.update(candidate_to_row(prefix, evidence_map.get(prefix)))

            debug_rows.append(debug_row)

        total_wall_time = time.perf_counter() - total_started_at
        rescue_attempt_count = len(rescue_indices)

        submission_df = pd.DataFrame({
            "id": ids,
            "svg": [candidate["final_svg"] for candidate in final_candidates],
        })
        debug_df = pd.DataFrame(debug_rows)

        summary_payload = build_summary_payload(
            ids=ids,
            debug_df=debug_df,
            summary_json_path=summary_json_path,
            pass1_wall_time=pass1_wall_time,
            rescue_wall_time=rescue_wall_time,
            total_wall_time=total_wall_time,
            rescue_attempt_count=rescue_attempt_count,
            rescue_replace_count=rescue_replace_count,
            limit=limit,
            rescue_enabled=enable_rescue,
        )

        submission_df.to_csv(output_csv_path, index=False)
        debug_df.to_csv(debug_csv_path, index=False)
        Path(summary_json_path).write_text(json.dumps(summary_payload, indent=2))

        print(f"Saved submission: {output_csv_path}")
        print(f"Saved debug: {debug_csv_path}")
        print(f"Saved summary: {summary_json_path}")
        print(f"Rescue attempted: {rescue_attempt_count}")
        print(f"Rescue replaced: {rescue_replace_count}")
        print(f"Final strict-contract pass rate: {summary_payload['final_strict_contract_pass_rate']:.4f}")
        print(f"Final token-cap count: {summary_payload['final_token_cap_count']}")

        return submission_df, debug_df, summary_payload
    '''
).strip() + "\n"


CELL5_TEMPLATE = textwrap.dedent(
    '''
    smoke_submission_df, smoke_debug_df, smoke_summary = build_submission_csv(
        output_csv_path=SMOKE_SUBMISSION_CSV,
        debug_csv_path=SMOKE_DEBUG_CSV,
        summary_json_path=SMOKE_SUMMARY_JSON,
        limit=20,
        enable_rescue=True,
    )

    smoke_pass1only_submission_df, smoke_pass1only_debug_df, smoke_pass1only_summary = build_submission_csv(
        output_csv_path=SMOKE_PASS1ONLY_SUBMISSION_CSV,
        debug_csv_path=SMOKE_PASS1ONLY_DEBUG_CSV,
        summary_json_path=SMOKE_PASS1ONLY_SUMMARY_JSON,
        limit=20,
        enable_rescue=False,
    )

    pass1_only_identity = bool((smoke_pass1only_debug_df["selected_source"] == "pass1").all())

    print("\\nSmoke rescue summary:")
    print(json.dumps(smoke_summary, indent=2))
    print("\\nSmoke pass1-only summary:")
    print(json.dumps(smoke_pass1only_summary, indent=2))
    print(f"Pass1-only selected_source all pass1: {pass1_only_identity}")
    display(smoke_debug_df.head())
    '''
).strip() + "\n"


CELL6_TEMPLATE = textwrap.dedent(
    '''
    from IPython.display import HTML


    def svg_preview_card(row):
        prompt_text = str(row["prompt"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        svg_markup = row["final_svg"]
        return f"""
        <div style="border:1px solid #d0d7de;border-radius:10px;padding:12px;background:#fff;">
            <div style="font-size:12px;color:#57606a;margin-bottom:8px;">
                <strong>ID:</strong> {row["id"]}<br/>
                <strong>Source:</strong> {row["selected_source"]}<br/>
                <strong>Reason:</strong> {row["reason"]}<br/>
                <strong>Strict:</strong> {row["strict_contract"]}<br/>
                <strong>Token cap:</strong> {row["hit_token_cap"]}
            </div>
            <div style="width:256px;height:256px;border:1px solid #d8dee4;background:#f6f8fa;margin-bottom:8px;display:flex;align-items:center;justify-content:center;overflow:hidden;">
                {svg_markup}
            </div>
            <div style="font-size:12px;line-height:1.35;color:#24292f;max-width:256px;white-space:normal;">
                {prompt_text}
            </div>
        </div>
        """


    smoke_preview_rows = smoke_debug_df[
        [
            "id",
            "prompt",
            "selected_source",
            "reason",
            "strict_contract",
            "hit_token_cap",
            "final_svg",
        ]
    ].to_dict("records")

    smoke_preview_html = "".join(svg_preview_card(row) for row in smoke_preview_rows)
    display(
        HTML(
            f"""
            <div>
                <h3 style="margin:0 0 12px 0;">Smoke Test Preview: 20 Samples</h3>
                <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;align-items:start;">
                    {smoke_preview_html}
                </div>
            </div>
            """
        )
    )
    '''
).strip() + "\n"


CELL7_TEMPLATE = textwrap.dedent(
    '''
    submission_df, debug_df, summary_payload = build_submission_csv(
        output_csv_path=SUBMISSION_CSV,
        debug_csv_path=DEBUG_CSV,
        summary_json_path=SUMMARY_JSON,
        enable_rescue=True,
    )

    print("\\nFull-run summary:")
    print(json.dumps(summary_payload, indent=2))
    display(submission_df.head())
    display(
        debug_df[
            [
                "id",
                "selected_source",
                "reason",
                "strict_contract",
                "render_success",
                "hit_token_cap",
                "normalized_model_score",
            ]
        ].head()
    )
    '''
).strip() + "\n"


EXPERIMENTS = [
    {
        "experiment_name": "Best-of-N Bad Rows",
        "experiment_kind": "best_of_n_bad_rows",
        "path": EVALUATION_NOTEBOOKS_DIR / "DL_Midterm_Eval_best_of_n_bad_rows.ipynb",
        "submission_csv": "submission_best_of_n_bad_rows.csv",
        "debug_csv": "submission_best_of_n_bad_rows_debug.csv",
        "summary_json": "submission_best_of_n_bad_rows_summary.json",
        "smoke_submission_csv": "smoke_submission_best_of_n_bad_rows.csv",
        "smoke_debug_csv": "smoke_submission_best_of_n_bad_rows_debug.csv",
        "smoke_summary_json": "smoke_submission_best_of_n_bad_rows_summary.json",
        "smoke_pass1only_submission_csv": "smoke_pass1only_submission_best_of_n_bad_rows.csv",
        "smoke_pass1only_debug_csv": "smoke_pass1only_submission_best_of_n_bad_rows_debug.csv",
        "smoke_pass1only_summary_json": "smoke_pass1only_submission_best_of_n_bad_rows_summary.json",
    },
    {
        "experiment_name": "Hybrid Retry",
        "experiment_kind": "hybrid_retry",
        "path": EVALUATION_NOTEBOOKS_DIR / "DL_Midterm_Eval_hybrid_retry.ipynb",
        "submission_csv": "submission_hybrid_retry.csv",
        "debug_csv": "submission_hybrid_retry_debug.csv",
        "summary_json": "submission_hybrid_retry_summary.json",
        "smoke_submission_csv": "smoke_submission_hybrid_retry.csv",
        "smoke_debug_csv": "smoke_submission_hybrid_retry_debug.csv",
        "smoke_summary_json": "smoke_submission_hybrid_retry_summary.json",
        "smoke_pass1only_submission_csv": "smoke_pass1only_submission_hybrid_retry.csv",
        "smoke_pass1only_debug_csv": "smoke_pass1only_submission_hybrid_retry_debug.csv",
        "smoke_pass1only_summary_json": "smoke_pass1only_submission_hybrid_retry_summary.json",
    },
    {
        "experiment_name": "Adaptive Token Retry",
        "experiment_kind": "adaptive_token_retry",
        "path": EVALUATION_NOTEBOOKS_DIR / "DL_Midterm_Eval_adaptive_token_retry.ipynb",
        "submission_csv": "submission_adaptive_token_retry.csv",
        "debug_csv": "submission_adaptive_token_retry_debug.csv",
        "summary_json": "submission_adaptive_token_retry_summary.json",
        "smoke_submission_csv": "smoke_submission_adaptive_token_retry.csv",
        "smoke_debug_csv": "smoke_submission_adaptive_token_retry_debug.csv",
        "smoke_summary_json": "smoke_submission_adaptive_token_retry_summary.json",
        "smoke_pass1only_submission_csv": "smoke_pass1only_submission_adaptive_token_retry.csv",
        "smoke_pass1only_debug_csv": "smoke_pass1only_submission_adaptive_token_retry_debug.csv",
        "smoke_pass1only_summary_json": "smoke_pass1only_submission_adaptive_token_retry_summary.json",
    },
]


def as_source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def as_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": as_source_lines(text),
    }


def render_cell4(experiment: dict) -> str:
    text = CELL4_TEMPLATE
    replacements = {
        "__EXPERIMENT_NAME__": experiment["experiment_name"],
        "__EXPERIMENT_KIND__": experiment["experiment_kind"],
        "__SUBMISSION_CSV__": experiment["submission_csv"],
        "__DEBUG_CSV__": experiment["debug_csv"],
        "__SUMMARY_JSON__": experiment["summary_json"],
        "__SMOKE_SUBMISSION_CSV__": experiment["smoke_submission_csv"],
        "__SMOKE_DEBUG_CSV__": experiment["smoke_debug_csv"],
        "__SMOKE_SUMMARY_JSON__": experiment["smoke_summary_json"],
        "__SMOKE_PASS1ONLY_SUBMISSION_CSV__": experiment["smoke_pass1only_submission_csv"],
        "__SMOKE_PASS1ONLY_DEBUG_CSV__": experiment["smoke_pass1only_debug_csv"],
        "__SMOKE_PASS1ONLY_SUMMARY_JSON__": experiment["smoke_pass1only_summary_json"],
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def main() -> None:
    base_notebook = json.loads(BASE_NOTEBOOK.read_text())
    prefix_cells = base_notebook["cells"][:4]

    for experiment in EXPERIMENTS:
        notebook = json.loads(BASE_NOTEBOOK.read_text())
        notebook["cells"] = prefix_cells + [
            as_code_cell(render_cell4(experiment)),
            as_code_cell(CELL5_TEMPLATE),
            as_code_cell(CELL6_TEMPLATE),
            as_code_cell(CELL7_TEMPLATE),
        ]
        experiment["path"].write_text(json.dumps(notebook, indent=1))
        print(f"Wrote {experiment['path'].name}")


if __name__ == "__main__":
    main()
