from __future__ import annotations

import gc
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cairosvg
import pandas as pd
import torch
from PIL import Image
from lxml import etree
from tqdm import tqdm

SVG_NS = "http://www.w3.org/2000/svg"
HELPER_VERSION = "2026-03-29-debug-fix-v1"

DEFAULT_ALLOWED_TAGS = (
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
)

DEFAULT_FALLBACK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" '
    'width="256" height="256">'
    '<rect width="256" height="256" fill="white"/>'
    "</svg>"
)

STAGE_RANK = {
    "fallback": 0,
    "xml": 1,
    "repaired": 2,
    "valid": 3,
}


@dataclass(frozen=True)
class InferenceConfig:
    pass1_max_new_tokens: int = 2048
    pass2_max_new_tokens: int = 3072
    retry_mode: str = "long_greedy"
    retry_on_stages: tuple[str, ...] = ("xml", "fallback")
    retry_on_capped: bool = True
    sample_temperature: float = 0.2
    sample_top_p: float = 0.95
    sample_top_k: int = 20
    pass1_batch_size: int = 250
    pass2_batch_size: int = 128
    min_batch_size: int = 1
    clear_cuda_cache_between_batches: bool = True
    render_size: int = 256
    max_svg_length: int = 16000
    max_path_count: int = 256
    allowed_tags: tuple[str, ...] = DEFAULT_ALLOWED_TAGS
    fallback_svg: str = DEFAULT_FALLBACK_SVG


@dataclass
class SvgCandidate:
    final_svg: str
    stage: str
    gate_reason: str
    strict_pass: bool
    strict_issues: str
    raw_text: str
    extracted_svg: str
    normalized_svg: str
    collapsed: bool
    source_gate_reason: str = ""
    normalized_gate_reason: str = ""
    normalization_status: str = ""
    raw_gate_reason: str = ""
    repaired_gate_reason: str = ""
    xml_gate_reason: str = ""
    failure_reasons: str = ""
    generated_tokens: int = 0
    hit_token_cap: bool = False


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


def render_svg(svg: str, size: int):
    try:
        png = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        Image.open(io.BytesIO(png)).convert("RGB")
        return True
    except Exception:
        return False


def validity_gate(svg: str, config: InferenceConfig) -> tuple[int, str]:
    if not isinstance(svg, str) or not svg.strip():
        return 0, "svg_too_long_or_empty"

    svg = svg.strip()

    if len(svg) > config.max_svg_length:
        return 0, "svg_too_long_or_empty"

    try:
        root = ET.fromstring(svg)
    except Exception:
        return 0, "parse_failed"

    if strip_namespace(root.tag) != "svg":
        return 0, "root_not_svg"

    path_count = 0
    for elem in root.iter():
        tag = strip_namespace(elem.tag)
        if tag not in config.allowed_tags:
            return 0, f"disallowed_tag:{tag}"
        if tag == "path":
            path_count += 1

    if path_count > config.max_path_count:
        return 0, "too_many_paths"

    if not render_svg(svg, size=config.render_size):
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

    paths = [elem for elem in root.iter() if strip_namespace(elem.tag) == "path"]
    nonempty_paths = [path for path in paths if path.attrib.get("d", "").strip()]

    if paths and not nonempty_paths:
        return True

    total_elems = sum(1 for _ in root.iter())
    return total_elems <= 1


def parse_numeric_dimension(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or "%" in value:
        return None
    match = re.match(r"^(-?\d+(?:\.\d+)?)", value)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def clean_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def get_attr_value(opening_tag: str, attr_name: str) -> str | None:
    pattern = rf"(\s{re.escape(attr_name)}\s*=\s*)([\"'])(.*?)\2"
    match = re.search(pattern, opening_tag, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(3)


def upsert_attr(opening_tag: str, attr_name: str, attr_value: str) -> str:
    pattern = rf"(\s{re.escape(attr_name)}\s*=\s*)([\"'])(.*?)\2"

    def replacer(match: re.Match[str]) -> str:
        prefix = match.group(1)
        quote = match.group(2)
        return f"{prefix}{quote}{attr_value}{quote}"

    if re.search(pattern, opening_tag, flags=re.IGNORECASE | re.DOTALL):
        return re.sub(
            pattern,
            replacer,
            opening_tag,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return opening_tag[:-1] + f' {attr_name}="{attr_value}">'


def normalize_root_attributes(svg: str, config: InferenceConfig) -> str:
    match = re.search(r"<svg\b[^>]*>", svg, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return svg

    opening_tag = match.group(0)
    normalized_tag = opening_tag

    if re.search(r"\sxmlns\s*=", normalized_tag) is None:
        normalized_tag = normalized_tag[:-1] + f' xmlns="{SVG_NS}">'

    width_value = parse_numeric_dimension(get_attr_value(opening_tag, "width"))
    height_value = parse_numeric_dimension(get_attr_value(opening_tag, "height"))
    viewbox_value = get_attr_value(opening_tag, "viewBox")

    if viewbox_value is None:
        if width_value and height_value:
            inferred_viewbox = f"0 0 {clean_number(width_value)} {clean_number(height_value)}"
        else:
            inferred_viewbox = f"0 0 {config.render_size} {config.render_size}"
        normalized_tag = upsert_attr(normalized_tag, "viewBox", inferred_viewbox)

    normalized_tag = upsert_attr(normalized_tag, "width", str(config.render_size))
    normalized_tag = upsert_attr(normalized_tag, "height", str(config.render_size))

    return svg.replace(opening_tag, normalized_tag, 1)


def summarize_stage_failure(source_gate_reason: str, normalized_gate_reason: str) -> str:
    parts: list[str] = []
    if source_gate_reason and source_gate_reason != "valid":
        parts.append(f"source:{source_gate_reason}")
    if normalized_gate_reason and normalized_gate_reason != "valid" and normalized_gate_reason != source_gate_reason:
        parts.append(f"normalized:{normalized_gate_reason}")
    if parts:
        return ";".join(parts)
    return source_gate_reason or normalized_gate_reason or "unknown_failure"


def strict_contract_issues(svg: str, config: InferenceConfig) -> list[str]:
    issues: list[str] = []

    if f'xmlns="{SVG_NS}"' not in svg:
        issues.append("missing_xmlns")

    try:
        root = ET.fromstring(svg)
    except Exception:
        issues.append("strict_parse_failed")
        return issues

    if strip_namespace(root.tag) != "svg":
        issues.append("root_not_svg")
        return issues

    expected_viewbox = f"0 0 {config.render_size} {config.render_size}"
    if root.attrib.get("viewBox") != expected_viewbox:
        issues.append("viewbox_not_exact")

    if root.attrib.get("width") != str(config.render_size) or root.attrib.get("height") != str(config.render_size):
        issues.append("width_height_not_exact")

    return issues


def candidate_from_svg(
    raw_text: str,
    extracted_svg: str,
    svg: str,
    stage: str,
    config: InferenceConfig,
) -> tuple[SvgCandidate | None, str]:
    source_valid, source_gate_reason = validity_gate(svg, config)
    normalized = normalize_root_attributes(svg, config)
    normalized_valid, normalized_gate_reason = validity_gate(normalized, config)

    normalization_status = "unchanged"
    final_svg = svg
    gate_reason = source_gate_reason

    if normalized != svg:
        if normalized_valid == 1:
            normalization_status = "applied"
            final_svg = normalized
            gate_reason = normalized_gate_reason
        elif source_valid == 1:
            normalization_status = "reverted_after_failed_normalize"
        else:
            normalization_status = "failed"
    elif normalized_valid == 1:
        gate_reason = normalized_gate_reason

    if source_valid == 0 and normalized_valid == 0:
        return None, summarize_stage_failure(source_gate_reason, normalized_gate_reason)

    collapsed = looks_collapsed(final_svg)
    if stage in {"repaired", "xml"} and (collapsed or len(final_svg) < 80):
        failure_reason = "collapsed_or_too_short"
        return None, failure_reason

    strict_issues = strict_contract_issues(final_svg, config)
    candidate = SvgCandidate(
        final_svg=final_svg,
        stage=stage,
        gate_reason=gate_reason,
        strict_pass=not strict_issues,
        strict_issues=",".join(strict_issues),
        raw_text=raw_text,
        extracted_svg=extracted_svg,
        normalized_svg=normalized,
        collapsed=collapsed,
        source_gate_reason=source_gate_reason,
        normalized_gate_reason=normalized_gate_reason,
        normalization_status=normalization_status,
    )
    return candidate, gate_reason


def clean_svg_output(raw_text: str, config: InferenceConfig) -> SvgCandidate:
    extracted_svg = extract_svg(raw_text)

    valid_candidate, raw_gate_reason = candidate_from_svg(raw_text, extracted_svg, extracted_svg, "valid", config)
    if valid_candidate is not None:
        valid_candidate.raw_gate_reason = raw_gate_reason
        valid_candidate.repaired_gate_reason = raw_gate_reason
        valid_candidate.xml_gate_reason = raw_gate_reason
        valid_candidate.failure_reasons = ""
        return valid_candidate

    repaired_svg = repair_svg(extracted_svg)
    repaired_candidate, repaired_gate_reason = candidate_from_svg(raw_text, extracted_svg, repaired_svg, "repaired", config)
    if repaired_candidate is not None:
        repaired_candidate.raw_gate_reason = raw_gate_reason
        repaired_candidate.repaired_gate_reason = repaired_gate_reason
        repaired_candidate.xml_gate_reason = repaired_gate_reason
        repaired_candidate.failure_reasons = f"raw={raw_gate_reason}"
        return repaired_candidate

    xml_svg = recover_svg_with_lxml(repaired_svg)
    xml_candidate, xml_gate_reason = candidate_from_svg(raw_text, extracted_svg, xml_svg, "xml", config)
    if xml_candidate is not None:
        xml_candidate.raw_gate_reason = raw_gate_reason
        xml_candidate.repaired_gate_reason = repaired_gate_reason
        xml_candidate.xml_gate_reason = xml_gate_reason
        xml_candidate.failure_reasons = f"raw={raw_gate_reason}|repaired={repaired_gate_reason}"
        return xml_candidate

    fallback_issues = strict_contract_issues(config.fallback_svg, config)
    failure_reasons = f"raw={raw_gate_reason}|repaired={repaired_gate_reason}|xml={xml_gate_reason}"
    return SvgCandidate(
        final_svg=config.fallback_svg,
        stage="fallback",
        gate_reason=xml_gate_reason or repaired_gate_reason or raw_gate_reason or "fallback",
        strict_pass=not fallback_issues,
        strict_issues=",".join(fallback_issues),
        raw_text=raw_text,
        extracted_svg=extracted_svg,
        normalized_svg=config.fallback_svg,
        collapsed=False,
        source_gate_reason="",
        normalized_gate_reason="",
        normalization_status="fallback",
        raw_gate_reason=raw_gate_reason,
        repaired_gate_reason=repaired_gate_reason,
        xml_gate_reason=xml_gate_reason,
        failure_reasons=failure_reasons,
    )


def build_model_input(prompt: str) -> str:
    return f"Prompt: {prompt}\nSVG:\n"


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in message


def generate_batch_candidates(
    batch_prompts: list[str],
    tokenizer,
    model,
    *,
    config: InferenceConfig,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[SvgCandidate]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs_text = [build_model_input(prompt) for prompt in batch_prompts]

    inputs = tokenizer(
        inputs_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k

    outputs = None
    generated_only: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        generated_only = [
            outputs[j, int(prompt_lengths[j]):].detach().cpu()
            for j in range(outputs.shape[0])
        ]
    finally:
        del outputs
        del inputs
        if config.clear_cuda_cache_between_batches:
            clear_cuda_memory()

    decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    candidates: list[SvgCandidate] = []
    for raw_text, generated_ids in zip(decoded, generated_only):
        candidate = clean_svg_output(raw_text, config)
        candidate.generated_tokens = int(generated_ids.shape[0])
        candidate.hit_token_cap = candidate.generated_tokens >= max_new_tokens
        candidates.append(candidate)

    return candidates


def run_generation_pass(
    prompts: list[str],
    tokenizer,
    model,
    *,
    config: InferenceConfig,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[SvgCandidate]:
    candidates: list[SvgCandidate] = []
    effective_batch_size = max(batch_size, config.min_batch_size)
    progress = tqdm(total=len(prompts))
    index = 0

    while index < len(prompts):
        current_batch_size = min(effective_batch_size, len(prompts) - index)
        batch_prompts = prompts[index:index + current_batch_size]

        try:
            batch_candidates = generate_batch_candidates(
                batch_prompts,
                tokenizer,
                model,
                config=config,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        except Exception as error:
            if not is_cuda_oom(error):
                progress.close()
                raise

            clear_cuda_memory()
            if current_batch_size <= config.min_batch_size:
                progress.close()
                raise RuntimeError(
                    "CUDA OOM even at the minimum batch size. "
                    f"Lower max_new_tokens or retry with a smaller starting batch size. "
                    f"max_new_tokens={max_new_tokens}, batch_size={current_batch_size}"
                ) from error

            next_batch_size = max(config.min_batch_size, current_batch_size // 2)
            if next_batch_size == current_batch_size:
                next_batch_size = current_batch_size - 1

            print(
                "CUDA OOM during generation. "
                f"Reducing batch size from {current_batch_size} to {next_batch_size} "
                f"for max_new_tokens={max_new_tokens}."
            )
            effective_batch_size = next_batch_size
            continue

        candidates.extend(batch_candidates)
        index += current_batch_size
        progress.update(current_batch_size)

    progress.close()
    return candidates


def should_retry(candidate: SvgCandidate, config: InferenceConfig) -> bool:
    if config.retry_mode == "off":
        return False
    if candidate.stage in config.retry_on_stages:
        return True
    if config.retry_on_capped and candidate.hit_token_cap:
        return True
    return False


def candidate_sort_key(candidate: SvgCandidate) -> tuple[int, int, int, int]:
    return (
        STAGE_RANK[candidate.stage],
        int(candidate.strict_pass),
        int(not candidate.hit_token_cap),
        min(len(candidate.final_svg), 8000),
    )


def choose_candidate(pass1_candidate: SvgCandidate, retry_candidate: SvgCandidate | None) -> tuple[SvgCandidate, bool]:
    if retry_candidate is None:
        return pass1_candidate, False
    if candidate_sort_key(retry_candidate) > candidate_sort_key(pass1_candidate):
        return retry_candidate, True
    return pass1_candidate, False


def candidate_to_row(prefix: str, candidate: SvgCandidate | None) -> dict[str, object]:
    if candidate is None:
        return {
            f"{prefix}_reason": "",
            f"{prefix}_gate_reason": "",
            f"{prefix}_source_gate_reason": "",
            f"{prefix}_normalized_gate_reason": "",
            f"{prefix}_normalization_status": "",
            f"{prefix}_strict_contract": False,
            f"{prefix}_strict_issues": "",
            f"{prefix}_raw_gate_reason": "",
            f"{prefix}_repaired_gate_reason": "",
            f"{prefix}_xml_gate_reason": "",
            f"{prefix}_failure_reasons": "",
            f"{prefix}_generated_tokens": 0,
            f"{prefix}_hit_token_cap": False,
        }

    return {
        f"{prefix}_reason": candidate.stage,
        f"{prefix}_gate_reason": candidate.gate_reason,
        f"{prefix}_source_gate_reason": candidate.source_gate_reason,
        f"{prefix}_normalized_gate_reason": candidate.normalized_gate_reason,
        f"{prefix}_normalization_status": candidate.normalization_status,
        f"{prefix}_strict_contract": candidate.strict_pass,
        f"{prefix}_strict_issues": candidate.strict_issues,
        f"{prefix}_raw_gate_reason": candidate.raw_gate_reason,
        f"{prefix}_repaired_gate_reason": candidate.repaired_gate_reason,
        f"{prefix}_xml_gate_reason": candidate.xml_gate_reason,
        f"{prefix}_failure_reasons": candidate.failure_reasons,
        f"{prefix}_generated_tokens": candidate.generated_tokens,
        f"{prefix}_hit_token_cap": candidate.hit_token_cap,
    }


def build_submission_csv(
    *,
    test_csv_path,
    output_csv_path,
    debug_csv_path,
    model,
    tokenizer,
    config: InferenceConfig,
    batch_size: int | None = None,
    pass1_batch_size: int | None = None,
    pass2_batch_size: int | None = None,
    limit: int | None = None,
):
    pass1_batch_size = pass1_batch_size or batch_size or config.pass1_batch_size
    pass2_batch_size = pass2_batch_size or batch_size or config.pass2_batch_size
    print(
        "Generation config:",
        {
            "helper_version": HELPER_VERSION,
            "pass1_batch_size": pass1_batch_size,
            "pass2_batch_size": pass2_batch_size,
            "min_batch_size": config.min_batch_size,
            "pass1_max_new_tokens": config.pass1_max_new_tokens,
            "pass2_max_new_tokens": config.pass2_max_new_tokens,
            "retry_mode": config.retry_mode,
        },
    )

    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.dropna(subset=["id", "prompt"]).copy()
    test_df["prompt"] = test_df["prompt"].astype(str)

    if limit is not None:
        test_df = test_df.head(limit).copy()

    prompts = test_df["prompt"].tolist()
    ids = test_df["id"].tolist()

    pass1_candidates = run_generation_pass(
        prompts,
        tokenizer,
        model,
        config=config,
        batch_size=pass1_batch_size,
        max_new_tokens=config.pass1_max_new_tokens,
        do_sample=False,
    )

    print("Pass 1 reason counts:")
    print(pd.Series([candidate.stage for candidate in pass1_candidates]).value_counts())

    final_candidates = list(pass1_candidates)
    retry_candidates_by_index: dict[int, SvgCandidate] = {}
    retry_used = [False] * len(pass1_candidates)

    retry_indices = [
        index
        for index, candidate in enumerate(pass1_candidates)
        if should_retry(candidate, config)
    ]

    if retry_indices:
        retry_prompts = [prompts[index] for index in retry_indices]
        retry_do_sample = config.retry_mode == "sample"

        retry_candidates = run_generation_pass(
            retry_prompts,
            tokenizer,
            model,
            config=config,
            batch_size=pass2_batch_size,
            max_new_tokens=config.pass2_max_new_tokens,
            do_sample=retry_do_sample,
            temperature=config.sample_temperature if retry_do_sample else None,
            top_p=config.sample_top_p if retry_do_sample else None,
            top_k=config.sample_top_k if retry_do_sample else None,
        )

        recovered = 0
        for index, retry_candidate in zip(retry_indices, retry_candidates):
            retry_candidates_by_index[index] = retry_candidate
            selected_candidate, used_retry = choose_candidate(pass1_candidates[index], retry_candidate)
            final_candidates[index] = selected_candidate
            retry_used[index] = used_retry
            if used_retry:
                recovered += 1

        print(f"Retry attempted rows: {len(retry_indices)}")
        print(f"Retry replacements: {recovered}")

    debug_rows = []
    for row_index, (prompt, candidate) in enumerate(zip(prompts, final_candidates)):
        retry_candidate = retry_candidates_by_index.get(row_index)
        debug_row = {
            "prompt": prompt,
            "reason": candidate.stage,
            "gate_reason": candidate.gate_reason,
            "source_gate_reason": candidate.source_gate_reason,
            "normalized_gate_reason": candidate.normalized_gate_reason,
            "normalization_status": candidate.normalization_status,
            "strict_contract": candidate.strict_pass,
            "strict_issues": candidate.strict_issues,
            "raw_text": candidate.raw_text,
            "extracted_svg": candidate.extracted_svg,
            "final_svg": candidate.final_svg,
            "raw_gate_reason": candidate.raw_gate_reason,
            "repaired_gate_reason": candidate.repaired_gate_reason,
            "xml_gate_reason": candidate.xml_gate_reason,
            "failure_reasons": candidate.failure_reasons,
            "retry_attempted": row_index in retry_candidates_by_index,
            "retry_used": retry_used[row_index],
            "final_generated_tokens": candidate.generated_tokens,
            "final_hit_token_cap": candidate.hit_token_cap,
        }
        debug_row.update(candidate_to_row("pass1", pass1_candidates[row_index]))
        debug_row.update(candidate_to_row("retry", retry_candidate))
        debug_rows.append(debug_row)

    debug_df = pd.DataFrame(debug_rows)
    print("Final reason counts:")
    print(debug_df["reason"].value_counts())
    print("Final strict-contract pass rate:", float(debug_df["strict_contract"].mean()))

    submission_df = pd.DataFrame(
        {
            "id": ids,
            "svg": [candidate.final_svg for candidate in final_candidates],
        }
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_csv_path, index=False)
    debug_df.insert(0, "id", ids)
    debug_df.to_csv(debug_csv_path, index=False)
    return submission_df
