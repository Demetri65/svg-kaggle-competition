#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from lxml import etree
except Exception:  # pragma: no cover - optional dependency
    etree = None

SVG_NS = "http://www.w3.org/2000/svg"
RENDER_SIZE = 256
MAX_SVG_LENGTH = 16000
MAX_PATH_COUNT = 256

ALLOWED_TAGS = (
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

COMPETITION_VIEWBOX = f"0 0 {RENDER_SIZE} {RENDER_SIZE}"
COMPETITION_SVG_OPEN = (
    f'<svg xmlns="{SVG_NS}" viewBox="{COMPETITION_VIEWBOX}" '
    f'width="{RENDER_SIZE}" height="{RENDER_SIZE}">'
)
COMPETITION_FALLBACK_SVG = (
    f'{COMPETITION_SVG_OPEN}<rect width="{RENDER_SIZE}" '
    f'height="{RENDER_SIZE}" fill="white"/></svg>'
)
SVG_BODY_RE = re.compile(r"<svg\b[^>]*>(.*)</svg>", flags=re.IGNORECASE | re.DOTALL)


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
    if etree is None or not svg or "<svg" not in svg:
        return svg

    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(svg.encode("utf-8"), parser=parser)
        if root is None:
            return svg
        return etree.tostring(root, encoding="unicode")
    except Exception:
        return svg


def validity_gate(svg: str) -> tuple[int, str]:
    if not isinstance(svg, str) or not svg.strip():
        return 0, "svg_too_long_or_empty"

    svg = svg.strip()
    if len(svg) > MAX_SVG_LENGTH:
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
        if tag not in ALLOWED_TAGS:
            return 0, f"disallowed_tag:{tag}"
        if tag == "path":
            path_count += 1

    if path_count > MAX_PATH_COUNT:
        return 0, "too_many_paths"

    return 1, "valid"


def is_competition_wrapper(svg: str) -> bool:
    if not isinstance(svg, str) or not svg.strip():
        return False

    svg = svg.strip()
    match = re.search(r"<svg\b[^>]*>", svg, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return False

    opening_tag = match.group(0)
    if f'xmlns="{SVG_NS}"' not in opening_tag:
        return False

    try:
        root = ET.fromstring(svg)
    except Exception:
        return False

    return (
        strip_namespace(root.tag) == "svg"
        and root.attrib.get("viewBox") == COMPETITION_VIEWBOX
        and root.attrib.get("width") == str(RENDER_SIZE)
        and root.attrib.get("height") == str(RENDER_SIZE)
    )


def ensure_competition_svg_wrapper(svg: str) -> tuple[str, str]:
    raw_svg = "" if svg is None else str(svg).strip()
    if not raw_svg:
        return COMPETITION_FALLBACK_SVG, "fallback_empty"

    extracted_svg = extract_svg(raw_svg)
    candidate_svg = repair_svg(extracted_svg or raw_svg)

    if "<svg" not in candidate_svg.lower():
        inner_svg = candidate_svg
    else:
        wrapper_match = SVG_BODY_RE.search(candidate_svg)
        inner_svg = wrapper_match.group(1).strip() if wrapper_match else candidate_svg

    wrapped_svg = f"{COMPETITION_SVG_OPEN}{inner_svg}</svg>"
    valid, reason = validity_gate(wrapped_svg)
    if valid == 1:
        return wrapped_svg, "rewrapped"

    recovered_svg = recover_svg_with_lxml(wrapped_svg)
    valid, recovered_reason = validity_gate(recovered_svg)
    if valid == 1:
        return recovered_svg, f"rewrapped_recovered:{reason}"

    return COMPETITION_FALLBACK_SVG, f"fallback:{reason}->{recovered_reason}"


def rewrite_submission(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        if not fieldnames or "svg" not in fieldnames:
            raise ValueError(f"{input_path} must contain an 'svg' column.")
        rows = list(reader)

    rewritten_rows = 0
    fallback_rows = 0
    recovery_rows = 0

    for row in rows:
        original_svg = row.get("svg", "")
        normalized_svg, status = ensure_competition_svg_wrapper(original_svg)
        if normalized_svg != original_svg:
            rewritten_rows += 1
        if status.startswith("fallback"):
            fallback_rows += 1
        if status.startswith("rewrapped_recovered"):
            recovery_rows += 1
        row["svg"] = normalized_svg

    noncanonical_rows = [
        index
        for index, row in enumerate(rows, start=2)
        if not is_competition_wrapper(row["svg"])
    ]
    if noncanonical_rows:
        first_bad_row = noncanonical_rows[0]
        raise ValueError(
            f"Found {len(noncanonical_rows)} rows without the canonical wrapper; "
            f"first bad CSV line is {first_bad_row}."
        )

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(output_path)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows processed: {len(rows)}")
    print(f"Rows rewritten: {rewritten_rows}")
    print(f"Rows recovered after rewrap: {recovery_rows}")
    print(f"Rows replaced with fallback: {fallback_rows}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite every SVG in a submission CSV to the exact competition wrapper."
    )
    parser.add_argument("input_csv", type=Path, help="Submission CSV to rewrite.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV path. Defaults to rewriting the input file in place.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_csv.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    return rewrite_submission(input_path, output_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
