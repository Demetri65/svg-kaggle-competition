#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

RAW_BASELINE_ARTIFACT_PREFIX = "qwen25coder15b_raw_onepass_len1024"
RAW_BASELINE_FORMAT = "Prompt: {prompt}\nSVG:\n{svg}"
RAW_BASELINE_NOTES = [
    "This is the canonical raw-data one-pass baseline used for the midterm-compliant repo surface.",
    "The training set is the original train.csv without canonicalization or prompt conflict resolution.",
    "max_length=1024 intentionally preserves the original truncated training regime.",
]
REQUIRED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "numpy",
    "peft",
    "pillow",
    "torch",
    "transformers",
    "trl",
)


@dataclass(frozen=True)
class TrainingConfigSnapshot:
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    bf16: bool = True
    logging_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 1000
    eval_strategy: str = "no"
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = False
    max_grad_norm: float = 0.3
    report_to: str = "none"
    max_length: int = 1024
    packing: bool = False
    train_eval_split_seed: int = 42
    train_eval_split_test_size: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the canonical raw-data one-pass SVG midterm baseline and save an adapter bundle."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("train.csv"),
        help="Path to the raw training CSV. Defaults to train.csv in the repo root.",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        required=True,
        help="Local directory containing the offline base model snapshot.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/raw_baseline"),
        help="Directory where checkpoints, adapter files, and metadata will be written.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional deterministic cap on the number of filtered rows used for a smoke run.",
    )
    parser.add_argument(
        "--token-diagnostic-samples",
        type=int,
        default=128,
        help="How many formatted training rows to sample for token length diagnostics.",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Also reload the base model, merge the adapter, and save a merged model snapshot.",
    )
    return parser.parse_args()


def ensure_required_paths(args: argparse.Namespace) -> None:
    missing: list[str] = []
    if not args.train_csv.exists():
        missing.append(str(args.train_csv))
    if not args.base_model_dir.exists():
        missing.append(str(args.base_model_dir))
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def ensure_runtime_support() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the canonical raw-baseline training path. "
            "Run this script on a GPU runtime with bitsandbytes support."
        )


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in REQUIRED_PACKAGES:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "missing"
    return versions


def format_svg_sample(prompt: str, svg_code: str) -> str:
    return RAW_BASELINE_FORMAT.format(prompt=prompt, svg=svg_code)


def is_valid_row(example: dict[str, Any]) -> bool:
    return (
        example.get("prompt") is not None
        and example.get("svg") is not None
        and str(example["prompt"]).strip() != ""
        and str(example["svg"]).strip() != ""
    )


def to_training_text(example: dict[str, Any]) -> dict[str, str]:
    return {
        "text": format_svg_sample(
            prompt=str(example["prompt"]).strip(),
            svg_code=str(example["svg"]).strip(),
        )
    }


def select_rows(dataset: Dataset, max_rows: int | None) -> Dataset:
    if max_rows is None or max_rows >= len(dataset):
        return dataset
    return dataset.select(range(max_rows))


def collect_token_length_summary(dataset: Dataset, tokenizer, sample_size: int) -> dict[str, float | int]:
    if len(dataset) == 0:
        return {
            "rows_measured": 0,
            "min_tokens": 0,
            "median_tokens": 0,
            "mean_tokens": 0,
            "max_tokens": 0,
        }

    rows_to_measure = min(sample_size, len(dataset))
    lengths: list[int] = []
    for index in range(rows_to_measure):
        token_ids = tokenizer(dataset[index]["text"], add_special_tokens=True)["input_ids"]
        lengths.append(len(token_ids))

    ordered = sorted(lengths)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 0:
        median_value = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    else:
        median_value = ordered[midpoint]

    return {
        "rows_measured": rows_to_measure,
        "min_tokens": min(ordered),
        "median_tokens": median_value,
        "mean_tokens": float(sum(ordered)) / float(rows_to_measure),
        "max_tokens": max(ordered),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_quantized_base_model(base_model_dir: Path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


def build_lora_model(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def create_training_args(output_dir: Path, config: TrainingConfigSnapshot) -> SFTConfig:
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        report_to=config.report_to,
        max_length=config.max_length,
        packing=config.packing,
        seed=config.seed,
    )


def build_manifest_snapshot(
    *,
    args: argparse.Namespace,
    output_root: Path,
    checkpoint_dir: Path,
    adapter_dir: Path,
    merged_dir: Path,
    metadata_dir: Path,
    config: TrainingConfigSnapshot,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    token_length_summary: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "artifact_prefix": RAW_BASELINE_ARTIFACT_PREFIX,
        "base_model_dir": str(args.base_model_dir.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "format": RAW_BASELINE_FORMAT,
            "max_train_rows": args.max_train_rows,
            "raw_train_csv": str(args.train_csv.resolve()),
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
        },
        "expected_outputs": {
            "adapter_dir": str(adapter_dir.resolve()),
            "checkpoints_dir": str(checkpoint_dir.resolve()),
            "merged_dir": str(merged_dir.resolve()),
            "metadata_dir": str(metadata_dir.resolve()),
            "output_root": str(output_root.resolve()),
        },
        "notes": RAW_BASELINE_NOTES,
        "package_versions": collect_package_versions(),
        "save_merged": bool(args.save_merged),
        "seeds": {
            "global_seed": config.seed,
            "train_eval_split_seed": config.train_eval_split_seed,
        },
        "token_length_summary": token_length_summary,
        "training_config": asdict(config),
    }


def save_merged_snapshot(base_model_dir: Path, adapter_dir: Path, merged_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)


def main() -> int:
    args = parse_args()
    ensure_required_paths(args)
    ensure_runtime_support()

    config = TrainingConfigSnapshot()
    output_root = args.output_root
    checkpoint_dir = output_root / "checkpoints"
    adapter_dir = output_root / "adapter"
    merged_dir = output_root / "merged"
    metadata_dir = output_root / "metadata"
    for path in (checkpoint_dir, adapter_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_merged:
        merged_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(config.seed)
    tokenizer, base_model = load_quantized_base_model(args.base_model_dir)
    model = build_lora_model(base_model)
    model.print_trainable_parameters()

    raw_dataset = load_dataset("csv", data_files=str(args.train_csv))["train"]
    filtered_dataset = raw_dataset.filter(is_valid_row)
    filtered_dataset = select_rows(filtered_dataset, args.max_train_rows)
    training_text_dataset = filtered_dataset.map(to_training_text, remove_columns=filtered_dataset.column_names)
    split_dataset = training_text_dataset.train_test_split(
        test_size=config.train_eval_split_test_size,
        seed=config.train_eval_split_seed,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    token_length_summary = collect_token_length_summary(
        train_dataset,
        tokenizer,
        sample_size=args.token_diagnostic_samples,
    )
    baseline_reference = build_manifest_snapshot(
        args=args,
        output_root=output_root,
        checkpoint_dir=checkpoint_dir,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        metadata_dir=metadata_dir,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        token_length_summary=token_length_summary,
    )
    write_json(metadata_dir / "baseline_reference.json", baseline_reference)

    trainer = SFTTrainer(
        model=model,
        args=create_training_args(checkpoint_dir, config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    summary_payload = dict(baseline_reference)
    summary_payload["training_result"] = {
        "global_step": trainer.state.global_step,
        "log_history_tail": trainer.state.log_history[-10:],
    }
    write_json(metadata_dir / "run_summary.json", summary_payload)

    if args.save_merged:
        save_merged_snapshot(args.base_model_dir, adapter_dir, merged_dir)

    print("Raw baseline training complete.")
    print(f"Output root: {output_root.resolve()}")
    print(f"Adapter dir: {adapter_dir.resolve()}")
    print(f"Metadata dir: {metadata_dir.resolve()}")
    if args.save_merged:
        print(f"Merged dir: {merged_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
