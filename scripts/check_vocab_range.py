#!/usr/bin/env python3
"""
Check tokenized dataset for token ids outside the tokenizer vocabulary range.

Usage:
  python scripts/check_vocab_range.py \
    --dataset_path /path/to/tokenized/dataset \
    --model_name BSC-LT/roberta-base-bne \
    [--ignore_index -100]

Exits with code 1 if any violations are found; 0 otherwise.
Prints per-split min/max and counts of invalid ids.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from datasets import load_from_disk, Dataset, DatasetDict
except Exception as e:
    print(f"Failed to import datasets: {e}")
    sys.exit(2)

try:
    from transformers import AutoTokenizer
except Exception as e:
    print(f"Failed to import transformers: {e}")
    sys.exit(2)


def analyze_split(split_ds: Dataset, vocab_size: int, ignore_index: int, split_name: str, max_report: int = 10) -> Dict:
    violations: Dict[str, List[Tuple[int, int, int]]] = {
        "input_ids": [],  # (row_idx, position, token_id)
        "labels": [],     # (row_idx, position, token_id)
    }

    # Track stats
    min_input = None
    max_input = None
    min_label = None
    max_label = None
    min_label_excl = None
    max_label_excl = None

    batch_size = 1024
    total_rows = len(split_ds)

    # Ensure required columns exist
    required_cols = ["input_ids", "attention_mask", "labels"]
    for col in required_cols:
        if col not in split_ds.column_names:
            raise ValueError(f"Missing required column '{col}' in split '{split_name}'")

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = split_ds[start:end]
        input_ids: List[List[int]] = batch["input_ids"]
        labels: List[List[int]] = batch["labels"]

        # Convert to numpy for vectorized checks
        input_np = np.asarray(input_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)

        # Update stats
        if input_np.size:
            cur_min_input = int(input_np.min())
            cur_max_input = int(input_np.max())
            min_input = cur_min_input if min_input is None else min(min_input, cur_min_input)
            max_input = cur_max_input if max_input is None else max(max_input, cur_max_input)

        if labels_np.size:
            cur_min_label = int(labels_np.min())
            cur_max_label = int(labels_np.max())
            min_label = cur_min_label if min_label is None else min(min_label, cur_min_label)
            max_label = cur_max_label if max_label is None else max(max_label, cur_max_label)

            valid_mask = labels_np != ignore_index
            if np.any(valid_mask):
                valid_labels = labels_np[valid_mask]
                cur_min_excl = int(valid_labels.min())
                cur_max_excl = int(valid_labels.max())
                min_label_excl = cur_min_excl if min_label_excl is None else min(min_label_excl, cur_min_excl)
                max_label_excl = cur_max_excl if max_label_excl is None else max(max_label_excl, cur_max_excl)

        # Violations: input_ids must be in [0, vocab_size)
        if input_np.size:
            bad_mask_inp = (input_np < 0) | (input_np >= vocab_size)
            if np.any(bad_mask_inp):
                rows, cols = np.where(bad_mask_inp)
                for r, c in zip(rows, cols):
                    if len(violations["input_ids"]) < max_report:
                        violations["input_ids"].append((start + int(r), int(c), int(input_np[r, c])))

        # Violations: labels (excluding ignore_index) must be in [0, vocab_size)
        if labels_np.size:
            valid_mask = labels_np != ignore_index
            if np.any(valid_mask):
                valid_labels = labels_np[valid_mask]
                # Build full-shaped mask for out-of-range valid labels
                bad_vals = (valid_labels < 0) | (valid_labels >= vocab_size)
                if np.any(bad_vals):
                    # Reconstruct indices
                    rows, cols = np.where(((labels_np < 0) | (labels_np >= vocab_size)) & valid_mask)
                    for r, c in zip(rows, cols):
                        if len(violations["labels"]) < max_report:
                            violations["labels"].append((start + int(r), int(c), int(labels_np[r, c])))

    return {
        "split": split_name,
        "min_input": min_input,
        "max_input": max_input,
        "min_label": min_label,
        "max_label": max_label,
        "min_label_excl": min_label_excl,
        "max_label_excl": max_label_excl,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check token ids against tokenizer vocab size")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to tokenized dataset (load_from_disk)")
    parser.add_argument("--model_name", type=str, required=True, help="HF model/tokenizer name")
    parser.add_argument("--ignore_index", type=int, default=-100, help="Ignore index used in labels")
    parser.add_argument("--max_report", type=int, default=10, help="Max examples to report per split per field")
    args = parser.parse_args()

    ds_path = Path(args.dataset_path)
    if not ds_path.exists():
        print(f"Dataset path does not exist: {ds_path}")
        sys.exit(2)

    # Load tokenizer to get vocab size and mask id
    tok = AutoTokenizer.from_pretrained(args.model_name)
    vocab_size = len(tok.get_vocab()) if hasattr(tok, "get_vocab") else tok.vocab_size
    print(f"Tokenizer: {args.model_name} | vocab_size={vocab_size}")

    dataset = load_from_disk(str(ds_path))

    splits: Dict[str, Dataset]
    if isinstance(dataset, DatasetDict):
        splits = dataset
    else:
        splits = {"dataset": dataset}

    any_violations = False
    results = []
    for split_name, split_ds in splits.items():
        print(f"Checking split '{split_name}' with {len(split_ds)} rows...")
        res = analyze_split(split_ds, vocab_size=vocab_size, ignore_index=args.ignore_index, split_name=split_name, max_report=args.max_report)
        results.append(res)

    print("\nSummary:")
    for res in results:
        print(
            f"- {res['split']}: "
            f"input_ids min/max=({res['min_input']},{res['max_input']}), "
            f"labels min/max=({res['min_label']},{res['max_label']}), "
            f"labels(excl) min/max=({res['min_label_excl']},{res['max_label_excl']})"
        )
        v_inp = res["violations"]["input_ids"]
        v_lab = res["violations"]["labels"]
        if v_inp:
            any_violations = True
            print(f"  input_ids violations (first {len(v_inp)}):")
            for row_idx, pos, val in v_inp:
                print(f"    row={row_idx} pos={pos} token_id={val}")
        if v_lab:
            any_violations = True
            print(f"  labels violations (first {len(v_lab)}):")
            for row_idx, pos, val in v_lab:
                print(f"    row={row_idx} pos={pos} token_id={val}")

    if any_violations:
        print("\nResult: VIOLATIONS FOUND (token ids out of vocab range)")
        sys.exit(1)
    else:
        print("\nResult: No violations found. All token ids are within vocab range.")
        sys.exit(0)


if __name__ == "__main__":
    main()


