#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict


def load_config_dict(model_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in: {model_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_model_weights_file(model_dir: str) -> str | None:
    # Prefer safetensors if present
    st_path = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(st_path):
        return st_path
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    return bin_path if os.path.isfile(bin_path) else None


def check_kan_annotations(config: Dict[str, Any]) -> tuple[bool, str]:
    # We expect the converter to annotate this metadata
    meta = config.get("lmtk_replace_ffn")
    if meta is None:
        return False, "Missing lmtk_replace_ffn in config.json"
    if not isinstance(meta, dict):
        return False, "lmtk_replace_ffn is not a dict"
    if meta.get("type") != "kan_chebyshev":
        return False, f"Unexpected lmtk_replace_ffn.type: {meta.get('type')}"
    order = meta.get("cheb_order")
    try:
        order_int = int(order)
    except Exception:
        return False, f"Invalid cheb_order value: {order}"
    if order_int < 1:
        return False, f"cheb_order must be >=1, got {order_int}"
    return True, f"KAN FFN annotation found (type=kan_chebyshev, cheb_order={order_int})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify KAN-FFN metadata in a saved HF model")
    parser.add_argument("model_dir", help="Path to the HF model directory (contains config.json)")
    parser.add_argument("--expect-order", type=int, default=None, help="Optional expected cheb_order to assert")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"❌ Not a directory: {model_dir}", file=sys.stderr)
        return 2

    # Basic file presence checks
    cfg_dict = load_config_dict(model_dir)
    weights_file = find_model_weights_file(model_dir)
    if weights_file is None:
        print("⚠️  No model weight file found (model.safetensors or pytorch_model.bin)")
    else:
        print(f"✅ Found weights: {os.path.basename(weights_file)}")

    ok, msg = check_kan_annotations(cfg_dict)
    if not ok:
        print(f"❌ {msg}", file=sys.stderr)
        return 1
    print(f"✅ {msg}")

    # Optional strict order assertion
    if args.expect_order is not None:
        order = int(cfg_dict.get("lmtk_replace_ffn", {}).get("cheb_order", -1))
        if order != args.expect_order:
            print(
                f"❌ cheb_order mismatch: expected {args.expect_order}, found {order}",
                file=sys.stderr,
            )
            return 1
        print(f"✅ cheb_order matches expectation: {args.expect_order}")

    # Summary
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


