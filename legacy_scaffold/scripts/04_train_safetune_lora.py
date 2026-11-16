"""Stage 04: Train-Time Safety Hardening with SafeTune QLoRA.

Trains a 4-bit NormalFloat (NF4) quantized parameter-efficient safety adapter
targeting Cohere Tiny Aya using SafeTune/PEFT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mitigation.trainer import SafeTuneTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SafeTune 4-bit QLoRA safety adapter for Cohere Tiny Aya."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "safetune_lora.yaml",
        help="Path to SafeTune QLoRA training YAML configuration.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "safety_train_pairs.jsonl",
        help="Path to curated safety training pairs JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "adapter_checkpoint",
        help="Output directory to save the trained PEFT adapter checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run training in fast decoupled dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trainer = SafeTuneTrainer()

    print(f"[04_train_safetune_lora] Loading configuration from: {args.config}")
    print(f"[04_train_safetune_lora] Output adapter directory: {args.output_dir}")

    adapter_config = trainer.train(
        config_path=args.config,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        dry_run=args.dry_run,
    )

    print(f"[04_train_safetune_lora] Hardened adapter checkpoint saved to: {args.output_dir}")
    print(f"[04_train_safetune_lora] Base model: {adapter_config.get('base_model_name_or_path')}")
    print(f"[04_train_safetune_lora] LoRA Rank: {adapter_config.get('r')}, Alpha: {adapter_config.get('lora_alpha')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
