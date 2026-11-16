"""Stage 05: Inference-Time Activation Steering Calibration.

Calibrates refusal direction vectors and configures activation steering hooks
using SafeTune to provide dual-layer inference alignment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mitigation.steering import ActivationSteeringManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and export SafeTune activation steering vectors."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "steering_config.yaml",
        help="Path to steering YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "steering",
        help="Output directory to save refusal direction vector and metadata.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha multiplier override (nominal calibration range: 0.10 - 0.60).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run calibration in fast decoupled dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steering_mgr = ActivationSteeringManager()

    print(f"[05_apply_steering] Loading configuration from: {args.config}")
    print(f"[05_apply_steering] Output steering directory: {args.output_dir}")

    meta = steering_mgr.calibrate_refusal_direction(
        config_path=args.config,
        output_dir=args.output_dir,
        alpha_override=args.alpha,
        dry_run=args.dry_run,
    )

    print(f"[05_apply_steering] Calibrated refusal direction vector.")
    print(f"[05_apply_steering] Steered layers: {meta.get('target_layers')}")
    print(f"[05_apply_steering] Calibrated Alpha: {meta.get('alpha')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
