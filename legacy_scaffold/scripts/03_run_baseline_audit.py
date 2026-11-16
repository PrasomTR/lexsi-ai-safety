"""Stage 03: Baseline Red-Teaming and Vulnerability Evaluation.

Executes baseline evaluation of Cohere Tiny Aya across safety and robustness probes
using AuditKIT, generating baseline metrics and SHA-256 fingerprint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.engine import AuditKitEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AuditKIT baseline red-teaming audit against Cohere Tiny Aya."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "eval_baseline.yaml",
        help="Path to baseline evaluation YAML configuration.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "data" / "redteam_benchmark.jsonl",
        help="Path to benchmark JSONL probes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline_eval",
        help="Directory to save evaluation results and run fingerprint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run audit in fast decoupled simulation mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = AuditKitEngine()

    print(f"[03_run_baseline_audit] Loading configuration from: {args.config}")
    print(f"[03_run_baseline_audit] Output directory: {args.output_dir}")

    results, fingerprint = engine.run_baseline_pipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        benchmark_path=args.benchmark,
        dry_run=args.dry_run,
    )

    print(f"[03_run_baseline_audit] Baseline ASR: {results.get('asr', 0.0) * 100:.1f}%")
    print(f"[03_run_baseline_audit] Refusal Rate: {results.get('refusal_rate', 0.0) * 100:.1f}%")
    print(f"[03_run_baseline_audit] Over-Refusal Rate (ORR): {results.get('orr', 0.0) * 100:.1f}%")
    print(f"[03_run_baseline_audit] DEFCON Grade: {results.get('defcon_grade', 'N/A')}")
    print(f"[03_run_baseline_audit] Cryptographic Fingerprint: {fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
