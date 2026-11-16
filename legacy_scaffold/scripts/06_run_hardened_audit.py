"""Stage 06: Post-Hardening Differential Verification Audit.

Evaluates the hardened model (QLoRA adapter + activation steering) using AuditKIT,
compares results against the baseline, and produces differential report delta_report.md.
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
        description="Run AuditKIT post-hardening differential evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "eval_hardened.yaml",
        help="Path to hardened evaluation YAML configuration.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline_eval" / "results.json",
        help="Path to baseline evaluation results.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hardened_eval",
        help="Directory to save hardened evaluation results and fingerprint.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "comparison",
        help="Directory to save differential comparison delta_report.md.",
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

    print(f"[06_run_hardened_audit] Loading configuration from: {args.config}")
    print(f"[06_run_hardened_audit] Baseline results from: {args.baseline_results}")
    print(f"[06_run_hardened_audit] Hardened results to: {args.output_dir}")
    print(f"[06_run_hardened_audit] Differential comparison to: {args.comparison_dir}")

    hardened_results, fingerprint, report_text = engine.run_hardened_pipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        comparison_dir=args.comparison_dir,
        baseline_results_path=args.baseline_results,
        dry_run=args.dry_run,
    )

    print(f"[06_run_hardened_audit] Hardened ASR: {hardened_results.get('asr', 0.0) * 100:.1f}%")
    print(f"[06_run_hardened_audit] Refusal Rate: {hardened_results.get('refusal_rate', 0.0) * 100:.1f}%")
    print(f"[06_run_hardened_audit] Over-Refusal Rate (ORR): {hardened_results.get('orr', 0.0) * 100:.1f}%")
    print(f"[06_run_hardened_audit] DEFCON Grade: {hardened_results.get('defcon_grade', 'N/A')}")
    print(f"[06_run_hardened_audit] Cryptographic Fingerprint: {fingerprint}")
    print(f"[06_run_hardened_audit] Differential report generated at: {args.comparison_dir / 'delta_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
