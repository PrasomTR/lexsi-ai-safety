"""Lexsi AI Safety Hackathon - Track 2: AI Red Teaming, Safety and Alignment.

Standalone E2E Test Runner CLI.
Usage:
    python tests/e2e/test_runner.py --tier all
    python tests/e2e/test_runner.py --tier 1
    python tests/e2e/test_runner.py --tier 2
    python tests/e2e/test_runner.py --tier 3
    python tests/e2e/test_runner.py --tier 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "e2e_test_report.json"

TIER_DIRECTORIES = {
    "1": PROJECT_ROOT / "tests" / "e2e" / "tier1_features",
    "2": PROJECT_ROOT / "tests" / "e2e" / "tier2_boundaries",
    "3": PROJECT_ROOT / "tests" / "e2e" / "tier3_combinations",
    "4": PROJECT_ROOT / "tests" / "e2e" / "tier4_workflows",
    "all": PROJECT_ROOT / "tests" / "e2e",
}

TIER_MARKERS = {
    "1": "tier1",
    "2": "tier2",
    "3": "tier3",
    "4": "tier4",
}


def build_pytest_command(
    tier: str = "all",
    verbose: bool = True,
    fail_fast: bool = False,
    marker: Optional[str] = None,
    keyword: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Tuple[List[str], Path]:
    """Construct the pytest subprocess argument list and target directory."""
    target_dir = TIER_DIRECTORIES.get(tier, PROJECT_ROOT / "tests" / "e2e")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(target_dir),
    ]

    if verbose:
        cmd.append("-v")

    if fail_fast:
        cmd.append("-x")

    if marker:
        cmd.extend(["-m", marker])

    if keyword:
        cmd.extend(["-k", keyword])

    if extra_args:
        cmd.extend(extra_args)

    return cmd, target_dir


def count_test_files(target_dir: Path) -> int:
    """Count python test files matching test_*.py in target_dir."""
    if not target_dir.exists():
        return 0
    return len(list(target_dir.rglob("test_*.py")))


def run_test_suite(
    tier: str = "all",
    verbose: bool = True,
    fail_fast: bool = False,
    allow_empty: bool = False,
    marker: Optional[str] = None,
    keyword: Optional[str] = None,
    report_path: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run pytest test suite for the requested tier and generate a summary report.

    Returns the report data dictionary.
    """
    start_time = time.time()
    out_report = report_path or DEFAULT_REPORT_PATH
    out_report.parent.mkdir(parents=True, exist_ok=True)

    cmd, target_dir = build_pytest_command(
        tier=tier,
        verbose=verbose,
        fail_fast=fail_fast,
        marker=marker,
        keyword=keyword,
        extra_args=extra_args,
    )

    test_file_count = count_test_files(target_dir)

    print("=" * 70)
    print(f"LEXSI AI SAFETY E2E TEST RUNNER")
    print(f"Selected Tier:   {tier.upper()}")
    print(f"Target Path:     {target_dir}")
    print(f"Test Files:      {test_file_count} found")
    print(f"Report Target:   {out_report}")
    print("=" * 70)
    sys.stdout.flush()

    # Clean stale report prior to execution to prevent false-positive reuse
    if out_report.exists():
        try:
            out_report.unlink()
        except OSError:
            pass

    # Prepare environment with project root on PYTHONPATH and UTF-8
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["E2E_TIER"] = tier  # Pass tier to conftest hook
    env["E2E_REPORT_PATH"] = str(out_report.resolve())
    cur_pythonpath = env.get("PYTHONPATH", "")
    project_str = str(PROJECT_ROOT)
    if project_str not in cur_pythonpath:
        env["PYTHONPATH"] = f"{project_str};{cur_pythonpath}" if cur_pythonpath else project_str

    # Run pytest subprocess
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=False,  # Stream directly to stdout/stderr
    )

    duration = round(time.time() - start_time, 4)
    exit_code = proc.returncode

    # If allow_empty is enabled and pytest returned 5 (NO_TESTS_COLLECTED), consider it passed
    is_empty_pass = (exit_code == 5 and allow_empty)

    # Inspect or load the generated report safely directly from out_report
    report_data: Dict[str, Any]
    report_exists = out_report.exists() and out_report.stat().st_size > 0
    if report_exists and exit_code in (0, 1):
        try:
            with open(out_report, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except Exception:
            report_data = _build_fallback_report(tier, exit_code, duration, allow_empty=allow_empty)
    else:
        report_data = _build_fallback_report(tier, exit_code, duration, allow_empty=allow_empty)

    # Strict exit-code enforcement to prevent false-positive passes
    if is_empty_pass:
        report_data["passed"] = True
        report_data["exit_code"] = 0
    elif exit_code != 0:
        report_data["passed"] = False
        report_data["exit_code"] = exit_code
        summary = report_data.setdefault("summary", {})
        if summary.get("failed", 0) == 0 and summary.get("errors", 0) == 0:
            summary["failed"] = 1

    # Guarantee top-level schema contract adherence (tier and details)
    report_data["tier"] = tier
    if "details" not in report_data:
        report_data["details"] = report_data.get("tests", [])
    if "tests" not in report_data:
        report_data["tests"] = report_data.get("details", [])

    # Always ensure report file is written
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    _print_summary_table(report_data, tier)

    return report_data


def _build_fallback_report(
    tier: str,
    exit_code: int,
    duration: float,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    """Fallback report generator if conftest hook was bypassed or no tests ran."""
    passed = (exit_code == 0) or (exit_code == 5 and allow_empty)
    records: List[Dict[str, Any]] = []
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier": tier,
        "exit_code": 0 if passed else exit_code,
        "passed": passed,
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0 if passed else 1,
            "skipped": 0,
            "errors": 0,
            "duration_seconds": duration,
        },
        "tests": records,
        "details": records,
    }


def _print_summary_table(report: Dict[str, Any], tier: str) -> None:
    """Print an ASCII summary table to stdout."""
    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    errors = summary.get("errors", 0)
    duration = summary.get("duration_seconds", 0.0)
    status_str = "PASSED" if report.get("passed", False) else "FAILED"

    print("\n" + "=" * 70)
    print(f"E2E TEST RUN SUMMARY [{status_str}] - Tier: {tier.upper()}")
    print("-" * 70)
    print(f"Total Tests Executed: {total:>5}")
    print(f"Passed:               {passed:>5}")
    print(f"Failed:               {failed:>5}")
    print(f"Skipped:              {skipped:>5}")
    print(f"Errors:               {errors:>5}")
    print(f"Execution Duration:   {duration:>8.2f}s")
    print("=" * 70)


def main() -> int:
    """CLI entry point for tests/e2e/test_runner.py."""
    parser = argparse.ArgumentParser(
        description="Lexsi AI Safety Hackathon E2E Test Suite Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Test tier to execute: 1 (features), 2 (boundaries), 3 (combinations), 4 (workflows), or all.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Destination path for the generated machine-readable JSON report.",
    )
    parser.add_argument(
        "-m",
        "--marker",
        type=str,
        default=None,
        help="Pytest marker expression (e.g. 'tier1', 'infra').",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        default=None,
        help="Pytest keyword expression to filter test functions.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        default=False,
        help="Treat exit code 5 (no tests collected) as passing (useful during early pipeline bootstrapping).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose test output.",
    )
    parser.add_argument(
        "-x",
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop pytest on first test failure.",
    )

    args, unknown_args = parser.parse_known_args()

    # Forward any unknown CLI arguments directly to pytest
    forwarded_extra = []
    if unknown_args:
        forwarded_extra.extend(unknown_args)

    report = run_test_suite(
        tier=args.tier,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        allow_empty=args.allow_empty,
        marker=args.marker,
        keyword=args.keyword,
        report_path=args.report_path,
        extra_args=forwarded_extra if forwarded_extra else None,
    )

    # Exit code: 0 on all pass, 1 on any failure
    return 0 if report.get("passed", False) else 1


if __name__ == "__main__":
    sys.exit(main())
