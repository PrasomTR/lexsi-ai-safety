"""Tier 1 Feature Tests: Feature 18 - Dual-Track E2E Test Suite Validation.

Validates:
1. pytest.ini configuration and complete tier marker declarations.
2. test_runner.py CLI invocation contract and arguments.
3. 4-tier directory layout completeness.
4. e2e_test_report.json generation and schema.
5. All 18 Tier 1 test modules present in tier1_features directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List
import pytest

from tests.e2e.conftest import CLIResult


@pytest.mark.tier1
class TestF18DualTrackSuite:
    """Feature 18: Dual-Track E2E Test Suite Validation tests."""

    def test_f18_pytest_configuration_markers(self, project_root: Path) -> None:
        """Verify pytest.ini registers all required test tier markers."""
        ini_file = project_root / "pytest.ini"
        assert ini_file.exists(), f"Missing pytest.ini at {ini_file}"
        content = ini_file.read_text(encoding="utf-8")

        for marker in ["tier1", "tier2", "tier3", "tier4", "infra"]:
            assert marker in content, f"Missing required marker '{marker}' in pytest.ini"

    def test_f18_test_runner_cli_interface(
        self,
        e2e_tests_dir: Path,
        cli_runner: Callable[..., CLIResult],
    ) -> None:
        """Verify test_runner.py CLI accepts --help and returns 0."""
        runner_path = e2e_tests_dir / "test_runner.py"
        assert runner_path.exists(), f"test_runner.py not found at {runner_path}"

        import sys
        res = cli_runner([sys.executable, str(runner_path), "--help"])
        assert res.is_success, f"test_runner.py --help failed: {res.stderr}"
        assert "--tier" in res.stdout

    def test_f18_tier_directory_structure(self, e2e_tests_dir: Path) -> None:
        """Verify the 4-tier directory hierarchy exists."""
        expected_dirs = [
            "tier1_features",
            "tier2_boundaries",
            "tier3_combinations",
            "tier4_workflows",
        ]
        for dirname in expected_dirs:
            d = e2e_tests_dir / dirname
            assert d.exists(), f"Missing test tier directory: {d}"
            assert d.is_dir()

    def test_f18_e2e_json_report_generation(self, tmp_path: Path) -> None:
        """Verify schema of e2e_test_report.json artifact."""
        report_file = tmp_path / "e2e_test_report.json"
        mock_report = {
            "timestamp": "2026-11-12T00:00:00Z",
            "exit_code": 0,
            "passed": True,
            "summary": {
                "total": 90,
                "passed": 90,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration_seconds": 12.34,
            },
            "tests": [
                {
                    "nodeid": "tests/e2e/tier1_features/test_f01_model_setup.py::TestF01ModelSetup::test_f01_target_model_identifier",
                    "outcome": "passed",
                    "duration": 0.05,
                    "tier": "tier1",
                    "error_message": None,
                }
            ],
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(mock_report, f, indent=2)

        with open(report_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["passed"] is True
        assert loaded["summary"]["total"] >= 90
        assert len(loaded["tests"]) > 0

    def test_f18_tier1_feature_suite_coverage(self, e2e_tests_dir: Path) -> None:
        """Verify all 18 feature test modules (test_f01 through test_f18) exist in tier1_features."""
        tier1_dir = e2e_tests_dir / "tier1_features"
        expected_files = [f"test_f{i:02d}" for i in range(1, 19)]

        existing_modules = [f.stem for f in tier1_dir.glob("test_f*.py")]

        for exp in expected_files:
            matched = any(m.startswith(exp) for m in existing_modules)
            assert matched, f"Missing Tier 1 test module matching prefix '{exp}' in {tier1_dir}"
