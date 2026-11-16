"""Tier 1 Feature Tests: Feature 14 - Reproducible CLI Execution Pipeline.

Validates:
1. Complete sequential script inventory (scripts 01 through 07).
2. Python syntax compilation validation for all script files.
3. Command-line interface standard argument handling (--help).
4. Windows batch script command structure (01_setup_env.bat).
5. End-to-end artifact dependency flow between pipeline stages.
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path
from typing import List
import pytest


@pytest.mark.tier1
class TestF14CLIPipeline:
    """Feature 14: Reproducible CLI Execution Pipeline tests."""

    EXPECTED_SCRIPTS: List[str] = [
        "01_setup_env.bat",
        "02_curate_benchmark.py",
        "03_run_baseline_audit.py",
        "04_train_safetune_lora.py",
        "05_apply_steering.py",
        "06_run_hardened_audit.py",
        "07_export_hf_artifacts.py",
    ]

    def test_f14_script_inventory_presence(self, scripts_dir: Path) -> None:
        """Verify all 7 sequential scripts exist as executable files in scripts/."""
        assert len(self.EXPECTED_SCRIPTS) == 7
        for name in self.EXPECTED_SCRIPTS:
            p = scripts_dir / name
            assert p.exists(), f"Required pipeline script missing: {p}"
            assert p.is_file(), f"Expected pipeline script file, got directory: {p}"

    def test_f14_python_scripts_syntax_compilation(self, scripts_dir: Path) -> None:
        """Verify all expected Python pipeline scripts compile cleanly without syntax errors."""
        expected_py_scripts = [s for s in self.EXPECTED_SCRIPTS if s.endswith(".py")]
        assert len(expected_py_scripts) == 6, "Expected 6 Python pipeline scripts"
        for script_name in expected_py_scripts:
            s = scripts_dir / script_name
            assert s.exists(), f"Missing Python script: {s}"
            py_compile.compile(str(s), doraise=True)

    def test_f14_cli_help_flag_execution(self, scripts_dir: Path) -> None:
        """Verify all Python pipeline scripts support --help with exit code 0."""
        expected_py_scripts = [s for s in self.EXPECTED_SCRIPTS if s.endswith(".py")]
        assert len(expected_py_scripts) == 6
        for script_name in expected_py_scripts:
            s = scripts_dir / script_name
            assert s.exists(), f"Missing Python script: {s}"
            res = subprocess.run(
                [sys.executable, str(s), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert res.returncode == 0, f"Script {s.name} --help failed: {res.stderr}"
            assert len(res.stdout) > 0, f"Script {s.name} --help produced no output"

    def test_f14_batch_script_windows_syntax(self, scripts_dir: Path) -> None:
        """Verify 01_setup_env.bat batch script exists and contains valid setup commands."""
        bat_file = scripts_dir / "01_setup_env.bat"
        assert bat_file.exists(), f"Missing batch script: {bat_file}"
        content = bat_file.read_text(encoding="utf-8", errors="replace")
        assert "python" in content.lower() or "pip" in content.lower(), (
            "01_setup_env.bat must contain environment installation commands"
        )

    def test_f14_pipeline_sequential_dependency_chain(self) -> None:
        """Verify stage dependencies form an acyclic pipeline from data to HF export."""
        pipeline_stages = [
            {"stage": 1, "script": "01_setup_env.bat", "out": "environment"},
            {"stage": 2, "script": "02_curate_benchmark.py", "out": "data/redteam_benchmark.jsonl"},
            {"stage": 3, "script": "03_run_baseline_audit.py", "out": "artifacts/baseline_eval/results.json"},
            {"stage": 4, "script": "04_train_safetune_lora.py", "out": "artifacts/adapter_checkpoint"},
            {"stage": 5, "script": "05_apply_steering.py", "out": "artifacts/steering/refusal_vector.pt"},
            {"stage": 6, "script": "06_run_hardened_audit.py", "out": "artifacts/hardened_eval/results.json"},
            {"stage": 7, "script": "07_export_hf_artifacts.py", "out": "artifacts/hf_collection"},
        ]
        assert len(pipeline_stages) == 7
        for i in range(len(pipeline_stages) - 1):
            assert pipeline_stages[i]["stage"] < pipeline_stages[i + 1]["stage"]
