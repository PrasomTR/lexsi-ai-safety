"""Tier 1 Feature Tests: Feature 8 - Baseline Metric & Fingerprint Logging.

Validates:
1. Baseline safety metric ranges (ASR, Refusal Rate, ORR in [0.0, 1.0]).
2. DEFCON grade classification string formatting and mapping.
3. Cryptographic SHA-256 evaluation run fingerprint format.
4. run_fingerprint.txt file persistence and exact formatting.
5. results.json persistence and roundtrip JSON deserialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pytest

from tests.e2e.conftest import SchemaValidator, SyntheticDataGenerator


@pytest.mark.tier1
class TestF08BaselineMetrics:
    """Feature 8: Baseline Metric & Fingerprint Logging tests."""

    def test_f08_baseline_metric_ranges(
        self, synthetic_baseline_results: Dict[str, Any]
    ) -> None:
        """Verify baseline metrics are well-formed numbers in the unit interval [0, 1]."""
        asr = synthetic_baseline_results["asr"]
        refusal = synthetic_baseline_results["refusal_rate"]
        orr = synthetic_baseline_results["orr"]

        assert 0.0 <= asr <= 1.0, f"ASR out of bounds: {asr}"
        assert 0.0 <= refusal <= 1.0, f"Refusal rate out of bounds: {refusal}"
        assert 0.0 <= orr <= 1.0, f"ORR out of bounds: {orr}"

        # Baseline model exhibits vulnerability (ASR > 0.50)
        assert asr > 0.50, f"Expected vulnerable baseline ASR, got: {asr}"

    def test_f08_defcon_grade_classification(
        self, synthetic_baseline_results: Dict[str, Any]
    ) -> None:
        """Verify DEFCON grade follows competition threat tier nomenclature."""
        grade = synthetic_baseline_results["defcon_grade"]
        valid_grades = {"DEFCON 1", "DEFCON 2", "DEFCON 3", "DEFCON 4", "DEFCON 5"}
        assert grade in valid_grades, f"Unexpected DEFCON grade: {grade}"

    def test_f08_sha256_fingerprint_cryptographic_format(
        self,
        schema_validator: SchemaValidator,
        synthetic_fingerprint: str,
    ) -> None:
        """Verify evaluation fingerprint is a valid 64-character hexadecimal SHA-256 hash."""
        valid, err = schema_validator.validate_sha256(synthetic_fingerprint)
        assert valid, f"SHA-256 validation failed: {err}"
        assert len(synthetic_fingerprint) == 64
        assert all(c in "0123456789abcdefABCDEF" for c in synthetic_fingerprint)

    def test_f08_run_fingerprint_file_persistence(
        self, tmp_path: Path, synthetic_generators: SyntheticDataGenerator
    ) -> None:
        """Verify writing and reading run_fingerprint.txt preserving hash integrity."""
        fp_file = tmp_path / "run_fingerprint.txt"
        test_hash = synthetic_generators.sha256_fingerprint("auditkit_eval_seed_2026")
        fp_file.write_text(test_hash.strip() + "\n", encoding="utf-8")

        read_hash = fp_file.read_text(encoding="utf-8").strip()
        assert read_hash == test_hash
        assert len(read_hash) == 64

    def test_f08_baseline_results_json_persistence(
        self,
        tmp_path: Path,
        schema_validator: SchemaValidator,
        synthetic_baseline_results: Dict[str, Any],
    ) -> None:
        """Verify serialization and deserialization of results.json."""
        out_file = tmp_path / "results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(synthetic_baseline_results, f, indent=2)

        with open(out_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        valid, err = schema_validator.validate_audit_results(loaded_data)
        assert valid, f"Persisted results failed validation: {err}"
        assert loaded_data["asr"] == synthetic_baseline_results["asr"]
        assert loaded_data["defcon_grade"] == synthetic_baseline_results["defcon_grade"]
