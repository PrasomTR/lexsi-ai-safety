"""Tier 1 Feature Tests: Feature 4 - Red-Teaming Benchmark Dataset Curation.

Validates:
1. Red-teaming record schema adherence ({id, category, prompt, source}).
2. Category diversity across jailbreak, prompt injection, and encoding bypass.
3. Provenance and attribution integrity to benchmark sources.
4. Dataset line integrity, non-emptiness, and unique record identifiers.
5. Full JSONL file validation using SchemaValidator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List
import pytest

from tests.e2e.conftest import SchemaValidator, SyntheticDataGenerator


@pytest.mark.tier1
class TestF04RedteamDataset:
    """Feature 4: Red-Teaming Benchmark Dataset Curation tests."""

    def test_f04_redteam_record_schema_conformance(
        self,
        schema_validator: SchemaValidator,
        synthetic_redteam_batch: List[Dict[str, str]],
    ) -> None:
        """Verify individual redteam records adhere strictly to schema."""
        for rec in synthetic_redteam_batch:
            valid, err = schema_validator.validate_redteam_record(rec)
            assert valid, f"Record {rec.get('id')} failed schema validation: {err}"
            assert isinstance(rec["id"], str) and len(rec["id"]) > 0
            assert isinstance(rec["category"], str) and len(rec["category"]) > 0
            assert isinstance(rec["prompt"], str) and len(rec["prompt"]) > 0
            assert isinstance(rec["source"], str) and len(rec["source"]) > 0

    def test_f04_category_coverage_diversity(
        self, synthetic_redteam_batch: List[Dict[str, str]]
    ) -> None:
        """Verify red-teaming dataset includes required threat vector categories."""
        categories = {r["category"] for r in synthetic_redteam_batch}
        expected_categories = {"jailbreak", "prompt_injection", "encoding_bypass"}
        # Ensure intersection has major categories
        assert expected_categories.issubset(categories), (
            f"Missing essential redteam categories. Found: {categories}"
        )

    def test_f04_source_provenance_integrity(
        self, synthetic_generators: SyntheticDataGenerator
    ) -> None:
        """Verify provenance tracking records legitimate benchmark origin sources."""
        sources = ["AdvBench", "HarmBench", "CuratorKIT_Scrubbed", "Synthetic_Frontier"]
        for idx, src in enumerate(sources, 1):
            record = synthetic_generators.redteam_record(idx=idx, category="jailbreak", source=src)
            assert record["source"] == src
            assert len(record["source"]) > 3

    def test_f04_dataset_line_integrity_and_unique_ids(
        self, synthetic_generators: SyntheticDataGenerator
    ) -> None:
        """Verify record IDs are globally unique and prompts are non-trivial."""
        batch = synthetic_generators.redteam_dataset(count=20)
        ids = [r["id"] for r in batch]
        assert len(ids) == len(set(ids)), "Found duplicate IDs in redteam benchmark dataset"

        for r in batch:
            assert len(r["prompt"]) >= 10, f"Prompt too short for realistic probe: {r['prompt']}"

    def test_f04_redteam_dataset_file_or_synthetic_validation(
        self,
        data_dir: Path,
        schema_validator: SchemaValidator,
        create_temp_jsonl: Callable[[str, List[Dict[str, str]]], Path],
        synthetic_redteam_batch: List[Dict[str, str]],
    ) -> None:
        """Validate live data/redteam_benchmark.jsonl or synthetic JSONL via validate_jsonl_file."""
        live_file = data_dir / "redteam_benchmark.jsonl"
        target_file = live_file if live_file.exists() else create_temp_jsonl(
            "temp_redteam.jsonl", synthetic_redteam_batch
        )

        valid, errors = schema_validator.validate_jsonl_file(target_file, "redteam")
        assert valid, f"JSONL validation failed for {target_file}: {errors}"
        assert len(errors) == 0
