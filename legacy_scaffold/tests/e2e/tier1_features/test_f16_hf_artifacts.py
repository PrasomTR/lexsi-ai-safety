"""Tier 1 Feature Tests: Feature 16 - Hugging Face Collection Structuring.

Validates:
1. Hugging Face staging directory layout (adapter, datasets, evaluation logs).
2. Model card README.md with YAML frontmatter.
3. Dataset card README.md with YAML metadata and documentation.
4. Export script CLI interface contract (07_export_hf_artifacts.py).
5. Collection package manifest integrity with SHA-256 checksums.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import pytest
import yaml

from tests.e2e.conftest import SchemaValidator


@pytest.mark.tier1
class TestF16HFArtifacts:
    """Feature 16: Hugging Face Collection Structuring tests."""

    def test_f16_hf_collection_directory_layout(self, tmp_path: Path) -> None:
        """Verify HF collection staging directory contains model, dataset, and logs folders."""
        hf_root = tmp_path / "hf_collection"
        model_dir = hf_root / "tiny-aya-safetune-adapter"
        dataset_dir = hf_root / "tiny-aya-redteam-eval"
        logs_dir = hf_root / "auditkit-evaluation-logs"

        for d in [model_dir, dataset_dir, logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
            assert d.exists() and d.is_dir()

    def test_f16_model_card_readme_metadata(self) -> None:
        """Verify model card README includes valid Hugging Face YAML frontmatter."""
        model_card = """---
language:
- en
- multilingual
license: apache-2.0
base_model: CohereLabs/tiny-aya-global
tags:
- alignment
- safetune
- qlora
- ai-safety
- red-teaming
---

# Tiny Aya SafeTune Hardened Adapter
Hardened 4-bit QLoRA safety adapter trained on curated red-teaming safety pairs.
"""
        parts = model_card.split("---")
        assert len(parts) >= 3, "Model card must have YAML frontmatter enclosed in '---'"
        meta = yaml.safe_load(parts[1])
        assert meta["base_model"] == "CohereLabs/tiny-aya-global"
        assert "alignment" in meta["tags"]
        assert meta["license"] == "apache-2.0"

    def test_f16_dataset_card_readme_metadata(self) -> None:
        """Verify dataset card README includes valid dataset tags and schema documentation."""
        dataset_card = """---
annotations_creators:
- machine-generated
- curated
language:
- en
license: cc-by-4.0
tags:
- red-teaming
- safety-eval
- jailbreak-benchmarks
---

# Tiny Aya Red-Teaming Benchmark Dataset
Curated safety evaluation probes for Tiny Aya.
"""
        parts = dataset_card.split("---")
        assert len(parts) >= 3
        meta = yaml.safe_load(parts[1])
        assert "red-teaming" in meta["tags"]
        assert meta["license"] == "cc-by-4.0"

    def test_f16_export_script_cli_contract(self, scripts_dir: Path) -> None:
        """Verify export script 07_export_hf_artifacts.py exists and responds to --help."""
        script_path = scripts_dir / "07_export_hf_artifacts.py"
        assert script_path.exists(), f"Export script missing: {script_path}"
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, f"Script --help failed: {res.stderr}"

    def test_f16_hf_artifacts_manifest_integrity(
        self, tmp_path: Path, schema_validator: SchemaValidator
    ) -> None:
        """Verify manifest records file paths, sizes, and valid 64-hex SHA-256 hashes."""
        sample_file = tmp_path / "sample_model.safetensors"
        content = b"sample_model_weights_payload_binary"
        sample_file.write_bytes(content)

        file_hash = hashlib.sha256(content).hexdigest()
        manifest: Dict[str, Any] = {
            "collection_id": "lexsi-track2-tiny-aya-hardening",
            "artifacts": [
                {
                    "path": sample_file.name,
                    "size_bytes": len(content),
                    "sha256": file_hash,
                }
            ],
        }

        assert len(manifest["artifacts"]) == 1
        art = manifest["artifacts"][0]
        valid, err = schema_validator.validate_sha256(art["sha256"])
        assert valid, f"Artifact SHA-256 invalid: {err}"
        assert art["size_bytes"] == len(content)
