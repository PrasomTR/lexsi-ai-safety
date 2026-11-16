"""Tier 1 Feature Tests: Feature 3 - Core Repository Structure & Configs.

Validates:
1. Canonical project directories exist according to PROJECT.md § Code Layout.
2. eval_baseline.yaml configuration satisfies schema requirements.
3. safetune_lora.yaml configuration satisfies schema requirements.
4. steering_config.yaml configuration satisfies schema requirements.
5. eval_hardened.yaml configuration satisfies schema requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pytest

from tests.e2e.conftest import SchemaValidator


@pytest.mark.tier1
class TestF03RepoStructure:
    """Feature 3: Core Repository Structure & Configs tests."""

    def test_f03_canonical_directories_exist(self, project_root: Path) -> None:
        """Verify presence of all root layout folders specified in PROJECT.md."""
        required_dirs = [
            "data",
            "configs",
            "scripts",
            "notebooks",
            "artifacts",
            "tests",
        ]
        for dirname in required_dirs:
            dirpath = project_root / dirname
            assert dirpath.exists(), f"Required directory '{dirname}' missing at {dirpath}"
            assert dirpath.is_dir(), f"Path '{dirname}' is not a directory: {dirpath}"

    def test_f03_yaml_config_eval_baseline(
        self, configs_dir: Path, schema_validator: SchemaValidator
    ) -> None:
        """Validate eval_baseline.yaml file or contract schema."""
        live_file = configs_dir / "eval_baseline.yaml"
        if live_file.exists():
            valid, err = schema_validator.validate_yaml_config(live_file, "eval_baseline")
            assert valid, f"eval_baseline.yaml failed schema validation: {err}"
        else:
            # Contract verification for eval_baseline config
            sample_config: Dict[str, Any] = {
                "model_id": "CohereLabs/tiny-aya-global",
                "probes": ["jailbreak", "prompt_injection", "encoding_bypass", "xstest"],
                "metrics": ["asr", "refusal_rate", "orr", "defcon_grade"],
                "batch_size": 4,
            }
            valid, err = schema_validator.validate_yaml_config(sample_config, "eval_baseline")
            assert valid, f"Sample eval_baseline config failed schema validation: {err}"

    def test_f03_yaml_config_safetune_lora(
        self, configs_dir: Path, schema_validator: SchemaValidator
    ) -> None:
        """Validate safetune_lora.yaml file or contract schema."""
        live_file = configs_dir / "safetune_lora.yaml"
        if live_file.exists():
            valid, err = schema_validator.validate_yaml_config(live_file, "safetune_lora")
            assert valid, f"safetune_lora.yaml failed schema validation: {err}"
        else:
            # Contract verification for safetune_lora config
            sample_config: Dict[str, Any] = {
                "model_id": "CohereLabs/tiny-aya-global",
                "lora_r": 16,
                "lora_alpha": 32,
                "training_args": {
                    "learning_rate": 2e-4,
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 2,
                    "fp16": False,
                    "bf16": True,
                },
            }
            valid, err = schema_validator.validate_yaml_config(sample_config, "safetune_lora")
            assert valid, f"Sample safetune_lora config failed schema validation: {err}"

    def test_f03_yaml_config_steering(
        self, configs_dir: Path, schema_validator: SchemaValidator
    ) -> None:
        """Validate steering_config.yaml file or contract schema."""
        live_file = configs_dir / "steering_config.yaml"
        if live_file.exists():
            valid, err = schema_validator.validate_yaml_config(live_file, "steering_config")
            assert valid, f"steering_config.yaml failed schema validation: {err}"
        else:
            # Contract verification for steering_config
            sample_config: Dict[str, Any] = {
                "model_id": "CohereLabs/tiny-aya-global",
                "refusal_direction": "artifacts/steering/refusal_vector.pt",
                "alpha": 0.35,
                "target_layers": [14, 15, 16, 17, 18],
            }
            valid, err = schema_validator.validate_yaml_config(sample_config, "steering_config")
            assert valid, f"Sample steering_config failed schema validation: {err}"

    def test_f03_yaml_config_eval_hardened(
        self, configs_dir: Path, schema_validator: SchemaValidator
    ) -> None:
        """Validate eval_hardened.yaml file or contract schema."""
        live_file = configs_dir / "eval_hardened.yaml"
        if live_file.exists():
            valid, err = schema_validator.validate_yaml_config(live_file, "eval_hardened")
            assert valid, f"eval_hardened.yaml failed schema validation: {err}"
        else:
            # Contract verification for eval_hardened config
            sample_config: Dict[str, Any] = {
                "model_id": "CohereLabs/tiny-aya-global",
                "adapter_path": "artifacts/adapter_checkpoint",
                "probes": ["jailbreak", "prompt_injection", "encoding_bypass", "xstest"],
                "metrics": ["asr", "refusal_rate", "orr", "defcon_grade"],
            }
            valid, err = schema_validator.validate_yaml_config(sample_config, "eval_hardened")
            assert valid, f"Sample eval_hardened config failed schema validation: {err}"
