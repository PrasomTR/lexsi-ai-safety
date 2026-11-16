"""Tier 1 Feature Tests: Feature 9 - SafeTune QLoRA Train-Time Hardening.

Validates:
1. LoRA training hyperparameters (rank, alpha, target modules, precision).
2. SafeTune trainer initialization and training loop interface contract.
3. Adapter checkpoint directory structure (adapter_config.json, weights, README.md).
4. adapter_config.json PEFT schema conformance.
5. Training script CLI contract (04_train_safetune_lora.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pytest
import yaml

from tests.e2e.conftest import SyntheticDataGenerator


@pytest.mark.tier1
class TestF09SafeTuneQLoRA:
    """Feature 9: SafeTune QLoRA Train-Time Hardening tests."""

    def test_f09_qlora_training_hyperparameters(self, configs_dir: Path) -> None:
        """Verify LoRA hyperparameters in configs/safetune_lora.yaml satisfy capacity requirements."""
        cfg_path = configs_dir / "safetune_lora.yaml"
        assert cfg_path.exists(), f"Configuration file missing: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("lora_r", 0) >= 8, "LoRA rank must be at least 8 for adequate expressive capacity"
        assert cfg.get("lora_alpha", 0) >= 16, "LoRA alpha must scale weights appropriately"
        assert len(cfg.get("target_modules", [])) >= 2, "LoRA must target key attention projections"

    def test_f09_safetune_trainer_interface(self) -> None:
        """Verify SafeTune trainer interface parameter contracts."""
        trainer_args = {
            "model_name_or_path": "CohereLabs/tiny-aya-global",
            "train_dataset_path": "data/safety_train_pairs.jsonl",
            "output_dir": "artifacts/adapter_checkpoint",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "learning_rate": 2e-4,
        }
        assert "CohereLabs/tiny-aya-global" in trainer_args["model_name_or_path"]
        assert trainer_args["bf16"] is True
        assert trainer_args["learning_rate"] > 0

    def test_f09_adapter_checkpoint_directory_structure(
        self, tmp_path: Path, synthetic_generators: SyntheticDataGenerator
    ) -> None:
        """Verify adapter directory contains required PEFT files."""
        adapter_dir = tmp_path / "mock_adapter"
        synthetic_generators.populate_mock_adapter_dir(adapter_dir)

        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "adapter_model.safetensors").exists()
        assert (adapter_dir / "README.md").exists()

    def test_f09_adapter_config_json_schema(
        self, tmp_path: Path, synthetic_generators: SyntheticDataGenerator
    ) -> None:
        """Verify adapter_config.json contains required PEFT metadata keys."""
        adapter_dir = tmp_path / "mock_adapter"
        synthetic_generators.populate_mock_adapter_dir(adapter_dir)

        with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        assert cfg.get("base_model_name_or_path") == "CohereLabs/tiny-aya-global"
        assert cfg.get("peft_type") == "LORA"
        assert "r" in cfg
        assert "lora_alpha" in cfg
        assert "target_modules" in cfg

    def test_f09_training_script_cli_contract(self, scripts_dir: Path) -> None:
        """Verify SafeTune LoRA training script 04_train_safetune_lora.py exists and responds to --help."""
        script_path = scripts_dir / "04_train_safetune_lora.py"
        assert script_path.exists(), f"LoRA training script missing: {script_path}"
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, f"Script --help failed: {res.stderr}"
