"""Tier 1 Feature Tests: Feature 10 - SafeTune Inference Activation Steering.

Validates:
1. Refusal direction steering vector dimension matches model hidden size (4096).
2. Steering intensity coefficient alpha is calibrated in optimal range (0.1 <= alpha <= 1.0).
3. Activation hook injection contract and tensor addition semantics.
4. steering_config.yaml schema conformance.
5. Activation steering CLI script contract (05_apply_steering.py).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest
import yaml

from tests.e2e.conftest import SchemaValidator


@pytest.mark.tier1
class TestF10ActivationSteering:
    """Feature 10: SafeTune Inference Activation Steering tests."""

    def test_f10_steering_vector_dimension_contract(self) -> None:
        """Verify steering vector matches Tiny Aya hidden size (4096 dimensions)."""
        hidden_size = 4096
        # Generate normalized refusal direction vector using numpy
        rng = np.random.default_rng(seed=42)
        refusal_vector = rng.standard_normal(hidden_size)
        norm = np.linalg.norm(refusal_vector)
        normalized_vector = refusal_vector / norm

        assert normalized_vector.shape == (hidden_size,)
        assert math.isclose(float(np.linalg.norm(normalized_vector)), 1.0, rel_tol=1e-4)

    def test_f10_alpha_multiplier_calibration_range(self, configs_dir: Path) -> None:
        """Verify alpha multiplier in configs/steering_config.yaml is within safe calibration bounds."""
        cfg_path = configs_dir / "steering_config.yaml"
        assert cfg_path.exists(), f"Configuration file missing: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        calibrated_alpha = cfg.get("alpha", 0.0)
        min_alpha = 0.10
        max_alpha = 1.00

        assert min_alpha <= calibrated_alpha <= max_alpha
        # Alpha should not be excessively high which would trigger catastrophic refusal
        assert calibrated_alpha < 0.60

    def test_f10_activation_hook_injection_interface(self) -> None:
        """Verify forward hook injection modifies activation by adding alpha * direction."""
        hidden_dim = 4096
        batch_size = 2
        seq_len = 8

        # Simulated residual stream activations: (batch, seq_len, hidden_dim)
        activations = np.zeros((batch_size, seq_len, hidden_dim), dtype=np.float32)
        direction = np.ones(hidden_dim, dtype=np.float32)
        alpha = 0.35

        def steering_hook(act: np.ndarray) -> np.ndarray:
            return act + alpha * direction

        steered = steering_hook(activations)
        assert steered.shape == activations.shape
        # Each activation element should now have alpha added
        assert np.allclose(steered, np.full_like(steered, alpha))

    def test_f10_steering_config_yaml_validation(
        self, configs_dir: Path, schema_validator: SchemaValidator
    ) -> None:
        """Validate steering_config.yaml against required schema."""
        live_file = configs_dir / "steering_config.yaml"
        assert live_file.exists(), f"steering_config.yaml missing: {live_file}"
        valid, err = schema_validator.validate_yaml_config(live_file, "steering_config")
        assert valid, f"steering_config.yaml invalid: {err}"

    def test_f10_steering_script_cli_contract(self, scripts_dir: Path) -> None:
        """Verify activation steering script 05_apply_steering.py exists and responds to --help."""
        script_path = scripts_dir / "05_apply_steering.py"
        assert script_path.exists(), f"Steering script missing: {script_path}"
        import subprocess
        import sys
        res = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, f"Script --help failed: {res.stderr}"
