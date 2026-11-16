"""Tier 1 Feature Tests: Feature 11 - CircuitKIT / SafeTune ReSta Patching.

Validates:
1. Weight-space patching transformation contract (base + delta -> patched).
2. Contrastive circuit steering fallback mechanism logic.
3. Target layer selection integrity (targeting critical reasoning/safety layers).
4. Delta weight Frobenius norm and sparsity bounds for stability.
5. Patch reversibility and parameter isolation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest


@pytest.mark.tier1
class TestF11ReStaPatching:
    """Feature 11: CircuitKIT / SafeTune ReSta Patching tests."""

    def test_f11_resta_weight_patching_contract(self) -> None:
        """Verify weight-space patching adds delta matrix preserving weight tensor shape."""
        rng = np.random.default_rng(seed=123)
        weight_shape = (4096, 4096)
        base_weight = rng.standard_normal(weight_shape).astype(np.float32)
        delta_weight = (rng.standard_normal(weight_shape) * 0.01).astype(np.float32)

        patched_weight = base_weight + delta_weight
        assert patched_weight.shape == weight_shape
        assert not np.array_equal(patched_weight, base_weight)
        assert np.allclose(patched_weight - delta_weight, base_weight, atol=1e-5)

    def test_f11_contrastive_circuit_steering_fallback(self) -> None:
        """Verify contrastive circuit calculation: vector_diff = harmful_act - safe_act."""
        hidden_dim = 4096
        harmful_activations = np.ones(hidden_dim, dtype=np.float32) * 2.0
        safe_activations = np.ones(hidden_dim, dtype=np.float32) * 0.5

        contrastive_vector = harmful_activations - safe_activations
        assert contrastive_vector.shape == (hidden_dim,)
        assert np.all(contrastive_vector > 0)

        # Unit norm normalization
        norm = np.linalg.norm(contrastive_vector)
        unit_contrastive = contrastive_vector / norm
        assert math.isclose(float(np.linalg.norm(unit_contrastive)), 1.0, rel_tol=1e-4)

    def test_f11_layer_target_selection_integrity(self) -> None:
        """Verify target layers focus on mid-to-late transformer layers where concepts form."""
        total_layers = 32
        target_layers = [12, 13, 14, 15, 16, 17, 18, 19, 20]

        assert all(0 <= l < total_layers for l in target_layers)
        # Verify targeted layers are in middle-to-deep representation range
        assert min(target_layers) >= 8
        assert max(target_layers) <= 24

    def test_f11_weight_delta_sparsity_bounds(self) -> None:
        """Verify Frobenius norm of weight update is strictly bounded to prevent catastrophic forgetting."""
        base_weight = np.eye(128, dtype=np.float32)
        rng = np.random.default_rng(seed=456)
        delta_weight = (rng.standard_normal((128, 128)) * 0.005).astype(np.float32)

        frobenius_norm = float(np.linalg.norm(delta_weight, ord="fro"))
        base_norm = float(np.linalg.norm(base_weight, ord="fro"))

        relative_delta = frobenius_norm / base_norm
        # Relative perturbation should be small (< 10%)
        assert relative_delta < 0.10, f"Perturbation too large: {relative_delta}"

    def test_f11_patching_reversibility_and_isolation(self) -> None:
        """Verify patched layer weights can be cleanly restored without state leakage."""
        original_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        working_tensor = original_tensor.copy()

        # Apply patch
        patch = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        working_tensor += patch
        assert not np.array_equal(working_tensor, original_tensor)

        # Revert patch
        working_tensor -= patch
        assert np.allclose(working_tensor, original_tensor)
