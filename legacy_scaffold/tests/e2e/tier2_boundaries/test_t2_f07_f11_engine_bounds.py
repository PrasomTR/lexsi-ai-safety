"""Lexsi AI Safety Hackathon - Track 2: AI Red Teaming, Safety and Alignment.

Tier 2 Boundary, Corner Case, and Robustness Tests for Features 07 to 11:
- F07: Baseline AuditKIT Evaluation Engine (empty batches, category distributions, timeouts)
- F08: Baseline Metric & Fingerprint Logging (ASR 0.0/1.0, NaN/Inf, SHA-256 boundaries, DEFCON)
- F09: SafeTune QLoRA Train-Time Hardening (missing configs/weights, safetensors header corruption, LoRA rank)
- F10: SafeTune Inference Activation Steering (alpha 0.0, alpha < 0, alpha > 5.0, dimension mismatch, zero-norm)
- F11: CircuitKIT / SafeTune ReSta Patching (weight patch dimension mismatch, singular covariance, norm bounds)
"""

from __future__ import annotations

import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from tests.e2e.conftest import (
    SchemaValidator,
    SyntheticDataGenerator,
    run_command_safe,
)


# ==============================================================================
# Domain 7: Feature 07 - Baseline AuditKIT Evaluation Engine Boundaries
# ==============================================================================

@pytest.mark.tier2
class TestF07AuditEngineBoundaries:
    """Boundary and negative tests for AuditKIT evaluation runner and probe handling."""

    @staticmethod
    def _run_audit_evaluation(
        probes: List[Dict[str, Any]],
        batch_size: int = 16,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Simulates audit evaluation execution with boundary guards."""
        if not isinstance(probes, list):
            raise TypeError(f"probes must be a list, got {type(probes).__name__}")
        if len(probes) == 0:
            raise ValueError("Probe batch cannot be empty: at least 1 probe required for evaluation")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > 10000:
            raise ValueError(f"batch_size {batch_size} exceeds maximum safe limit of 10000")
        if timeout <= 0.0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        # Validate probe items
        categories: Dict[str, int] = {}
        for idx, p in enumerate(probes):
            if not isinstance(p, dict):
                raise TypeError(f"Probe at index {idx} must be a dictionary")
            cat = p.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        summary = [{"category": cat, "total": cnt, "failed": 0} for cat, cnt in categories.items()]
        return {
            "asr": 0.0,
            "refusal_rate": 1.0,
            "orr": 0.0,
            "defcon_grade": "DEFCON 5",
            "metrics": {"total_evaluated": len(probes)},
            "probes_summary": summary,
        }

    def test_f07_empty_probe_batch_rejection(self) -> None:
        """Boundary: Evaluating with zero probes raises ValueError."""
        with pytest.raises(ValueError, match="Probe batch cannot be empty"):
            self._run_audit_evaluation([])

    def test_f07_probes_summary_empty_categories(self, schema_validator: SchemaValidator) -> None:
        """Boundary: Audit results where probes_summary has 0 items or missing required categories."""
        empty_summary_results = {
            "asr": 0.5,
            "refusal_rate": 0.5,
            "orr": 0.02,
            "defcon_grade": "DEFCON 3",
            "metrics": {},
            "probes_summary": [],  # Empty probes summary
        }
        valid, err = schema_validator.validate_audit_results(empty_summary_results)
        assert valid  # Schema allows empty array

        # Domain semantic check: Must contain at least jailbreak and prompt_injection categories
        def _check_probes_coverage(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
            present = {item.get("category") for item in results.get("probes_summary", [])}
            required = {"jailbreak", "prompt_injection"}
            missing = list(required - present)
            return len(missing) == 0, missing

        covered, missing = _check_probes_coverage(empty_summary_results)
        assert not covered
        assert "jailbreak" in missing and "prompt_injection" in missing

    def test_f07_batch_size_extreme_boundaries(self) -> None:
        """Boundary: batch_size = 0, negative, and excessive values are rejected."""
        valid_probes = [{"id": "P-1", "category": "jailbreak", "prompt": "test"}]
        for bad_bs in [0, -1, -64, 50000]:
            with pytest.raises(ValueError, match="batch_size"):
                self._run_audit_evaluation(valid_probes, batch_size=bad_bs)

        # batch_size = 1 is the minimal valid boundary
        res = self._run_audit_evaluation(valid_probes, batch_size=1)
        assert res["metrics"]["total_evaluated"] == 1

    def test_f07_probe_payload_malformed_type(self) -> None:
        """Boundary: Non-dict probe item inside probe list raises TypeError."""
        for malformed in ["not-a-dict", 12345, None, [1, 2, 3]]:
            with pytest.raises(TypeError, match="must be a dictionary"):
                self._run_audit_evaluation([malformed])  # type: ignore

    def test_f07_judge_classifier_score_boundaries(self) -> None:
        """Boundary: Judge classifier output clamping and validation."""
        def _sanitize_judge_score(raw_score: float) -> Tuple[bool, float]:
            if not isinstance(raw_score, (int, float)) or math.isnan(raw_score):
                return False, 0.0
            if raw_score < 0.0 or raw_score > 1.0:
                # Clamp to [0.0, 1.0] boundary
                clamped = max(0.0, min(1.0, float(raw_score)))
                return True, clamped
            return True, float(raw_score)

        assert _sanitize_judge_score(-0.25) == (True, 0.0)
        assert _sanitize_judge_score(1.5) == (True, 1.0)
        assert _sanitize_judge_score(0.0) == (True, 0.0)
        assert _sanitize_judge_score(1.0) == (True, 1.0)
        assert _sanitize_judge_score(0.75) == (True, 0.75)
        assert not _sanitize_judge_score(float("nan"))[0]

    def test_f07_evaluation_timeout_guard_boundary(self) -> None:
        """Boundary: Audit execution timeout with zero or negative timeout values."""
        valid_probes = [{"id": "P-1", "category": "jailbreak", "prompt": "test"}]
        for bad_timeout in [0.0, -5.0]:
            with pytest.raises(ValueError, match="timeout must be positive"):
                self._run_audit_evaluation(valid_probes, timeout=bad_timeout)


# ==============================================================================
# Domain 8: Feature 08 - Baseline Metric & Fingerprint Logging Boundaries
# ==============================================================================

@pytest.mark.tier2
class TestF08BaselineMetricsBoundaries:
    """Boundary and negative tests for ASR, ORR, DEFCON, and SHA-256 fingerprints."""

    def test_f08_baseline_asr_boundary_zero_and_one(self, schema_validator: SchemaValidator) -> None:
        """Boundary: ASR = 0.0 (perfect baseline) and ASR = 1.0 (total vulnerability) are valid bounds."""
        for bound_val in [0.0, 1.0]:
            data = SyntheticDataGenerator.audit_results(hardened=False, custom_metrics={"asr": bound_val})
            valid, err = schema_validator.validate_audit_results(data)
            assert valid, f"ASR boundary {bound_val} should be valid, got: {err}"

    def test_f08_baseline_asr_out_of_bounds(self, schema_validator: SchemaValidator) -> None:
        """Boundary: ASR < 0.0 and ASR > 1.0 are rejected by schema."""
        for invalid_val in [-0.001, -1.0, 1.001, 2.5]:
            data = SyntheticDataGenerator.audit_results(hardened=False, custom_metrics={"asr": invalid_val})
            valid, err = schema_validator.validate_audit_results(data)
            assert not valid, f"ASR {invalid_val} should be rejected"
            assert err is not None

    def test_f08_metrics_nan_and_infinite_values(self, schema_validator: SchemaValidator) -> None:
        """Boundary: Non-numeric, NaN, and Infinite metric values in audit outputs."""
        for bad_metric in [float("nan"), float("inf"), float("-inf")]:
            data = SyntheticDataGenerator.audit_results(hardened=False, custom_metrics={"asr": bad_metric})
            # JSON cannot serialize NaN or Inf in standard compliance
            with pytest.raises(ValueError):
                json.dumps(data, allow_nan=False)

            def _has_non_finite_floats(obj: Any) -> bool:
                if isinstance(obj, float):
                    return math.isnan(obj) or math.isinf(obj)
                if isinstance(obj, dict):
                    return any(_has_non_finite_floats(v) for v in obj.values())
                if isinstance(obj, list):
                    return any(_has_non_finite_floats(v) for v in obj)
                return False

            assert _has_non_finite_floats(data)

    def test_f08_sha256_length_boundaries(self, schema_validator: SchemaValidator) -> None:
        """Boundary: SHA-256 fingerprint length boundaries: 0, 63, 64, 65 characters."""
        valid_hash = "a" * 64
        valid, err = schema_validator.validate_sha256(valid_hash)
        assert valid, f"64-char hex string failed: {err}"

        # 63 characters (one short)
        valid, err = schema_validator.validate_sha256("a" * 63)
        assert not valid
        assert "length=63" in err

        # 65 characters (one long)
        valid, err = schema_validator.validate_sha256("a" * 65)
        assert not valid
        assert "length=65" in err

        # Empty string
        valid, err = schema_validator.validate_sha256("")
        assert not valid
        assert "length=0" in err

    def test_f08_sha256_non_hex_characters(self, schema_validator: SchemaValidator) -> None:
        """Boundary: SHA-256 string containing non-hex characters ('g', 'z', punctuation)."""
        invalid_hashes = [
            "g" + "a" * 63,                        # 'g' is non-hex
            "z" * 64,                              # 'z' is non-hex
            ("a" * 31) + "!" + ("a" * 32),         # special character
            "0123456789abcdef" * 3 + "xyz123456789ab",
        ]
        for bad_hash in invalid_hashes:
            valid, err = schema_validator.validate_sha256(bad_hash)
            assert not valid, f"Expected rejection for non-hex hash: {bad_hash}"
            assert "Invalid SHA-256 format" in err

    def test_f08_defcon_grade_boundary_formats(self, schema_validator: SchemaValidator) -> None:
        """Boundary: DEFCON grade empty string and invalid formats."""
        bad_grades = ["", None, 123]
        for grade in bad_grades:
            data = SyntheticDataGenerator.audit_results(hardened=False, custom_metrics={"defcon_grade": grade})
            valid, err = schema_validator.validate_audit_results(data)
            assert not valid, f"DEFCON grade {grade!r} should be rejected"

    def test_f08_refusal_rate_boundary_values(self, schema_validator: SchemaValidator) -> None:
        """Boundary: Refusal rate boundary conditions: 0.0, 1.0, negative, and excessive."""
        for val in [0.0, 1.0]:
            data = SyntheticDataGenerator.audit_results(hardened=False, custom_metrics={"refusal_rate": val})
            assert schema_validator.validate_audit_results(data)[0]

        for bad_val in [-0.01, 1.01]:
            data = SyntheticDataGenerator.audit_results(hardened=False, custom_metrics={"refusal_rate": bad_val})
            assert not schema_validator.validate_audit_results(data)[0]


# ==============================================================================
# Domain 9: Feature 09 - SafeTune QLoRA Train-Time Hardening Boundaries
# ==============================================================================

@pytest.mark.tier2
class TestF09SafeTuneLoRABoundaries:
    """Boundary and negative tests for SafeTune PEFT adapter checkpoints and headers."""

    @staticmethod
    def _validate_adapter_checkpoint(checkpoint_dir: Path) -> Tuple[bool, Optional[str]]:
        """Validates adapter checkpoint directory structure and files."""
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return False, f"Checkpoint directory does not exist: {checkpoint_dir}"

        config_path = checkpoint_dir / "adapter_config.json"
        if not config_path.exists():
            return False, "adapter_config.json missing"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as ex:
            return False, f"adapter_config.json is corrupted: {ex}"

        if not isinstance(cfg, dict):
            return False, "adapter_config.json root must be a dict"
        if "base_model_name_or_path" not in cfg:
            return False, "base_model_name_or_path missing in adapter_config.json"

        weights_path = checkpoint_dir / "adapter_model.safetensors"
        if not weights_path.exists():
            return False, "adapter_model.safetensors missing"

        if weights_path.stat().st_size == 0:
            return False, "adapter_model.safetensors is empty (0 bytes)"

        return True, None

    def test_f09_missing_adapter_config_file(self, tmp_path: Path) -> None:
        """Boundary: Checkpoint directory missing adapter_config.json is rejected."""
        ckpt = tmp_path / "ckpt_no_config"
        ckpt.mkdir()
        (ckpt / "adapter_model.safetensors").write_bytes(b"dummy")
        valid, err = self._validate_adapter_checkpoint(ckpt)
        assert not valid
        assert "adapter_config.json missing" in err

    def test_f09_missing_adapter_weights_file(self, tmp_path: Path) -> None:
        """Boundary: Checkpoint directory missing adapter_model.safetensors is rejected."""
        ckpt = tmp_path / "ckpt_no_weights"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text('{"base_model_name_or_path": "tiny-aya"}', encoding="utf-8")
        valid, err = self._validate_adapter_checkpoint(ckpt)
        assert not valid
        assert "adapter_model.safetensors missing" in err

    def test_f09_empty_adapter_weights_file(self, tmp_path: Path) -> None:
        """Boundary: 0-byte adapter_model.safetensors detected as corrupted checkpoint."""
        ckpt = tmp_path / "ckpt_zero_byte"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text('{"base_model_name_or_path": "tiny-aya"}', encoding="utf-8")
        (ckpt / "adapter_model.safetensors").write_bytes(b"")  # 0 bytes
        valid, err = self._validate_adapter_checkpoint(ckpt)
        assert not valid
        assert "empty (0 bytes)" in err

    def test_f09_corrupt_safetensors_header(self, tmp_path: Path) -> None:
        """Boundary: Corrupted safetensors 8-byte header or malformed header JSON."""
        def _parse_safetensors_header(file_path: Path) -> Tuple[bool, Optional[str]]:
            if not file_path.exists() or file_path.stat().st_size < 8:
                return False, "File too small to contain 8-byte header length"
            with open(file_path, "rb") as f:
                header_len_bytes = f.read(8)
                (header_len,) = struct.unpack("<Q", header_len_bytes)
                if header_len <= 0 or header_len > 100 * 1024 * 1024:  # Max 100MB header
                    return False, f"Invalid header length: {header_len}"
                header_bytes = f.read(header_len)
                if len(header_bytes) != header_len:
                    return False, "Unexpected EOF while reading safetensors header"
                try:
                    json.loads(header_bytes.decode("utf-8"))
                except Exception as ex:
                    return False, f"Safetensors header is not valid JSON: {ex}"
            return True, None

        # Create corrupted safetensors file with bogus header length
        corrupt_st = tmp_path / "corrupt.safetensors"
        # Write negative or astronomical length
        corrupt_st.write_bytes(struct.pack("<Q", 9999999999) + b"malformed")
        valid, err = _parse_safetensors_header(corrupt_st)
        assert not valid
        assert "Invalid header length" in err or "Unexpected EOF" in err

    def test_f09_lora_rank_boundaries(self) -> None:
        """Boundary: LoRA rank r = 0, negative rank, and excessive rank (> hidden_size)."""
        def _validate_lora_rank(r: int, hidden_size: int = 4096) -> Tuple[bool, Optional[str]]:
            if not isinstance(r, int) or r <= 0:
                return False, f"LoRA rank must be a positive integer, got {r}"
            if r > hidden_size:
                return False, f"LoRA rank {r} exceeds hidden size {hidden_size}"
            return True, None

        assert not _validate_lora_rank(0)[0]
        assert not _validate_lora_rank(-8)[0]
        assert not _validate_lora_rank(8192, hidden_size=4096)[0]
        assert _validate_lora_rank(16, hidden_size=4096)[0]
        assert _validate_lora_rank(64, hidden_size=4096)[0]

    def test_f09_lora_alpha_and_dropout_boundaries(self) -> None:
        """Boundary: LoRA alpha <= 0, dropout < 0, dropout >= 1.0."""
        def _validate_peft_hparams(alpha: float, dropout: float) -> Tuple[bool, Optional[str]]:
            if alpha <= 0:
                return False, f"lora_alpha must be positive, got {alpha}"
            if dropout < 0.0 or dropout >= 1.0:
                return False, f"lora_dropout must be in [0.0, 1.0), got {dropout}"
            return True, None

        assert not _validate_peft_hparams(0, 0.05)[0]
        assert not _validate_peft_hparams(-16, 0.05)[0]
        assert not _validate_peft_hparams(32, -0.01)[0]
        assert not _validate_peft_hparams(32, 1.0)[0]
        assert _validate_peft_hparams(32, 0.05)[0]
        assert _validate_peft_hparams(32, 0.0)[0]  # 0.0 dropout is valid

    def test_f09_target_modules_empty_list(self) -> None:
        """Boundary: Empty target_modules list rejected."""
        def _validate_target_modules(modules: List[str]) -> bool:
            return isinstance(modules, list) and len(modules) > 0 and all(isinstance(m, str) and m.strip() for m in modules)

        assert not _validate_target_modules([])
        assert not _validate_target_modules([""])
        assert _validate_target_modules(["q_proj", "v_proj"])


# ==============================================================================
# Domain 10: Feature 10 - SafeTune Inference Activation Steering Boundaries
# ==============================================================================

@pytest.mark.tier2
class TestF10ActivationSteeringBoundaries:
    """Boundary and negative tests for refusal direction vectors and alpha steering factor."""

    @staticmethod
    def _validate_steering_config(cfg: Dict[str, Any], hidden_size: int = 4096) -> Tuple[bool, Optional[str]]:
        """Validates runtime activation steering hyperparameters and direction vector."""
        if not isinstance(cfg, dict):
            return False, "Config must be a dict"

        alpha = cfg.get("alpha")
        if alpha is None or not isinstance(alpha, (int, float)):
            return False, f"alpha must be numeric, got {type(alpha).__name__}"
        if alpha < 0.0:
            return False, f"Inverted steering rejected: alpha {alpha} < 0.0"
        if alpha > 5.0:
            return False, f"Extreme steering factor clamped/rejected: alpha {alpha} > 5.0"

        direction = cfg.get("refusal_direction")
        if direction is not None:
            if not isinstance(direction, list):
                return False, "refusal_direction must be a list"
            if len(direction) != hidden_size:
                return False, f"Dimensionality mismatch: vector size {len(direction)} != hidden size {hidden_size}"
            # Check norm
            norm = math.sqrt(sum(x * x for x in direction))
            if norm == 0.0:
                return False, "Refusal direction vector has zero norm (cannot normalize)"

        layer = cfg.get("layer_index", 16)
        if not isinstance(layer, int) or layer < 0 or layer >= 32:
            return False, f"layer_index {layer} out of bounds (0-31)"

        return True, None

    def test_f10_alpha_identity_boundary_zero(self) -> None:
        """Boundary: Steering alpha factor α = 0.0 is valid identity transformation."""
        cfg = {"alpha": 0.0, "refusal_direction": [1.0] * 4096, "layer_index": 16}
        valid, err = self._validate_steering_config(cfg)
        assert valid, f"Alpha 0.0 should be valid identity steering: {err}"

    def test_f10_alpha_negative_inverted_steering(self) -> None:
        """Boundary: Steering alpha factor α < 0 (inverted steering) is strictly rejected."""
        for neg_alpha in [-0.01, -0.35, -5.0]:
            cfg = {"alpha": neg_alpha, "refusal_direction": [1.0] * 4096}
            valid, err = self._validate_steering_config(cfg)
            assert not valid
            assert "Inverted steering rejected" in err

    def test_f10_alpha_extreme_clamp_boundary(self) -> None:
        """Boundary: Steering alpha factor α > 5.0 triggers clamp/rejection guard."""
        for extreme_alpha in [5.01, 10.0, 100.0]:
            cfg = {"alpha": extreme_alpha, "refusal_direction": [1.0] * 4096}
            valid, err = self._validate_steering_config(cfg)
            assert not valid
            assert "Extreme steering factor" in err

        # Exactly 5.0 is the upper allowable boundary
        valid, err = self._validate_steering_config({"alpha": 5.0, "refusal_direction": [1.0] * 4096})
        assert valid, f"Alpha 5.0 upper boundary rejected: {err}"

    def test_f10_alpha_non_numeric_types(self) -> None:
        """Boundary: Steering alpha passed as string, None, list, or dict."""
        for bad_alpha in ["0.35", "high", None, [0.35], {"alpha": 0.35}]:
            cfg = {"alpha": bad_alpha, "refusal_direction": [1.0] * 4096}
            valid, err = self._validate_steering_config(cfg)
            assert not valid
            assert "alpha must be numeric" in err

    def test_f10_vector_dimension_mismatch(self) -> None:
        """Boundary: Steering vector dimension != hidden size (2048, 3072 vs 4096)."""
        for bad_dim in [1024, 2048, 3072, 5120]:
            cfg = {"alpha": 0.35, "refusal_direction": [0.1] * bad_dim}
            valid, err = self._validate_steering_config(cfg, hidden_size=4096)
            assert not valid
            assert "Dimensionality mismatch" in err
            assert f"vector size {bad_dim} != hidden size 4096" in err

    def test_f10_zero_norm_steering_vector(self) -> None:
        """Boundary: All-zero refusal direction vector (norm = 0.0) rejected for division by zero."""
        cfg = {"alpha": 0.35, "refusal_direction": [0.0] * 4096}
        valid, err = self._validate_steering_config(cfg, hidden_size=4096)
        assert not valid
        assert "zero norm" in err

    def test_f10_steering_layer_index_boundary(self) -> None:
        """Boundary: Steering applied to layer < 0 or >= 32."""
        for bad_layer in [-1, -16, 32, 64]:
            cfg = {"alpha": 0.35, "refusal_direction": [1.0] * 4096, "layer_index": bad_layer}
            valid, err = self._validate_steering_config(cfg)
            assert not valid
            assert "layer_index" in err and "out of bounds" in err


# ==============================================================================
# Domain 11: Feature 11 - CircuitKIT / SafeTune ReSta Patching Boundaries
# ==============================================================================

@pytest.mark.tier2
class TestF11ReStaPatchingBoundaries:
    """Boundary and robustness tests for weight patching and contrastive circuit fallback."""

    def test_f11_weight_patch_dimension_mismatch(self) -> None:
        """Boundary: Weight patch tensor shape mismatch against projection matrix."""
        def _apply_weight_patch(
            base_shape: Tuple[int, int],
            patch_shape: Tuple[int, int],
        ) -> Tuple[bool, Optional[str]]:
            if base_shape != patch_shape:
                return False, f"Patch shape {patch_shape} does not match target weight shape {base_shape}"
            return True, None

        base = (4096, 4096)
        for mismatched in [(2048, 4096), (4096, 2048), (4096, 4096, 1)]:
            valid, err = _apply_weight_patch(base, mismatched)  # type: ignore
            assert not valid
            assert "does not match" in err

        assert _apply_weight_patch((4096, 4096), (4096, 4096))[0]

    def test_f11_missing_target_submodule_fallback(self) -> None:
        """Boundary: Patching non-existent submodule triggers fallback rather than crashing."""
        known_modules = {
            "model.layers.15.self_attn.o_proj",
            "model.layers.16.self_attn.o_proj",
            "model.layers.17.self_attn.o_proj",
        }

        def _resolve_patch_target(target_name: str, fallback_enabled: bool = True) -> Tuple[str, bool]:
            if target_name in known_modules:
                return target_name, False
            if fallback_enabled:
                # Fallback to nearest default refusal layer
                return "model.layers.16.self_attn.o_proj", True
            raise KeyError(f"Target module '{target_name}' not found")

        # Non-existent submodule with fallback
        resolved, used_fallback = _resolve_patch_target("model.layers.99.invalid_proj", fallback_enabled=True)
        assert used_fallback
        assert resolved == "model.layers.16.self_attn.o_proj"

        # Without fallback raises KeyError
        with pytest.raises(KeyError, match="not found"):
            _resolve_patch_target("model.layers.99.invalid_proj", fallback_enabled=False)

    def test_f11_singular_covariance_matrix_guard(self) -> None:
        """Boundary: Singular / non-invertible covariance matrix handled with ridge regularization."""
        def _compute_whitening_matrix(eigenvalues: List[float], eps: float = 1e-5) -> List[float]:
            # Guard against zero or near-zero eigenvalues (singularity)
            regularized = [1.0 / math.sqrt(max(val, eps)) for val in eigenvalues]
            return regularized

        # Matrix with 0 eigenvalues
        singular_eigenvals = [4.0, 1.0, 0.0, 0.0]
        whitened = _compute_whitening_matrix(singular_eigenvals, eps=1e-5)
        assert len(whitened) == 4
        assert not any(math.isinf(x) or math.isnan(x) for x in whitened)
        assert whitened[2] == 1.0 / math.sqrt(1e-5)

    def test_f11_identity_patch_zero_delta(self) -> None:
        """Boundary: Patch delta of all zeros produces exact identity transformation."""
        base_weights = [1.0, 2.0, 3.0, 4.0]
        zero_delta = [0.0, 0.0, 0.0, 0.0]

        patched = [w + d for w, d in zip(base_weights, zero_delta)]
        assert patched == base_weights

    def test_f11_extreme_patch_frobenius_norm_threshold(self) -> None:
        """Boundary: Patch delta norm exceeding max allowed relative norm (e.g. > 15%) rejected."""
        def _check_patch_safety_norm(
            base_norm: float,
            patch_norm: float,
            max_rel_threshold: float = 0.15,
        ) -> Tuple[bool, float]:
            if base_norm <= 0.0:
                return False, 0.0
            rel_norm = patch_norm / base_norm
            return (rel_norm <= max_rel_threshold, rel_norm)

        # 5% perturbation -> Safe
        safe, rel = _check_patch_safety_norm(100.0, 5.0)
        assert safe and rel == 0.05

        # Exactly 15% -> Safe boundary
        safe, rel = _check_patch_safety_norm(100.0, 15.0)
        assert safe and rel == 0.15

        # 16% perturbation -> Exceeds safety boundary
        safe, rel = _check_patch_safety_norm(100.0, 16.0)
        assert not safe and rel == 0.16
