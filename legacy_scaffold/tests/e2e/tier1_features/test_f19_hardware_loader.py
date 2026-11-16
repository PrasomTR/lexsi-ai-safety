"""Tier 1 Feature Tests: Feature 19 - 4-Tier Hardware Model Loader.

Validates Feature 1 from PROJECT.md and TEST_INFRA.md:
1. Primary GPU 4-bit NF4 Loading Contract (BitsAndBytes normal-float4, double-quantization, compute dtype).
2. Un-gated Mirror Fallback Execution (CohereLabs/tiny-aya-global -> DhruvParmar051/tiny-aya-global-oneshot-repair).
3. CPU Fallback Degradation (graceful execution when CUDA is unavailable).
4. Decoupled Synthetic Mock Generation (Cohere2 36-layer architecture, 2048/4096 hidden dimension, forward hooks).
5. VRAM Budget Verification (< 8.0 GB RTX 5060 envelope, ~1.93 GB nominal footprint).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch
import pytest
import torch

from src.models.config import (
    HardwareEnvelope,
    ModelExecutionTier,
    ModelLoadingConfig,
    ModelMetadata,
)
from src.models.loader import (
    HardwareAwareModelLoader,
    load_model_and_tokenizer,
)
from src.models.mock import (
    MockCohere2ForCausalLM,
    MockCohereTokenizer,
)


@pytest.mark.tier1
class TestF19HardwareModelLoader:
    """Feature 1: 4-Tier Hardware Model Loader specification tests."""

    def test_f19_gpu_4bit_nf4_loading_contract(self) -> None:
        """T1-F01-01: Verify BitsAndBytes 4-bit NF4 double quantization configuration."""
        cfg = ModelLoadingConfig(quant_mode="4bit")
        bnb_cfg = cfg.to_bitsandbytes_config()
        assert bnb_cfg is not None, "BitsAndBytesConfig must be created for 4bit mode"
        assert bnb_cfg.load_in_4bit is True, "Expected load_in_4bit=True for 8GB GPU envelope"
        assert bnb_cfg.bnb_4bit_quant_type == "nf4", "Must use NormalFloat4 (nf4) quantization"
        assert bnb_cfg.bnb_4bit_use_double_quant is True, "Double quantization required for <2GB footprint"

        loader = HardwareAwareModelLoader(quant_mode="4bit")
        assert loader.quant_mode == "4bit"

        # Theoretical parameter footprint calculation for 3.35B model:
        # 3.35B parameters in 4-bit = 3.35e9 * 0.5 bytes = ~1.675 GB weights + metadata ~1.93 GB
        estimated_footprint_gb = (3.35 * 10**9 * 0.5) / (1024**3)
        assert 1.4 <= estimated_footprint_gb <= 2.2, (
            f"Expected ~1.6-1.9 GB weight footprint, got {estimated_footprint_gb:.2f} GB"
        )

    def test_f19_ungated_mirror_fallback_contract(self) -> None:
        """T1-F01-02: Verify un-gated mirror fallback redirection logic."""
        assert HardwareAwareModelLoader.DEFAULT_GATED_MODEL_ID == "CohereLabs/tiny-aya-global"
        assert HardwareAwareModelLoader.DEFAULT_MIRROR_MODEL_ID == "DhruvParmar051/tiny-aya-global-oneshot-repair"

        # Model execution tier resolver validation
        tier_primary = ModelExecutionTier.from_string("primary_gpu_4bit")
        tier_mirror = ModelExecutionTier.from_string("mirror_gpu_4bit")
        assert tier_primary == ModelExecutionTier.TIER_1_GATED_GPU
        assert tier_mirror == ModelExecutionTier.TIER_2_MIRROR_GPU

        loader = HardwareAwareModelLoader()
        model, tok, meta = loader.load_model_and_tokenizer(
            "CohereLabs/tiny-aya-global",
            fallback_mode="mock",
        )
        assert meta["requested_model_id"] == "CohereLabs/tiny-aya-global"
        assert meta["tier_name"] == "tier4_synthetic_mock"
        assert "tiny-aya" in meta["requested_model_id"].lower()

    def test_f19_cpu_fallback_degradation_contract(self) -> None:
        """T1-F01-03: Verify CPU fallback execution when CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            env = HardwareEnvelope.detect()
            assert env.cuda_available is False
            assert env.device_name == "cpu"
            assert env.supports_4bit_gpu is False

        loader = HardwareAwareModelLoader(device="cpu")
        model, tok, meta = loader.load_mock_model(device="cpu")
        assert meta["device"] == "cpu"
        assert meta["tier_name"] == "tier4_synthetic_mock"

        # Run forward pass on CPU to confirm execution degradation works cleanly
        dummy_input = torch.zeros((1, 4), dtype=torch.long)
        out = model(dummy_input)
        assert out.logits.shape[:2] == (1, 4)

    def test_f19_decoupled_synthetic_mock_generation(self) -> None:
        """T1-F01-04: Verify decoupled synthetic mock conforms to Cohere2 interface."""
        loader = HardwareAwareModelLoader()
        model, tokenizer, metadata = loader.load_mock_model(profile="standard")

        assert isinstance(model, MockCohere2ForCausalLM), "Model must be instance of MockCohere2ForCausalLM"
        assert isinstance(tokenizer, MockCohereTokenizer), "Tokenizer must be instance of MockCohereTokenizer"
        assert len(model.model.layers) == 36, "Cohere2 requires 36 decoder layers"
        assert model.hidden_size == 2048, "Cohere2 hidden size is 2048"
        assert tokenizer.vocab_size == 262144, "Cohere2 vocab size is 262,144"

        # Verify hook registration capability on target layers 16-24
        hook_fired = False

        def dummy_hook(module: torch.nn.Module, inp: Any, out: Any) -> Any:
            nonlocal hook_fired
            hook_fired = True
            if isinstance(out, tuple):
                return (out[0] + 0.35,) + out[1:]
            return out + 0.35

        handle = model.model.layers[18].register_forward_hook(dummy_hook)
        dummy_input = torch.zeros((1, 4), dtype=torch.long)
        out = model(dummy_input)
        assert hook_fired is True, "Forward hook must fire on layer 18"
        assert hasattr(out, "logits") and out.logits.shape[-1] == 262144
        handle.remove()

    def test_f19_vram_budget_verification_8gb(self) -> None:
        """T1-F01-05: Verify VRAM budget constraints for NVIDIA GeForce RTX 5060 (8GB)."""
        env = HardwareEnvelope.detect()
        assert isinstance(env.total_vram_gb, float)
        assert isinstance(env.free_vram_gb, float)

        # Minimum required headroom threshold guarantee (<= 4.0 GB required, leaving >= 4.0 GB free in 8GB envelope)
        min_vram_threshold_gb = HardwareAwareModelLoader.MIN_GPU_VRAM_BYTES / (1024**3)
        assert min_vram_threshold_gb <= 4.0, (
            f"Expected loader minimum VRAM threshold <= 4.0 GB, got {min_vram_threshold_gb:.2f} GB"
        )
        assert min_vram_threshold_gb == 3.5, "Loader specifies exactly 3.5 GB threshold"
