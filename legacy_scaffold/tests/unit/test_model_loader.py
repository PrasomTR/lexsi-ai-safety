"""Unit tests for Hardware-Aware Model Execution Engine and Fallback Hierarchy.

Verifies:
- Hardware detection envelope and RTX 5060 capabilities
- ModelExecutionTier enum parsing and properties
- ModelLoadingConfig and BitsAndBytesConfig generation
- MockCohere2ForCausalLM architecture, forward hooks, generation, and dual-mode profiles
- MockCohereTokenizer chat templates, encode/decode, and batch encoding
- HardwareAwareModelLoader 4-tier cascading and error trapping
- Environment variable overrides and metadata contract completeness
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from src.models.config import (
    HardwareEnvelope,
    ModelExecutionTier,
    ModelLoadingConfig,
    ModelMetadata,
)
from src.models.mock import (
    MockCohere2ForCausalLM,
    MockCohereTokenizer,
    MockCohere2Config,
)
from src.models.loader import (
    HardwareAwareModelLoader,
    load_model_and_tokenizer,
)


# ==============================================================================
# 1. Hardware Envelope & Detection Tests
# ==============================================================================

class TestHardwareEnvelope:
    """Tests hardware envelope detection and serialization."""

    def test_hardware_detection_returns_envelope(self) -> None:
        env = HardwareEnvelope.detect()
        assert isinstance(env, HardwareEnvelope)
        assert isinstance(env.cuda_available, bool)
        assert isinstance(env.total_vram_gb, float)
        assert isinstance(env.free_vram_gb, float)
        assert isinstance(env.bitsandbytes_available, bool)
        assert isinstance(env.is_rtx_5060, bool)
        assert isinstance(env.supports_4bit_gpu, bool)

    def test_hardware_envelope_to_dict(self) -> None:
        env = HardwareEnvelope.detect()
        d = env.to_dict()
        assert isinstance(d, dict)
        required_keys = [
            "cuda_available", "device_name", "device_index",
            "compute_capability", "total_vram_bytes", "total_vram_gb",
            "free_vram_bytes", "free_vram_gb", "bitsandbytes_available",
            "bfloat16_supported", "total_ram_gb", "available_ram_gb",
            "is_rtx_5060", "supports_4bit_gpu",
        ]
        for key in required_keys:
            assert key in d, f"Missing key '{key}' in HardwareEnvelope.to_dict()"

    def test_envelope_simulated_cpu_only(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            env = HardwareEnvelope.detect()
            assert env.cuda_available is False
            assert env.device_name == "cpu"
            assert env.supports_4bit_gpu is False
            assert env.total_vram_gb == 0.0


# ==============================================================================
# 2. ModelExecutionTier Enum Tests
# ==============================================================================

class TestModelExecutionTier:
    """Tests ModelExecutionTier enum mappings and aliases."""

    def test_tier_enum_values(self) -> None:
        assert ModelExecutionTier.TIER_1_GATED_GPU == 1
        assert ModelExecutionTier.TIER_2_MIRROR_GPU == 2
        assert ModelExecutionTier.TIER_3_CPU_FALLBACK == 3
        assert ModelExecutionTier.TIER_4_SYNTHETIC_MOCK == 4

    def test_tier_names(self) -> None:
        assert ModelExecutionTier.TIER_1_GATED_GPU.tier_name == "tier1_gated_gpu"
        assert ModelExecutionTier.TIER_2_MIRROR_GPU.tier_name == "tier2_mirror_gpu"
        assert ModelExecutionTier.TIER_3_CPU_FALLBACK.tier_name == "tier3_cpu_fallback"
        assert ModelExecutionTier.TIER_4_SYNTHETIC_MOCK.tier_name == "tier4_synthetic_mock"

    @pytest.mark.parametrize("input_str,expected", [
        ("tier1", ModelExecutionTier.TIER_1_GATED_GPU),
        ("tier_1", ModelExecutionTier.TIER_1_GATED_GPU),
        ("gated", ModelExecutionTier.TIER_1_GATED_GPU),
        ("1", ModelExecutionTier.TIER_1_GATED_GPU),
        ("tier2", ModelExecutionTier.TIER_2_MIRROR_GPU),
        ("mirror", ModelExecutionTier.TIER_2_MIRROR_GPU),
        ("2", ModelExecutionTier.TIER_2_MIRROR_GPU),
        ("tier3", ModelExecutionTier.TIER_3_CPU_FALLBACK),
        ("cpu", ModelExecutionTier.TIER_3_CPU_FALLBACK),
        ("3", ModelExecutionTier.TIER_3_CPU_FALLBACK),
        ("tier4", ModelExecutionTier.TIER_4_SYNTHETIC_MOCK),
        ("mock", ModelExecutionTier.TIER_4_SYNTHETIC_MOCK),
        ("synthetic", ModelExecutionTier.TIER_4_SYNTHETIC_MOCK),
        ("4", ModelExecutionTier.TIER_4_SYNTHETIC_MOCK),
    ])
    def test_from_string_valid_aliases(self, input_str: str, expected: ModelExecutionTier) -> None:
        assert ModelExecutionTier.from_string(input_str) == expected

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model execution tier"):
            ModelExecutionTier.from_string("invalid_tier")


# ==============================================================================
# 3. ModelLoadingConfig & Quantization Tests
# ==============================================================================

class TestModelLoadingConfig:
    """Tests quantization config and BitsAndBytes parameters."""

    def test_config_defaults(self) -> None:
        cfg = ModelLoadingConfig()
        assert cfg.quant_mode == "4bit"
        assert cfg.quant_type == "nf4"
        assert cfg.double_quant is True
        assert cfg.low_cpu_mem_usage is True
        assert cfg.device_map == "auto"

    def test_to_bitsandbytes_config_construction(self) -> None:
        cfg = ModelLoadingConfig(quant_mode="4bit", quant_type="nf4", double_quant=True)
        bnb_cfg = cfg.to_bitsandbytes_config()
        if bnb_cfg is not None:
            assert getattr(bnb_cfg, "load_in_4bit", None) is True
            assert getattr(bnb_cfg, "bnb_4bit_quant_type", None) == "nf4"
            assert getattr(bnb_cfg, "bnb_4bit_use_double_quant", None) is True

    def test_non_4bit_mode_returns_none(self) -> None:
        cfg = ModelLoadingConfig(quant_mode="none")
        assert cfg.to_bitsandbytes_config() is None

    def test_get_torch_dtype(self) -> None:
        cfg_bf16 = ModelLoadingConfig(compute_dtype=None, torch_dtype="bfloat16")
        assert cfg_bf16.get_torch_dtype() == torch.bfloat16

        cfg_fp16 = ModelLoadingConfig(compute_dtype=None, torch_dtype="float16")
        assert cfg_fp16.get_torch_dtype() == torch.float16

        cfg_fp32 = ModelLoadingConfig(compute_dtype=None, torch_dtype="float32")
        assert cfg_fp32.get_torch_dtype() == torch.float32


# ==============================================================================
# 4. Tier 4 Synthetic Mock Model Tests
# ==============================================================================

class TestMockCohere2Model:
    """Tests MockCohere2ForCausalLM structural compliance and behavior."""

    def test_model_layer_structure_standard(self) -> None:
        model = MockCohere2ForCausalLM(profile="standard")
        assert len(model.model.layers) == 36
        assert model.hidden_size == 2048
        assert model.vocab_size == 262144

        # Verify each layer exposes attention, MLP, and layernorm
        layer0 = model.model.layers[0]
        assert hasattr(layer0, "self_attn")
        assert hasattr(layer0, "mlp")
        assert hasattr(layer0, "input_layernorm")

        # Verify attention projection modules
        assert hasattr(layer0.self_attn, "q_proj")
        assert hasattr(layer0.self_attn, "k_proj")
        assert hasattr(layer0.self_attn, "v_proj")
        assert hasattr(layer0.self_attn, "o_proj")

        # Verify MLP projection modules
        assert hasattr(layer0.mlp, "gate_proj")
        assert hasattr(layer0.mlp, "up_proj")
        assert hasattr(layer0.mlp, "down_proj")

    def test_model_layers_alias_property(self) -> None:
        model = MockCohere2ForCausalLM()
        # safetune and circuitkit require model.model.layers or model.layers
        assert model.layers is model.model.layers
        assert len(model.layers) == 36

    def test_forward_hook_registration_and_execution(self) -> None:
        model = MockCohere2ForCausalLM()
        hook_fired = []

        def steer_hook(module: Any, input_tensor: Any, output_tensor: Any) -> Any:
            hook_fired.append(True)
            # Add small perturbation to activation tensor
            return output_tensor + 0.5

        # Register hook on layer 18 (inside steering range 16-24)
        handle = model.model.layers[18].register_forward_hook(steer_hook)
        dummy_input = torch.tensor([[5, 120, 255001]])

        out = model(dummy_input)
        assert len(hook_fired) == 1, "Registered forward hook did not execute during model forward pass!"
        handle.remove()

    def test_forward_output_types_and_shapes(self) -> None:
        model = MockCohere2ForCausalLM(hidden_size=2048, vocab_size=262144)
        input_ids = torch.tensor([[5, 42, 100, 255001]])

        # Forward pass without labels
        out = model(input_ids)
        assert out.loss is None
        assert out.logits.shape == torch.Size([1, 4, 262144])
        assert len(out.hidden_states) >= 1

        # Forward pass with labels (calculates mock loss)
        labels = torch.tensor([[5, 42, 100, 255001]])
        out_with_loss = model(input_ids, labels=labels)
        assert out_with_loss.loss is not None
        assert out_with_loss.loss.item() > 0

    def test_generate_method(self) -> None:
        model = MockCohere2ForCausalLM()
        input_ids = torch.tensor([[5, 100]])
        gen = model.generate(input_ids, max_new_tokens=10)
        assert gen.shape == torch.Size([1, 12])
        # Verify prompt tokens preserved at head of sequence
        assert gen[0, 0].item() == 5
        assert gen[0, 1].item() == 100

    def test_safe_device_movement(self) -> None:
        model = MockCohere2ForCausalLM()
        # Moving to CPU always works
        model.to("cpu")
        assert model.device.type == "cpu"

        # Moving to CUDA when simulated unavailable falls back to CPU cleanly
        with patch("torch.cuda.is_available", return_value=False):
            model.to("cuda:0")
            assert model.device.type == "cpu"

    def test_forward_hook_modifies_output_logits(self) -> None:
        """Verifies that forward hooks modifying residual activations measurably change output logits."""
        model = MockCohere2ForCausalLM()
        inputs = torch.tensor([[5, 10, 20]])

        # Baseline logits before hook
        base_logits = model(inputs).logits.clone()

        # Steering hook on layer 18 adding perturbation
        def steer_hook(m: Any, i: Any, o: Any) -> Any:
            return o + 1.0

        handle = model.model.layers[18].register_forward_hook(steer_hook)
        steered_logits = model(inputs).logits.clone()
        handle.remove()

        # Logits must reflect the hidden state modification
        diff = (steered_logits - base_logits).abs().sum().item()
        assert diff > 0, "Logits did not reflect forward hook modification!"

        # Removing hook restores unsteered logits
        restored_logits = model(inputs).logits.clone()
        restore_diff = (restored_logits - base_logits).abs().sum().item()
        assert restore_diff == 0.0, "Removing hook did not restore baseline logits!"

    def test_safe_device_parsing_invalid_strings(self) -> None:
        """Verifies invalid or unknown device strings do not crash model.to()."""
        model = MockCohere2ForCausalLM()
        # Invalid device strings should be caught and handled gracefully
        model.to("invalid_xyz")
        assert model.device.type in ("cpu", "cuda")
        model.to("custom_npu:0")
        assert model.device.type in ("cpu", "cuda")
        model.to(device="unknown_accel:42")
        assert model.device.type in ("cpu", "cuda")

    def test_dual_mode_legacy_mock_profile(self) -> None:
        model = MockCohere2ForCausalLM(profile="legacy-mock")
        assert model.hidden_size == 4096
        assert model.vocab_size == 256000
        assert model.num_layers == 32
        assert len(model.model.layers) == 32

    def test_config_duck_typing(self) -> None:
        model = MockCohere2ForCausalLM(hidden_size=2048, vocab_size=262144, num_layers=36)
        cfg = model.config
        # Attribute access
        assert cfg.hidden_size == 2048
        assert cfg.vocab_size == 262144
        assert cfg.num_hidden_layers == 36
        assert "Cohere2ForCausalLM" in cfg.architectures

        # Dict access
        assert cfg["hidden_size"] == 2048
        assert cfg.get("hidden_size") == 2048
        assert cfg.get("model_type") == "cohere2"


# ==============================================================================
# 5. Tier 4 Synthetic Mock Tokenizer Tests
# ==============================================================================

class TestMockCohereTokenizer:
    """Tests MockCohereTokenizer chat template, encoding, and decoding."""

    def test_chat_template_cohere_turn_tokens(self) -> None:
        tokenizer = MockCohereTokenizer()
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "How do I make a safe password?"},
            {"role": "assistant", "content": "Use numbers, letters, and symbols."},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful AI assistant.<|END_OF_TURN_TOKEN|>" in prompt
        assert "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>How do I make a safe password?<|END_OF_TURN_TOKEN|>" in prompt
        assert "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Use numbers, letters, and symbols.<|END_OF_TURN_TOKEN|>" in prompt
        assert prompt.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")

    def test_encode_and_decode(self) -> None:
        tokenizer = MockCohereTokenizer()
        text = "Hello world from Tiny Aya safety evaluation"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert isinstance(tokens, list)
        assert tokens[0] == tokenizer.bos_token_id
        assert tokens[-1] == tokenizer.eos_token_id

        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_batch_encode_plus(self) -> None:
        tokenizer = MockCohereTokenizer()
        batch = [
            "Short prompt",
            "This is a longer prompt that requires padding to align across tensors",
        ]
        res = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True)
        assert "input_ids" in res
        assert "attention_mask" in res
        input_ids = res["input_ids"]
        mask = res["attention_mask"]
        assert input_ids.shape[0] == 2
        assert mask.shape == input_ids.shape
        # Shorter sequence has padding token id at tail
        assert input_ids[0, -1].item() == tokenizer.pad_token_id
        assert mask[0, -1].item() == 0

    def test_callable_tokenizer(self) -> None:
        tokenizer = MockCohereTokenizer()
        out = tokenizer("Testing direct call interface")
        assert "input_ids" in out
        assert "attention_mask" in out

    def test_small_and_boundary_vocabularies(self) -> None:
        """Verifies tokenizer handles small and boundary vocabularies without ZeroDivisionError or negative IDs."""
        for v in [1000, 500, 100, 10, 2]:
            tok = MockCohereTokenizer(vocab_size=v)
            encoded = tok.encode("hello world safety evaluation test")
            assert isinstance(encoded, list)
            assert len(encoded) > 0
            assert all(isinstance(t, int) for t in encoded)
            assert all(0 <= t < v for t in encoded), f"Token ID out of bounds [0, {v-1}] for vocab_size={v}: {encoded}"


# ==============================================================================
# 6. HardwareAwareModelLoader 4-Tier Fallback Tests
# ==============================================================================

class TestHardwareAwareModelLoader:
    """Tests loader 4-tier hierarchy, cascading, and exception handling."""

    def test_direct_tier4_mock_loading(self) -> None:
        loader = HardwareAwareModelLoader()
        model, tokenizer, metadata = loader.load_model_and_tokenizer(fallback_mode="tier4")

        assert isinstance(model, MockCohere2ForCausalLM)
        assert isinstance(tokenizer, MockCohereTokenizer)
        assert metadata["tier_loaded"] == 4
        assert metadata["tier"] == "tier4"
        assert metadata["tier_name"] == "tier4_synthetic_mock"
        assert metadata["hidden_size"] == 2048
        assert metadata["vocab_size"] == 262144
        assert metadata["num_layers"] == 36

    def test_tier1_gated_failure_cascades_to_tier2(self) -> None:
        """Simulates 401 GatedRepoError on Tier 1; verifies Tier 2 mirror succeeds."""
        loader = HardwareAwareModelLoader()

        mock_tier2_model = MockCohere2ForCausalLM()
        mock_tier2_tok = MockCohereTokenizer()
        mock_tier2_meta = {
            "tier": "tier2",
            "tier_loaded": 2,
            "tier_name": "tier2_mirror_gpu",
            "model_id": "DhruvParmar051/tiny-aya-global-oneshot-repair",
            "requested_model_id": "CohereLabs/tiny-aya-global",
            "device": "cuda:0",
            "quant_mode": "4bit",
            "quantization": "4bit_nf4",
            "torch_dtype": "bfloat16",
            "vram_allocated_gb": 1.93,
            "vram_total_gb": 7.96,
            "fallback_occurred": True,
            "fallback_trace": [
                "Tier 1 (CohereLabs/tiny-aya-global) failed: OSError: 401 Client Error: Access restricted.",
                "Tier 2 mirror loaded successfully with 4-bit NF4.",
            ],
            "fallback_chain": [],
            "fallback_reason": "Tier 1 (CohereLabs/tiny-aya-global) failed: OSError: 401 Client Error: Access restricted.",
            "hidden_size": 2048,
            "num_layers": 36,
            "vocab_size": 262144,
        }

        with patch.object(loader, "_load_tier1_gpu", side_effect=OSError("401 Client Error (gated repository)")):
            with patch.object(loader, "_load_tier2_gpu", return_value=(mock_tier2_model, mock_tier2_tok, mock_tier2_meta)):
                model, tok, meta = loader.load_model_and_tokenizer(fallback_mode="auto")
                assert meta["tier_loaded"] == 2
                assert meta["tier"] == "tier2"
                assert "401 Client Error" in meta["fallback_reason"]

    def test_tier1_and_tier2_failure_cascades_to_tier3(self) -> None:
        """Simulates GPU OOM on Tier 2; verifies Tier 3 CPU succeeds."""
        loader = HardwareAwareModelLoader()

        mock_cpu_model = MockCohere2ForCausalLM()
        mock_cpu_tok = MockCohereTokenizer()
        mock_cpu_meta = {
            "tier": "tier3",
            "tier_loaded": 3,
            "tier_name": "tier3_cpu_fallback",
            "model_id": "DhruvParmar051/tiny-aya-global-oneshot-repair",
            "requested_model_id": "CohereLabs/tiny-aya-global",
            "device": "cpu",
            "quant_mode": "bfloat16",
            "quantization": "bfloat16",
            "torch_dtype": "bfloat16",
            "vram_allocated_gb": 0.0,
            "vram_total_gb": 7.96,
            "fallback_occurred": True,
            "fallback_trace": [
                "Tier 1 failed: 401 Client Error",
                "Tier 2 failed: CUDA OOM",
                "Tier 3 CPU loaded successfully.",
            ],
            "fallback_chain": [],
            "fallback_reason": "Tier 2 failed: CUDA OOM",
            "hidden_size": 2048,
            "num_layers": 36,
            "vocab_size": 262144,
        }

        with patch.object(loader, "_load_tier1_gpu", side_effect=OSError("401 Gated")):
            with patch.object(loader, "_load_tier2_gpu", side_effect=RuntimeError("CUDA out of memory")):
                with patch.object(loader, "_load_tier3_cpu", return_value=(mock_cpu_model, mock_cpu_tok, mock_cpu_meta)):
                    model, tok, meta = loader.load_model_and_tokenizer(fallback_mode="auto")
                    assert meta["tier_loaded"] == 3
                    assert meta["device"] == "cpu"

    def test_full_cascade_to_tier4(self) -> None:
        """Simulates failures across Tiers 1, 2, and 3; verifies Tier 4 always succeeds."""
        loader = HardwareAwareModelLoader()

        with patch.object(loader, "_load_tier1_gpu", side_effect=OSError("401 Gated")):
            with patch.object(loader, "_load_tier2_gpu", side_effect=RuntimeError("CUDA OOM")):
                with patch.object(loader, "_load_tier3_cpu", side_effect=MemoryError("CPU RAM exhausted")):
                    model, tok, meta = loader.load_model_and_tokenizer(fallback_mode="auto")
                    assert meta["tier_loaded"] == 4
                    assert meta["tier"] == "tier4"
                    assert meta["fallback_occurred"] is True
                    assert len(meta["fallback_trace"]) >= 3

    def test_explicit_tier1_failure_raises_exception(self) -> None:
        """When explicit tier1 is selected, it should fail immediately without cascading."""
        loader = HardwareAwareModelLoader()
        with patch.object(loader, "_load_tier1_gpu", side_effect=OSError("401 Gated")):
            with pytest.raises(RuntimeError, match="Explicit Tier 1 loading failed"):
                loader.load_model_and_tokenizer(fallback_mode="tier1")

    def test_explicit_tier2_failure_raises_exception(self) -> None:
        loader = HardwareAwareModelLoader()
        with patch.object(loader, "_load_tier2_gpu", side_effect=RuntimeError("CUDA OOM")):
            with pytest.raises(RuntimeError, match="Explicit Tier 2 loading failed"):
                loader.load_model_and_tokenizer(fallback_mode="tier2")

    def test_env_var_force_mock_override(self) -> None:
        loader = HardwareAwareModelLoader()
        with patch.dict(os.environ, {"FORCE_MOCK_MODEL": "1"}):
            model, tok, meta = loader.load_model_and_tokenizer(fallback_mode="auto")
            assert meta["tier_loaded"] == 4
            assert meta["tier"] == "tier4"

    def test_env_var_model_execution_tier_override(self) -> None:
        loader = HardwareAwareModelLoader()
        with patch.dict(os.environ, {"MODEL_EXECUTION_TIER": "tier4"}):
            model, tok, meta = loader.load_model_and_tokenizer(fallback_mode="auto")
            assert meta["tier_loaded"] == 4

    def test_metadata_contract_completeness(self) -> None:
        loader = HardwareAwareModelLoader()
        model, tok, meta = loader.load_model_and_tokenizer(fallback_mode="tier4")

        required_metadata_keys = [
            "tier", "tier_loaded", "tier_name", "model_id",
            "requested_model_id", "device", "quant_mode", "quantization",
            "torch_dtype", "vram_allocated_gb", "vram_total_gb",
            "fallback_occurred", "fallback_trace", "fallback_chain",
            "fallback_reason", "hidden_size", "num_layers", "vocab_size",
        ]
        for key in required_metadata_keys:
            assert key in meta, f"Missing required metadata key: '{key}'"

    def test_standalone_load_function(self) -> None:
        model, tok, meta = load_model_and_tokenizer(fallback_mode="tier4")
        assert isinstance(model, MockCohere2ForCausalLM)
        assert isinstance(tok, MockCohereTokenizer)
        assert meta["tier_loaded"] == 4

    def test_classmethod_invocation_patterns(self) -> None:
        """Verifies load_model_and_tokenizer works as classmethod, instance method, and standalone."""
        # 1. Classmethod invocation with keyword args
        m1, t1, meta1 = HardwareAwareModelLoader.load_model_and_tokenizer(fallback_mode="tier4")
        assert meta1["tier"] == "tier4"
        assert isinstance(m1, MockCohere2ForCausalLM)
        assert isinstance(t1, MockCohereTokenizer)

        # 2. Classmethod invocation with positional model id
        m2, t2, meta2 = HardwareAwareModelLoader.load_model_and_tokenizer(
            "CohereLabs/tiny-aya-global", fallback_mode="tier4"
        )
        assert meta2["tier"] == "tier4"

        # 3. Instance invocation
        loader = HardwareAwareModelLoader()
        m3, t3, meta3 = loader.load_model_and_tokenizer(fallback_mode="tier4")
        assert meta3["tier"] == "tier4"

    def test_kwargs_passthrough_no_collision(self) -> None:
        """Verifies passing profile, dimensions, and device kwargs does not cause TypeError."""
        loader = HardwareAwareModelLoader()
        # Explicit Tier 4
        m, t, meta = loader.load_model_and_tokenizer(
            fallback_mode="tier4",
            profile="legacy-mock",
            hidden_size=4096,
            num_layers=32,
            vocab_size=256000,
            device="cpu",
        )
        assert m.hidden_size == 4096
        assert m.num_layers == 32
        assert t.vocab_size == 256000
        assert meta["tier"] == "tier4"

        # Classmethod invocation with profile kwarg
        m_cls, t_cls, meta_cls = HardwareAwareModelLoader.load_model_and_tokenizer(
            fallback_mode="tier4",
            profile="legacy-mock",
        )
        assert m_cls.hidden_size == 4096

    def test_invalid_fallback_mode_raises_value_error(self) -> None:
        """Verifies invalid tier names fail fast with clean ValueError rather than cascading to Tier 1 network."""
        loader = HardwareAwareModelLoader()
        with pytest.raises(ValueError, match="Invalid fallback_mode 'invalid_tier_mode'"):
            loader.load_model_and_tokenizer(fallback_mode="invalid_tier_mode")

        with pytest.raises(ValueError, match="Invalid fallback_mode 'nonexistent_tier'"):
            HardwareAwareModelLoader.load_model_and_tokenizer(fallback_mode="nonexistent_tier")

        with patch.dict(os.environ, {"MODEL_EXECUTION_TIER": "bad_env_tier"}):
            with pytest.raises(ValueError, match="Invalid MODEL_EXECUTION_TIER 'bad_env_tier'"):
                loader.load_model_and_tokenizer(fallback_mode="auto")
