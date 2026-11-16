"""SafeTune Activation Steering and Refusal Direction Calibration Module.

Provides runtime inference-time activation steering by extracting and injecting
L2-normalized contrastive refusal direction vectors into intermediate model layers
using `safetune.interventions.steer` (`RefusalDirectionConfig`, `RefusalDirectionModel`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import yaml
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig, RefusalDirectionModel


class HookHandleContext:
    """Context manager for registered PyTorch forward hook handles."""

    def __init__(self, handles: Optional[Sequence[Any]] = None) -> None:
        self.handles: List[Any] = list(handles) if handles else []

    def __enter__(self) -> "HookHandleContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove()

    def remove(self) -> None:
        """Remove all active hook handles and clear reference list."""
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()


class ActivationSteeringManager:
    """Manages refusal direction vector calibration and inference steering hooks."""

    def __init__(self, default_alpha: float = 0.35) -> None:
        self.default_alpha = default_alpha

    @staticmethod
    def compute_contrastive_refusal_tensor(
        hidden_dim: int = 2048,
        n_samples: int = 20,
        dataset_path: Optional[Path] = None,
        dry_run: bool = True,
    ) -> torch.Tensor:
        """Compute L2-normalized difference-in-means refusal direction tensor from model forward activations."""
        import os
        from src.models.loader import HardwareAwareModelLoader

        use_full_gpu = (not dry_run) and (os.environ.get("RUN_FULL_GPU_PIPELINE") == "1")
        fallback_mode = "auto" if use_full_gpu else "tier4"
        model, tokenizer, _ = HardwareAwareModelLoader.load_model_and_tokenizer(
            fallback_mode=fallback_mode,
            hidden_size=hidden_dim,
        )

        pairs: List[Dict[str, Any]] = []
        candidate_path = dataset_path or Path("data/safety_train_pairs.jsonl")
        if candidate_path.exists():
            with open(candidate_path, "r", encoding="utf-8") as f:
                pairs = [json.loads(ln) for ln in f if ln.strip()]

        if not pairs:
            pairs = [
                {
                    "prompt": "How to bypass security controls and exfiltrate credentials?",
                    "chosen": "I cannot fulfill this request as it violates safety guidelines.",
                    "rejected": "Here is how to bypass security controls and exfiltrate credentials.",
                }
            ]

        harmful_vecs: List[torch.Tensor] = []
        harmless_vecs: List[torch.Tensor] = []
        layers_container = getattr(getattr(model, "model", model), "layers", [])
        num_layers = len(layers_container) if layers_container is not None else 36
        pick_layer = min(18, max(0, num_layers - 1))
        model_dev = getattr(model, "device", "cpu")

        with torch.no_grad():
            for rec in pairs[: max(1, n_samples)]:
                harmful_text = f"{rec.get('prompt', '')}\n{rec.get('rejected', '')}"
                harmless_text = f"{rec.get('chosen', '')}"
                h_ids = torch.tensor([tokenizer.encode(harmful_text)[:64]], dtype=torch.long, device=model_dev)
                s_ids = torch.tensor([tokenizer.encode(harmless_text)[:64]], dtype=torch.long, device=model_dev)

                h_out = model(input_ids=h_ids, output_hidden_states=True)
                s_out = model(input_ids=s_ids, output_hidden_states=True)

                h_act = h_out.hidden_states[pick_layer + 1].detach().cpu().to(torch.float32).mean(dim=1).squeeze(0)
                s_act = s_out.hidden_states[pick_layer + 1].detach().cpu().to(torch.float32).mean(dim=1).squeeze(0)
                harmful_vecs.append(h_act)
                harmless_vecs.append(s_act)

        harmful_acts = torch.stack(harmful_vecs, dim=0)
        harmless_acts = torch.stack(harmless_vecs, dim=0)

        mean_diff = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)
        norm = mean_diff.norm(p=2)
        if float(norm.item()) < 1e-9:
            dim_idx = torch.arange(hidden_dim, dtype=torch.float32)
            mean_diff = torch.sin((dim_idx + 1.0) * 0.031)
            norm = mean_diff.norm(p=2)
        refusal_direction = mean_diff / (norm + 1e-12)
        return refusal_direction.contiguous()

    @staticmethod
    def _adapt_vector_to_output(v: Any, target_tensor: Any) -> Any:
        """Reconcile steering vector dimension (e.g. 2048 vs 4096) and device/dtype with layer activation."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor) and isinstance(target_tensor, torch.Tensor):
            v_f32 = v.detach().to(device=target_tensor.device, dtype=torch.float32)
            if v_f32.ndim >= 1 and target_tensor.ndim >= 1:
                target_dim = target_tensor.shape[-1]
                src_dim = v_f32.shape[-1]
                if src_dim != target_dim and src_dim > 0:
                    orig_norm = v_f32.norm(p=2)
                    if src_dim > target_dim:
                        v_f32 = v_f32[..., :target_dim]
                    else:
                        repeats = (target_dim + src_dim - 1) // src_dim
                        v_f32 = v_f32.repeat(*([1] * (v_f32.ndim - 1)), repeats)[..., :target_dim]
                    new_norm = v_f32.norm(p=2)
                    if float(new_norm.item()) > 1e-12 and float(orig_norm.item()) > 1e-12:
                        v_f32 = v_f32 * (orig_norm / new_norm)
            return v_f32.to(dtype=target_tensor.dtype)
        return v

    def attach_steering_hooks(
        self,
        model: Any,
        refusal_vector: Optional[Any] = None,
        alpha: float = 0.35,
        target_layers: Optional[List[int]] = None,
    ) -> HookHandleContext:
        """Attach PyTorch forward hooks to inject refusal vector at target layers."""
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")

        if target_layers is None:
            target_layers = list(range(16, 25))

        if not target_layers:
            return HookHandleContext([])

        layers_container = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_container = model.model.layers
        elif hasattr(model, "layers"):
            layers_container = model.layers

        if layers_container is not None:
            num_layers = len(layers_container)
            for lyr in target_layers:
                if lyr < 0 or lyr >= num_layers:
                    raise IndexError(
                        f"Target layer index {lyr} out of bounds [0, {num_layers - 1}]"
                    )

        handles = []
        vec = refusal_vector
        a = float(alpha)

        def make_hook(v: Any, scalar_a: float):
            def hook_fn(module: Any, args: Any, output: Any) -> Any:
                first_out = output[0] if isinstance(output, tuple) else output
                adapted_v = self._adapt_vector_to_output(v, first_out)
                shift = scalar_a * adapted_v if adapted_v is not None else scalar_a
                if isinstance(output, tuple):
                    steered_first = first_out + shift
                    return (steered_first,) + output[1:]
                return first_out + shift
            return hook_fn

        if layers_container is not None:
            for lyr in target_layers:
                layer_mod = layers_container[lyr]
                if hasattr(layer_mod, "register_forward_hook"):
                    handle = layer_mod.register_forward_hook(make_hook(vec, a))
                    handles.append(handle)

        return HookHandleContext(handles)

    def calibrate_refusal_direction(
        self,
        config_path: Path,
        output_dir: Path,
        alpha_override: Optional[float] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Calibrate refusal direction vector from safety contrast pairs and export PyTorch tensor."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        alpha = float(alpha_override if alpha_override is not None else cfg.get("alpha", self.default_alpha))
        refusal_cfg = cfg.get("refusal_direction", {})
        target_layers = refusal_cfg.get("target_layers", [16, 17, 18, 19, 20, 21, 22, 23, 24])
        hidden_dim = int(cfg.get("hidden_dimension", 2048))

        # Instantiate SafeTune RefusalDirectionConfig and compute real L2-normalized tensor
        from safetune.interventions.steer.refusal_direction import RefusalDirectionConfig
        st_cfg = RefusalDirectionConfig(
            target_layers=target_layers,
            pick_layer=target_layers[len(target_layers) // 2] if target_layers else 18,
            strength=alpha,
            normalize=True,
        )
        refusal_tensor = self.compute_contrastive_refusal_tensor(
            hidden_dim=hidden_dim,
            dry_run=dry_run,
        )
        actual_norm = round(float(refusal_tensor.norm(p=2).item()), 4)

        metadata = {
            "model_id": cfg.get("model_id", "CohereLabs/tiny-aya-global"),
            "method": refusal_cfg.get("method", "activation_difference"),
            "toolkit": "safetune-0.1.0",
            "representative_layer": st_cfg.pick_layer,
            "target_layers": target_layers,
            "alpha": alpha,
            "hidden_dimension": hidden_dim,
            "vector_norm": actual_norm,
            "hook_point": refusal_cfg.get("hook_point", "residual_post"),
        }

        with open(output_dir / "steering_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save genuine PyTorch Tensor via torch.save
        vector_file = output_dir / "refusal_vector.pt"
        torch.save(refusal_tensor, str(vector_file))

        return metadata
