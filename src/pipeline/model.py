"""Real model runner: transformers generation + activation capture on CPU.

No mock, no synthetic completions. Every metric in the shipped pipeline is derived
from completions produced by this module.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np


class ModelRunner:
    """Thin, explicit wrapper around a causal LM (HF transformers)."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "auto",
        max_new_tokens: int = 48,
        batch_size: int = 4,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.model = None
        self.tokenizer = None
        self.meta: Dict[str, object] = {}

    # ------------------------------------------------------------------ loading
    def load(self) -> "ModelRunner":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = None
        if self.dtype_name == "float32":
            torch_dtype = torch.float32
        elif self.dtype_name == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.dtype_name == "float16":
            torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        self.model.eval()
        self.model.to(self.device)

        cfg = self.model.config
        self.meta = {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "hidden_size": int(getattr(cfg, "hidden_size", 0)),
            "num_layers": int(getattr(cfg, "num_hidden_layers", 0)),
            "vocab_size": int(getattr(cfg, "vocab_size", 0)),
        }
        return self

    # --------------------------------------------------------------- prompting
    def format_prompt(self, user_text: str) -> str:
        """Apply the model's own chat template; fall back to a plain prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        prefix = (self.system_prompt + "\n\n") if self.system_prompt else ""
        return f"{prefix}User: {user_text}\nAssistant:"

    # -------------------------------------------------------------- generation
    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Greedy-decode a list of raw prompts; returns decoded completions."""
        import torch

        if self.model is None:
            raise RuntimeError("ModelRunner.load() must be called before generate()")
        max_new = max_new_tokens or self.max_new_tokens
        bs = batch_size or self.batch_size
        outs: List[str] = []
        for i in range(0, len(prompts), bs):
            chunk = list(prompts[i : i + bs])
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for row in range(gen.shape[0]):
                new_tokens = gen[row, enc["input_ids"].shape[1] :]
                outs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outs

    def chat(self, user_texts: Sequence[str], max_new_tokens: Optional[int] = None,
             batch_size: Optional[int] = None) -> List[str]:
        return self.generate([self.format_prompt(t) for t in user_texts], max_new_tokens, batch_size)

    # ------------------------------------------------------- activation capture
    def last_hidden_layers(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None,
                           batch_size: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Last-token residual-stream activations for *all* layers in a single forward pass."""
        import torch

        if self.model is None:
            raise RuntimeError("ModelRunner.load() must be called before last_hidden_layers()")
        bs = batch_size or self.batch_size
        out: Dict[int, List[np.ndarray]] = {}
        for i in range(0, len(prompts), bs):
            chunk = list(prompts[i : i + bs])
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                # logits_to_keep=1 -> we only need the last position's logits, otherwise the
                # LM head materialises (batch x seq x vocab) floats and OOMs on CPU.
                res = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
            for layer, hs in enumerate(res.hidden_states):
                if layers is not None and layer not in layers:
                    continue
                out.setdefault(layer, []).append(
                    hs[:, -1, :].detach().to(torch.float32).cpu().numpy()
                )
        return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in out.items()}

    def last_hidden(self, prompts: Sequence[str], layer: int,
                    batch_size: Optional[int] = None) -> np.ndarray:
        """Last-token residual-stream activations at `layer` (numpy, float32)."""
        import torch

        if self.model is None:
            raise RuntimeError("ModelRunner.load() must be called before last_hidden()")
        bs = batch_size or self.batch_size
        feats: List[np.ndarray] = []
        for i in range(0, len(prompts), bs):
            chunk = list(prompts[i : i + bs])
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
            hs = out.hidden_states[layer]
            last = hs[:, -1, :].detach().to(torch.float32).cpu().numpy()
            feats.append(last)
        return np.concatenate(feats, axis=0).astype(np.float32)

    # --------------------------------------------------------------- utilities
    def layers(self) -> int:
        return int(self.meta.get("num_layers", 0))

    def hidden_size(self) -> int:
        return int(self.meta.get("hidden_size", 0))


def timed(label: str, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    print(f"[{label}] {time.time() - t0:.1f}s", flush=True)
    return out
