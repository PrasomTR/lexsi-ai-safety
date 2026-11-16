"""Activation steering on the residual stream (real forward hooks, no simulation).

The steering vector is injected additively into the residual stream after the
decoder layer that carries it, which is exactly where the refusal direction was
localised by the circuit analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def get_decoder_layers(model):
    for attr in ("model.model.layers", "model.layers", "transformer.h"):
        obj = model
        ok = True
        for part in attr.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__"):
            return obj
    raise RuntimeError("Could not locate decoder layers on this model")


class ActivationSteerer:
    """Adds `alpha * vector` to the residual stream at `layer` during forward passes."""

    def __init__(self, model, layer: int, vector: np.ndarray, alpha: float = 0.0) -> None:
        self.model = model
        self.layer = int(layer)
        self.vector = np.asarray(vector, dtype=np.float32)
        self.alpha = float(alpha)
        self._handle = None
        self._layers = get_decoder_layers(model)

    def _hook(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        v = torch_tensor(self.vector, output)
        if isinstance(output, tuple):
            hidden = output[0]
            steered = hidden + self.alpha * v
            return (steered,) + tuple(output[1:])
        return output + self.alpha * v

    def attach(self) -> "ActivationSteerer":
        import torch

        self._handle = self._layers[self.layer].register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ActivationSteerer":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.remove()


def torch_tensor(vector: np.ndarray, like):
    import torch

    v = torch.from_numpy(np.asarray(vector, dtype=np.float32))
    return v.to(device=like.device, dtype=like.dtype)
