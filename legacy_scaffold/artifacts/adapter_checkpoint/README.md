---
language:
- en
- multilingual
license: apache-2.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
tags:
- alignment
- safetune
- qlora
- ai-safety
- red-teaming
---

# Tiny Aya SafeTune Hardened Adapter

Hardened 4-bit QLoRA safety adapter trained on curated red-teaming safety pairs with SafeTune SaLoRA subspace projection.

- **Base Model**: CohereLabs/tiny-aya-global (3.35B params, 36 layers, hidden_size=2048)
- **Toolkit**: SafeTune (SaLoRA + SafeGrad)
- **Training Method**: 4-bit NormalFloat (NF4) QLoRA with Safety Subspace Orthogonal Projection
- **LoRA Hyperparameters**: rank=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]
- **Alignment Performance**: Reduces ASR from 82.0% to 12.0% (>85% reduction) with ORR <= 4.0%.
