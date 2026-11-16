# Superseded prototype (archived, NOT part of the submission)

Everything in this directory was the first prototype. It has two defects that make it
unusable as evidence:

1. `audit/engine.py` returned **hard-coded metric constants** (`asr = 0.82`, `orr = 0.02`, ...)
   and its "model probe pass" used a mock model and then substituted a canned completion
   string, so the reported numbers were never measured.
2. The shipped LoRA adapter was never trained - its `lora_B` matrices were byte-identical
   random tensors, and the "refusal vector" was a random unit vector.

It also depends on the external `auditkit` / `curatorkit` / `safetune` packages, which are
not part of this bundle, so it cannot even be imported here.

The submission is `src/pipeline/` + `scripts/` (see the top-level README). The archived code
is kept only to document the design evolution and to show what the anti-slop scanner
(`scripts/audit_ai_slop.py`) now protects against.
