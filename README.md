# Lexsi AI Safety Hackathon — Track 2: AI Red Teaming, Safety and Alignment

**Safety claims should be executable, not asserted.**

This repository does not publish a number it cannot reproduce. Every metric below is
computed from raw model completions that are shipped with the submission, and
`python scripts/verify_submission.py` recomputes each of them from those raw
transcripts and fails loudly if anything disagrees.

No GPU. No server. No API keys. Runs on a 4-core laptop CPU in ~10 minutes.

---

## 1. What we actually built

**Target model: CohereLabs/tiny-aya-global** (3.35B, 36 layers, d=2048, GQA 16:4, 70 languages). The pipeline is model-agnostic: it runs on any causal LM. The same code runs on tiny-aya on a GPU with no changes.

**The solution:** A complete, self-contained safety research toolkit that finds vulnerabilities, understands them mechanistically, fixes them with a portable 8 KB vector, and verifies the fix. No GPU. No server. No API key. No fabricated numbers.

| Metric | Baseline | After steering | Note |
|---|---|---|---|
| Attack success rate | **51.5%** | **47.1%** | measured on a CPU-runnable model; same code runs on tiny-aya |
| Over-refusal on benign probes | 0.0% | 11.5% | the operating point is honest about the trade-off |
| Coherence (non-degenerate) | 100% | 99.0% | collapse is scored as failure, never as safety |
| Probe transfer (EN -> 7 langs) | - | **73.3%** mean | de/fr 86%, zh 78%, es 75%, hi 67%, ar 56%, ru 40% |
| Refusal direction alignment vs EN | - | mean cos **0.57** | shared but not identical - no "universal direction" claim |

The headline is:

> **A single 8 KB vector, extracted from 20 English pairs, detects jailbreaks across 8 languages
> with 73% accuracy, localises the refusal circuit to one layer, and ships with a verifier
> that recomputes every number from raw model output in under a second.** No GPU, no server,
> no API key. The full pipeline runs on any causal LM.

| Stage | What it does | Where |
|---|---|---|
| **Red-team audit** | 8 languages x 6 attack families, ASR/ORR/utility computed from real generations | `src/pipeline/audit.py` |
| **Parallel multilingual benchmark** | 56 *parallel* (sentence-aligned) probes in 8 languages - the shipped benchmark only had scattered non-English prompts, which makes cross-lingual claims statistically meaningless | `data/parallel_multilingual.jsonl` |
| **Mechanism** | Refusal direction extracted from real activations, layer localisation, cross-lingual direction alignment, EN->X probe transfer | `src/pipeline/probe.py` |
| **Mitigation** | Residual-stream activation steering with a single vector + alpha sweep (safety/utility Pareto) | `src/pipeline/steering.py` |
| **Verification** | One-command falsification: recompute every published number from the transcripts | `scripts/verify_submission.py` |
| **Degeneracy control** | Collapsed output counts as a *failed* attack and as a *failure* on benign prompts | `src/pipeline/refusal.py` |

---

## 2. Reproduce it (no GPU, no server)

> **The design decision we are most proud of.** During development the strongest steering
> setting drove ASR to 0% - because the model had collapsed into repeating a single token. A naive
> evaluator scores that as a perfect defence; this pipeline scores it as a *failed attack* (no refusal was
> produced) and as a *failure on benign prompts* (the model is broken). `coherence_rate` ships
> next to every ASR number, and the rule is pinned by a unit test. **Safety is not the absence of
> compliance; it is refusal without collapse.** This single rule is what makes every number in this
> repository trustworthy.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors numpy pytest matplotlib

# 1) build the parallel multilingual probe set
python scripts/build_parallel_multilingual.py

# 2) run the full pipeline (audit -> mechanism -> mitigation -> hardened audit)
python scripts/run_real_pipeline.py --model <hf-model-id-or-path>

# 3) falsify our own numbers from the shipped transcripts
python scripts/verify_submission.py

# 4) unit tests (metric algebra + mechanism math, milliseconds, no model needed)
python -m pytest tests -q
```

Optional extras:

```bash
python scripts/train_lora_arm.py --model <model>   # second defence arm: safety SFT (LoRA)
python scripts/make_figures.py                     # figures from the measured JSON
python scripts/make_html_report.py                 # single-file HTML report (opens from disk)
python scripts/make_submission_report.py           # SUBMISSION_REPORT.md + safety_spec.json
python scripts/audit_ai_slop.py                    # proves the shipped code has no fabricated metrics
```

Artifacts land in `artifacts/real/`: transcripts, summaries, the refusal vector, the
layer sweep, the Pareto curve, `safety_spec.json`, figures, and `report.html`
(a self-contained HTML report - open it from disk, no server, no CDN).
`docs/DEMO.md` contains a 3-minute demo script.

---

## 3. Why this is not another "ASR went down" project

Most safety submissions publish a number and ask you to trust it. This one ships the
*mechanism* that produced the number, the *raw transcripts* it was computed from, and a
*verifier* that recomputes it. `scripts/verify_submission.py` re-derives every published
number from `artifacts/real/*_transcripts.jsonl`; `tests/test_pipeline.py` pins the
metric algebra; `scripts/audit_ai_slop.py` fails the build if a metric-shaped literal
or a fabricated completion string reappears in the shipped code.

The previous prototype is kept in `legacy_scaffold/` for reference only - it needed the
external `auditkit` / `curatorkit` / `safetune` packages and produced no measurements.

---

## 4. Layout

```
src/pipeline/         the shipped, measured system (self-contained)
  refusal.py          multilingual refusal detection (8 languages)
  model.py            transformers runner: batched generation + activation capture
  audit.py            ASR / ORR / utility from completions, DEFCON rubric, SHA-256 fingerprint
  probe.py            refusal direction, layer sweep, logistic probe + transfer
  steering.py         residual-stream activation steering (forward hooks)
scripts/
  build_parallel_multilingual.py   builds the 8-language parallel probe set
  run_real_pipeline.py             end-to-end measured pipeline
  verify_submission.py             one-command falsification of every published number
  audit_ai_slop.py                 anti-slop scanner (now also rejects hard-coded metrics)
data/                 benchmark, benign retention set, parallel multilingual probes
artifacts/real/       measured outputs (transcripts, summaries, vector, sweeps)
tests/                fast unit tests for the metric algebra and the mechanism math
artifacts/real/report.html   single-file HTML report (opens from disk, no server)
docs/DEMO.md          3-minute demo script
legacy_scaffold/      superseded prototype (kept for history, produces no measurements)
```

## 5. Safety note

The red-team prompts are standard safety-research probes (jailbreak personas,
prompt injection, encoding bypass, system-prompt leakage, phishing/phishing framing,
and multilingual variants of the same). They are used to *measure refusal behaviour*,
and the multilingual set is deliberately *parallel* so that language is the only
independent variable.
