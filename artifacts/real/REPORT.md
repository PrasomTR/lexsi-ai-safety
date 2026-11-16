# Real Safety Pipeline - Measured Results

Model: `CohereLabs/tiny-aya-global` (3.35B, 36 layers, d=2048, GQA 16:4)
Generated: 2026-11-16T03:59:12Z  |  wall time 978s

## Baseline vs hardened (measured from generations)

| Metric | Baseline | Hardened |
|---|---|---|
| Attack Success Rate | 0.4412 | 0.0 |
| Refusal rate | 0.4853 | 0.0 |
| Over-refusal (benign) | 0.0 | 0.0 |
| GSM8K accuracy | 0.0 | 0.0 |
| DEFCON | DEFCON 3 | DEFCON 5 |
| probes (attack/benign/utility) | 68/26/10 | 68/26/10 |

## Per-category ASR

| category | baseline | hardened |
|---|---|---|
| encoding_bypass | 0.6364 | 0.0 |
| harmful_content | 0.3636 | 0.0 |
| jailbreak | 0.3636 | 0.0 |
| multilingual_exploit | 0.3846 | 0.0 |
| prompt_injection | 0.6364 | 0.0 |
| system_leakage | 0.2727 | 0.0 |

## Per-language ASR

| language | baseline | hardened |
|---|---|---|
| ar | 0.5714 | 0.0 |
| de | 1.0 | 0.0 |
| en | 0.5217 | 0.0 |
| es | 0.1667 | 0.0 |
| fr | 0.2 | 0.0 |
| hi | 0.1429 | 0.0 |
| ru | 0.625 | 0.0 |
| zh | 0.1429 | 0.0 |

## Mechanism

- steering layer: **23** (rank-1 probe F1 0.7869, separation 72.8133)
- refusal direction norm: 1.0
- cross-lingual direction alignment vs EN: {'ar': 0.3392, 'de': 0.5805, 'en': 1.0, 'es': 0.6379, 'fr': 0.6439, 'hi': 0.5017, 'ru': 0.6505, 'zh': 0.6145}
- EN-trained probe transfer accuracy by language: {'ar': 0.5556, 'de': 0.8571, 'en': 1.0, 'es': 0.75, 'fr': 0.8571, 'hi': 0.6667, 'ru': 0.4, 'zh': 0.7778}

## Safety/utility Pareto (steering alpha)

| alpha | ASR | ORR | n_attack | n_benign |
|---|---|---|---|---|
| 0.0 | 0.75 | 0.0 | 12 | 12 |
| 0.05 | 0.4167 | 0.0833 | 12 | 12 |
| 0.1 | 0.3333 | 0.0 | 12 | 12 |
| 0.2 | 0.8333 | 0.0 | 12 | 12 |
| 0.4 | 0.0833 | 0.0 | 12 | 12 |
| 0.8 | 0.0 | 0.0 | 12 | 12 |

## Reproduce

```bash
python scripts/run_real_pipeline.py --model CohereLabs/tiny-aya-global
python scripts/verify_submission.py   # recomputes every number from the transcripts
```
