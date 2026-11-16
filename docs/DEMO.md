# 3-minute demo (no server, no GPU, no internet)

Everything below runs offline on a 4-core laptop CPU. Total wall time ~10 minutes.

## Beat 1 (30 s) - the claim

> "Most safety reports are numbers in a README. Ours are executable: every number we publish
> can be recomputed from the raw transcripts we ship, in one command, in under a second."

```bash
python scripts/verify_submission.py
```

It re-classifies every completion with an independent detector, recomputes ASR / ORR /
utility per category and per language, compares them against the published summary and
prints `ALL CHECKS PASSED`. If a single number does not reproduce, it exits non-zero.

## Beat 2 (30 s) - the anti-slop guard

```bash
python scripts/audit_ai_slop.py
```

Shows that the shipped code contains no metric-shaped literals, no fabricated completion
strings, no mock models, and that every published summary has a transcript file next to
it. (The first prototype of this project failed this scan - that is why it exists.)

## Beat 3 (60 s) - the mechanism

```bash
python -c "import json;m=json.load(open('artifacts/real/mechanism.json'));\
print('steering layer', m['steering_layer']);\
print('layer sweep', [(s['layer'], s['f1']) for s in m['layer_sweep']]);\
print('direction alignment vs EN', m['direction_alignment_vs_en']);\
print('probe transfer', m['probe_transfer'].get('transfer_accuracy_by_language'))"
```

Narrative: refusal is *one* direction in the residual stream. It shows up at a specific
layer, it points the same way in eight languages, and a probe trained only on English
transfers to the other seven. So the fix is not "fine-tune harder" - it is "find the
direction and move it".

## Beat 4 (60 s) - the fix and the trade-off

```bash
python -c "import json;p=json.load(open('artifacts/real/pareto.json'));\
[print(x) for x in p]"
```

One knob (steering strength), two measured quantities (attack success, over-refusal).
The shipped operating point is in `artifacts/real/steering_config.json`.

## Beat 5 (60 s) - re-measure it yourself

```bash
python scripts/run_real_pipeline.py --model CohereLabs/tiny-aya-global
```

Full pipeline on CPU: baseline audit -> layer localisation -> direction extraction ->
alpha sweep -> hardened audit -> report. No server, no API key, no GPU.

## Backup questions

- *"Is this just keyword matching?"* - The detector is deliberately conservative and
  multilingual; the ASR counts only completions that both avoid refusal markers **and**
  carry substantive content. Truncated or empty completions are counted as neither.
- *"How do you know the safety improvement is not over-refusal?"* - Every configuration is
  reported as a pair (ASR, over-refusal on benign probes), and the operating point is
  chosen under an explicit 5% over-refusal budget.
- *"Why a 0.5B model?"* - Because the point of the submission is that the *method* is
  cheap enough to run anywhere. The same command runs on
  `CohereLabs/tiny-aya-global` on a GPU without code changes.
