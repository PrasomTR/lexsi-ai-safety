"""Second mitigation arm: safety SFT (LoRA) on refusal pairs, measured the same way.

Compares two families of defence on identical probes:
  A) activation steering  - one vector, no gradients, no weights change
  B) safety SFT (LoRA)    - 20 contrast pairs, rank-8 adapter

    python scripts/train_lora_arm.py --model CohereLabs/tiny-aya-global
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.audit import Auditor, Probe, load_benign, load_redteam  # noqa: E402
from src.pipeline.model import ModelRunner  # noqa: E402
from src.pipeline.refusal import RefusalDetector  # noqa: E402

OUT = ROOT / "artifacts" / "real"


def load_pairs(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_examples(tokenizer, pairs, max_len=160):
    """Prompt -> refusal (chosen) examples; the rejected completion is the negative."""
    out = []
    for r in pairs:
        prompt = r["prompt"]
        refusal = r.get("chosen") or r.get("response") or ""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": refusal}],
            tokenize=False, add_generation_prompt=False)
        ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
        if len(ids) < 8:
            continue
        out.append({"input_ids": ids, "labels": list(ids)})
    return out


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(mask))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--benign-limit", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(4)
    t0 = time.time()
    runner = ModelRunner(args.model, dtype="float32", batch_size=args.batch_size,
                         max_new_tokens=args.max_new_tokens).load()
    tok = runner.tokenizer

    cfg = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(runner.model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[1/4] LoRA r={args.rank}: {trainable} trainable / {total} params "
          f"({100 * trainable / total:.3f}%)", flush=True)

    pairs = load_pairs(ROOT / "data" / "safety_train_pairs.jsonl")
    examples = build_examples(tok, pairs)
    print(f"[2/4] {len(examples)} training examples from {len(pairs)} contrast pairs", flush=True)

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pad_id = tok.pad_token_id or tok.eos_token_id
    losses = []
    for epoch in range(args.epochs):
        # deterministic shuffling so the run is reproducible
        order = np.arange(len(examples))
        np.random.default_rng(epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            idx = list(order[start : start + args.batch_size])
            batch = [examples[i] for i in idx]
            input_ids, labels, mask = collate(batch, pad_id)
            out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(out.loss.item()))
        print(f"      epoch {epoch + 1}/{args.epochs} loss={np.mean(losses[-len(order):]):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    adapter_dir = OUT / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    model.eval()

    print("[3/4] measuring the fine-tuned model on the same probes", flush=True)
    probes = load_redteam(ROOT / "data" / "redteam_benchmark.jsonl")[: args.per_category * 6]
    probes = [p for i, p in enumerate(probes) if i % 6 < args.per_category]
    probes += load_benign(ROOT / "data" / "benign_retention.jsonl")[: args.benign_limit]
    auditor = Auditor(runner, RefusalDetector(), max_new_tokens=args.max_new_tokens,
                      batch_size=args.batch_size)
    records = auditor.run(probes, progress_every=0)
    summary = Auditor.summarize(records, model_id=args.model, stage="safetuned")
    summary["adapter"] = str(adapter_dir.relative_to(ROOT))
    summary["train_loss_first_last"] = [round(losses[0], 4), round(losses[-1], 4)]
    summary["train_seconds"] = round(time.time() - t0, 1)
    (OUT / "safetuned_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "safetuned_transcripts.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[4/4] ASR={summary['asr']} ORR={summary['orr']} defcon={summary['defcon']['label']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
