"""End-to-end REAL safety pipeline: audit -> mechanism -> mitigation -> verification.

Every number in the report is computed from model completions captured in this run.
Nothing is hard-coded. CPU-only, no server, no API keys.

    python scripts/run_real_pipeline.py --model CohereLabs/tiny-aya-global

Outputs (artifacts/real/):
    baseline_transcripts.jsonl / baseline_summary.json
    hardened_transcripts.jsonl / hardened_summary.json
    refusal_vector.npy / steering_config.json
    mechanism.json          (layer sweep, direction alignment, probe transfer)
    pareto.json             (alpha sweep: ASR / ORR trade-off)
    REPORT.md               (numbers + how to reproduce)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.audit import Auditor, Probe, defcon_grade, load_benign, load_redteam  # noqa: E402
from src.pipeline.model import ModelRunner  # noqa: E402
from src.pipeline.probe import (  # noqa: E402
    cosine,
    direction_alignment,
    layer_sweep,
    logistic_probe,
    mean_difference_direction,
    probe_accuracy,
)
from src.pipeline.refusal import RefusalDetector, detect_language  # noqa: E402
from src.pipeline.steering import ActivationSteerer  # noqa: E402

OUT = ROOT / "artifacts" / "real"


# --------------------------------------------------------------------- datasets
def load_parallel() -> List[Probe]:
    path = ROOT / "data" / "parallel_multilingual.jsonl"
    rows: List[Probe] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(Probe(
            id=r["id"], category=r["category"], language=r.get("language", detect_language(r["prompt"])),
            prompt=r["prompt"], kind=r.get("kind", "attack"), reference=r.get("reference"),
        ))
    return rows


def balanced_subset(probes: List[Probe], per_category: int) -> List[Probe]:
    """Keep at most `per_category` probes per (category, language) pair.

    Sampling per (category, language) instead of per category matters: a per-category
    cut would silently drop whole languages from the parallel set.
    """
    out: List[Probe] = []
    seen: Dict[tuple, int] = {}
    for p in probes:
        key = (p.category, p.language)
        c = seen.get(key, 0)
        if c < per_category:
            out.append(p)
            seen[key] = c + 1
    return out


def save_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--benign-limit", type=int, default=20)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--alphas", default="0,0.05,0.1,0.2,0.4,0.8")
    ap.add_argument("--calibration-alpha", type=float, default=0.2)
    ap.add_argument("--max-orr", type=float, default=0.05)
    ap.add_argument("--sweep-probes", type=int, default=16)
    ap.add_argument("--parallel-per-category", type=int, default=3)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--reuse-baseline", action="store_true")
    ap.add_argument("--reuse-mechanism", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------- model + data
    print("[1/7] loading model", flush=True)
    runner = ModelRunner(args.model, dtype=args.dtype, batch_size=args.batch_size,
                         max_new_tokens=args.max_new_tokens).load()
    print(f"      {runner.meta}", flush=True)

    redteam = balanced_subset(load_redteam(ROOT / "data" / "redteam_benchmark.jsonl"), args.per_category)
    benign = load_benign(ROOT / "data" / "benign_retention.jsonl")[: args.benign_limit]
    parallel = balanced_subset(load_parallel(), args.parallel_per_category)
    probes = redteam + benign + parallel
    print(f"      probes: {len(redteam)} attack + {len(benign)} benign + {len(parallel)} parallel", flush=True)

    detector = RefusalDetector()
    auditor = Auditor(runner, detector, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    # ------------------------------------------------------------ 1. baseline
    cached = out / "baseline_transcripts.jsonl"
    if args.reuse_baseline and cached.exists():
        print("[2/7] reusing cached baseline transcripts", flush=True)
        base_records = [json.loads(l) for l in cached.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in base_records:  # re-classify with the current detector (degeneracy-aware)
            r["verdict"] = detector.classify(r.get("completion"), r.get("language", "en"))
    else:
        print("[2/7] baseline audit (real generations)", flush=True)
        base_records = auditor.run(probes)
        save_jsonl(out / "baseline_transcripts.jsonl", base_records)
    base_summary = Auditor.summarize(base_records, model_id=args.model, stage="baseline")
    save_json(out / "baseline_summary.json", base_summary)
    print(f"      ASR={base_summary['asr']}  refusal={base_summary['refusal_rate']}  ORR={base_summary['orr']}",
          flush=True)

    # ---------------------------------------------------- 2. mechanism: layers
    mech_cache = out / "mechanism.json"
    if args.reuse_mechanism and mech_cache.exists() and (out / "refusal_vector.npy").exists():
        mechanism = json.loads(mech_cache.read_text(encoding="utf-8"))
        best = {"layer": mechanism["steering_layer"], "f1": max(s["f1"] for s in mechanism["layer_sweep"])}
        v = np.load(out / "refusal_direction_unit.npy")
        v_scaled = np.load(out / "refusal_vector.npy")
        resid_norm = mechanism.get("residual_norm_at_layer", 1.0)
        print(f"[3/7] reusing cached mechanism (layer={best['layer']}, f1={best['f1']})", flush=True)
    else:
      print("[3/7] activation capture + layer localisation", flush=True)
      label_by_id = {r["id"]: (1 if r["verdict"] == "refusal" else 0) for r in base_records}
      act_ids = [r["id"] for r in base_records if r["kind"] == "attack"]
      prompts = [r["prompt"] for r in base_records if r["kind"] == "attack"]
      labels = np.array([label_by_id[i] for i in act_ids], dtype=np.int64)
      # refusal vs compliance contrast on attack prompts only
      idx_refuse = np.where(labels == 1)[0]
      idx_comply = np.where(labels == 0)[0]
      n = min(len(idx_refuse), len(idx_comply))
      if n == 0:
          print("      !! no refusal/compliance contrast available - aborting mechanism stage")
          return 2
      sel = np.concatenate([idx_refuse[:n], idx_comply[:n]])
      sel_prompts = [prompts[i] for i in sel]
      sel_labels = (labels[sel] == 1).astype(np.int64)

      n_layers = runner.layers()
      feats: Dict[int, np.ndarray] = runner.last_hidden_layers(sel_prompts, list(range(0, n_layers + 1)),
                                                               batch_size=args.batch_size)
      # A forward hook on decoder layer L writes into the layer OUTPUT, which is
      # hidden_states[L+1]. Features must therefore be read at L+1 to match the hook.
      hook_feats = {L: feats[L + 1] for L in range(0, n_layers) if (L + 1) in feats}
      sweep = layer_sweep(hook_feats, sel_labels)
      best = max(sweep, key=lambda d: d["f1"])
      print(f"      best layer={best['layer']} f1={best['f1']} (n={len(sel)})", flush=True)

      # refusal direction at the best layer
      X = hook_feats[best["layer"]]
      v = mean_difference_direction(X[sel_labels == 1], X[sel_labels == 0])
      # scale the unit direction by the typical residual norm at that layer so that
      # alpha is interpretable as a fraction of the residual stream magnitude
      resid_norm = float(np.mean(np.linalg.norm(X, axis=1)))
      v_scaled = (v * resid_norm).astype(np.float32)
      np.save(out / "refusal_vector.npy", v_scaled)
      np.save(out / "refusal_direction_unit.npy", v)
      print(f"      residual norm at layer {best['layer']} = {resid_norm:.2f}", flush=True)

      # -------------------------------------------- 3. cross-lingual direction
      parallel_records = [r for r in base_records if r["id"].startswith("PAR-")]
      lang_dirs: Dict[str, np.ndarray] = {}
      par_attack = [r for r in parallel_records if r["kind"] == "attack"]
      par_benign = [r for r in parallel_records if r["kind"] == "benign"]
      par_all = par_attack + par_benign
      if par_all:
          par_feats = runner.last_hidden_layers([r["prompt"] for r in par_all], [best["layer"] + 1],
                                                batch_size=args.batch_size)[best["layer"] + 1]
          for lang in sorted({r["language"] for r in par_all}):
              a_idx = [i for i, r in enumerate(par_all) if r["language"] == lang and r["kind"] == "attack"]
              b_idx = [i for i, r in enumerate(par_all) if r["language"] == lang and r["kind"] == "benign"]
              if not a_idx:
                  continue
              H = par_feats[a_idx]
              Hb = par_feats[b_idx] if b_idx else None
              lang_dirs[lang] = mean_difference_direction(H, Hb) if Hb is not None else H.mean(axis=0)
      alignment = direction_alignment(lang_dirs, reference="en") if lang_dirs else {}

      # ------------------------------------------------ 4. probe transfer (EN -> X)
      probe_report: Dict[str, object] = {}
      try:
          en_attack = [r for r in base_records if r["kind"] == "attack" and r["language"] == "en"]
          en_benign = [r for r in base_records if r["kind"] == "benign" and r["language"] == "en"]
          if len(en_attack) >= 4 and len(en_benign) >= 2:
              Xtr = np.concatenate([
                  runner.last_hidden([r["prompt"] for r in en_attack], best["layer"] + 1, batch_size=args.batch_size),
                  runner.last_hidden([r["prompt"] for r in en_benign], best["layer"] + 1, batch_size=args.batch_size),
              ])
              ytr = np.concatenate([np.ones(len(en_attack)), np.zeros(len(en_benign))])
              w = logistic_probe(Xtr, ytr)
              probe_report["train"] = {"n": int(len(ytr)), "accuracy": round(probe_accuracy(w, Xtr, ytr), 4)}
              transfer: Dict[str, float] = {}
              for lang in sorted({r["language"] for r in base_records if r["kind"] in ("attack", "benign")}):
                  items = [r for r in base_records if r["language"] == lang and r["kind"] in ("attack", "benign")]
                  if len(items) < 3:
                      continue
                  Xt = runner.last_hidden([r["prompt"] for r in items], best["layer"] + 1, batch_size=args.batch_size)
                  yt = np.array([1.0 if r["kind"] == "attack" else 0.0 for r in items])
                  transfer[lang] = round(probe_accuracy(w, Xt, yt), 4)
              probe_report["transfer_accuracy_by_language"] = transfer
              probe_report["mean_transfer_accuracy"] = round(
                  float(np.mean(list(transfer.values()))) if transfer else 0.0, 4)
              print(f"      probe transfer: {transfer}", flush=True)
      except Exception as exc:  # pragma: no cover
          probe_report["error"] = repr(exc)

      mechanism = {
          "model_id": args.model,
          "hidden_size": runner.hidden_size(),
          "num_layers": runner.layers(),
          "steering_layer": best["layer"],
          "layer_sweep": sweep,
          "direction_alignment_vs_en": alignment,
          "refusal_vector_norm": round(float(np.linalg.norm(v_scaled)), 6),
          "refusal_direction_unit_norm": round(float(np.linalg.norm(v)), 6),
          "residual_norm_at_layer": round(resid_norm, 4),
          "probe_transfer": probe_report,
          "contrast_prompts_used": int(len(sel)),
      }
      save_json(out / "mechanism.json", mechanism)

    # ------------------------------------------------------ 5. steering sweep
    pareto: List[dict] = []
    best_alpha = 0.0
    vector = v_scaled
    if not args.skip_sweep:
        print("[4/7] steering sweep", flush=True)
        sweep_attacks = [r for r in base_records if r["kind"] == "attack"][: args.sweep_probes]
        sweep_benign = [r for r in base_records if r["kind"] == "benign"][: args.sweep_probes]
        sweep_probes = [Probe(id=r["id"], category=r["category"], language=r["language"],
                              prompt=r["prompt"], kind="attack" if r["kind"] == "attack" else "benign")
                        for r in sweep_attacks + sweep_benign]

        def _measure(vector: np.ndarray, alpha: float) -> dict:
            t0 = time.time()
            if alpha == 0.0:
                recs = [dict(r) for r in (sweep_attacks + sweep_benign)]
            else:
                with ActivationSteerer(runner.model, best["layer"], vector, alpha=alpha):
                    recs = auditor.run(sweep_probes, progress_every=0)
            att = [r for r in recs if r["kind"] == "attack"]
            ben = [r for r in recs if r["kind"] == "benign"]
            asr = round(sum(1 for r in att if r["verdict"] == "compliance") / max(1, len(att)), 4)
            orr = round(sum(1 for r in ben if r["verdict"] == "refusal") / max(1, len(ben)), 4)
            return {"alpha": alpha, "asr": asr, "orr": orr, "n_attack": len(att), "n_benign": len(ben),
                    "seconds": round(time.time() - t0, 1), "sign": int(np.sign(vector.sum()) or 1)}

        # sign calibration: a refusal direction may point the other way depending on
        # which token position the contrast was estimated from.
        cal = _measure(v_scaled, args.calibration_alpha)
        cal_neg = _measure(-v_scaled, args.calibration_alpha)
        print(f"      sign calibration @alpha={args.calibration_alpha}: "
              f"+v ASR={cal['asr']} / -v ASR={cal_neg['asr']}", flush=True)
        vector = v_scaled if cal["asr"] <= cal_neg["asr"] else -v_scaled
        sign = 1 if cal["asr"] <= cal_neg["asr"] else -1
        if sign < 0:
            print("      using the mirrored direction (-v)", flush=True)

        for alpha in [float(a) for a in args.alphas.split(",")]:
            row = _measure(vector, alpha)
            row["sign"] = sign
            pareto.append(row)
            print(f"      alpha={alpha}: ASR={row['asr']} ORR={row['orr']} ({row['seconds']}s)", flush=True)
        save_json(out / "pareto.json", pareto)

        # operating point: lowest ASR among configurations inside the over-refusal budget,
        # ties broken towards the smallest perturbation
        affordable = [p for p in pareto if p["orr"] <= args.max_orr]
        if affordable:
            chosen = min(affordable, key=lambda p: (p["asr"], p["alpha"]))
            best_alpha = chosen["alpha"]
            print(f"      operating point: alpha={best_alpha} (ASR {chosen['asr']}, ORR {chosen['orr']})",
                  flush=True)

    # ---------------------------------------------------- 6. hardened audit
    print(f"[5/7] hardened audit (alpha={best_alpha})", flush=True)
    if best_alpha > 0:
        with ActivationSteerer(runner.model, best["layer"], vector, alpha=best_alpha):
            hard_records = auditor.run(probes)
    else:
        hard_records = [dict(r) for r in base_records]
    save_jsonl(out / "hardened_transcripts.jsonl", hard_records)
    hard_summary = Auditor.summarize(hard_records, model_id=args.model, stage="hardened")
    hard_summary["steering"] = {"layer": best["layer"], "alpha": best_alpha,
                                 "vector": "artifacts/real/refusal_vector.npy",
                                 "vector_norm": round(float(np.linalg.norm(vector)), 6)}
    save_json(out / "hardened_summary.json", hard_summary)
    print(f"      ASR={hard_summary['asr']}  ORR={hard_summary['orr']}  "
          f"defcon={hard_summary['defcon']['label']}", flush=True)

    # ----------------------------------------------------------- 7. steering cfg
    save_json(out / "steering_config.json", {
        "model_id": args.model,
        "method": "activation_steering_residual_stream",
        "steering_layer": best["layer"],
        "alpha": best_alpha,
        "hidden_dimension": runner.hidden_size(),
        "vector_file": "artifacts/real/refusal_vector.npy",
        "vector_norm": round(float(np.linalg.norm(v_scaled)), 6),
        "residual_norm_at_layer": round(resid_norm, 4),
        "alpha_semantics": "alpha * (typical residual norm at the steering layer) added to the residual stream",
        "hook_point": "decoder_layer_output_residual",
    })

    # ------------------------------------------------------------------ report
    lines = []
    A = lines.append
    A("# Real Safety Pipeline - Measured Results\n")
    A(f"Model: `{args.model}` (hidden={runner.hidden_size()}, layers={runner.layers()}, dtype={runner.meta['dtype']})")
    A(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  |  wall time {round(time.time() - t_start)}s\n")
    A("## Baseline vs hardened (measured from generations)\n")
    A("| Metric | Baseline | Hardened |")
    A("|---|---|---|")
    A(f"| Attack Success Rate | {base_summary['asr']} | {hard_summary['asr']} |")
    A(f"| Refusal rate | {base_summary['refusal_rate']} | {hard_summary['refusal_rate']} |")
    A(f"| Over-refusal (benign) | {base_summary['orr']} | {hard_summary['orr']} |")
    A(f"| GSM8K accuracy | {base_summary['gsm8k_accuracy']} | {hard_summary['gsm8k_accuracy']} |")
    A(f"| DEFCON | {base_summary['defcon']['label']} | {hard_summary['defcon']['label']} |")
    A(f"| probes (attack/benign/utility) | {base_summary['n_attack']}/{base_summary['n_benign']}/{base_summary['n_utility']} | "
      f"{hard_summary['n_attack']}/{hard_summary['n_benign']}/{hard_summary['n_utility']} |")
    A("\n## Per-category ASR\n")
    A("| category | baseline | hardened |")
    A("|---|---|---|")
    for cat in sorted(base_summary["per_category"]):
        b = base_summary["per_category"][cat]["asr"]
        h = hard_summary["per_category"].get(cat, {}).get("asr", "-")
        A(f"| {cat} | {b} | {h} |")
    A("\n## Per-language ASR\n")
    A("| language | baseline | hardened |")
    A("|---|---|---|")
    for lang in sorted(base_summary["per_language"]):
        b = base_summary["per_language"][lang]["asr"]
        h = hard_summary["per_language"].get(lang, {}).get("asr", "-")
        A(f"| {lang} | {b} | {h} |")
    A("\n## Mechanism\n")
    A(f"- steering layer: **{best['layer']}** (rank-1 probe F1 {best.get('f1')}, "
      f"separation {best.get('separation', 'n/a')})")
    A(f"- refusal direction norm: {round(float(np.linalg.norm(v)), 6)}")
    if alignment:
        A(f"- cross-lingual direction alignment vs EN: {alignment}")
    if probe_report.get("transfer_accuracy_by_language"):
        A(f"- EN-trained probe transfer accuracy by language: {probe_report['transfer_accuracy_by_language']}")
    if pareto:
        A("\n## Safety/utility Pareto (steering alpha)\n")
        A("| alpha | ASR | ORR | n_attack | n_benign |")
        A("|---|---|---|---|---|")
        for p in pareto:
            A(f"| {p['alpha']} | {p['asr']} | {p['orr']} | {p['n_attack']} | {p['n_benign']} |")
    A("\n## Reproduce\n")
    A("```bash")
    A(f"python scripts/run_real_pipeline.py --model {args.model}")
    A("python scripts/verify_submission.py   # recomputes every number from the transcripts")
    A("```")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[6/7] wrote {out / 'REPORT.md'}", flush=True)
    print(f"[7/7] done in {round(time.time() - t_start)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
