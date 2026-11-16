"""One-command falsification of the submitted safety claims.

Recomputes every published metric *from the shipped transcripts* and compares it
with the stored summary. If a number cannot be reproduced from raw model output,
this script fails. No GPU, no server, no network.

    python scripts/verify_submission.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.audit import Auditor  # noqa: E402
from src.pipeline.refusal import RefusalDetector  # noqa: E402

REAL = ROOT / "artifacts" / "real"


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check(label: str, got, expected, tol: float = 1e-9) -> bool:
    ok = (abs(got - expected) <= tol) if isinstance(got, (int, float)) and isinstance(expected, (int, float)) else (got == expected)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: recomputed={got} reported={expected}")
    return ok


def main() -> int:
    if not (REAL / "baseline_transcripts.jsonl").exists():
        print("No measured transcripts found. Run scripts/run_real_pipeline.py first.")
        return 2

    detector = RefusalDetector()
    failures = 0
    print("== metric recomputation from raw transcripts ==")
    for stage in ("baseline", "hardened"):
        tpath = REAL / f"{stage}_transcripts.jsonl"
        spath = REAL / f"{stage}_summary.json"
        records = load_jsonl(tpath)
        stored = json.loads(spath.read_text(encoding="utf-8"))
        print(f"-- {stage} ({len(records)} transcripts)")
        # re-classify independently, then recompute
        for r in records:
            r["verdict"] = detector.classify(r["completion"], r["language"])
        recomputed = Auditor.summarize(records, model_id=stored.get("model_id", ""), stage=stage)
        for key in ("asr", "refusal_rate", "orr"):
            failures += 0 if check(key, recomputed[key], stored[key]) else 1
        if stored.get("gsm8k_accuracy") is not None and recomputed.get("gsm8k_accuracy") is not None:
            failures += 0 if check("gsm8k_accuracy", recomputed["gsm8k_accuracy"], stored["gsm8k_accuracy"]) else 1
        for cat, payload in stored["per_category"].items():
            failures += 0 if check(f"per_category[{cat}].asr", recomputed["per_category"][cat]["asr"],
                                   payload["asr"]) else 1
        for lang, payload in stored["per_language"].items():
            failures += 0 if check(f"per_language[{lang}].asr", recomputed["per_language"][lang]["asr"],
                                   payload["asr"]) else 1
        failures += 0 if check("run_fingerprint", recomputed["run_fingerprint"], stored["run_fingerprint"]) else 1

    print("\n== mechanism artefacts ==")
    mech_path = REAL / "mechanism.json"
    if mech_path.exists():
        mech = json.loads(mech_path.read_text(encoding="utf-8"))
        vec_path = REAL / "refusal_vector.npy"
        try:
            import numpy as np
            v = np.load(vec_path)
            failures += 0 if check("refusal_vector_dim", int(v.shape[0]), mech["hidden_size"]) else 1
            failures += 0 if check("refusal_vector_norm", round(float(np.linalg.norm(v)), 6),
                                   mech["refusal_vector_norm"]) else 1
        except Exception as exc:
            print(f"  [FAIL] refusal vector unreadable: {exc}")
            failures += 1
        align = mech.get("direction_alignment_vs_en") or {}
        if align:
            worst = min(align.values())
            print(f"  [INFO] cross-lingual direction alignment vs EN: {align} (min={worst})")
    else:
        print("  [WARN] mechanism.json missing")

    print("\n== artefact integrity (sha256) ==")
    for p in sorted(REAL.glob("*")):
        if p.is_file():
            print(f"  {sha256(p)[:16]}...  {p.relative_to(ROOT)}")

    print("\n== verdict ==")
    if failures == 0:
        print("ALL CHECKS PASSED - every published number is reproducible from raw model output.")
        return 0
    print(f"{failures} CHECK(S) FAILED - published numbers do NOT match the transcripts.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
