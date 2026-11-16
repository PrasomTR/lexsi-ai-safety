"""Builds a single-file HTML report (opens from disk, no server, no CDN).

    python scripts/make_html_report.py -> artifacts/real/report.html
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "artifacts" / "real"

CSS = """
:root{--bg:#0e1116;--fg:#e6edf3;--muted:#8b949e;--red:#f85149;--green:#3fb950;--line:#21262d;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:38px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
p,li{color:#c9d1d9}
.sub{color:var(--muted);margin:0 0 26px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:22px 0}
.card{background:#161b22;border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.card .v{font-size:28px;font-weight:650;margin-top:6px;letter-spacing:-.02em}
.card .d{font-size:13px;color:var(--muted);margin-top:4px}
.good{color:var(--green)} .bad{color:var(--red)} .acc{color:var(--acc)}
table{width:100%;border-collapse:collapse;margin:10px 0 6px;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code{background:#161b22;padding:2px 6px;border-radius:5px;color:#79c0ff;font-size:13px}
pre{background:#161b22;border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow:auto}
.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--muted);margin-right:6px}
.note{border-left:3px solid var(--acc);padding:10px 16px;background:#12181f;border-radius:0 8px 8px 0;margin:16px 0}
footer{color:var(--muted);font-size:13px;margin-top:48px;border-top:1px solid var(--line);padding-top:16px}
"""


def load(name):
    p = REAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table(headers, rows):
    out = ["<table><thead><tr>"]
    out += [f"<th class='num'>{h}</th>" if isinstance(h, tuple) else f"<th>{h}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(
            f"<td class='num'>{c}</td>" if isinstance(c, tuple) else f"<td>{c}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main():
    base, hard = load("baseline_summary.json"), load("hardened_summary.json")
    mech, pareto = load("mechanism.json"), load("pareto.json")
    if not base or not hard:
        print("no measured artifacts yet")
        return 1
    h = []
    h.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    h.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    h.append("<title>Lexsi AI Safety - measured results</title>")
    h.append(f"<style>{CSS}</style></head><body><div class='wrap'>")
    h.append("<h1>Safety claims should be executable, not asserted</h1>")
    h.append(f"<p class='sub'>Lexsi AI Safety Hackathon - Track 2 (AI Red Teaming, Safety and Alignment) | "
             f"model <code>{base['model_id']}</code> | {base['n_attack']} attack / {base['n_benign']} benign / "
             f"{base['n_utility']} utility probes | 8 languages</p>")

    d_asr = (hard["asr"] - base["asr"]) * 100
    h.append("<div class='cards'>")
    for k, v, d, cls in [
        ("Attack success rate", pct(base["asr"]), f"-> {pct(hard['asr'])}", "good" if hard["asr"] < base["asr"] else "bad"),
        ("Refusal rate", pct(base["refusal_rate"]), f"-> {pct(hard['refusal_rate'])}", "acc"),
        ("Over-refusal (benign)", pct(base["orr"]), f"-> {pct(hard['orr'])}", "good"),
        ("Coherence", pct(base.get("coherence_rate")), f"-> {pct(hard.get('coherence_rate'))}", "acc"),
        ("DEFCON", base["defcon"]["label"], f"-> {hard['defcon']['label']}", "acc"),
        ("GSM8K accuracy", pct(base.get("gsm8k_accuracy")), f"-> {pct(hard.get('gsm8k_accuracy'))}", "acc"),
    ]:
        h.append(f"<div class='card'><div class='k'>{k}</div><div class='v {cls}'>{v}</div><div class='d'>{d}</div></div>")
    h.append("</div>")

    h.append("<div class='note'>Every number on this page is recomputed from the raw completions in "
             "<code>artifacts/real/*_transcripts.jsonl</code>. Run "
             "<code>python scripts/verify_submission.py</code> to falsify them yourself.</div>")

    h.append("<h2>Per attack family</h2>")
    rows = []
    for c in sorted(base["per_category"]):
        b = base["per_category"][c]["asr"]
        s = hard["per_category"].get(c, {}).get("asr")
        rows.append((c, (pct(b), ""), (pct(s), "")))
    h.append(table(["family", ("baseline", ""), ("steered", "")], rows))

    if base.get("per_language"):
        h.append("<h2>Per language (parallel sentences)</h2>")
        rows = []
        for l in sorted(base["per_language"]):
            rows.append((l, (pct(base["per_language"][l]["asr"]), ""),
                         (pct(hard.get("per_language", {}).get(l, {}).get("asr")), "")))
        h.append(table(["language", ("baseline", ""), ("steered", "")], rows))

    if mech:
        h.append("<h2>Mechanism</h2>")
        h.append("<ul>")
        h.append(f"<li>Refusal becomes linearly decodable at decoder layer <b>{mech['steering_layer']}</b> "
                 f"(rank-1 probe F1 = {max(s['f1'] for s in mech.get('layer_sweep', [{'f1': 0}]))}).</li>")
        al = mech.get("direction_alignment_vs_en") or {}
        if al:
            vals = [v for k, v in al.items() if k != "en"]
            h.append(f"<li>The refusal direction extracted independently per language points the same way: "
                     f"cosine vs. English min {min(vals):.3f}, mean {sum(vals)/len(vals):.3f}.</li>")
        pt = mech.get("probe_transfer", {})
        if pt.get("transfer_accuracy_by_language"):
            h.append(f"<li>A probe trained only on English transfers to the other languages: "
                     f"{pt['transfer_accuracy_by_language']} (mean {pct(pt.get('mean_transfer_accuracy'))}).</li>")
        h.append(f"<li>Mitigation = one {mech['hidden_size']}-d vector, no retraining, no second model.</li>")
        h.append("</ul>")
        sweep = mech.get("layer_sweep", [])
        if sweep:
            h.append(table(["layer", ("probe F1", "")], [(s["layer"], (s["f1"], "")) for s in sweep]))

    if pareto:
        h.append("<h2>Safety / utility Pareto</h2>")
        h.append(table(["alpha", ("ASR", ""), ("over-refusal", "")],
                       [(p["alpha"], (pct(p["asr"]), ""), (pct(p["orr"]), "")) for p in pareto]))

    h.append("<h2>Reproduce</h2>")
    h.append("<pre>python scripts/run_real_pipeline.py --model &lt;model-id&gt;\n"
             "python scripts/verify_submission.py\n"
             "python scripts/audit_ai_slop.py\n"
             "python -m pytest tests -q</pre>")
    h.append("<h2>Disclosure</h2>")
    h.append("<p>The first prototype of this submission reported attack-success rates that were hard-coded "
             "constants in the evaluator and shipped an untrained adapter. Both were removed: a metric may "
             "now only be produced by a function that consumed model output, and "
             "<code>scripts/audit_ai_slop.py</code> fails the build if a metric-shaped literal or a "
             "fabricated completion string reappears.</p>")
    h.append("<footer>Generated from <code>artifacts/real/*.json</code> - no server, no CDN, no network.</footer>")
    h.append("</div></body></html>")
    out = REAL / "report.html"
    out.write_text("\n".join(h), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
