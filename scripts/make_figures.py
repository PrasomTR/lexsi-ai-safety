"""Figures for the measured pipeline (reads artifacts/real/*.json, no model needed).

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "artifacts" / "real"
OUT = ROOT / "artifacts" / "real" / "figures"


def load(name):
    p = REAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_category(base, hard):
    cats = sorted(base["per_category"])
    b = [base["per_category"][c]["asr"] for c in cats]
    h = [hard["per_category"].get(c, {}).get("asr", 0.0) for c in cats]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = range(len(cats))
    ax.bar([i - 0.2 for i in x], b, width=0.4, label="baseline", color="#c0392b")
    ax.bar([i + 0.2 for i in x], h, width=0.4, label="steered", color="#1e8449")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"Attack success rate per family (n={base['n_attack']} probes, measured)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_asr_per_family.png", dpi=300)
    plt.close(fig)


def fig_language(base, hard):
    langs = sorted(base["per_language"])
    b = [base["per_language"][l]["asr"] for l in langs]
    h = [hard["per_language"].get(l, {}).get("asr", 0.0) for l in langs]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(langs))
    ax.bar([i - 0.2 for i in x], b, width=0.4, label="baseline", color="#b03a2e")
    ax.bar([i + 0.2 for i in x], h, width=0.4, label="steered", color="#1e8449")
    ax.set_xticks(list(x))
    ax.set_xticklabels(langs)
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Parallel multilingual attack success rate (same sentences, 8 languages)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_asr_per_language.png", dpi=300)
    plt.close(fig)


def fig_pareto(pareto):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([p["orr"] for p in pareto], [p["asr"] for p in pareto], "o-", color="#2e86c1")
    for p in pareto:
        ax.annotate(f"a={p['alpha']}", (p["orr"], p["asr"]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("over-refusal rate (benign probes)")
    ax.set_ylabel("attack success rate")
    ax.set_title("Safety / utility Pareto frontier (steering strength)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_pareto.png", dpi=300)
    plt.close(fig)


def fig_layers(mech):
    sweep = mech.get("layer_sweep", [])
    if not sweep:
        return
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot([s["layer"] for s in sweep], [s["f1"] for s in sweep], "o-", color="#8e44ad")
    best = mech.get("steering_layer")
    if best is not None:
        ax.axvline(best, color="#c0392b", ls="--", label=f"steering layer {best}")
        ax.legend()
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("refusal-vs-compliance F1")
    ax.set_title("Where refusal becomes linearly decodable in the residual stream")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_layer_localisation.png", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base, hard, pareto, mech = (load("baseline_summary.json"), load("hardened_summary.json"),
                                load("pareto.json"), load("mechanism.json"))
    if base and hard:
        fig_category(base, hard)
        if base.get("per_language"):
            fig_language(base, hard)
    if pareto:
        fig_pareto(pareto)
    if mech:
        fig_layers(mech)
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
