"""Tier 1 Feature Tests: Feature 20 - CircuitKIT Circuit Identification & EAP.

Validates Feature 5 from PROJECT.md and TEST_INFRA.md:
1. Edge Attribution Patching (EAP) execution and scoring calculation.
2. Safety attention head localization (Layers 16-24 refusal nexus >= 60% attribution).
3. Grouped Query Attention (GQA) KV-head deduplication (16 query heads : 4 KV heads, 4:1 ratio).
4. D3.js Graph Format Serialization (nodes, edges, refusal nexus highlight flags).
5. Safety Neuron Attribution Ranking Table (layer index, neuron ID, contrast score).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.models.mock import (
    MockCohere2Attention,
    MockCohere2ForCausalLM,
    MockCohere2MLP,
)
from src.interpretability.circuit import (
    CircuitDiscoveryEngine,
    CircuitGraph,
    deduplicate_gqa_heads,
    export_d3_graph,
    rank_safety_neurons,
)


@pytest.mark.tier1
class TestF20CircuitKITEAP:
    """Feature 5: CircuitKIT Circuit Identification & EAP tests."""

    def test_f20_eap_attribution_scoring_computation(self) -> None:
        """T1-F05-01: Verify Edge Attribution Patching (EAP) scoring contract."""
        model = MockCohere2ForCausalLM()
        engine = CircuitDiscoveryEngine(model=model)
        scores = engine.compute_eap_scores()

        assert scores["num_layers"] == 36, "Cohere2 model depth must be 36 layers"
        assert len(scores["head_scores"]) == 36 * 16, "Must compute scores for all 576 attention heads"
        assert len(scores["mlp_scores"]) == 36, "Must compute scores for all 36 MLP blocks"
        assert len(scores["edge_scores"]) > 0, "Attribution graph edges must be generated"

        head_vals = list(scores["head_scores"].values())
        assert all(s >= 0 for s in head_vals), "Attribution scores must be non-negative"
        assert max(head_vals) > 0.05, "Top attribution scores must show significant magnitude"

    def test_f20_refusal_nexus_layer_localization(self) -> None:
        """T1-F05-02: Verify >= 60% of top refusal heads localize to layers 16-24."""
        engine = CircuitDiscoveryEngine()
        nexus = engine.get_refusal_nexus_layers()
        assert nexus == list(range(16, 25)), "Refusal nexus must span layers 16 to 24"

        scores = engine.compute_eap_scores()
        head_scores_items = list(scores["head_scores"].items())
        head_scores_items.sort(key=lambda item: item[1], reverse=True)

        # Top 5% heads (top 28 of 576 heads)
        top_k = int(0.05 * len(head_scores_items))
        top_heads = head_scores_items[:top_k]

        nexus_count = 0
        for head_id, _ in top_heads:
            # head_id format: "L{layer}_H{head}"
            layer_idx = int(head_id.split("_")[0][1:])
            if layer_idx in nexus:
                nexus_count += 1

        nexus_fraction = nexus_count / len(top_heads)
        assert nexus_fraction >= 0.60, (
            f"Expected >= 60% of top heads in layers 16-24, observed {nexus_fraction:.1%}"
        )

    def test_f20_gqa_kv_head_deduplication(self) -> None:
        """T1-F05-03: Verify GQA 16:4 head deduplication prevents 4x over-counting."""
        attn = MockCohere2Attention(hidden_size=2048, num_heads=16, num_kv_heads=4)
        assert attn.num_heads == 16, "Tiny Aya has 16 query heads"
        assert attn.num_kv_heads == 4, "Tiny Aya has 4 KV heads"
        group_ratio = attn.num_heads // attn.num_kv_heads
        assert group_ratio == 4, "Tiny Aya GQA ratio must be exactly 4:1"

        # Deduplicate query heads to unique KV heads
        kv_attributions = deduplicate_gqa_heads(num_query_heads=16, num_kv_heads=4)
        assert len(kv_attributions) == 4, "Must deduplicate to 4 unique KV head nodes"
        assert all(0 <= k < 4 for k in kv_attributions.keys()), "KV indices must be in [0, 3]"

    def test_f20_d3_graph_format_serialization(self, tmp_path: Path) -> None:
        """T1-F05-04: Verify D3.js circuit graph schema and refusal nexus flagging."""
        circuit_data = export_d3_graph(threshold=0.5)

        assert len(circuit_data["nodes"]) >= 4, "Graph must contain nodes"
        assert len(circuit_data["edges"]) >= 3, "Graph must contain edges"

        for node in circuit_data["nodes"]:
            assert "id" in node
            assert "layer" in node and 0 <= node["layer"] < 36
            assert node["type"] in ("head", "mlp")
            assert "score" in node
            assert "is_refusal_nexus" in node
            if node["is_refusal_nexus"]:
                assert 16 <= node["layer"] <= 24, "Refusal nexus nodes must be within layers 16-24"

        for edge in circuit_data["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge and edge["weight"] >= 0.0

        # Verify serialization and deserialization
        out_file = tmp_path / "circuit_graph.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(circuit_data, f, indent=2)

        with open(out_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert "nodes" in loaded and "edges" in loaded
        assert len(loaded["nexus_layers"]) == 9

    def test_f20_safety_neuron_attribution_ranking_table(self) -> None:
        """T1-F05-05: Verify safety neuron attribution table is sorted descending with valid contrast scores."""
        mlp = MockCohere2MLP(hidden_size=2048, intermediate_size=11008)
        assert mlp.intermediate_size == 11008, "Tiny Aya MLP intermediate size is 11,008"

        neurons = rank_safety_neurons(mlp_module=mlp, top_k=5)
        assert len(neurons) == 5, "Must return top 5 ranked safety neurons"

        # Verify strictly sorted descending by contrast score
        for i in range(len(neurons) - 1):
            assert neurons[i]["contrast_score"] >= neurons[i + 1]["contrast_score"], (
                "Neurons must be sorted descending by contrast score"
            )

        # Verify all neuron indices fall within intermediate_size bounds
        for n in neurons:
            assert 0 <= n["neuron_idx"] < mlp.intermediate_size
            assert 16 <= n["layer"] <= 24, "Safety neurons must reside in refusal nexus layers 16-24"
            assert n["contrast_score"] > 0.5, "Contrast score must be non-trivial"
