"""Tier 1 Feature Tests: Feature 15 - Interactive Demonstration Notebook.

Validates:
1. Notebook JSON structure and nbformat specification (nbformat >= 4).
2. Key narrative sections present in markdown cells.
3. Python syntax validity across all notebook code cells.
4. Visualization and metric presentation cells.
5. Notebook kernel and language runtime metadata.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List
import pytest


@pytest.mark.tier1
class TestF15DemoNotebook:
    """Feature 15: Interactive Demonstration Notebook tests."""

    def test_f15_notebook_file_format_and_json(self, notebooks_dir: Path) -> None:
        """Verify notebook exists and is well-formed JSON conforming to nbformat v4."""
        nb_path = notebooks_dir / "tiny_aya_safety_demo.ipynb"
        assert nb_path.exists(), f"Required demonstration notebook missing at {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "cells" in data, "Notebook JSON must contain 'cells' array"
        assert data.get("nbformat", 0) >= 4, f"nbformat must be >= 4, got {data.get('nbformat')}"
        assert isinstance(data["cells"], list), "'cells' property must be a list"
        assert len(data["cells"]) > 0, "Notebook must contain cells"

    def test_f15_notebook_required_sections_present(self, notebooks_dir: Path) -> None:
        """Verify markdown cells cover the full hackathon narrative."""
        nb_path = notebooks_dir / "tiny_aya_safety_demo.ipynb"
        assert nb_path.exists(), f"Required demonstration notebook missing at {nb_path}"
        expected_keywords = ["Tiny Aya", "SafeTune", "AuditKIT", "Baseline", "Differential"]

        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        markdown_text = " ".join(
            "".join(c.get("source", []))
            for c in data.get("cells", [])
            if c.get("cell_type") == "markdown"
        )
        for kw in expected_keywords:
            assert kw.lower() in markdown_text.lower(), f"Missing required keyword '{kw}' in notebook narrative"

    def test_f15_notebook_code_cells_validity(self, notebooks_dir: Path) -> None:
        """Verify all code cells in the notebook parse without syntax errors."""
        nb_path = notebooks_dir / "tiny_aya_safety_demo.ipynb"
        assert nb_path.exists(), f"Required demonstration notebook missing at {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        code_cells = [c for c in data.get("cells", []) if c.get("cell_type") == "code"]
        assert len(code_cells) >= 3, f"Notebook must contain at least 3 code cells, found {len(code_cells)}"
        for idx, cell in enumerate(code_cells):
            code_src = "".join(cell.get("source", []))
            # Ignore empty cells or ipython magics for AST parse
            cleaned_code = "\n".join(
                line for line in code_src.splitlines()
                if not line.strip().startswith("%") and not line.strip().startswith("!")
            )
            if cleaned_code.strip():
                try:
                    ast.parse(cleaned_code)
                except SyntaxError as err:
                    pytest.fail(f"Cell #{idx} syntax error: {err}")

    def test_f15_notebook_visualization_cells(self, notebooks_dir: Path) -> None:
        """Verify notebook includes visualization or tabular output components."""
        nb_path = notebooks_dir / "tiny_aya_safety_demo.ipynb"
        assert nb_path.exists(), f"Required demonstration notebook missing at {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        code_text = " ".join(
            "".join(c.get("source", []))
            for c in data.get("cells", [])
            if c.get("cell_type") == "code"
        )
        has_viz = any(k in code_text for k in ["matplotlib", "plt.", "pandas", "tabulate", "rich", "table"])
        assert has_viz, "Notebook must include visualization or tabular comparison rendering"

    def test_f15_notebook_execution_metadata(self, notebooks_dir: Path) -> None:
        """Verify notebook metadata specifies Python 3 kernel."""
        nb_path = notebooks_dir / "tiny_aya_safety_demo.ipynb"
        assert nb_path.exists(), f"Required demonstration notebook missing at {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        lang_info = meta.get("language_info", {})
        assert lang_info.get("name", "python") == "python", f"Notebook kernel must be python, got {lang_info}"
