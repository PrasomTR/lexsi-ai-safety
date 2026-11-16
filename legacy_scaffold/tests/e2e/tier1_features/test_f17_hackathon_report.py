"""Tier 1 Feature Tests: Feature 17 - Comprehensive Hackathon README/Report.

Validates:
1. README.md existence and UTF-8 markdown formatting in project root.
2. Essential hackathon submission sections and Track 2 scope.
3. Before-and-after quantitative evaluation comparison table.
4. Concrete reproducibility instructions and execution commands.
5. Official competition links and toolkit citations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import pytest


@pytest.mark.tier1
class TestF17HackathonReport:
    """Feature 17: Comprehensive Hackathon README/Report tests."""

    def test_f17_readme_file_presence_and_format(self, project_root: Path) -> None:
        """Verify README.md exists in project root with substantial competition content."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), f"Hackathon README.md missing from project root: {readme_path}"
        content = readme_path.read_text(encoding="utf-8")
        assert len(content) > 100, f"README.md too short ({len(content)} chars) for competition submission"

    def test_f17_required_hackathon_sections(self, project_root: Path) -> None:
        """Verify report covers all required competition submission topics."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), f"Hackathon README.md missing from project root: {readme_path}"
        required_topics = [
            "Track 2",
            "Tiny Aya",
            "SafeTune",
            "AuditKIT",
            "Results",
            "Reproducibility",
        ]
        content = readme_path.read_text(encoding="utf-8").lower()
        for t in required_topics:
            assert t.lower() in content, f"README missing section/topic: {t}"

    def test_f17_before_after_metric_table(self, project_root: Path) -> None:
        """Verify presence of quantitative before-and-after comparison table."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), f"Hackathon README.md missing from project root: {readme_path}"
        content = readme_path.read_text(encoding="utf-8")
        # Check for markdown table indicators and metric names
        assert "|" in content, "README must contain a markdown comparison table"
        assert "ASR" in content or "Attack Success Rate" in content, "README must contain ASR metrics"
        assert "Refusal" in content or "ORR" in content, "README must contain Refusal/ORR metrics"

    def test_f17_reproducibility_run_instructions(self, project_root: Path) -> None:
        """Verify reproduction instructions detail running scripts in sequence."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), f"Hackathon README.md missing from project root: {readme_path}"
        content = readme_path.read_text(encoding="utf-8")
        assert "python" in content.lower(), "README must include python command instructions"
        assert "01_" in content or "02_" in content or "scripts" in content, "README must list script commands"

    def test_f17_official_competition_links_and_citations(self, project_root: Path) -> None:
        """Verify inclusion of official hackathon URL and Cohere/toolkit references."""
        readme_path = project_root / "README.md"
        assert readme_path.exists(), f"Hackathon README.md missing from project root: {readme_path}"
        content = readme_path.read_text(encoding="utf-8")
        assert "hackathon.lexsi.ai" in content or "lexsi" in content.lower(), "Missing hackathon URL"
        assert "cohere" in content.lower(), "Missing Cohere model citation"
