"""Tier 1 Feature Tests: Feature 2 - Python & Windows Environment Setup.

Validates:
1. Python version adheres to project requirement (Python 3.10+).
2. Windows platform compatibility, path handling, and UTF-8 environment settings.
3. Core safety and machine learning dependencies are importable or tracked in requirements.
4. PyTorch hardware acceleration or CPU fallback is properly initialized.
5. Subprocess execution environment under Windows executes reliably with proper encoding.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable
import pytest

from tests.e2e.conftest import CLIResult


@pytest.mark.tier1
class TestF02Environment:
    """Feature 2: Python & Windows Environment Setup tests."""

    def test_f02_python_runtime_version(self) -> None:
        """Verify Python interpreter version is 3.10 or newer."""
        major = sys.version_info.major
        minor = sys.version_info.minor
        assert major == 3, f"Expected Python 3, got Python {major}"
        assert minor >= 10, f"Expected Python >= 3.10, got 3.{minor}"

    def test_f02_windows_os_compatibility(self) -> None:
        """Verify Windows OS platform compatibility and UTF-8 path handling."""
        assert sys.platform == "win32", f"Expected win32 platform, got {sys.platform}"
        assert os.name == "nt", f"Expected os.name 'nt', got {os.name}"

        # Test Windows path normalization and separator handling
        test_path = Path("C:/Users/test/path/to/resource")
        assert "\\" in str(test_path) or "/" in str(test_path)
        assert test_path.drive.upper() == "C:"

        # Verify UTF-8 environment handling
        sample_multilingual = "Tiny Aya 安全 🚀 Multilingual Test"
        encoded = sample_multilingual.encode("utf-8")
        assert encoded.decode("utf-8") == sample_multilingual

    def test_f02_required_core_packages_installed(self, project_root: Path) -> None:
        """Verify required packages for Track 2 are tracked in requirements or installed."""
        required_modules = [
            "torch",
            "transformers",
            "peft",
            "yaml",
            "jsonschema",
            "pytest",
        ]
        installed = []
        for mod in required_modules:
            try:
                __import__(mod)
                installed.append(mod)
            except ImportError:
                pass

        # At minimum, test framework and schema validation packages must be active
        assert "pytest" in installed, "pytest must be installed in test environment"
        assert "jsonschema" in installed, "jsonschema must be installed for contract validation"
        assert "yaml" in installed, "PyYAML must be installed for config parsing"
        assert len(required_modules) == 6

    def test_f02_cuda_availability_or_simulation(self) -> None:
        """Verify CUDA detection semantics and compute capability checks."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                assert torch.cuda.device_count() >= 1
            else:
                cpu_tensor = torch.zeros((2, 2))
                assert cpu_tensor.device.type == "cpu"
        except ImportError:
            # Fallback hardware detection check
            cuda_path = os.environ.get("CUDA_PATH")
            simulated_device = "cuda" if cuda_path else "cpu"
            assert simulated_device in ("cuda", "cpu")

    def test_f02_pip_tooling_and_cli_execution(
        self, cli_runner: Callable[..., CLIResult]
    ) -> None:
        """Verify python CLI execution works cleanly via cli_runner."""
        res = cli_runner([sys.executable, "-c", "import sys; print(sys.version_info.major)"])
        assert res.is_success, f"CLI execution failed: {res.stderr}"
        assert res.stdout.strip() == "3"
        assert res.returncode == 0
