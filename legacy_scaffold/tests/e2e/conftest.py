"""Lexsi AI Safety Hackathon - Track 2: AI Red Teaming, Safety and Alignment.

Global End-to-End (E2E) Test Suite Configuration and Fixtures.
Provides:
1. Canonical project path fixtures.
2. Comprehensive schema validation helpers for datasets, audit results, and configs.
3. Windows-safe CLI execution helpers with timeout and environment handling.
4. Synthetic fixture generators for decoupled / mock testing without an 80GB GPU.
5. Pytest session tracking and report generation integration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import jsonschema
import pytest
import yaml

# ==============================================================================
# 1. Project Path Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the absolute root directory of the lexsi_ai_safety project."""
    # conftest.py is at: <project_root>/tests/e2e/conftest.py
    root = Path(__file__).resolve().parent.parent.parent
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Returns the path to the data directory."""
    d = project_root / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def artifacts_dir(project_root: Path) -> Path:
    """Returns the path to the artifacts directory."""
    d = project_root / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Returns the path to the configs directory."""
    d = project_root / "configs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def scripts_dir(project_root: Path) -> Path:
    """Returns the path to the scripts directory."""
    d = project_root / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def notebooks_dir(project_root: Path) -> Path:
    """Returns the path to the notebooks directory."""
    d = project_root / "notebooks"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def tests_dir(project_root: Path) -> Path:
    """Returns the path to the tests directory."""
    d = project_root / "tests"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def e2e_tests_dir(tests_dir: Path) -> Path:
    """Returns the path to the tests/e2e directory."""
    d = tests_dir / "e2e"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ==============================================================================
# 2. Schema Definitions and Validation Helpers
# ==============================================================================

REDTEAM_RECORD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "category": {"type": "string", "minLength": 1},
        "prompt": {"type": "string", "minLength": 1},
        "source": {"type": "string", "minLength": 1},
    },
    "required": ["id", "category", "prompt", "source"],
    "additionalProperties": True,
}

BENIGN_RECORD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "type": {"type": "string", "minLength": 1},
        "prompt": {"type": "string", "minLength": 1},
        "reference": {"type": "string", "minLength": 1},
    },
    "required": ["id", "type", "prompt", "reference"],
    "additionalProperties": True,
}

SAFETY_PAIR_RECORD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "minLength": 1},
        "response": {"type": "string"},
        "chosen": {"type": "string", "minLength": 1},
        "rejected": {"type": "string", "minLength": 1},
    },
    "required": ["prompt", "response", "chosen", "rejected"],
    "additionalProperties": True,
}

AUDIT_RESULTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metrics": {"type": "object"},
        "probes_summary": {"type": ["array", "object"]},
        "asr": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "refusal_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "orr": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "defcon_grade": {"type": "string", "minLength": 1},
    },
    "required": ["metrics", "probes_summary", "asr", "refusal_rate", "orr", "defcon_grade"],
    "additionalProperties": True,
}

YAML_CONFIG_REQUIRED_KEYS: Dict[str, List[str]] = {
    "eval_baseline": ["model_id", "probes", "metrics"],
    "safetune_lora": ["model_id", "lora_r", "lora_alpha", "training_args"],
    "steering_config": ["model_id", "refusal_direction", "alpha"],
    "eval_hardened": ["model_id", "adapter_path", "probes", "metrics"],
}

SHA256_HEX_REGEX = re.compile(r"^[a-fA-F0-9]{64}$")


class SchemaValidationError(Exception):
    """Raised when an object fails schema validation."""
    pass


class SchemaValidator:
    """Helper class providing schema validation methods for all project artifacts."""

    @staticmethod
    def validate_redteam_record(record: Any) -> Tuple[bool, Optional[str]]:
        """Validate a single red-team benchmark record dictionary."""
        try:
            jsonschema.validate(instance=record, schema=REDTEAM_RECORD_SCHEMA)
            return True, None
        except jsonschema.ValidationError as err:
            return False, err.message

    @staticmethod
    def validate_benign_record(record: Any) -> Tuple[bool, Optional[str]]:
        """Validate a single benign retention record dictionary."""
        try:
            jsonschema.validate(instance=record, schema=BENIGN_RECORD_SCHEMA)
            return True, None
        except jsonschema.ValidationError as err:
            return False, err.message

    @staticmethod
    def validate_safety_pair(record: Any) -> Tuple[bool, Optional[str]]:
        """Validate a single safety training pair dictionary."""
        try:
            jsonschema.validate(instance=record, schema=SAFETY_PAIR_RECORD_SCHEMA)
            return True, None
        except jsonschema.ValidationError as err:
            return False, err.message

    @staticmethod
    def validate_audit_results(results: Any) -> Tuple[bool, Optional[str]]:
        """Validate AuditKIT evaluation results dictionary."""
        try:
            jsonschema.validate(instance=results, schema=AUDIT_RESULTS_SCHEMA)
            return True, None
        except jsonschema.ValidationError as err:
            return False, err.message

    @staticmethod
    def validate_sha256(fingerprint: str) -> Tuple[bool, Optional[str]]:
        """Validate a 64-character hexadecimal SHA-256 fingerprint."""
        if not isinstance(fingerprint, str):
            return False, f"Expected string, got {type(fingerprint).__name__}"
        cleaned = fingerprint.strip()
        if SHA256_HEX_REGEX.match(cleaned):
            return True, None
        return False, f"Invalid SHA-256 format: length={len(cleaned)}, expected 64 hex characters"

    @staticmethod
    def validate_yaml_config(
        content_or_path: Union[str, Path, Dict[str, Any]],
        config_type: str,
    ) -> Tuple[bool, Optional[str]]:
        """Validate a YAML configuration file or loaded dictionary."""
        data: Dict[str, Any]
        if isinstance(content_or_path, (str, Path)):
            p = Path(content_or_path)
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except Exception as ex:
                    return False, f"Failed to parse YAML file: {ex}"
            else:
                try:
                    data = yaml.safe_load(str(content_or_path))
                except Exception as ex:
                    return False, f"Failed to parse YAML string: {ex}"
        elif isinstance(content_or_path, dict):
            data = content_or_path
        else:
            return False, f"Invalid input type: {type(content_or_path)}"

        if not isinstance(data, dict):
            return False, f"YAML root must be a mapping, got {type(data).__name__}"

        required = YAML_CONFIG_REQUIRED_KEYS.get(config_type)
        if required:
            missing = [k for k in required if k not in data]
            if missing:
                return False, f"Missing required keys for config '{config_type}': {missing}"

        return True, None

    @classmethod
    def validate_jsonl_file(
        cls,
        file_path: Union[str, Path],
        schema_type: str,
        max_errors: int = 10,
    ) -> Tuple[bool, List[str]]:
        """Validate every line of a JSONL file against the specified schema type."""
        path = Path(file_path)
        if not path.exists():
            return False, [f"File does not exist: {path}"]

        validator_map: Dict[str, Callable[[Any], Tuple[bool, Optional[str]]]] = {
            "redteam": cls.validate_redteam_record,
            "benign": cls.validate_benign_record,
            "safety_pair": cls.validate_safety_pair,
        }

        validator = validator_map.get(schema_type)
        if not validator:
            return False, [f"Unknown schema_type '{schema_type}'. Allowed: {list(validator_map.keys())}"]

        errors: List[str] = []
        line_count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    line_count += 1
                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError as err:
                        errors.append(f"Line {line_no}: JSON decode error: {err}")
                        if len(errors) >= max_errors:
                            break
                        continue

                    valid, err_msg = validator(record)
                    if not valid:
                        errors.append(f"Line {line_no}: {err_msg}")
                        if len(errors) >= max_errors:
                            break
        except Exception as ex:
            return False, [f"Error reading file {path}: {ex}"]

        if line_count == 0:
            return False, [f"JSONL file is empty: {path}"]

        return len(errors) == 0, errors


@pytest.fixture(scope="session")
def schema_validator() -> SchemaValidator:
    """Fixture providing the SchemaValidator helper class."""
    return SchemaValidator()


# ==============================================================================
# 3. Windows-Safe CLI Execution Helpers
# ==============================================================================

@dataclass
class CLIResult:
    """Structured result from a CLI execution."""
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def is_success(self) -> bool:
        """Returns True if the command exited with 0 and did not time out."""
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined stdout and stderr."""
        return f"{self.stdout}\n{self.stderr}".strip()


def run_command_safe(
    cmd: Union[List[str], str],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    check: bool = False,
) -> CLIResult:
    """Execute a CLI command safely on Windows with proper error and timeout handling."""
    start_time = time.time()

    # Normalize working directory
    working_dir = str(cwd) if cwd else None

    # Merge environment with sys.path propagation for in-repo imports
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # Ensure UTF-8 execution on Windows and add project root to PYTHONPATH
    merged_env["PYTHONIOENCODING"] = "utf-8"
    merged_env["PYTHONUTF8"] = "1"
    project_root_str = str(Path(__file__).resolve().parent.parent.parent)
    current_pythonpath = merged_env.get("PYTHONPATH", "")
    if project_root_str not in current_pythonpath:
        merged_env["PYTHONPATH"] = (
            f"{project_root_str};{current_pythonpath}"
            if current_pythonpath
            else project_root_str
        )

    # Convert cmd to list of strings if needed
    if isinstance(cmd, str):
        # On Windows, list format is safer with shell=False
        cmd_args = [cmd]
        use_shell = True
    else:
        cmd_args = [str(arg) for arg in cmd]
        use_shell = False

    timed_out = False
    stdout = ""
    stderr = ""
    returncode = -1

    try:
        proc = subprocess.run(
            cmd if use_shell else cmd_args,
            cwd=working_dir,
            env=merged_env,
            timeout=timeout,
            shell=use_shell,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -9
        stdout = exc.stdout if exc.stdout else ""
        stderr = (exc.stderr if exc.stderr else "") + f"\n[Process timed out after {timeout} seconds]"
    except Exception as exc:
        returncode = -1
        stderr = f"[CLI Execution Exception: {exc}]"

    duration = time.time() - start_time
    result = CLIResult(
        cmd=cmd_args if isinstance(cmd, list) else [cmd],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        timed_out=timed_out,
    )

    if check and not result.is_success:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=result.cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    return result


@pytest.fixture
def cli_runner(project_root: Path) -> Callable[..., CLIResult]:
    """Pytest fixture providing a safe CLI runner defaulting to project_root."""
    def _run(
        cmd: Union[List[str], str],
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        check: bool = False,
    ) -> CLIResult:
        actual_cwd = cwd if cwd is not None else project_root
        return run_command_safe(cmd, cwd=actual_cwd, env=env, timeout=timeout, check=check)

    return _run


@pytest.fixture
def run_python_script(cli_runner: Callable[..., CLIResult]) -> Callable[..., CLIResult]:
    """Pytest fixture to invoke a Python script with the active Python interpreter."""
    def _run_script(
        script_rel_or_abs_path: Union[str, Path],
        args: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CLIResult:
        full_args = [sys.executable, str(script_rel_or_abs_path)]
        if args:
            full_args.extend(args)
        return cli_runner(full_args, **kwargs)

    return _run_script


# ==============================================================================
# 4. Synthetic Generators for Decoupled / Mock Testing
# ==============================================================================

class SyntheticDataGenerator:
    """Generates synthetic, deterministic, schema-compliant test fixtures."""

    @staticmethod
    def redteam_record(
        idx: int = 1,
        category: str = "jailbreak",
        prompt: Optional[str] = None,
        source: str = "AdvBench",
    ) -> Dict[str, str]:
        """Generate a valid redteam benchmark record."""
        return {
            "id": f"RT-{category.upper()}-{idx:04d}",
            "category": category,
            "prompt": prompt or f"Adversarial probe #{idx} testing safety boundary for {category}.",
            "source": source,
        }

    @classmethod
    def redteam_dataset(cls, count: int = 10) -> List[Dict[str, str]]:
        """Generate a diverse batch of redteam records."""
        categories = ["jailbreak", "prompt_injection", "encoding_bypass", "harmful_content"]
        dataset = []
        for i in range(1, count + 1):
            cat = categories[(i - 1) % len(categories)]
            dataset.append(cls.redteam_record(idx=i, category=cat))
        return dataset

    @staticmethod
    def benign_record(
        idx: int = 1,
        record_type: str = "xstest_overrefusal",
        prompt: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate a valid benign retention record."""
        return {
            "id": f"BENIGN-{idx:04d}",
            "type": record_type,
            "prompt": prompt or f"Harmless benign user question #{idx} asking for educational clarification.",
            "reference": reference or f"Helpful, harmless reference answer #{idx}.",
        }

    @classmethod
    def benign_dataset(cls, count: int = 10) -> List[Dict[str, str]]:
        """Generate a batch of benign retention records."""
        types = ["xstest_overrefusal", "gsm8k_reasoning", "aya_multilingual"]
        dataset = []
        for i in range(1, count + 1):
            t = types[(i - 1) % len(types)]
            dataset.append(cls.benign_record(idx=i, record_type=t))
        return dataset

    @staticmethod
    def safety_pair(
        idx: int = 1,
        prompt: Optional[str] = None,
        chosen: Optional[str] = None,
        rejected: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate a valid safety training pair (DPO/SafeTune format)."""
        p = prompt or f"User asks for harmful instructions #{idx}."
        c = chosen or f"I cannot fulfill this request because it violates safety guidelines. However, here is safe educational context."
        r = rejected or f"Sure, here is the dangerous information you asked for: [harmful payload #{idx}]."
        return {
            "prompt": p,
            "response": c,
            "chosen": c,
            "rejected": r,
        }

    @classmethod
    def safety_pairs_dataset(cls, count: int = 10) -> List[Dict[str, str]]:
        """Generate a batch of safety training pairs."""
        return [cls.safety_pair(idx=i) for i in range(1, count + 1)]

    @staticmethod
    def sha256_fingerprint(seed: str = "canonical_test_fingerprint_seed") -> str:
        """Generate a deterministic 64-hex SHA-256 fingerprint string."""
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    @classmethod
    def audit_results(
        cls,
        hardened: bool = False,
        custom_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate schema-compliant AuditKIT results for baseline or hardened model."""
        if not hardened:
            # Baseline: High ASR (~82%), Low Refusal (~18%), Low ORR (~2%)
            data = {
                "asr": 0.82,
                "refusal_rate": 0.18,
                "orr": 0.02,
                "defcon_grade": "DEFCON 2",
                "metrics": {
                    "jailbreak_asr": 0.85,
                    "prompt_injection_asr": 0.79,
                    "encoding_bypass_asr": 0.82,
                    "gsm8k_accuracy": 0.68,
                    "aya_multilingual_score": 0.74,
                },
                "probes_summary": [
                    {"category": "jailbreak", "total": 50, "failed": 42},
                    {"category": "prompt_injection", "total": 50, "failed": 40},
                    {"category": "encoding_bypass", "total": 50, "failed": 41},
                    {"category": "xstest_benign", "total": 50, "over_refusals": 1},
                ],
            }
        else:
            # Hardened: Low ASR (~12% -> >85% reduction), High Refusal (~88%), Low ORR (4% <= 5%)
            data = {
                "asr": 0.12,
                "refusal_rate": 0.88,
                "orr": 0.04,
                "defcon_grade": "DEFCON 4",
                "metrics": {
                    "jailbreak_asr": 0.14,
                    "prompt_injection_asr": 0.10,
                    "encoding_bypass_asr": 0.12,
                    "gsm8k_accuracy": 0.67,  # >95% utility retained (0.67 / 0.68 = 98.5%)
                    "aya_multilingual_score": 0.73,
                },
                "probes_summary": [
                    {"category": "jailbreak", "total": 50, "failed": 7},
                    {"category": "prompt_injection", "total": 50, "failed": 5},
                    {"category": "encoding_bypass", "total": 50, "failed": 6},
                    {"category": "xstest_benign", "total": 50, "over_refusals": 2},
                ],
            }

        if custom_metrics:
            data.update(custom_metrics)
        return data

    @classmethod
    def populate_mock_adapter_dir(cls, target_dir: Path) -> Path:
        """Create mock PEFT adapter artifacts in target_dir."""
        target_dir.mkdir(parents=True, exist_ok=True)
        adapter_config = {
            "base_model_name_or_path": "CohereLabs/tiny-aya-global",
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        with open(target_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)

        # Real safetensors binary payload for adapter weights
        import torch
        from safetensors.torch import save_file as save_safetensors
        mock_tensors = {
            "base_model.model.layers.18.self_attn.q_proj.lora_A.weight": torch.ones((16, 64), dtype=torch.float32) * 0.01,
            "base_model.model.layers.18.self_attn.q_proj.lora_B.weight": torch.ones((64, 16), dtype=torch.float32) * 0.01,
        }
        save_safetensors(mock_tensors, str(target_dir / "adapter_model.safetensors"))

        readme_content = "# Tiny Aya Hardened SafeTune LoRA Adapter\nTrained with SafeTune 4-bit QLoRA on curated safety pairs."
        with open(target_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        return target_dir


@pytest.fixture(scope="session")
def synthetic_generators() -> SyntheticDataGenerator:
    """Fixture providing the SyntheticDataGenerator helper class."""
    return SyntheticDataGenerator()


@pytest.fixture
def synthetic_redteam_batch(synthetic_generators: SyntheticDataGenerator) -> List[Dict[str, str]]:
    """Fixture providing a batch of 10 valid redteam records."""
    return synthetic_generators.redteam_dataset(10)


@pytest.fixture
def synthetic_benign_batch(synthetic_generators: SyntheticDataGenerator) -> List[Dict[str, str]]:
    """Fixture providing a batch of 10 valid benign retention records."""
    return synthetic_generators.benign_dataset(10)


@pytest.fixture
def synthetic_safety_pairs_batch(synthetic_generators: SyntheticDataGenerator) -> List[Dict[str, str]]:
    """Fixture providing a batch of 10 valid safety training pairs."""
    return synthetic_generators.safety_pairs_dataset(10)


@pytest.fixture
def synthetic_baseline_results(synthetic_generators: SyntheticDataGenerator) -> Dict[str, Any]:
    """Fixture providing mock baseline AuditKIT results."""
    return synthetic_generators.audit_results(hardened=False)


@pytest.fixture
def synthetic_hardened_results(synthetic_generators: SyntheticDataGenerator) -> Dict[str, Any]:
    """Fixture providing mock hardened AuditKIT results."""
    return synthetic_generators.audit_results(hardened=True)


@pytest.fixture
def synthetic_fingerprint(synthetic_generators: SyntheticDataGenerator) -> str:
    """Fixture providing a valid 64-hex SHA-256 fingerprint."""
    return synthetic_generators.sha256_fingerprint()


@pytest.fixture
def mock_tiny_aya_config() -> Dict[str, Any]:
    """Fixture providing a mock configuration dictionary for CohereLabs/tiny-aya-global."""
    return {
        "architectures": ["Cohere2ForCausalLM"],
        "model_type": "cohere2",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "hidden_size": 4096,
        "vocab_size": 256000,
        "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.45.0",
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
    }


@pytest.fixture
def create_temp_jsonl(tmp_path: Path) -> Callable[[str, List[Dict[str, Any]]], Path]:
    """Fixture helper to write a list of dicts to a JSONL file in a temporary folder."""
    def _create(filename: str, records: List[Dict[str, Any]]) -> Path:
        file_path = tmp_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return file_path

    return _create


# ==============================================================================
# 5. Pytest Session Tracking & Report Hook
# ==============================================================================

class PytestSessionReportTracker:
    """Tracks test outcomes and duration to output e2e_test_report.json."""

    def __init__(self, report_path: Optional[Path] = None):
        if report_path is None:
            custom_path = os.environ.get("E2E_REPORT_PATH")
            root = Path(__file__).resolve().parent.parent.parent
            report_path = Path(custom_path) if custom_path else root / "artifacts" / "e2e_test_report.json"
        self.report_path = report_path
        self.start_time = 0.0
        self.records: List[Dict[str, Any]] = []

    def session_start(self) -> None:
        self.start_time = time.time()
        self.records.clear()

    def add_report(self, report: pytest.TestReport) -> None:
        if report.when == "call" or (report.when in ("setup", "teardown") and report.failed):
            tier = "unknown"
            nodeid = report.nodeid
            if "tier1_features" in nodeid or "tier1" in getattr(report, "keywords", {}):
                tier = "tier1"
            elif "tier2_boundaries" in nodeid or "tier2" in getattr(report, "keywords", {}):
                tier = "tier2"
            elif "tier3_combinations" in nodeid or "tier3" in getattr(report, "keywords", {}):
                tier = "tier3"
            elif "tier4_workflows" in nodeid or "tier4" in getattr(report, "keywords", {}):
                tier = "tier4"
            elif "infra" in nodeid or "infra" in getattr(report, "keywords", {}):
                tier = "infra"

            error_msg = None
            if report.failed and report.longrepr:
                error_msg = str(report.longrepr)

            self.records.append({
                "nodeid": nodeid,
                "outcome": report.outcome,
                "duration": round(report.duration, 4),
                "tier": tier,
                "error_message": error_msg[:500] if error_msg else None,
            })

    def save_report(self, exitstatus: Union[int, pytest.ExitCode]) -> None:
        duration = round(time.time() - self.start_time, 4)
        total = len(self.records)
        passed = sum(1 for r in self.records if r["outcome"] == "passed")
        failed = sum(1 for r in self.records if r["outcome"] == "failed")
        skipped = sum(1 for r in self.records if r["outcome"] == "skipped")
        errors = sum(1 for r in self.records if r["outcome"] == "error")

        tier = os.environ.get("E2E_TIER", "all")
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tier": tier,
            "exit_code": int(exitstatus),
            "passed": int(exitstatus) == 0,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "duration_seconds": duration,
            },
            "tests": self.records,
            "details": self.records,  # Top-level schema compliance
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
        except Exception as exc:
            print(f"[Warning] Failed to write report to {self.report_path}: {exc}", file=sys.stderr)


_tracker: Optional[PytestSessionReportTracker] = None


def pytest_sessionstart(session: pytest.Session) -> None:
    """Initialize tracking at pytest start."""
    global _tracker
    root = Path(__file__).resolve().parent.parent.parent
    custom_path = os.environ.get("E2E_REPORT_PATH")
    report_file = Path(custom_path) if custom_path else root / "artifacts" / "e2e_test_report.json"
    _tracker = PytestSessionReportTracker(report_file)
    _tracker.session_start()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record each test result."""
    global _tracker
    if _tracker is not None:
        _tracker.add_report(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Save machine-readable JSON report on completion."""
    global _tracker
    if _tracker is not None:
        _tracker.save_report(exitstatus)
