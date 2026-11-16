"""Real, reproducible safety pipeline (no mock, no hard-coded metrics).

Modules:
    refusal.py  multilingual refusal detection (8 languages)
    model.py    transformers runner: generation + activation capture (CPU friendly)
    audit.py    ASR / ORR / utility computed from completions
    probe.py    refusal direction, layer localisation, probe transfer
    steering.py residual-stream activation steering via forward hooks
"""

from src.pipeline.audit import Auditor, Probe, defcon_grade, load_benign, load_redteam
from src.pipeline.refusal import RefusalDetector, detect_language

__all__ = [
    "Auditor",
    "Probe",
    "RefusalDetector",
    "defcon_grade",
    "detect_language",
    "load_benign",
    "load_redteam",
]
