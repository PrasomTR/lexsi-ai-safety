"""Stage 02: Benchmark Dataset Curation and Provenance Pipeline.

Curates adversarial red-teaming benchmarks, benign retention sets, and safety training pairs
using CuratorKIT hygiene filters and generates cryptographic provenance manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.curation import CuratorKitScrubber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate adversarial benchmark and benign evaluation datasets using CuratorKIT."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw candidates JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Output directory for curated datasets.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Destination path for provenance manifest JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run curation in fast decoupled dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scrubber = CuratorKitScrubber()

    raw_records = None
    if args.input and args.input.exists():
        with open(args.input, "r", encoding="utf-8") as f:
            raw_records = [json.loads(line) for line in f if line.strip()]

    output_dir = args.output_dir
    manifest_path = args.manifest_path or (output_dir / "curator_manifest.json")

    print(f"[02_curate_benchmark] Curating red-teaming benchmark to: {output_dir}")
    scrubbed, manifest = scrubber.curate_benchmark_pipeline(
        input_records=raw_records,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )

    print(f"[02_curate_benchmark] Successfully curated {len(scrubbed)} records.")
    print(f"[02_curate_benchmark] SHA-256 Provenance Hash: {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
