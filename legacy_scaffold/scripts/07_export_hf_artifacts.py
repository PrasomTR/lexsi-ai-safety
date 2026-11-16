"""Stage 07: Hugging Face Artifact Collection Export.

Packages trained model adapters, benchmark evaluation datasets, and differential
audit logs into structured Hugging Face repositories with dataset/model cards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hf_export import HFCollectionExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package project deliverables into Hugging Face collection repositories."
    )
    parser.add_argument(
        "--source-artifacts",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Path to source artifacts directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Path to curated data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "hf_collection",
        help="Destination directory for Hugging Face collection packages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run export packaging in fast decoupled dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exporter = HFCollectionExporter()

    print(f"[07_export_hf_artifacts] Packaging artifacts from: {args.source_artifacts}")
    print(f"[07_export_hf_artifacts] Export destination: {args.output_dir}")

    manifest = exporter.export_collection(
        artifacts_dir=args.source_artifacts,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )

    print(f"[07_export_hf_artifacts] Packaging complete. Exported files: {len(manifest.get('files', {}))}")
    print(f"[07_export_hf_artifacts] Manifest saved to: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
