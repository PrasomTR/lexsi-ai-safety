"""CLI Entrypoint for the Interactive Demonstration Web Application.

Launches the zero-dependency Python web server supporting Live Safety Playground,
CircuitKIT D3 Explorer, Real-Time Steering Calibration, and Provenance Explorer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.server import create_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Lexsi AI Safety Hackathon Interactive Demonstration Server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind server to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port number to listen on (default: 8501).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless server mode without attempting browser launch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate server binding and socket initialization, then exit immediately.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[launch_ui] Initializing server on {args.host}:{args.port} (headless={args.headless})")

    try:
        server = create_server(host=args.host, port=args.port)
    except Exception as e:
        print(f"[launch_ui] Failed to bind server: {e}", file=sys.stderr)
        return 1

    print(f"[launch_ui] Server successfully listening at http://{args.host}:{args.port}/")

    if args.dry_run:
        print("[launch_ui] Dry-run mode completed successfully. Exiting cleanly.")
        server.server_close()
        return 0

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[launch_ui] Server interrupted by user. Shutting down...")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
