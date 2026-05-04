from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from quadrl.main import run

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quadrl",
        description="Quadruped Residual RL — training entry point",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without running training",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console log verbosity (default: INFO)",
    )
    return parser


def cli_main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run(
        config_path=args.config,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    cli_main()
