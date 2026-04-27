#*************************************************************************/
# File Name: ./src/thermoviz/cli.py
# Author: Beomjun Chung
# Updated: 2026-03-22
#
# Description:
#   cli entrypoint for thermoviz
#
# Example:
#       py -m thermoviz.cli --config config/config.yaml --dry-run --log-level DEBUG
#*************************************************************************/

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from quadrl.main import run

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermoviz",
        description="Thermoviz CLI",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config.yaml",
    )

    parser.add_argument(
        "--sim-config",
        type=Path,
        default=Path("config/sim_config.yaml"),
        help="Optional path to sim_config.yaml. If omitted, config loading resolves the default.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print execution info without running analysis",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console log level",
    )

    return parser

def cli_main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    run(
        config_path=args.config,
        sim_config_path=args.sim_config,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )

if __name__ == "__main__":
    cli_main()