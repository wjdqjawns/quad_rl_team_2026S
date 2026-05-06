#*************************************************************************/
# File Name: ./src/quadrl/main.py
# Author: Beomjun Chung
# Updated: 2026-05-05
#
# Description:
#   QuadRL main training entry point. Initializes components and starts training.
#*************************************************************************/

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from quadrl import __version__, __project_name__, __author__, __updated__
from quadrl.utils.logger import setup_logging
from quadrl.utils.config import load_config
from quadrl.

_PROGRAM_VERSION = __version__
_PROGRAM_NAME = __project_name__
_PROGRAM_AUTHOR = __author__
_PROGRAM_UPDATED = __updated__

def run(
    config_path: Path,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> None:
    """Application entry point.

    Args:
        config_path: Path to the YAML config file.
        dry_run:     Validate config and exit without training.
        log_level:   Console log verbosity.
    """
    logger = setup_logging(level=log_level, config=None, start_time=None)

    try:
        cfg = load_config(config_path)
        logger.info("Config loaded from '%s'", config_path)

        if dry_run:
            logger.info("[dry-run] Config is valid. Exiting without training.")
            logger.debug("Resolved config: %s", cfg)
            return

        # Defer heavy imports (mujoco, torch) until after logging is ready

        from quadrl.training.trainer import Trainer
        trainer = Trainer(cfg)
        trainer.train()

    except Exception as e: