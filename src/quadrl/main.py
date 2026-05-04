from __future__ import annotations

from pathlib import Path

from quadrl.utils.config import load_config
from quadrl.utils.logger import setup_logging, get_logger


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
    cfg = load_config(config_path)
    setup_logging(level=log_level, log_dir=cfg.logging.export_dir)
    log = get_logger("main")

    log.info("Config loaded from '%s'", config_path)

    if dry_run:
        log.info("[dry-run] Config is valid. Exiting without training.")
        log.debug("Resolved config: %s", cfg)
        return

    # Defer heavy imports (mujoco, torch) until after logging is ready
    from quadrl.training.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()
