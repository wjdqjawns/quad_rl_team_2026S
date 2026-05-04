from __future__ import annotations

import csv
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── ANSI colors ────────────────────────────────────────────────────────────────
_COLOR = {
    "DEBUG":    "\033[36m",    # cyan
    "INFO":     "\033[32m",    # green
    "WARNING":  "\033[33m",    # yellow
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[35;1m",  # bold magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"

_LOG_FMT_PLAIN = "[{asctime}] [{levelname:<8s}] {name}: {message}"
_DATE_CONSOLE  = "%H:%M:%S"
_DATE_FILE     = "%Y-%m-%d %H:%M:%S"


class _ColorFormatter(logging.Formatter):
    """Console formatter with ANSI level colors."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLOR.get(record.levelname, "")
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{color}{_BOLD}{record.levelname}{_RESET}"
        fmt = logging.Formatter(_LOG_FMT_PLAIN, datefmt=_DATE_CONSOLE, style="{")
        return fmt.format(record)


class _PlainFormatter(logging.Formatter):
    """File formatter without ANSI codes."""

    def __init__(self) -> None:
        super().__init__(_LOG_FMT_PLAIN, datefmt=_DATE_FILE, style="{")


# ── Public API ─────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> None:
    """Initialize the quadrl package logger.

    Call once at program startup (in main.py / cli.py).

    Args:
        level:   Console verbosity (DEBUG / INFO / WARNING / ERROR / CRITICAL).
        log_dir: Directory for rotating log files.
                 None → file logging disabled.
    """
    root = logging.getLogger("quadrl")
    if root.handlers:
        return  # already configured; skip re-init

    root.setLevel(logging.DEBUG)   # let handlers filter

    # Console ─────────────────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(_ColorFormatter())
    root.addHandler(console)

    # Rotating file ────────────────────────────────────────────────────────────
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fh = logging.handlers.RotatingFileHandler(
            log_path / f"{stamp}.log",
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_PlainFormatter())
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``quadrl`` namespace.

    Args:
        name: Dotted submodule path, e.g. ``"envs.go1"`` or ``"training.trainer"``.
    """
    return logging.getLogger(f"quadrl.{name}")


# ── Metric logger ──────────────────────────────────────────────────────────────

class MetricLogger:
    """Logs scalar training metrics to a CSV file (and optionally TensorBoard).

    Usage::

        ml = MetricLogger(log_dir="export/temp/log", use_tensorboard=True)
        ml.log(step=1000, policy_loss=0.12, value_loss=0.34, fps=2500)
        ml.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str | None = None,
        use_tensorboard: bool = False,
    ) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = f"{run_name}_{stamp}" if run_name else stamp

        # CSV
        self._csv_path = log_dir / f"{prefix}_metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer: csv.DictWriter | None = None   # created on first log()

        # TensorBoard (optional)
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self._tb_writer = SummaryWriter(log_dir=str(log_dir / prefix))
            except ImportError:
                logging.getLogger("quadrl.utils.logger").warning(
                    "TensorBoard not installed; skipping TB logging."
                )

    def log(self, step: int, **metrics: Any) -> None:
        """Record metrics at a given training step.

        Args:
            step:    Global environment step count.
            **metrics: Arbitrary scalar key-value pairs.
        """
        row = {"step": step, **{k: v for k, v in metrics.items()}}

        # CSV – write header on first call
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # TensorBoard
        if self._tb_writer is not None:
            for key, val in metrics.items():
                self._tb_writer.add_scalar(key, val, global_step=step)

    def close(self) -> None:
        """Flush and close all writers."""
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()