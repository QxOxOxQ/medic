from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_NAME_COLOR = "\x1b[35m"
_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",
    "INFO": "\x1b[32m",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
    "CRITICAL": "\x1b[1;37;41m",
}

_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "openai", "qdrant_client")
_DEFAULT_LOG_FILE = Path("logs/medic.log")


class ColorFormatter(logging.Formatter):
    """Human-friendly, optionally ANSI-colored log formatter."""

    def __init__(self, *, datefmt: str, use_color: bool) -> None:
        super().__init__(datefmt=datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname
        message = record.getMessage()
        if self._use_color:
            color = _LEVEL_COLORS.get(level, "")
            line = (
                f"{_DIM}{timestamp}{_RESET} "
                f"{color}{level:<8}{_RESET} "
                f"{_NAME_COLOR}{record.name}{_RESET} "
                f"{color}{message}{_RESET}"
            )
        else:
            line = f"{timestamp} {level:<8} {record.name} {message}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line


def configure_logging(
    *,
    level: int | None = None,
    log_file: str | Path | None = None,
    use_color: bool | None = None,
) -> Path | None:
    """Configure colorful console + detailed rotating file logging.

    Returns the active log file path, or ``None`` if file logging is disabled.
    Controlled by env vars: ``MEDIC_LOG_LEVEL``, ``MEDIC_LOG_FILE``,
    ``MEDIC_LOG_COLOR`` (``0``/``false`` to disable colors).
    """

    resolved_level = level if level is not None else _resolve_level()
    resolved_color = use_color if use_color is not None else _resolve_color()
    resolved_file = Path(log_file) if log_file is not None else _resolve_log_file()

    root = logging.getLogger()
    root.setLevel(resolved_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(ColorFormatter(datefmt="%H:%M:%S", use_color=resolved_color))
    root.addHandler(console)

    file_handler = _build_file_handler(resolved_file, use_color=resolved_color)
    if file_handler is not None:
        root.addHandler(file_handler)
        root.info("Detailed logs are written to %s", resolved_file)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return resolved_file if file_handler is not None else None


def _build_file_handler(path: Path, *, use_color: bool) -> logging.Handler | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            path,
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
    except OSError:
        logging.getLogger("medic").warning(
            "Could not open log file at %s; file logging disabled", path
        )
        return None
    handler.setFormatter(
        ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S", use_color=use_color)
    )
    return handler


def _resolve_level() -> int:
    raw = os.getenv("MEDIC_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw, logging.INFO)
    return level if isinstance(level, int) else logging.INFO


def _resolve_color() -> bool:
    raw = os.getenv("MEDIC_LOG_COLOR")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_log_file() -> Path:
    raw = os.getenv("MEDIC_LOG_FILE")
    return Path(raw) if raw else _DEFAULT_LOG_FILE
