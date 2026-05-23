"""Structured logging via loguru.

One configuration entrypoint for the whole project. CLI commands and tests
call ``configure_logging`` exactly once at startup. Library modules just do
``from loguru import logger`` and log normally.

Why loguru and not stdlib logging:
- Tracebacks are captured automatically with ``logger.exception``.
- No silent failures: any uncaught exception in a logged call is itself logged.
- Structured fields via ``logger.bind(...)`` show up in JSON output.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | None = None,
    json: bool = False,
) -> None:
    """Configure the global loguru logger.

    Args:
        level: Minimum level. One of DEBUG, INFO, WARNING, ERROR.
        log_dir: If given, also write rotating logs to ``{log_dir}/engineering-codes-rag.log``.
        json: Emit JSON records to stderr (useful for piping to a log shipper).
    """
    logger.remove()  # drop loguru's default handler

    fmt_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "<level>{level: <8}</level> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        level=level,
        format=fmt_console,
        backtrace=True,
        diagnose=False,  # keep variable values out of logs (avoid leaking secrets)
        serialize=json,
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "engineering-codes-rag.log",
            level=level,
            rotation="20 MB",
            retention=10,
            compression="zip",
            backtrace=True,
            diagnose=False,
            enqueue=True,  # process-safe
            serialize=True,  # always JSON on disk
        )


__all__ = ["configure_logging", "logger"]
