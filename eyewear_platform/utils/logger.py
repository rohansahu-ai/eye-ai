"""
Structured logging using structlog.
All log output is JSON in production, colored in development.
"""
import logging
import os
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Call once at app startup to configure structlog."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Determine if we're in a TTY (dev) or not (prod/CI)
    is_tty = sys.stdout.isatty()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if is_tty:
        # Colored, human-readable output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON for production
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to match
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a module-level structured logger."""
    return structlog.get_logger().bind(logger=name)


if __name__ == "__main__":
    configure_logging("DEBUG")
    logger = get_logger(__name__)
    logger.info("Logger configured", module=__name__)
    logger.warning("Test warning", key="value")
    logger.error("Test error", exception="None")
