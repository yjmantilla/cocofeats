# loggers.py
from __future__ import annotations

import logging
import os
import sys
import structlog

_CONFIGURED = False

def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    lvl = logging.getLevelName(level.upper())
    # logging.getLevelName returns int for known names, str otherwise
    if isinstance(lvl, int):
        return lvl
    raise ValueError(f"Unknown log level: {level!r}")

def configure_logging(*, json: bool | None = None, level: str | int | None = None) -> None:
    """
    Configure structlog + stdlib logging once.

    Args:
        json: Force JSON output (default: True if not a TTY or env LOG_FMT=json).
        level: Log level (default: INFO or env LOG_LEVEL).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # ---- defaults from environment / context ----
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    level = _coerce_level(level)

    if json is None:
        # Prefer JSON in non-TTY (batch/HPC) or when explicitly requested
        json = (os.getenv("LOG_FMT", "json").lower() == "json") or (not sys.stdout.isatty())

    # ---- stdlib logging baseline ----
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(level)

    # Remove pre-existing handlers to avoid duplicates (e.g., in notebooks or reloads)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Simple passthrough; structlog will render final message
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)

    # ---- structlog processors ----
    processors = [
        structlog.contextvars.merge_contextvars,            # include contextvars if used
        structlog.stdlib.filter_by_level,                   # drop events below level early
        structlog.processors.add_log_level,                 # add 'level' field (keep this one)
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,               # clean tracebacks
        structlog.stdlib.add_logger_name,                   # logger name field
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    render = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=processors + [render],
        wrapper_class=structlog.make_filtering_bound_logger(level),  # filter in structlog too
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None, **bind):
    """
    Convenience helper to get a bound structlog logger.
    """
    log = structlog.get_logger(name or __name__)
    return log.bind(**bind) if bind else log


# --- Optional: unify stdlib logs through structlog rendering (advanced) ---
# If you want *all* stdlib logs (from libraries) to go through structlog’s renderer,
# replace the handler formatter above with a ProcessorFormatter and add these lines:
#
# from structlog.stdlib import ProcessorFormatter
# pf = ProcessorFormatter(
#     foreign_pre_chain=[
#         structlog.contextvars.merge_contextvars,
#         structlog.stdlib.add_logger_name,
#         structlog.stdlib.add_log_level,
#         structlog.processors.TimeStamper(fmt="iso", utc=True),
#         structlog.processors.format_exc_info,
#     ],
#     processors=[render],  # same renderer picked above (JSON or Console)
# )
# handler.setFormatter(pf)
# structlog.configure(
#     processors=[
#         structlog.contextvars.merge_contextvars,
#         structlog.stdlib.filter_by_level,
#         ProcessorFormatter.remove_processors_meta,  # hand off to stdlib formatter
#     ],
#     wrapper_class=structlog.make_filtering_bound_logger(level),
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     cache_logger_on_first_use=True,
# )
