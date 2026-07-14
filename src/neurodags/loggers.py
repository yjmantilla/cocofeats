# loggers.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog

_CONFIGURED = False


def _coerce_level(level: str | int) -> int:
    """
    Convert a textual or numeric level to a stdlib logging level.

    Parameters
    ----------
    level : str or int
        Either a string such as ``"INFO"``/``"DEBUG"`` or a numeric level.

    Returns
    -------
    int
        A valid stdlib logging level (e.g., ``logging.INFO``).

    Raises
    ------
    ValueError
        If ``level`` is an unknown string.
    """
    if isinstance(level, int):
        return level
    lvl = logging.getLevelName(level.upper())
    # logging.getLevelName returns int for known names, str otherwise
    if isinstance(lvl, int):
        return lvl
    raise ValueError(f"Unknown log level: {level!r}")


def configure_logging(
    *,
    json: bool | None = None,
    level: str | int | None = None,
    route_stdlib: bool = False,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure structlog and stdlib logging once.

    This sets up a consistent logging pipeline. By default, structlog logs are
    rendered as pretty console output (TTY) or JSON (non-TTY). When ``log_file``
    is provided, all events are *also* written to that file in JSONL format
    (one JSON object per line), which can be loaded directly as a dataframe:

    .. code-block:: python

        import pandas as pd
        df = pd.read_json("run.jsonl", lines=True)

    Parameters
    ----------
    json : bool, optional
        Force JSON output on the console. If ``None`` (default), JSON is used
        when stdout is not a TTY or when the environment variable
        ``LOG_FMT=json`` is set.
    level : str or int, optional
        Global log level, e.g., ``"WARNING"`` or ``logging.WARNING``.
        Defaults to the value of ``$LOG_LEVEL`` or ``"INFO"``.
    route_stdlib : bool, optional
        Route stdlib logs through structlog's renderer for uniform formatting.
    log_file : str or Path, optional
        If given, also write all log events to this file in JSONL format.
        The file is opened in append mode.

    Returns
    -------
    None

    Notes
    -----
    - This function is idempotent and returns immediately on subsequent calls.
    - When ``log_file`` is set, ``route_stdlib`` is forced to ``True`` so that
      structlog events reach both the console and file handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    level = _coerce_level(level)
    os.environ["LOG_LEVEL"] = logging.getLevelName(level)

    if json is None:
        env_fmt = os.getenv("LOG_FMT", "").lower()
        if env_fmt == "json":
            json = True
        elif env_fmt == "console":
            json = False
        else:
            # Logs go to stderr, so base the pretty/JSON choice on stderr's TTY.
            json = not sys.stderr.isatty()

    if log_file is not None:
        route_stdlib = True

    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    if route_stdlib:
        from structlog.stdlib import ProcessorFormatter

        # Shared pre-processing chain for both structlog and stdlib (foreign) events
        shared = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        # Console handler — pretty or JSON depending on context
        console_render = (
            structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
        )
        # Route logs to stderr so stdout carries only command output (tables,
        # --format json, Mermaid). See #15.
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            ProcessorFormatter(
                foreign_pre_chain=shared,
                processors=[ProcessorFormatter.remove_processors_meta, console_render],
            )
        )
        root.addHandler(console_handler)

        # File handler — always JSONL
        if log_file is not None:
            fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(
                ProcessorFormatter(
                    foreign_pre_chain=shared,
                    processors=[
                        ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),
                    ],
                )
            )
            root.addHandler(fh)

        structlog.configure(
            processors=[*shared, ProcessorFormatter.wrap_for_formatter],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Simple path: structlog renders directly to stderr; no file output.
        # stderr keeps stdout clean for command output (see #15).
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))

        render = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.filter_by_level,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.PositionalArgumentsFormatter(),
                render,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str | None = None, **bind):
    """
    Get a bound structlog logger.

    Parameters
    ----------
    name : str or None, optional
        Logger name. If ``None``, uses the current module name.
    **bind
        Key/value pairs to bind immediately to the logger's context.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A logger that renders via the configured structlog pipeline.
    """
    log = structlog.get_logger(name or __name__)
    return log.bind(**bind) if bind else log


def _bootstrap_quiet_default() -> None:
    """Route structlog to stderr at a sane default *before* ``configure_logging``.

    Import-time log calls — most notably the built-in derivative registration
    that runs when ``neurodags.derivatives.pipeline`` is imported — would
    otherwise hit structlog's unconfigured default, which prints every level to
    **stdout** and pollutes command output (breaking ``--format json`` piping;
    see #15). This installs a stderr-routed, level-filtered structlog default so
    that chatter is filtered and never lands on stdout.

    It configures only structlog (no stdlib root handler) and does **not** set
    ``_CONFIGURED``, so :func:`configure_logging` still runs and fully
    reconfigures logging when the CLI/app starts.
    """
    level = _coerce_level(os.getenv("LOG_LEVEL", "INFO"))
    render = (
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            render,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


_bootstrap_quiet_default()
