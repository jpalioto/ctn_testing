"""Logging infrastructure for CTN testing."""

import os
import sys
import traceback
from datetime import datetime
from enum import IntEnum
from typing import Any


class LogLevel(IntEnum):
    """Log levels in increasing verbosity."""

    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3
    TRACE = 4


class Logger:
    """
    Configurable logger with pretty exception printing.

    Usage:
        log = Logger(level=LogLevel.DEBUG)
        log.info("Starting extraction")
        log.debug("Raw response", data=response[:200])
        log.error("Parse failed", exc=e, context={"text": text[:100]})
    """

    _instance: "Logger | None" = None

    def __init__(self, level: LogLevel | str = LogLevel.INFO, use_timestamps: bool = False):
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        self._level = level
        self._use_timestamps = use_timestamps

    @classmethod
    def get(cls) -> "Logger":
        """Get or create singleton logger."""
        if cls._instance is None:
            # Check environment for log level
            env_level = os.environ.get("CTN_LOG_LEVEL", "INFO")
            cls._instance = cls(level=env_level)
        return cls._instance

    @classmethod
    def configure(cls, level: LogLevel | str, use_timestamps: bool = False):
        """Configure the singleton logger."""
        cls._instance = cls(level=level, use_timestamps=use_timestamps)

    @property
    def level(self) -> LogLevel:
        return self._level

    def _format_prefix(self, level: LogLevel) -> str:
        """Format log prefix."""
        prefix = f"[{level.name}]"
        if self._use_timestamps:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            prefix = f"[{ts}] {prefix}"
        return prefix

    def _format_data(self, data: Any, indent: int = 2) -> str:
        """Format data for display."""
        if data is None:
            return ""

        import json

        try:
            if isinstance(data, (dict, list)):
                return json.dumps(data, indent=indent, default=str)
            return str(data)
        except Exception:
            return repr(data)

    def _log(self, level: LogLevel, msg: str, **kwargs):
        """Core logging method."""
        if level > self._level:
            return

        prefix = self._format_prefix(level)

        # Build output
        output = f"{prefix} {msg}"

        # Add any extra data
        data = kwargs.get("data")
        if data is not None:
            formatted = self._format_data(data)
            if "\n" in formatted:
                output += f"\n{formatted}"
            else:
                output += f" | {formatted}"

        # Add context dict
        context = kwargs.get("context")
        if context:
            output += f"\n  Context: {self._format_data(context)}"

        # Handle exception
        exc = kwargs.get("exc")
        if exc is not None:
            output += f"\n  Exception: {type(exc).__name__}: {exc}"
            if self._level >= LogLevel.DEBUG:
                tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
                output += "\n  Traceback:\n    " + "    ".join(tb)

        print(output, file=sys.stderr if level <= LogLevel.WARN else sys.stdout)

    def error(self, msg: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, msg, **kwargs)

    def warn(self, msg: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARN, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, msg, **kwargs)

    def trace(self, msg: str, **kwargs):
        """Log trace message (very verbose)."""
        self._log(LogLevel.TRACE, msg, **kwargs)


# Convenience functions using singleton
def get_logger() -> Logger:
    """Get the singleton logger."""
    return Logger.get()


def configure_logging(level: LogLevel | str = LogLevel.INFO, use_timestamps: bool = False):
    """Configure global logging."""
    Logger.configure(level, use_timestamps)
