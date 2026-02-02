"""
Structured logging configuration with correlation ID support.

Provides JSON-formatted logs for production and human-readable logs for development.
Each request gets a unique correlation ID for distributed tracing.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import Processor

from app.core.config import get_settings

# Context variable for correlation ID per request
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """
    Get the current request's correlation ID.
    
    Returns:
        str: Correlation ID for the current request context.
    """
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """
    Set correlation ID for the current request context.
    
    Args:
        correlation_id: Optional existing correlation ID. If None, generates new one.
    
    Returns:
        str: The correlation ID that was set.
    """
    cid = correlation_id or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def add_correlation_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Structlog processor to add correlation ID to log entries.
    
    Args:
        logger: The logger instance.
        method_name: The logging method name (info, error, etc.).
        event_dict: The log event dictionary.
    
    Returns:
        dict: Updated event dictionary with correlation_id.
    """
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def drop_color_message_key(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove color_message key from log events (used by uvicorn).
    
    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The log event dictionary.
    
    Returns:
        dict: Updated event dictionary without color_message.
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    Sets up structlog with JSON formatting for production and
    colored console output for development. Integrates with
    standard library logging for third-party packages.
    """
    settings = get_settings()
    
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Common processors for all environments
    common_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        add_correlation_id,
        drop_color_message_key,
    ]
    
    if settings.log_format == "json":
        # Production: JSON format
        processors: list[Processor] = [
            *common_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Colored console output
        processors = [
            *common_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ["uvicorn", "uvicorn.access", "sqlalchemy.engine"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Optional logger name. Defaults to caller's module name.
    
    Returns:
        BoundLogger: Configured structlog logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("plate_detected", plate_number="MH12AB1234", confidence=0.95)
    """
    return structlog.get_logger(name)
