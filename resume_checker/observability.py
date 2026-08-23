from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    from resume_checker.config import get_settings

    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def trace_event(name: str, payload: dict) -> None:
    if not langfuse_enabled():
        return
    try:
        from langfuse import Langfuse

        client = Langfuse()
        client.trace(name=name, metadata=payload)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Langfuse tracing skipped: %s", exc)
