"""Sentry instrumentation. Everything here is a no-op unless SENTRY_DSN is set, so the app runs
identically without it.

Products used beyond error monitoring: Tracing (one span per agent step, linked to the FastAPI
request), Logs (structured pipeline logs), Profiling (continuous, tied to traces) and AI agent
monitoring (OpenAI calls plus gen_ai.invoke_agent spans)."""
import os
from contextlib import contextmanager

import sentry_sdk
from sentry_sdk import logger as sentry_logger

from .config import settings

_enabled = False


def init(transport=None) -> bool:
    global _enabled
    if not settings.sentry_dsn or _enabled:
        return _enabled
    from sentry_sdk.integrations.openai import OpenAIIntegration

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=1.0,
        profile_session_sample_rate=1.0,
        profile_lifecycle="trace",
        enable_logs=True,
        send_default_pii=False,
        integrations=[OpenAIIntegration(include_prompts=True)],
        trace_propagation_targets=[],  # don't leak trace headers to OpenAI/Elastic; nothing downstream reads them
        transport=transport,
        debug=os.getenv("SENTRY_DEBUG") == "1",
    )
    _enabled = True
    return True


def enabled() -> bool:
    return _enabled


@contextmanager
def agent_span(name: str, **data):
    """A gen_ai.invoke_agent span so Sentry's AI agent view shows each pipeline stage."""
    with sentry_sdk.start_span(op="gen_ai.invoke_agent", name=f"invoke_agent {name}") as span:
        span.set_data("gen_ai.agent.name", name)
        for k, v in data.items():
            span.set_data(k, v)
        yield span


def log_pipeline(mode: str, evidence: int, audit: dict, ungrounded: list, derived: list, invalid: list, ms: int) -> None:
    """Structured log + alert for answers whose audit turned up something worth a human look."""
    sentry_sdk.set_tag("mode", mode)
    sentry_sdk.set_tag("answerable", str(audit["answerable"]).lower())
    sentry_logger.info(
        "answer complete: {evidence} passages, {conflicts} conflicts, {gaps} gaps, {ungrounded} ungrounded figures in {ms} ms",
        evidence=evidence,
        conflicts=len(audit["conflicts"]),
        gaps=len(audit["gaps"]),
        ungrounded=len(ungrounded),
        ms=ms,
    )
    if ungrounded or invalid:
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("audit", "ungrounded")
            scope.set_context("audit", {"ungrounded_numbers": ungrounded, "invalid_citations": invalid, "derived": derived})
            sentry_sdk.capture_message("Answer contained figures or citations not backed by evidence", level="warning")
    elif not audit["answerable"]:
        sentry_logger.warning("question not answerable from corpus: {gaps}", gaps="; ".join(audit["gaps"])[:300])
