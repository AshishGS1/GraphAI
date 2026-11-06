"""Progress events: each is logged to the console and yielded into the workflow event stream(streaming currently not set)."""

import logging
from typing import Any

from agent_framework import WorkflowContext

from graphai.models import RunEvent

logger = logging.getLogger("graphai.events")


async def emit(ctx: WorkflowContext[Any, RunEvent], run_id: str, source: str, kind: str, **detail: Any) -> None:
    """`source` is the emitting executor's id (log line only; the event carries it as executor_id)."""
    logger.info("run=%s event=%s source=%s %s", run_id, kind, source, detail)
    await ctx.yield_output(RunEvent(kind, run_id, detail))