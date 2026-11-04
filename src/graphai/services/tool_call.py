"""Run one MCP tool and record the result. No LLM involved."""

import logging
import time
from typing import Any

from fastmcp import Client
from graphai.models import ToolCallTrace

logger = logging.getLogger("graphai.tool_call")


def _tool_output(result: Any) -> Any:
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    return "\n".join(getattr(block, "text", "") for block in (result.content or []))


async def _call_tool(mcp: Client, name: str, args: dict[str, Any]) -> tuple[bool, Any]:
    try:
        result = await mcp.call_tool(name, args, raise_on_error=False)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return (not result.is_error), _tool_output(result)


def input_schema(tool: Any) -> dict:
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}


def clean_args(args: dict[str, Any]) -> dict[str, Any]:
    """Drop null values; the model often sends null for arguments it leaves out."""
    return {k: v for k, v in (args or {}).items() if v is not None}


def missing_required(tool: Any, args: dict[str, Any]) -> list[str]:
    return [k for k in input_schema(tool).get("required", []) if args.get(k) in (None, "")]


async def run_tool_call(
    *, run_id: str, mcp: Client, tool: Any, arguments: dict[str, Any]
) -> ToolCallTrace:
    """Call `tool` once. Errors are captured in the trace, not raised."""
    started = time.perf_counter()
    ok, output = await _call_tool(mcp, tool.name, arguments)
    logger.info(
        "run=%s tool=%s ok=%s duration_ms=%.1f",
        run_id, tool.name, ok, (time.perf_counter() - started) * 1000,
    )
    return ToolCallTrace(tool=tool.name, arguments=arguments, ok=ok, result=output)
