"""The ReviewerAgent's Gemini call: decide the next tool call, or write the final answer."""

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from graphai.consts import TOOL_ROUTES
from graphai.models import ToolCallTrace, ToolRequest, ReviewResult
from graphai.prompts import FORCED_ANSWER_NOTE, REVIEWER_PROMPT
from graphai.services.tool_call import clean_args, input_schema

logger = logging.getLogger("graphai.review")


def _declarations(mcp_tools: list[Any]) -> list[types.Tool]:
    """Only tools that have an agent in the graph."""
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t.name, description=t.description or "",
            parameters_json_schema=input_schema(t),
        )
        for t in mcp_tools if t.name in TOOL_ROUTES
    ])]


def _digest(prompt: str, trace: list[ToolCallTrace]) -> str:
    lines = [f"Question:\n{prompt}", "", "Tool calls made so far:"]
    for i, t in enumerate(trace, 1):
        lines.append(
            f"{i}. {t.tool}({json.dumps(t.arguments, default=str)}) -> "
            f"{'ok' if t.ok else 'ERROR'}: {json.dumps(t.result, default=str)}"
        )
    return "\n".join(lines)


def _text(response: Any) -> str:
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None:
        return ""
    return "".join(
        p.text for p in (candidate.content.parts or []) if p.text and not p.thought
    ).strip()


async def review(
    *,
    prompt: str,
    trace: list[ToolCallTrace],
    run_id: str,
    gemini: genai.Client,
    model: str,
    mcp_tools: list[Any],
    mcp_instructions: str | None,
    max_llm_calls: int,
) -> ReviewResult:
    system = "\n\n".join(filter(None, [mcp_instructions, REVIEWER_PROMPT]))
    tools = _declarations(mcp_tools)
    body = _digest(prompt, trace)

    def config(extra: str = "") -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system + extra,
            tools=tools,
        )

    response = await gemini.aio.models.generate_content(
        model=model, contents=body, config=config()
    )
    calls = response.function_calls
    if not calls:
        text = _text(response)
        if not text:
            return ReviewResult(1, message="Gemini returned no text.")
        return ReviewResult(1, answer=text)
    call=calls[0]

    args = clean_args(dict(call.args or {}))
    key = json.dumps(args, sort_keys=True, default=str)

    reason = None
    if call.name not in TOOL_ROUTES:
        reason = f"it asked for an unknown tool '{call.name}'"
    elif len(trace) >= max_llm_calls:
        reason = f"the tool-call limit ({max_llm_calls}) was reached"
    elif any(t.tool == call.name and json.dumps(t.arguments, sort_keys=True, default=str) == key for t in trace):
        reason = f"it repeated an identical call to {call.name}"
    if reason is None:
        logger.info("run=%s reviewer -> %s %s", run_id, call.name, args)
        return ReviewResult(1, request=ToolRequest(call.name, args))

    logger.warning("run=%s reviewer wanted %s %s but %s; forcing an answer", run_id, call.name, args, reason)
    forced = await gemini.aio.models.generate_content(
        model=model,
        contents=f"{body}\n\n(The call {call.name}({json.dumps(args, default=str)}) was NOT run.)",
        config=config("\n\n" + FORCED_ANSWER_NOTE),
    )
    note = f"Stopped early because {reason}; the answer uses only the data gathered so far."
    text = _text(forced)
    if not text:
        return ReviewResult(2, message=note + " Gemini returned no text.")
    return ReviewResult(2, answer=text, message=note)
