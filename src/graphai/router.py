"""HTTP route that runs the agent graph."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastmcp import Client
from google.genai import errors as genai_errors

from graphai.config import get_settings
from graphai.models import RunRequest, RunResponse
from graphai.workflow import RunDeps, build_workflow

logger = logging.getLogger("graphai.api")
router = APIRouter()


def _server_instructions(mcp: Client) -> str | None:
    if mcp.initialize_result is None:
        return None
    return mcp.initialize_result.instructions


def _find_cause(exc: BaseException, kind: type[BaseException]) -> BaseException | None:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, kind):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__  
    return None

@router.post("/agent/run", response_model=RunResponse)
async def run(req: RunRequest, request: Request) -> RunResponse:
    settings = get_settings()
    run_id = uuid.uuid4().hex[:12]
    logger.info("run=%s", run_id)

    try:
        async with Client(settings.mcp_url) as mcp:
            deps = RunDeps(
                run_id=run_id,
                gemini=request.app.state.gemini,
                model=settings.gemini_model,
                mcp=mcp,
                mcp_tools=await mcp.list_tools(),
                mcp_instructions=_server_instructions(mcp),
                max_llm_calls=settings.max_llm_calls,
            )
            result = await build_workflow(deps).run(req.prompt)
    except Exception as exc:
        api_err = _find_cause(exc, genai_errors.APIError)
        if api_err is not None:
            logger.error("run=%s Gemini error: %s %s", run_id, api_err.code, api_err.message)
            raise HTTPException(502, detail=f"Gemini API error ({api_err.code}): {api_err.message}")
        if isinstance(exc, (ConnectionError, OSError)) or (
            isinstance(exc, RuntimeError) and "failed to connect" in str(exc).lower()
        ):
            logger.error("run=%s MCP connection error: %s", run_id, exc)
            raise HTTPException(503, detail=f"Could not reach MCP server at {settings.mcp_url}")
        raise

    outputs = result.get_outputs()
    if not outputs:
        raise HTTPException(500, detail="Workflow produced no output")
    return outputs[-1]