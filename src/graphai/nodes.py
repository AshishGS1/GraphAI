"""Nodes for the workflow: RouterAgent, ToolExec, ReviewerAgent, OffTopic, Finalize."""

import logging

from agent_framework import (
    Executor,
    WorkflowContext,
    handler,
)
from typing_extensions import Never

from graphai.models import (
    AgentOutcome, CallDone, RouteDecision, RunDeps, RunResponse, RunState,
    RunEvent, ToolCallTrace, ToolRequest,
)
from graphai.events import emit
from graphai.services.route import route_prompt
from graphai.services.review import review
from graphai.services.tool_call import clean_args, missing_required, run_tool_call
from graphai.prompts import build_router_prompt
from graphai.consts import OFF_TOPIC_REPLY, UNROUTABLE_REPLY

logger = logging.getLogger("graphai.nodes")


class RouterAgent(Executor):
    def __init__(self, deps: RunDeps) -> None:
        super().__init__(id="router")
        self.deps = deps
        self.router_prompt = build_router_prompt(deps.mcp_tools)

    @handler
    async def route(self, prompt: str, ctx: WorkflowContext[RouteDecision, RunEvent]) -> None:
        run = RunState(prompt=prompt)
        await emit(ctx, self.deps.run_id, self.id, "llm_called", model=self.deps.model, purpose="route")
        decision = await route_prompt(self.deps.gemini, self.deps.model, prompt, self.router_prompt)
        await emit(ctx, self.deps.run_id, self.id, "llm_responded", route=decision.route)
        run.route, run.llm_calls = decision.route, 1
        ctx.set_state("run", run)
        await ctx.send_message(decision)


class ToolExec(Executor):
    """Bound to exactly one MCP tool; no LLM. Runs it once, records the result, hands to the ReviewerAgent.
    Entered from the router (first call) or from the ReviewerAgent (later calls).
    """

    def __init__(self, deps: RunDeps, *, id: str, tool_name: str) -> None:
        super().__init__(id=id)
        tool = next((t for t in deps.mcp_tools if t.name == tool_name), None)
        if tool is None:
            raise RuntimeError(f"MCP server does not expose tool '{tool_name}'")
        self.deps, self.tool = deps, tool

    @handler
    async def first_call(self, decision: RouteDecision, ctx: WorkflowContext[CallDone]) -> None:
        await self._execute(decision.arguments, ctx)

    
    @handler
    async def later_calls(self, request: ToolRequest, ctx: WorkflowContext[CallDone]) -> None:
        await self._execute(request.arguments, ctx)

    async def _execute(self, arguments: dict, ctx: WorkflowContext[CallDone]) -> None:
        run: RunState = ctx.get_state("run")
        args = clean_args(arguments)
        missing = missing_required(self.tool, args)
        if missing:
            names = ", ".join(missing)
            logger.info("run=%s tool=%s missing required args: %s", self.deps.run_id, self.tool.name, names)
            run.trace.append(ToolCallTrace(
                tool=self.tool.name, arguments=args, ok=False,
                result=f"Not run: missing required argument(s): {names}",
            ))
        else:
            run.trace.append(await run_tool_call(
                run_id=self.deps.run_id, mcp=self.deps.mcp, tool=self.tool, arguments=args,
            ))
        ctx.set_state("run", run)
        await ctx.send_message(CallDone(self.tool.name))
            

class ReviewerAgent(Executor):
    """One LLM call per hop: ask for another tool call, or write the final answer.
    Additional llm call for forced answer in case of invalid/repeated tool call."""

    def __init__(self, deps: RunDeps) -> None:
        super().__init__(id="reviewer")
        self.deps = deps

    @handler
    async def review(self, _done: CallDone, ctx: WorkflowContext[ToolRequest | AgentOutcome, RunEvent]) -> None:
        deps = self.deps
        run: RunState = ctx.get_state("run")
        async def on_event(kind: str, **detail) -> None:
            await emit(ctx, deps.run_id, self.id, kind, **detail)
        result = await review(
            prompt=run.prompt, trace=run.trace, run_id=deps.run_id, gemini=deps.gemini,
            model=deps.model, mcp_tools=deps.mcp_tools, mcp_instructions=deps.mcp_instructions,
            max_llm_calls=deps.max_llm_calls, on_event=on_event,
        )
        run.llm_calls += result.llm_calls
        ctx.set_state("run", run)
        if result.request is not None:
            await ctx.send_message(result.request)
        elif result.answer is not None:
            await ctx.send_message(AgentOutcome("completed", result.answer, result.message))
        else:
            await ctx.send_message(AgentOutcome("no_response", None, result.message))


class OffTopic(Executor):
    """Fixed reply, no Gemini call. Also catches 'unroutable' (router output unusable)."""

    def __init__(self) -> None:
        super().__init__(id="off_topic")

    @handler
    async def reject(self, decision: RouteDecision, ctx: WorkflowContext[AgentOutcome]) -> None:
        if decision.route == "off_topic":
            outcome = AgentOutcome(
                "off_topic", OFF_TOPIC_REPLY, "Question is outside the employee directory."
            )
        else:
            outcome = AgentOutcome(
                "no_response", UNROUTABLE_REPLY, "The router could not determine a route for this task."
            )
        await ctx.send_message(outcome)


class Finalize(Executor):
    def __init__(self, deps: RunDeps) -> None:
        super().__init__(id="finalize")
        self.deps = deps

    @handler
    async def finish(self, outcome: AgentOutcome, ctx: WorkflowContext[Never, RunResponse]) -> None:
        run: RunState = ctx.get_state("run")
        await ctx.yield_output(RunResponse(
            status=outcome.status,
            route=run.route,
            answer=outcome.answer,
            message=outcome.message,
            llm_calls=run.llm_calls,
            model=self.deps.model,
            tool_call=run.trace,
            run_id=self.deps.run_id,
        ))
