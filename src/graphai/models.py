"""Request / response models for the HTTP API."""

from typing import Any, Literal

from fastmcp import Client
from google import genai
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, description="What the user wants to know.")


Route = Literal[
    "search_employees", "get_employee_details", "get_department_statistics",
    "off_topic", "unroutable",
]

class RouteDecision(BaseModel):
    route: Route
    reason: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallTrace(BaseModel):
    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = Field(description="Tool output, or the error text if ok is false.")


@dataclass
class RunDeps:
    """Per-request dependencies handed to every executor."""

    run_id: str
    gemini: genai.Client
    model: str
    mcp: Client
    mcp_tools: list[Any]
    mcp_instructions: str | None
    max_llm_calls: int


@dataclass
class RunState:
    """Shared state, stored in the workflow context under 'run'."""

    prompt: str
    route: str = "unrouted"
    llm_calls: int = 0
    trace: list[ToolCallTrace] = field(default_factory=list)


@dataclass
class ToolRequest:
    """ReviewerAgent -> tool agent: run this tool next."""

    tool: str
    arguments: dict[str, Any]


@dataclass
class CallDone:
    """Tool agent -> ReviewerAgent: a tool call finished; its result is in RunState.trace."""

    tool: str


@dataclass
class AgentOutcome:
    """Final message passed from an agent / reviewer / off-topic node to Finalize."""

    status: Literal["completed", "no_response", "off_topic"]
    answer: str | None = None
    message: str | None = None


@dataclass
class ReviewResult:
    llm_calls: int
    request: ToolRequest | None = None
    answer: str | None = None
    message: str | None = None


class RunResponse(BaseModel):
    status: Literal["completed", "off_topic", "no_response"]
    route: str = Field(description="Which branch the router chose.")
    answer: str | None = Field(description="Final text.")
    message: str | None = Field(
        None, description="Set when status is not 'completed', or when the run was cut short."
    )
    llm_calls: int = Field(description="Every Gemini call: router + reviewer calls.")
    model: str
    tool_call: list[ToolCallTrace]
    run_id: str
