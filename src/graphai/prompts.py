import json
from typing import Any
from graphai.services.tool_call import input_schema 

_ROUTER_TEMPLATE = """You route questions about a company employee directory.
Questions are answered with calls to these tools; a later step may make further
calls. You only choose the tool for the FIRST call:
{tool_lines}

Pick exactly one route:
- search_employees: find people by name, department name and/or job title (the user
  gives no employee id). At least one of name / department / job_title is needed.
- get_employee_details: the full record of ONE employee, when the user gives a
  numeric employee id. Never invent an id; if the user gives only a name, use
  search_employees instead.
- get_department_statistics: headcount, salary statistics, budget, location or
  manager of departments. It takes only a numeric department_id; if the user names a
  department but gives no id, OMIT department_id (all departments are returned and
  a later step picks the right one).
- off_topic: unrelated to the employee directory (general knowledge, chit-chat,
  coding help, anything else).
Also fill "arguments" with a JSON object holding the arguments for the
chosen tool, following its input schema. Use only values the user gave; leave out
any argument the user did not provide (do not guess). Use "{{}}" for off_topic.
If the user clearly wants one of these tools but left out its required arguments, STILL
choose that route and use "{{}}" or the partial arguments; a later step asks the user for
what is missing. Never refuse because arguments are absent.
Return JSON with "route", a one-sentence "reason" and "arguments"."""


def build_router_prompt(mcp_tools: list[Any]) -> str:
    """RouterAgent prompt with the live MCP tool descriptions, so it can't drift from the server."""
    lines = "\n".join(
        f"- {t.name}: {' '.join((t.description or '').split())}\n"
        f"  input schema: {json.dumps(input_schema(t))}"
        for t in mcp_tools
    )
    return _ROUTER_TEMPLATE.format(tool_lines=lines or "(none)")


REVIEWER_PROMPT = """You answer questions about a company employee directory using the tools.
You are given the question and the tool calls made so far with their results.
- If the results are enough, reply with the final answer in plain text, using ONLY
  those results. Do not invent facts.
- If another tool call is needed (for example an id found in one result is required
  by the next tool), call exactly one tool. Never repeat a call that was already made.
- If a call was not run because a required argument was missing, do not guess the
  value. First try to get it from another tool using what the user told you (for
  example, search_employees by name to find an employee id). If the question gives
  nothing to look it up with, reply in plain text asking the user for the missing
  information, in terms they would know (a name, a department) rather than an
  internal id.
- If a result is any other error, correct your call if you can, otherwise say
  clearly what could not be determined."""

FORCED_ANSWER_NOTE = (
    "You cannot call any more tools. Answer now with ONLY the results above and say "
    "clearly what you could not determine."
)
