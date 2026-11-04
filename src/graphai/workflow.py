"""The agent graph.

    Router --(switch-case on route)--> SearchEmployees | EmployeeDetails | DepartmentStats
                                       | OffTopic (off_topic, and Default for 'unroutable')
    each tool executor --> ReviewerAgent           (a tool call finished)
    ReviewerAgent --(switch on tool name)--> a tool executor   (another call is needed)
    ReviewerAgent --> Finalize                  (final answer)
    OffTopic --> Finalize
"""

from agent_framework import (
    Case,
    Default,
    WorkflowBuilder,
)

from graphai.models import AgentOutcome, RunDeps, ToolRequest
from graphai.consts import SEARCH_EMPLOYEES, GET_EMPLOYEE_DETAILS, GET_DEPARTMENT_STATISTICS
from graphai.nodes import RouterAgent, ToolExec, ReviewerAgent, OffTopic, Finalize


def build_workflow(deps: RunDeps):
    router = RouterAgent(deps)
    search = ToolExec(deps, id="search_employees_agent", tool_name=SEARCH_EMPLOYEES)
    details = ToolExec(deps, id="employee_details_agent", tool_name=GET_EMPLOYEE_DETAILS)
    department = ToolExec(deps, id="department_stats_agent", tool_name=GET_DEPARTMENT_STATISTICS)
    reviewer = ReviewerAgent(deps)
    off_topic = OffTopic()
    finalize = Finalize(deps)

    builder = WorkflowBuilder(start_executor=router, name="graphai")
    builder.add_switch_case_edge_group(router, [
        Case(condition=lambda d: d.route == SEARCH_EMPLOYEES, target=search),
        Case(condition=lambda d: d.route == GET_EMPLOYEE_DETAILS, target=details),
        Case(condition=lambda d: d.route == GET_DEPARTMENT_STATISTICS, target=department),
        Default(target=off_topic),
    ])
    for exec, tool_name in (
        (search, SEARCH_EMPLOYEES), (details, GET_EMPLOYEE_DETAILS),
        (department, GET_DEPARTMENT_STATISTICS),
    ):
        builder.add_edge(exec, reviewer)
        builder.add_edge(
            reviewer, exec,
            condition=lambda m, n=tool_name: isinstance(m, ToolRequest) and m.tool == n,
        )
    builder.add_edge(reviewer, finalize, condition=lambda m: isinstance(m, AgentOutcome))
    builder.add_edge(off_topic, finalize)
    return builder.build()
