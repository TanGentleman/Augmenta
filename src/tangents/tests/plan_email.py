"""
Email Response Planner

A workflow-based system for generating and refining email responses.

Features:
- Fetches email content from specified source
- Generates initial response plan
- Iteratively refines plan through revision cycles

Example:
    from tangents.tests.plan_email import plan_email

    await plan_email(
        fetch_params={"source": "email.txt", "method": "get_email_content"},
        plan_params={"max_revisions": 2},
        extra_params={"system_message": "Custom instructions"}
    )
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4
import logging
import asyncio

from tangents.classes.states import AgentState
from tangents.template import get_default_config, get_initial_graph_state
from tangents.tan_graph import create_workflow, process_workflow_output_streaming
from tangents.classes.tasks import Task, Status, TaskType
from tangents.classes.actions import PlanActionType
from tangents.utils.action_utils import create_action


# Type definitions
@dataclass
class FetchParams:
    """Email content fetching configuration."""

    source: str  # Source file path or identifier
    method: str = 'get_email_content'  # Fetch method to use


@dataclass
class PlanParams:
    """Plan generation configuration."""

    max_revisions: int = 3  # Maximum revision cycles


@dataclass
class ExtraParams:
    """Additional configuration parameters."""

    system_message: Optional[str] = None  # Custom system instructions
    mock_inputs: list[str] = field(default_factory=list)  # Test inputs for workflow
    task_name: str = 'plan_from_email_task'  # Custom task identifier


# Constants
RECURSION_LIMIT = 50
DEFAULT_FETCH = FetchParams(source='example-email.txt')
DEFAULT_PLAN = PlanParams()
DEFAULT_EXTRA = ExtraParams()


def create_planning_task(fetch: FetchParams, plan: Optional[PlanParams] = None) -> Optional[Task]:
    """Create email planning task with sequential actions.

    Args:
        fetch: Email content fetch configuration
        plan: Optional planning parameters

    Returns:
        Configured Task or None if invalid params
    """
    if not fetch.source:
        logging.error('Missing source in fetch params')
        return None

    actions = [
        create_action(PlanActionType.FETCH, args={'source': fetch.source, 'method': fetch.method}),
        create_action(PlanActionType.PROPOSE_PLAN),
        create_action(
            PlanActionType.REVISE_PLAN,
            args={'max_revisions': (plan or DEFAULT_PLAN).max_revisions},
        ),
    ]

    return Task(type=TaskType.PLANNING, status=Status.NOT_STARTED, actions=actions, state=None)


def get_initial_state(task: Task, extra: ExtraParams = DEFAULT_EXTRA) -> AgentState:
    """Initialize workflow state.

    Args:
        task: Planning task to execute
        extra: Additional configuration parameters

    Returns:
        Initial agent state
    """
    config = get_default_config()
    if extra.system_message:
        config.chat_settings.system_message = extra.system_message

    return {
        'config': config,
        'action_count': 0,
        'task_dict': {extra.task_name: task},
        'user_input': None,
        'mock_inputs': extra.mock_inputs,
    }


async def execute_workflow(task: Task, extra: ExtraParams = DEFAULT_EXTRA) -> None:
    """Run the email planning workflow.

    Handles workflow initialization and output processing with fallback modes.
    """
    app = create_workflow()

    graph_state = get_initial_graph_state()
    graph_state['keys'] = get_initial_state(task, extra)

    app_config = {
        'recursion_limit': RECURSION_LIMIT,
        'configurable': {'thread_id': uuid4()},
    }

    # Try values mode first, fallback to updates if needed
    stream_mode = 'values'
    async for output in app.astream(graph_state, app_config, stream_mode=stream_mode):
        await process_workflow_output_streaming(output, app, app_config, stream_mode)

    # Check if human input needed
    if stream_mode == 'values':
        next_node = app.get_state(app_config).next
        if next_node and next_node[0] == 'human_node':
            print('Switching to updates mode for human input...')
            stream_mode = 'updates'
            async for output in app.astream(graph_state, app_config, stream_mode=stream_mode):
                await process_workflow_output_streaming(output, app, app_config, stream_mode)


async def plan_email(
    fetch_params: FetchParams,
    plan_params: Optional[PlanParams] = None,
    extra_params: Optional[ExtraParams] = None,
) -> None:
    """Plan an email response.

    Args:
        fetch_params: Email content fetch configuration
        plan_params: Optional planning parameters
        extra_params: Additional configuration parameters
    """
    task = create_planning_task(fetch_params, plan_params)
    if task:
        await execute_workflow(task, extra_params or DEFAULT_EXTRA)
    else:
        logging.error('Failed to create planning task')


if __name__ == '__main__':
    asyncio.run(plan_email(DEFAULT_FETCH, DEFAULT_PLAN))
