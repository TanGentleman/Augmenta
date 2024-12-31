"""
Email Planning Script

A script that uses a task-based workflow system to plan email responses. It:
1. Fetches email content from a source
2. Generates an initial response plan
3. Iteratively revises the plan

Configuration Options:
- fetch_params: Source and method for email content
- plan_params: Number of revision cycles
- stream_mode: "updates" or "values" output mode

Usage:
    python -m tangents.tests.plan-email

    # With custom parameters:
    asyncio.run(plan_email(
        fetch_params={
            "source": "example-email.txt",
            "method": "get_email_content"
        },
        plan_params={
            "max_revisions": 2
        }
    ))
"""

from uuid import uuid4
import logging
import asyncio
from typing_extensions import TypedDict
from tangents.classes.states import AgentState
from tangents.template import DEFAULT_CONFIG, INITIAL_GRAPH_STATE
from tangents.tan_graph import create_workflow, process_workflow_output_streaming
from tangents.classes.tasks import Task, Status, TaskType
from tangents.classes.actions import PlanActionType
from tangents.utils.action_utils import create_action

# Type definitions for configuration parameters
class FetchParams(TypedDict):
    """Parameters for email content fetching."""
    source: str  # Path or identifier of email source
    method: str  # Method to use for fetching content

class PlanParams(TypedDict):
    """Parameters for plan generation and revision."""
    max_revisions: int  # Maximum number of plan revision cycles

# Configuration constants
RECURSION_LIMIT = 50

# Default task parameters
DEFAULT_FETCH_PARAMS: FetchParams = {
    "source": "example-email.txt",
    "method": "get_email_content"
}

DEFAULT_PLAN_PARAMS: PlanParams = {
    "max_revisions": 3
}

def create_email_planning_task(
    fetch_params: FetchParams,
    plan_params: PlanParams | None = None
) -> Task | None:
    """Create an email planning task with the given configuration.
    
    Args:
        fetch_params: Parameters for fetching email content
        plan_params: Optional parameters for plan revision
        
    Returns:
        Task object if successful, None if required params missing
    """
    if not fetch_params.get('source'):
        logging.error("No source provided, failed to create email planning task.")
        return None

    # Build sequential action pipeline
    actions = [
        # 1. Fetch email content
        create_action(
            PlanActionType.FETCH,
            args={
                "source": fetch_params['source'],
                "method": fetch_params.get('method', DEFAULT_FETCH_PARAMS['method'])
            }
        ),
        # 2. Generate initial plan
        create_action(PlanActionType.PROPOSE_PLAN),
        # 3. Revise plan iteratively
        create_action(
            PlanActionType.REVISE_PLAN,
            args={
                "max_revisions": (plan_params or DEFAULT_PLAN_PARAMS)["max_revisions"]
            }
        )
    ]
    
    return Task(
        type=TaskType.PLANNING,
        status=Status.NOT_STARTED,
        conditions=None,
        actions=actions,
        state=None
    )

def get_state_dict(task: Task, mock_inputs: list[str] = []) -> AgentState:
    """Initialize workflow state with task and configuration.
    
    Args:
        task: The email planning task to execute
        mock_inputs: Optional list of predefined inputs for testing
    
    Returns:
        Initial agent state dictionary
    """
    config = DEFAULT_CONFIG
    # TODO: Fix aliasing of config
    # TODO: Support adjustments to config

    return {
        "config": config,
        "action_count": 0,
        "task_dict": {"plan_from_email_task": task},
        "user_input": None,
        "mock_inputs": mock_inputs
    }

async def main_async(task: Task):
    """Execute the email planning workflow asynchronously.
    
    Handles workflow initialization, streaming output processing,
    and fallback to updates mode if needed.
    """
    app = create_workflow()
    
    # Initialize graph state with email context
    graph_state = {
        "keys": {},
        "mutation_count": 0,
        "is_done": False
    }
    assert graph_state == INITIAL_GRAPH_STATE

    state_dict = get_state_dict(task)
    graph_state["keys"] = state_dict

    # Configure app settings
    app_config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {
            "thread_id": uuid4(),
        }
    }

    # Process workflow outputs with streaming
    stream_mode = "values"
    assert stream_mode in ["updates", "values"]
    
    async for output in app.astream(graph_state, app_config, stream_mode=stream_mode):
        await process_workflow_output_streaming(output, app, app_config, stream_mode=stream_mode)
    
    # Handle fallback to updates mode if human input needed
    if stream_mode == "values":
        next_node = app.get_state(app_config).next
        if next_node:
            next_node = next_node[0]
            # print("Node: ", str(next_node))
            if next_node == "human_node":
                FALLBACK_MODE = True
                if not FALLBACK_MODE:
                    raise SystemExit("Exit: TODO: Handle interruption in values mode.")
                print("Switching to updates mode...")
                stream_mode = "updates"
                async for output in app.astream(graph_state, app_config, stream_mode=stream_mode):
                    await process_workflow_output_streaming(output, app, app_config, stream_mode=stream_mode)
    
    print("Email planning workflow completed.")

async def plan_email(
    fetch_params: FetchParams,
    plan_params: PlanParams | None = None,
):
    """Plan an email response using the workflow system.
    
    Args:
        fetch_params: Configuration for source content fetching
        plan_params: Optional email planning parameters
    """
    email_task = create_email_planning_task(fetch_params, plan_params)
    if not email_task:
        print("Failed to create email planning task.")
        return
    await main_async(email_task)

if __name__ == "__main__":
    # Example usage with structured parameters
    # asyncio.run(plan_email(
    #     fetch_params={
    #         "source": "example-email.txt",
    #         "method": "get_email_content"
    #     },
    #     plan_params={
    #         "max_revisions": 2
    #     }
    # ))
    asyncio.run(plan_email(
        fetch_params=DEFAULT_FETCH_PARAMS,
        plan_params=DEFAULT_PLAN_PARAMS
    ))
