import json
from typing import TypedDict
from mcp.types import CallToolResult
from tangents.classes.states import AgentState
from tangents.classes.tasks import Task, TaskType
from tangents.classes.actions import ActionType, Status
from tangents.utils.action_utils import create_action
from tangents.template import get_default_config

class BriefEntity(TypedDict):
    name: str
    entityType: str

def parse_convex_result(res: CallToolResult) -> dict | None:
    try:
        p1 = json.loads(res.content[0].text)
        if p1["isError"]:
            raise ValueError(p1["error"])
        p2 = p1["content"][0]["text"]
        p3 = json.loads(p2)
        return p3["result"]
    except Exception as e:
        print(f"Error parsing convex result: {e}")
        return None

def get_experimental_state_dict() -> AgentState:
    return {
        'config': get_default_config(),
        'action_count': 0,
        'user_input': None,
        'task_dict': {'entity_task': get_experimental_task()},
        'mock_inputs': ["/quit"],
    }

def get_experimental_task() -> Task:
    return Task(
        type=TaskType.EXPERIMENTAL,
        status=Status.NOT_STARTED,
        actions=[],
        # actions=[create_action(ActionType.TOOL_CALL, args={'tool_name': 'create_entities', 'tool_args': {'entities': [{'name': 'table', 'entityType': 'furniture'}]}})],
        state=None,
    )