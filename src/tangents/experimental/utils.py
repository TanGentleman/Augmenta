import json
from typing import TypedDict
from mcp.types import CallToolResult
from tangents.classes.settings import ChatSettings, Config
from tangents.classes.states import AgentState
from tangents.classes.tasks import Task, TaskType
from tangents.classes.actions import Status
from tangents.template import get_default_config
from tangents.utils.chains import fast_get_llm

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
        'mock_inputs': ["1+1=", "/quit"],
    }

def get_experimental_task() -> Task:
    return Task(
        type=TaskType.CHAT,
        status=Status.NOT_STARTED,
        # actions=[create_action(ActionType.TOOL_CALL, args={'tool_name': 'create_entities', 'tool_args': {'entities': [{'name': 'table', 'entityType': 'furniture'}]}})],
        actions=[],
        state=None,
    )

def get_gradio_state_dict(user_message: str, history: list, model_name: str, system_message: str) -> AgentState:
    chat_config = ChatSettings(
        primary_model=model_name,
        stream=True,
        system_message=system_message,
        disable_system_message=False,
    )
    messages = []
    if history == [] or history[0]['role'] != 'system':
        if chat_config.disable_system_message is False:
            messages.insert(0, {'role': 'system', 'content': chat_config.system_message})
    # Add all messages from history except the last one if it's a user message
    for i, m in enumerate(history):
        # Skip the last message if it's from the user (we'll handle it separately)
        if i == len(history) - 1 and m['role'] == 'user':
            continue
        # Can add custom logic here to modify the messages
        messages.append(m)
        print(f'{m["role"]}: {m["content"]}')
    
    task_state = {
        'messages': messages,
        'active_chain': fast_get_llm(model_name),
        'stream': True,
    }
    state_dict = {
        'config': Config(
            chat_settings=chat_config,
        ),
        'action_count': 0,
        'user_input': None,
        'task_dict': {
            'chat_task': Task(
                type=TaskType.CHAT,
                status=Status.IN_PROGRESS,
                actions=[],
                state=task_state
            )
        },
        'mock_inputs': [user_message],
    }
    return state_dict