import json
import logging
from typing import Any, TypedDict
from mcp.types import CallToolResult
from tangents.classes.settings import ChatSettings, Config
from tangents.classes.states import AgentState
from tangents.classes.tasks import Task, TaskType
from tangents.classes.actions import ActionType, Status
from tangents.template import get_default_config
from tangents.utils.action_utils import create_action
from tangents.utils.chains import fast_get_llm

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

def get_gradio_state_dict(user_message: str, history: list, model_name: str, system_message: str, active_chain: Any | None = None) -> AgentState:
    chat_config = ChatSettings(
        primary_model=model_name,
        stream=True,
        system_message=system_message,
        disable_system_message=False,
    )
    messages = []
    if history == [] or history[0]['role'] != 'system':
        if chat_config.disable_system_message is False:
            messages.append(SystemMessage(content=chat_config.system_message))
    # Add all messages from history except the last one if it's a user message
    for i, m in enumerate(history):
        # Skip the last message if it's from the user (we'll handle it separately)
        if i == len(history) - 1:
            if m['role'] == 'user':
                # Added as a mock input
                if user_message != m['content']:
                    print(f"WARNING: User message mismatch: {user_message} != {m['content']}")
                continue
        if m['role'] == 'assistant':
            messages.append(AIMessage(content=m['content']))
        elif m['role'] == 'user':
            messages.append(HumanMessage(content=m['content']))
        elif m['role'] == 'system':
            print("WARNING: System message passed in history")
            messages.append(SystemMessage(content=m['content']))
    
    task_state = {
        'messages': messages,
        'chain': active_chain,
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
                actions=[create_action(ActionType.HEALTHCHECK)],
                state=task_state
            )
        },
        'mock_inputs': [user_message],
    }
    return state_dict

def get_proposed_plan(context: str, system_prompt: str) -> str | None:
    """
    Get a proposed plan from the LLM.
    """
    try:
        llm = fast_get_llm('nebius/meta-llama/Llama-3.3-70B-Instruct')
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logging.error(f"Error getting proposed plan: {e}")
        return None