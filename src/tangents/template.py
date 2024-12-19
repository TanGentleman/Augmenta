from typing import Optional
from tangents.graph_classes import Action, ActionArgs, ActionType, AgentState, ChatSettings, Config, PlanActionType, RAGSettings, Task, Status, TaskType, create_action

STATE_ZERO = {
    "keys": {},
    "mutation_count": 0,
    "is_done": False
}

DEFAULT_TASK: Task = {
    "type": TaskType.CHAT,
    "status": Status.NOT_STARTED,
    "conditions": None,
    "actions": []
}

RAG_TASK: Task = {
    "type": TaskType.RAG,
    "status": Status.IN_PROGRESS,
    "conditions": None,
    "actions": []
}

DEFAULT_CONFIG = Config(
    chat_settings=ChatSettings(
        primary_model="llama",
        stream=True,
        system_message="Speak with lots of emojis",
        disable_system_message=False
    ),
    rag_settings=RAGSettings(
        enabled=True
    )
)


class MOCK_INPUTS:
    SUMMARY = ["/mode summary", "/read", "/quit"]
    DEFAULT = ["What's the capital of France?", "/quit"]
    EMPTY = []

READ_EMAIL_ACTION = create_action(PlanActionType.FETCH,
    args = {
        "source": "example-email.txt",
        "method": "get_email_content"
    }
)
CREATE_PLAN_ACTION = create_action(PlanActionType.CREATE_PLAN)
REVISE_PLAN_ACTION = create_action(PlanActionType.REVISE_PLAN,
    args = {
        "revision_count": 0,
        "max_revisions": 1,
    }
)

EXAMPLE_PLANNING_TASK = Task(
    type=TaskType.PLANNING,
    status=Status.NOT_STARTED,
    conditions=None,
    actions=[READ_EMAIL_ACTION, CREATE_PLAN_ACTION, REVISE_PLAN_ACTION]
)

EXAMPLE_CHAT_TASK: Task = Task(
    type=TaskType.CHAT,
    status=Status.NOT_STARTED,
    conditions=None,
    actions=[]
)

INITIAL_STATE_DICT: AgentState = {
    "config": DEFAULT_CONFIG,
    "messages": [],
    "action_count": 0,
    "active_chain": None,
    "tool_choice": None,
    "task_dict": {"chat_task": DEFAULT_TASK},
    "user_input": None,
    "mock_inputs": MOCK_INPUTS.EMPTY
}

PLANNING_STATE_DICT = {
    "config": DEFAULT_CONFIG,
    "messages": [],
    "action_count": 0,
    "user_input": None,
    "active_chain": None,
    "tool_choice": None,
    "task_dict": {"plan_from_email_task": EXAMPLE_PLANNING_TASK,
                  "chat_task": EXAMPLE_CHAT_TASK},
    "mock_inputs": MOCK_INPUTS.DEFAULT,
    "plan_dict": {
        "context": None,
        "proposed_plan": None,
        "plan": None
    }
}
