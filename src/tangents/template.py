from tangents.classes.states import AgentState
from tangents.classes.tasks import Task, TaskType
from tangents.classes.actions import ActionType, PlanActionType, Status
from tangents.classes.settings import Config, ChatSettings, RAGSettings

from tangents.utils.action_utils import create_action
INITIAL_GRAPH_STATE = {
    "keys": {},
    "mutation_count": 0,
    "is_done": False
}

DEFAULT_TASK: Task = {
    "type": TaskType.CHAT,
    "status": Status.NOT_STARTED,
    "conditions": None,
    "actions": [],
    "state": None
}

RAG_TASK: Task = {
    "type": TaskType.RAG,
    "status": Status.IN_PROGRESS,
    "conditions": None,
    "actions": [],
    "state": None
}

DEFAULT_CONFIG = Config(
    chat_settings=ChatSettings(
        primary_model="open/google/gemini-flash-1.5-8b",
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

STASH_ACTION = create_action(ActionType.STASH)
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
        "max_revisions": 3,
    }
)

EXAMPLE_PLANNING_TASK = Task(
    type=TaskType.PLANNING,
    status=Status.NOT_STARTED,
    conditions=None,
    actions=[READ_EMAIL_ACTION, CREATE_PLAN_ACTION, REVISE_PLAN_ACTION],
    state=None
)

INITIAL_STATE_DICT: AgentState = {
    "config": DEFAULT_CONFIG,
    "action_count": 0,
    "task_dict": {"chat_task": DEFAULT_TASK},
    "user_input": None,
    "mock_inputs": MOCK_INPUTS.EMPTY
}

PLANNING_STATE_DICT = {
    "config": DEFAULT_CONFIG,
    "action_count": 0,
    "user_input": None,
    "task_dict": {"plan_from_email_task": EXAMPLE_PLANNING_TASK},
    "mock_inputs": MOCK_INPUTS.EMPTY,
}
