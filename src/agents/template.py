from agents.graph_classes import AgentState, ChatSettings, Config, RAGSettings, Task, TaskStatus, TaskType
from augmenta.utils import read_sample


STATE_ZERO = {
    "keys": {},
    "mutation_count": 0,
    "is_done": False
}

DEFAULT_TASK: Task = {
    "type": TaskType.CHAT,
    "status": TaskStatus.NOT_STARTED,
    "conditions": [],
    "actions": []
}

RAG_TASK: Task = {
    "type": TaskType.RAG,
    "status": TaskStatus.IN_PROGRESS,
    "conditions": [],
    "actions": []
}

DEFAULT_CONFIG = Config(
    chat_settings=ChatSettings(
        primary_model="samba",
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
