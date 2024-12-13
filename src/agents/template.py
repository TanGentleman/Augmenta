from agents.graph_classes import AgentState, ChatSettings, Config, RAGSettings, Task, TaskStatus, TaskType


STATE_ZERO = {
    "keys": {},
    "mutation_count": 0,
    "is_done": False
}

DEFAULT_TASK: Task = {
    "type": TaskType.CHAT,
    "status": TaskStatus.IN_PROGRESS,
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
        primary_model="llama",
        stream=True,
        system_message="Speak with lots of emojis",
        disable_system_message=False
    ),
    rag_settings=RAGSettings(
        enabled=True
    )
)

INITIAL_STATE_DICT: AgentState = {
    "config": DEFAULT_CONFIG,
    "messages": [],
    "action_count": 0,
    "active_chain": None,
    "tool_choice": None,
    "task_dict": {"chat_task": DEFAULT_TASK},
    "user_input": None
}
