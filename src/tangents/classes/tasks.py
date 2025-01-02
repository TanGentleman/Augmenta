from typing import Optional
from typing_extensions import TypedDict
from enum import Enum
from tangents.classes.actions import Status, Action

class TaskType(Enum):
    """Available task categories.
    
    Values:
        CHAT: Basic conversational tasks
        RAG: Retrieval-augmented generation tasks
        PLANNING: Multi-step planning tasks
    """
    CHAT = "chat"
    RAG = "rag"
    PLANNING = "planning"

# NOTE: Not sure if states.py is the right place for this
class TaskState(TypedDict):
    """Abstract base class for task-specific state."""
    pass

class Task(TypedDict):
    """Task definition and execution state.
    
    Attributes:
        type: Task category
        status: Current execution status
        actions: Ordered list of actions to execute
        state: Task-specific state
    """
    type: TaskType
    status: Status
    actions: list[Action]
    state: Optional[TaskState]
