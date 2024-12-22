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

class Task(TypedDict):
    """Task definition and execution state.
    
    Attributes:
        type: Task category
        status: Current execution status
        conditions: Optional execution prerequisites
        actions: Ordered list of actions to execute
    """
    type: TaskType
    status: Status
    conditions: Optional[dict]
    actions: list[Action]