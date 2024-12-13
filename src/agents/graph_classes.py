import logging
from typing import Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from enum import Enum

class ChatSettings(BaseModel):
    """Chat-specific configuration settings"""
    primary_model: str = "llama"
    stream: bool = True
    enable_system_message: bool = False
    system_message: str = ""
    disable_system_message: bool = False

class RAGSettings(BaseModel):
    """Retrieval-Augmented Generation settings"""
    enabled: bool = False
    # Add RAG-specific settings here

class Config(BaseModel):
    """Global configuration settings"""
    chat_settings: ChatSettings = ChatSettings()
    rag_settings: RAGSettings = RAGSettings()

class TaskStatus(Enum):
    """Status values for tasks in the system"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DONE = "done" 
    FAILED = "failed"

class TaskType(Enum):
    """Types of tasks in the system"""
    CHAT = "chat"
    RAG = "rag"


class Task(TypedDict):
    """Represents a single task in the system"""
    type: TaskType
    status: TaskStatus
    conditions: list[str]
    actions: list[str]

class AgentState(TypedDict):
    """Core state maintained by the agent"""
    config: Config
    messages: list[dict]
    response_count: int
    active_chain: Optional[object]
    tool_choice: Optional[str]
    task_dict: dict[str, Task]
    user_input: Optional[str]

class GraphState(TypedDict):
    """Overall graph state"""
    keys: AgentState
    mutation_count: int
    is_done: bool

class ActionType(Enum):
    """Types of actions that can be executed"""
    GENERATE = "generate"
    WEB_SEARCH = "web_search"
    SAVE_DATA = "save_data"
    TOOL_CALL = "tool_call"
    CUSTOM = "custom"

class ActionResult(TypedDict):
    """Result of an action execution"""
    success: bool
    data: Optional[dict]
    error: Optional[str]

class CommandType(Enum):
    """Available command types"""
    QUIT = "quit"
    HELP = "help"
    CLEAR = "clear"
    SETTINGS = "settings"
    SAVE = "save"
    LOAD = "load"
    DEBUG = "debug"
    RETRY = "retry"
    UNDO = "undo"
    TOOLS = "tools"

class Command:
    """Represents a parsed command"""
    def __init__(self, raw_input: str):
        parts = raw_input.lstrip('/').split(' ', 1)
        self.command = parts[0].lower()
        self.args = parts[1] if len(parts) > 1 else ""
        
    @property
    def is_valid(self) -> bool:
        try:
            CommandType(self.command)
            return True
        except ValueError:
            return False
    
    @property
    def type(self) -> Optional[CommandType]:
        try:
            return CommandType(self.command)
        except ValueError:
            return None
