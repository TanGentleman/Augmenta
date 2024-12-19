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
    PLAN_FROM_SOURCE = "plan_from_source"

class ActionType(Enum):
    """Types of actions that can be executed"""
    GENERATE = "generate"
    READ_EMAIL = "read_email"
    CREATE_PLAN = "create_plan"
    REVISE_PLAN = "revise_plan"
    WEB_SEARCH = "web_search"
    SAVE_DATA = "save_data"
    TOOL_CALL = "tool_call"
    CUSTOM = "custom"

class Task(TypedDict):
    """Represents a single task in the system"""
    type: TaskType
    status: TaskStatus
    conditions: list[str]
    actions: list[ActionType]

class AgentState(TypedDict):
    """Core state maintained by the agent"""
    config: Config
    messages: list[dict]
    response_count: int
    active_chain: Optional[object]
    tool_choice: Optional[str]
    task_dict: dict[str, Task]
    user_input: Optional[str]
    mock_inputs: list[str]

class GraphState(TypedDict):
    """Overall graph state"""
    keys: AgentState
    mutation_count: int
    is_done: bool

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
    UNDO = "undo"
    MODE = "mode"
    READ = "read"

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


class Plan(TypedDict):
    """Represents a plan"""
    plan_text: str
    plan_status: TaskStatus
    plan_actions: list[ActionType]

"""
Graph-Based Task Management System
--------------------------------

This module implements a state machine graph for managing AI agent tasks and conversations.
The system supports multiple task types, command processing, and action execution through
a flexible state machine architecture.

Key Components:
- Task Management: Handles task lifecycle (creation, execution, completion)
- Command Processing: Supports system commands prefixed with '/'
- Action Execution: Manages various action types like generation, email reading, etc.
- State Management: Maintains conversation and task state throughout execution

Task Types:
- CHAT: Basic conversational tasks
- RAG: Retrieval-augmented generation tasks
- PLAN_FROM_SOURCE: Multi-step planning tasks

Example Usage:
Initialize with default chat:
taskapp = create_workflow() [...]
#TODO: Finish this
"""