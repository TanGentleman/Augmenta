import logging
from typing import Any, Optional
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

class Status(Enum):
    """Status values for tasks in the system"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DONE = "done" 
    FAILED = "failed"

class TaskType(Enum):
    """Types of tasks in the system"""
    CHAT = "chat"
    RAG = "rag"
    PLANNING = "planning"

class ActionResult(TypedDict):
    """Result of an action execution"""
    success: bool
    data: Optional[dict]
    error: Optional[str]

class ActionType(Enum):
    """Types of actions that can be executed"""
    GENERATE = "generate"
    WEB_SEARCH = "web_search"
    SAVE_DATA = "save_data"
    TOOL_CALL = "tool_call"


class ActionArgs(TypedDict):
    """Arguments for an action"""
    pass


class Action(TypedDict):
    """Represents an action"""
    type: ActionType
    status: Status
    args: Optional[ActionArgs]
    result: Optional[ActionResult]
    

class PlanActionType(Enum):
    """Types of actions specific to plan execution"""
    FETCH = "fetch"
    CREATE_PLAN = "create_plan" 
    REVISE_PLAN = "revise_plan"

# We
class Task(TypedDict):
    """Represents a single task in the system"""
    type: TaskType
    status: Status
    conditions: Optional[dict]
    actions: list[Action]

class PlanActionArgs(ActionArgs):
    """Arguments for a plan action"""
    plan_context: Optional[str]
    proposed_plan: Optional[dict[str, Task]]
    plan: Optional[dict[str, Task]]

# TODO: Implement this in lieu of passing state_dict to execute_action
class GenerateActionArgs(ActionArgs):
    """Arguments for a generate action"""
    stream: Optional[bool]
    chain: Optional[Any]
    messages: Optional[list[dict]]
    pass

class AgentState(TypedDict, total=True):
    """Core state maintained by the agent"""
    config: Config
    messages: list[dict]
    response_count: int
    active_chain: Optional[object]
    tool_choice: Optional[str]
    user_input: Optional[str]
    mock_inputs: list[str]
    task_dict: dict[str, Task]

class GraphState(TypedDict):
    """Overall graph state"""
    keys: AgentState
    mutation_count: int
    is_done: bool


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
- PLANNING: Multi-step planning tasks

Example Usage:
Initialize with default chat:
taskapp = create_workflow() [...]
#TODO: Finish this
"""

def create_action(action_type: ActionType | PlanActionType, args: Optional[ActionArgs] = None) -> Action:
    assert isinstance(action_type, ActionType | PlanActionType)
    if args is not None:
        # TODO: Add validation for args
        assert isinstance(args, dict)

    return Action(
        type=action_type,
        status=Status.NOT_STARTED,
        args=args,
        result=None
    )