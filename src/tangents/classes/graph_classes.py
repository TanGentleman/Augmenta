"""Core classes and types for the graph-based task management system.

This module defines the foundational data structures and types used throughout the system,
including configuration, tasks, actions, and state management classes.

The type system uses TypedDict and Pydantic models to ensure type safety and validation
throughout the application lifecycle.
"""

from typing import Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from enum import Enum

class ChatSettings(BaseModel):
    """Configuration settings for chat interactions.
    
    Attributes:
        primary_model: The default language model to use (default: "llama")
        stream: Whether to stream responses (default: True)
        system_message: Custom system message to use (default: "")
        disable_system_message: Override to disable system messages (default: False)
    """
    primary_model: str = "llama"
    stream: bool = True
    system_message: str = ""
    disable_system_message: bool = False

class RAGSettings(BaseModel):
    """Settings for Retrieval-Augmented Generation functionality.
    
    Attributes:
        enabled: Whether RAG features are enabled (default: False)
    """
    enabled: bool = False
    # Add RAG-specific settings here

class Config(BaseModel):
    """Global configuration container.
    
    Attributes:
        chat_settings: Settings for chat interactions
        rag_settings: Settings for RAG functionality
    """
    chat_settings: ChatSettings = ChatSettings()
    rag_settings: RAGSettings = RAGSettings()

class Status(Enum):
    """Task and action status indicators.
    
    Values:
        NOT_STARTED: Initial state
        IN_PROGRESS: Currently executing
        DONE: Successfully completed
        FAILED: Failed to complete
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DONE = "done" 
    FAILED = "failed"

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

class ActionResult(TypedDict):
    """Result data from action execution.
    
    Attributes:
        success: Whether the action completed successfully
        data: Optional result data from the action
        error: Error message if action failed
    """
    success: bool
    data: Optional[dict]
    error: Optional[str]

class ActionType(Enum):
    """Supported action categories.
    
    Values:
        GENERATE: Text generation actions
        WEB_SEARCH: Internet search actions
        SAVE_DATA: Data persistence actions
        TOOL_CALL: External tool invocations
    """
    GENERATE = "generate"
    WEB_SEARCH = "web_search"
    SAVE_DATA = "save_data"
    TOOL_CALL = "tool_call"

class ActionArgs(TypedDict):
    """Base class for action arguments."""
    pass

class Action(TypedDict):
    """Action definition and state container.
    
    Attributes:
        type: The category of action
        status: Current execution status
        args: Action-specific arguments
        result: Execution result data
    """
    type: ActionType
    status: Status
    args: ActionArgs
    result: Optional[ActionResult]

class PlanActionType(Enum):
    """Planning-specific action types.
    
    Values:
        FETCH: Retrieve planning context
        CREATE_PLAN: Generate new plan
        REVISE_PLAN: Modify existing plan
    """
    FETCH = "fetch"
    CREATE_PLAN = "create_plan" 
    REVISE_PLAN = "revise_plan"

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

class PlanActionArgs(ActionArgs):
    """Arguments for planning actions.
    
    Attributes:
        plan_context: Background information for planning
        proposed_plan: Draft plan under consideration
        plan: Finalized task plan
    """
    plan_context: Optional[str]
    proposed_plan: Optional[dict[str, Task]]
    plan: Optional[dict[str, Task]]

class GenerateActionArgs(ActionArgs):
    """Arguments for generation actions.
    
    Attributes:
        stream: Whether to stream generation output
        chain: LangChain chain to use
        messages: Conversation history
    """
    stream: Optional[bool]
    chain: Optional[Any]
    messages: Optional[list[dict]]
    # hyperparameters: Optional[dict]

class AgentState(TypedDict, total=True):
    """Core agent runtime state.
    
    Attributes:
        config: Global configuration
        messages: Conversation history
        response_count: Number of agent responses
        active_chain: Currently executing chain
        tool_choice: Selected tool for execution
        user_input: Latest user input
        mock_inputs: Test inputs for simulation
        task_dict: Active tasks by ID
    """
    config: Config
    messages: list[dict]
    response_count: int
    active_chain: Optional[object]
    tool_choice: Optional[str]
    user_input: Optional[str]
    mock_inputs: list[str]
    task_dict: dict[str, Task]

class GraphState(TypedDict):
    """Overall graph execution state.
    
    Attributes:
        keys: Agent state container
        mutation_count: Number of state mutations
        is_done: Whether execution is complete
    """
    keys: AgentState
    mutation_count: int
    is_done: bool

class CommandType(Enum):
    """System command types.
    
    Values:
        QUIT: Exit the application
        HELP: Show help information
        CLEAR: Clear conversation history
        SETTINGS: Modify configuration
        SAVE: Save current state
        LOAD: Load saved state
        DEBUG: Toggle debug mode
        UNDO: Revert last action
        MODE: Change operation mode
        READ: Load external content
    """
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
    """System command parser and validator.
    
    Parses raw command strings into structured command objects.
    Commands must be prefixed with '/' and may include arguments.
    
    Attributes:
        command: The base command name
        args: Optional command arguments
        
    Properties:
        is_valid: Whether command is recognized
        type: Parsed command type
    """
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
