from typing import Any, Optional
from typing_extensions import TypedDict
from enum import Enum

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
    STASH = "stash"

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