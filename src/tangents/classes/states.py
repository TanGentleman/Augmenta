from typing import Any, Optional
from typing_extensions import TypedDict
from tangents.classes.settings import Config
from tangents.classes.tasks import Task

class AgentState(TypedDict):
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
    # Config can include overrides, like FORCE_LOCAL_MODELS.
    config: Config
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

# Should task states be an instance of TaskState in tasks.py?
class ChatTaskState(TypedDict):
    """Task-specific state for chat tasks.
    
    Attributes:
        messages: Conversation history
        response_count: Number of agent responses
        active_chain: Currently executing chain
        tool_choice: Selected tool for execution
        user_input: Latest user input
        mock_inputs: Test inputs for simulation
    """
    messages: list[dict]
    response_count: int
    active_chain: Optional[object]
    tool_choice: Optional[str]

class PlanState(TypedDict):
    """Arguments for planning actions.
    
    Attributes:
        plan_context: Background information for planning
        proposed_plan: Draft plan under consideration
        plan: Finalized task plan
    """
    plan_context: Optional[str]
    proposed_plan: Optional[dict[str, Task]]
    plan: Optional[dict[str, Task]]
