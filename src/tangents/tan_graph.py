"""
Directed graph implementation for stateful workflows.

Implements a state machine that coordinates processing steps through a directed graph
while maintaining context and task state. The graph manages transitions between nodes
based on state conditions and decision functions.

Core Components:
- GraphState: Container for workflow state and configuration
- Processing Nodes: Discrete operation handlers
  - start_node: Validates and initializes configuration
  - agent_node: Makes high-level workflow decisions
  - task_manager: Manages task lifecycle
  - human_node: Handles external input
  - processor_node: Routes input to handlers
  - action_node: Executes task operations
  - execute_command: Processes system commands

Flow:
START -> Initialize -> Decision -> [
    Input/Process |
    Task Management |
    Action Execution |
    END
]

Configuration:
- RECURSION_LIMIT: Maximum graph traversal depth (default: 50)
- MAX_MUTATIONS: Maximum state mutations per run (default: 50)
- MAX_ACTIONS: Maximum actions per task (default: 5)
"""
import logging
from uuid import uuid4
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from tangents.template import INITIAL_GRAPH_STATE, INITIAL_STATE_DICT, PLANNING_STATE_DICT
from .classes.states import GraphState
from .nodes import (
    start_node, agent_node, task_manager_node, human_node,
    processor_node, execute_command_node, action_node,
    decide_from_agent, decide_from_processor
)

# Maximum recursion depth for graph traversal
RECURSION_LIMIT = 50
DEFAULT_AGENT_STATE = PLANNING_STATE_DICT

def create_workflow() -> CompiledStateGraph:
    """Create and compile the conversation workflow graph.
    
    Constructs a directed graph with nodes for processing different aspects of the conversation:
    - State initialization and configuration
    - Agent coordination and decision making
    - Task and action management
    - User interaction and input processing
    
    Returns:
        CompiledStateGraph: Executable workflow graph
    """
    workflow = StateGraph(GraphState)
    
    # Add core processing nodes
    workflow.add_node("start_node", start_node)
    workflow.add_node("agent_node", agent_node) 
    workflow.add_node("task_manager", task_manager_node)
    workflow.add_node("human_node", human_node)
    workflow.add_node("processor_node", processor_node)
    workflow.add_node("execute_command", execute_command_node)
    workflow.add_node("action_node", action_node)
    
    # Define main graph structure
    workflow.add_edge(START, "start_node")
    workflow.add_edge("start_node", "agent_node")
    
    # Add conditional routing from agent node
    workflow.add_conditional_edges(
        "agent_node",
        decide_from_agent,
        {
            "human_node": "human_node",
            "task_manager": "task_manager",
            "action_node": "action_node", 
            "end_node": END
        }
    )
    
    # Add input processing flow
    workflow.add_edge("human_node", "processor_node")
    workflow.add_conditional_edges(
        "processor_node",
        decide_from_processor,
        {
            "execute_command": "execute_command",
            "agent_node": "agent_node"
        }
    )
    
    # Add return paths to agent node
    workflow.add_edge("execute_command", "agent_node")
    workflow.add_edge("action_node", "agent_node")
    workflow.add_edge("task_manager", "agent_node")
    
    return workflow.compile()

def main():
    """Run the workflow with initial state and configuration."""
    app = create_workflow()
    
    # Initialize graph state
    initial_graph_state: GraphState = {
        "keys": {},
        "mutation_count": 0,
        "is_done": False
    }
    assert initial_graph_state == INITIAL_GRAPH_STATE
    
    graph_state = initial_graph_state
    graph_state["keys"] = DEFAULT_AGENT_STATE

    # Configure app settings
    app_config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": uuid4()}
    }
    
    # Process and display workflow outputs
    for output in app.stream(graph_state, app_config):
        for key, value in output.items():
            print(f"\nNode: {key}")
            print(f"Mutations: {value['mutation_count']}")
            
            if 'task_dict' in value['keys']:
                print("\nTask Status:")
                for task_name, task in value['keys']['task_dict'].items():
                    print(f"{task_name}: {task['status']}")
                    break
        
        print("\n---\n")
    
    print("Workflow completed.")

def set_env_vars():
    """Load required environment variables from .env file.
    
    Required variables:
    - LANGCHAIN_TRACING_V2: Tracing configuration
    - LANGCHAIN_API_KEY: LangChain API credentials
    - LANGCHAIN_PROJECT: Project identifier
    - LITELLM_API_KEY: LiteLLM API credentials
    """
    from dotenv import load_dotenv
    load_dotenv()

if __name__ == "__main__":
    set_env_vars()
    main()