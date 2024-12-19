"""
State machine graph for managing conversational AI interactions.

This module implements a directed graph that manages the flow of conversation, task execution,
and action handling for an AI agent. The graph coordinates multiple components including:

- User input/output handling
- Task management and execution
- Action processing (generation, web search, etc.)
- Command processing (system commands prefixed with /)
- State management and persistence

Key Components:
- GraphState: Main state container tracking messages, tasks, config
- Nodes: Discrete processing steps (agent, human, action, etc.)
- Decision Functions: Route flow between nodes based on state
- Task Manager: Handles task lifecycle (start, complete, fail)
- Action Executor: Processes individual actions within tasks

Usage:
    app = create_workflow()
    initial_state = {
        "keys": {},
        "mutation_count": 0, 
        "is_done": False
    }
    app.stream(initial_state)

Graph Flow:
START -> start_node: Initialize state
start_node -> agent_node: Central coordinator
agent_node -> human_node: Get user input
agent_node -> task_manager: Handle tasks
agent_node -> action_node: Execute actions
agent_node -> END: Terminate workflow

Configuration:
- RECURSION_LIMIT: Max graph traversal depth (default: 50)
- MAX_MUTATIONS: Max state mutations (default: 50) # TODO: Move to config
- MAX_ACTIONS: Max actions per task (default: 5) # TODO: Move to config

See Also:
    GraphState: Main state container class
    Action: Action definition and execution
    Command: System command processing
"""
import logging
from uuid import uuid4
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph


from tangents.template import INITIAL_GRAPH_STATE
from .graph_classes import GraphState

# Constants
from .nodes import start_node, agent_node, task_manager_node, human_node, processor_node, execute_command_node, action_node
from .nodes import decide_from_agent, decide_from_processor
# Constants
RECURSION_LIMIT = 50

# Build graph
def create_workflow() -> CompiledStateGraph:
    """
    Creates and compiles the task management workflow graph.
    
    The workflow consists of several interconnected nodes:
    - start_node: Initializes graph state and configuration
    - agent_node: Central node managing agent state and decisions
    - task_manager: Handles task lifecycle and completion
    - human_node: Processes user input
    - processor_node: Routes input to appropriate handlers
    - action_node: Executes task-specific actions
    
    Returns:
        CompiledStateGraph: Compiled workflow ready for execution
    """
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("start_node", start_node)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("task_manager", task_manager_node)
    workflow.add_node("human_node", human_node)
    workflow.add_node("processor_node", processor_node)
    workflow.add_node("execute_command", execute_command_node)
    workflow.add_node("action_node", action_node)
    
    # Add edges
    # workflow.set_entry_point("start_node")
    workflow.add_edge(START, "start_node")
    workflow.add_edge("start_node", "agent_node")
    
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
    
    workflow.add_edge("human_node", "processor_node")
    workflow.add_conditional_edges(
        "processor_node",
        decide_from_processor,
        {
            "execute_command": "execute_command",
            "agent_node": "agent_node"
        }
    )
    workflow.add_edge("execute_command", "agent_node")
    workflow.add_edge("action_node", "agent_node")
    workflow.add_edge("task_manager", "agent_node")
    
    app = workflow.compile()
    return app

def main():
    """Main execution function"""
    app = create_workflow()
    
    initial_state: GraphState = {
        "keys": {},
        "mutation_count": 0,
        "is_done": False
    }
    assert initial_state == INITIAL_GRAPH_STATE

    app_config = {"recursion_limit": RECURSION_LIMIT,
                  "configurable": {"thread_id": uuid4()}}
    
    for output in app.stream(initial_state, app_config):
        for key, value in output.items():
            print(f"\nNode: {key}")
            print(f"Mutations: {value['mutation_count']}")
            
            # Debug task state
            if 'task_dict' in value['keys']:
                print("\nTask Status:")
                for task_name, task in value['keys']['task_dict'].items():
                    print(f"{task_name}: {task['status']}")
                    break
        
        print("\n---\n")
    
    print("Workflow completed.")

def set_env_vars():
    """
    This function sets the environment variables for the workflow.

    Configures:
    - Provider keys
    - Langsmith tracing 

    Environment variables:
    ---
    LANGCHAIN_TRACING_V2=""
    LANGCHAIN_API_KEY=""
    LANGCHAIN_PROJECT=""
    LITELLM_API_KEY=""
    """
    from dotenv import load_dotenv
    load_dotenv()

if __name__ == "__main__":
    set_env_vars()
    main()
    # save_graph()