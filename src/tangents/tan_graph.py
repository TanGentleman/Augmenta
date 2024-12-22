"""Directed graph implementation for managing conversational AI workflows.

This module provides a state machine that coordinates conversations between users and AI agents,
handling tasks, actions, and system commands. The graph manages state transitions between
different processing nodes while maintaining conversation context and task state.

Components:
    - GraphState: State container for messages, tasks, and config
    - Nodes: Processing steps (agent, human, action handlers)
    - Decision Functions: Route graph flow based on state
    - Task Manager: Manages task lifecycle
    - Action Executor: Handles individual task actions

Example:
    >>> app = create_workflow()
    >>> state = {"keys": {}, "mutation_count": 0, "is_done": False}
    >>> app.stream(state)

Flow:
    START -> start_node (init) -> agent_node (coordinator) -> [
        human_node (input) |
        task_manager (tasks) |
        action_node (execution) |
        END (termination)
    ]

Configuration:
    RECURSION_LIMIT: Maximum graph traversal depth (default: 50)
    MAX_MUTATIONS: Maximum state mutations per run (default: 50)
    MAX_ACTIONS: Maximum actions per task (default: 5)
"""
import logging
from uuid import uuid4
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from tangents.template import INITIAL_GRAPH_STATE
from .classes.graph_classes import GraphState
from .nodes import (
    start_node, agent_node, task_manager_node, human_node,
    processor_node, execute_command_node, action_node,
    decide_from_agent, decide_from_processor
)

# Maximum recursion depth for graph traversal
RECURSION_LIMIT = 50

def create_workflow() -> CompiledStateGraph:
    """Create and compile the conversation workflow graph.
    
    Constructs a directed graph with nodes for:
    - State initialization
    - Agent coordination
    - Task management
    - User interaction
    - Input processing
    - Action execution
    
    Returns:
        CompiledStateGraph: Executable workflow graph
    """
    workflow = StateGraph(GraphState)
    
    # Core processing nodes
    workflow.add_node("start_node", start_node)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("task_manager", task_manager_node)
    workflow.add_node("human_node", human_node)
    workflow.add_node("processor_node", processor_node)
    workflow.add_node("execute_command", execute_command_node)
    workflow.add_node("action_node", action_node)
    
    # Graph structure
    workflow.add_edge(START, "start_node")
    workflow.add_edge("start_node", "agent_node")
    
    # Agent decision routing
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
    
    # Input processing flow
    workflow.add_edge("human_node", "processor_node")
    workflow.add_conditional_edges(
        "processor_node",
        decide_from_processor,
        {
            "execute_command": "execute_command",
            "agent_node": "agent_node"
        }
    )
    
    # Return paths to agent
    workflow.add_edge("execute_command", "agent_node")
    workflow.add_edge("action_node", "agent_node")
    workflow.add_edge("task_manager", "agent_node")
    
    return workflow.compile()

def main():
    """Execute the workflow with initial state and configuration."""
    app = create_workflow()
    
    initial_state: GraphState = {
        "keys": {},
        "mutation_count": 0,
        "is_done": False
    }
    assert initial_state == INITIAL_GRAPH_STATE

    app_config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": uuid4()}
    }
    
    # Process workflow outputs
    for output in app.stream(initial_state, app_config):
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
    """Load environment variables from .env file.
    
    Required variables:
        LANGCHAIN_TRACING_V2: Tracing configuration
        LANGCHAIN_API_KEY: LangChain API credentials  
        LANGCHAIN_PROJECT: Project identifier
        LITELLM_API_KEY: LiteLLM API credentials
    """
    from dotenv import load_dotenv
    load_dotenv()

if __name__ == "__main__":
    set_env_vars()
    main()