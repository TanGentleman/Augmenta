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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command as ResumeCommand

from tangents.template import INITIAL_GRAPH_STATE, INITIAL_STATE_DICT, PLANNING_STATE_DICT
from .classes.states import GraphState
from .nodes import (
    start_node, agent_node, task_manager_node, human_node,
    processor_node, execute_command_node, action_node,
    decide_from_agent, decide_from_processor
)

# Maximum recursion depth for graph traversal
RECURSION_LIMIT = 50
DEFAULT_AGENT_STATE = INITIAL_STATE_DICT

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
    
    return workflow.compile(
        checkpointer=MemorySaver()
    )

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
    unique_thread = {"thread_id": uuid4()}
    thread_config = {
        "configurable": unique_thread
    }
    app_config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": unique_thread
    }
    
    def process_output(output, app_config):
        for key, value in output.items():
            if key == "__interrupt__":
                # print("interrupt received!")
                interrupt_value = value[0].value
                interrupt_prompt = ""
                if "prompt" in interrupt_value:
                    interrupt_prompt = interrupt_value["prompt"]
                else:
                    # TODO: Handle other interrupt cases
                    raise ValueError("Unhandled interrupt case")
                # current_state = app.get_state(thread_config)
                # print(f"Current state: {current_state.values}")
                # print(f"Current tasks: {current_state.tasks}")
                user_input = input(f"{interrupt_prompt}\n").strip()
                if not user_input:
                    print("No input provided, click enter again to quit...")
                    user_input = input().strip()
                    if not user_input:
                        user_input = "/quit"

                for interrupt_output in app.stream(ResumeCommand(resume=user_input), app_config):
                    process_output(interrupt_output, app_config)
                return
            
            print(f"\nNode: {key}")
            print(f"Hops: {value['mutation_count']}")
                
            if 'task_dict' in value['keys']:
                for task_name, task in value['keys']['task_dict'].items():
                    print(f"{task_name}: {task['status'].name}")
                    break
        
        print("\n---\n")

    # Process and display workflow outputs 
    for output in app.stream(graph_state, app_config):
        process_output(output, app_config)
    
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