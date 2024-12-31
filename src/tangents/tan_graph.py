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

logging.basicConfig(level=logging.WARNING)
from uuid import uuid4
import asyncio
import argparse
from typing import Dict, Any

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
from .output_handlers import OutputProcessor
from .input_handlers import InterruptHandler

# Configuration
RECURSION_LIMIT = 50
DEV_MODE = False
DEFAULT_AGENT_STATE = INITIAL_STATE_DICT if not DEV_MODE else PLANNING_STATE_DICT

def create_workflow() -> CompiledStateGraph:
    """Create and compile the async conversation workflow graph."""
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
    
    return workflow.compile(checkpointer=MemorySaver())

async def process_interrupt(interrupt_value: dict) -> str:
    """Handle workflow interrupts and get user input asynchronously."""
    if "prompt" not in interrupt_value:
        raise ValueError("Unhandled interrupt case - missing prompt")
    
    loop = asyncio.get_running_loop()
    user_input = await loop.run_in_executor(None, input, f"{interrupt_value['prompt']}\n")
    user_input = user_input.strip()
    
    if not user_input:
        print("No input provided, click enter again to quit...")
        user_input = await loop.run_in_executor(None, input)
        user_input = user_input.strip() or "/quit"
            
    return user_input

# TODO: Add a way to check interrupt_type and handle it accordingly
async def process_workflow_output_streaming(output: Dict[str, Any], app: CompiledStateGraph, app_config: dict, stream_mode: str = "updates"):
    """Process streaming output from workflow execution."""
    output_processor = OutputProcessor()
    interrupt_handler = InterruptHandler()

    async def handle_interrupt_stream(interrupt_value, depth=0):
        """Handle nested interrupt streams recursively."""
        if depth > app_config.get("recursion_limit", RECURSION_LIMIT):
            raise RecursionError("Maximum interrupt depth exceeded")
            
        user_input = await interrupt_handler.process_interrupt(interrupt_value)
        
        # Process resumed workflow with streaming
        async for chunk in app.astream(
            ResumeCommand(resume=user_input),
            app_config,
            stream_mode="updates"
        ):
            for node, updates in chunk.items():
                if node == "__interrupt__":
                    # TODO: Double check if this is correct
                    await handle_interrupt_stream(updates[0].value, depth + 1)
                else:
                    output_processor.process_updates(node, updates)

    if stream_mode == "updates":
        for key, value in output.items():
            if key == "__interrupt__":
                await handle_interrupt_stream(value[0].value)
            else:
                output_processor.process_updates(key, value)
    else:
        # if task dict is non-empty, and current task is in progress, process values
        output_processor.process_values(output)

async def main_async(stream_mode: str = "updates"):
    """Run the async workflow with streaming support."""
    app = create_workflow()
    
    # Initialize graph state
    graph_state = INITIAL_GRAPH_STATE.copy()
    graph_state["keys"] = DEFAULT_AGENT_STATE

    # Configure app settings
    app_config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": uuid4()}
    }

    # Process workflow outputs with streaming
    async for output in app.astream(graph_state, app_config, stream_mode=stream_mode):
        await process_workflow_output_streaming(output, app, app_config, stream_mode)
    
    print("Workflow completed.")

def main():
    """Run the workflow with command line configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream-mode",
        choices=["values", "updates"],
        default="updates",
        help="Streaming mode for workflow output"
    )
    args = parser.parse_args()
    
    asyncio.run(main_async(args.stream_mode))

def set_env_vars():
    """Load required environment variables from .env file."""
    from dotenv import load_dotenv
    load_dotenv()

if __name__ == "__main__":
    set_env_vars()
    main()