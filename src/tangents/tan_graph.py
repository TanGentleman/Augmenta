"""
Directed Graph Workflow System

A state machine implementation that coordinates processing steps through a directed graph
while maintaining workflow context and task state. The system manages transitions between
nodes based on state conditions and decision functions.

Key Components:
1. Graph Structure
   - Nodes represent discrete processing steps (e.g. agent, task manager, input handler)
   - Edges define valid transitions between nodes
   - Conditional routing based on state and decisions

2. State Management
   - GraphState tracks workflow context and configuration
   - Task lifecycle management
   - Input/output handling

3. Processing Flow
   START -> Initialize -> Decision Loop -> END
   Decision Loop:
   - Process user input
   - Manage tasks
   - Execute actions

Configuration Options:
- RECURSION_LIMIT: Max graph traversal depth (default: 50)
- Stream modes: "updates" or "values"
- Development mode toggle

Usage:
    python -m tangents.tan_graph --stream-mode updates
"""

import argparse
import asyncio
import logging
from typing import Any, Dict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command as ResumeCommand

from tangents.template import (
    get_default_state_dict,
    get_initial_graph_state,
    get_planning_state_dict,
)

from .classes.states import GraphState
from .input_handlers import InterruptHandler
from .nodes import (
    action_node,
    agent_node,
    decide_from_agent,
    decide_from_processor,
    execute_command_node,
    human_node,
    processor_node,
    start_node,
    task_manager_node,
)
from .output_handlers import OutputProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)

# System configuration
RECURSION_LIMIT = 50
DEV_MODE = True
if DEV_MODE:
    print('Dev mode enabled. Using planning state dict.')


def create_workflow() -> CompiledStateGraph:
    """
    Create and compile the workflow directed graph.

    Defines the core graph structure including:
    - Processing nodes for each workflow step
    - Direct edges between sequential nodes
    - Conditional edges based on decision functions

    Returns:
        CompiledStateGraph: The compiled workflow graph
    """
    workflow = StateGraph(GraphState)

    # Register all processing nodes
    workflow.add_node('start_node', start_node)
    workflow.add_node('agent_node', agent_node)
    workflow.add_node('task_manager', task_manager_node)
    workflow.add_node('human_node', human_node)
    workflow.add_node('processor_node', processor_node)
    workflow.add_node('execute_command', execute_command_node)
    workflow.add_node('action_node', action_node)

    # Define primary workflow path
    workflow.add_edge(START, 'start_node')
    workflow.add_edge('start_node', 'agent_node')

    # Configure agent node routing
    workflow.add_conditional_edges(
        'agent_node',
        decide_from_agent,
        {
            'human_node': 'human_node',
            'task_manager': 'task_manager',
            'action_node': 'action_node',
            'end_node': END,
        },
    )

    # Configure input handling flow
    workflow.add_edge('human_node', 'processor_node')
    workflow.add_conditional_edges(
        'processor_node',
        decide_from_processor,
        {'execute_command': 'execute_command', 'agent_node': 'agent_node'},
    )

    # Add return paths to agent node
    workflow.add_edge('execute_command', 'agent_node')
    workflow.add_edge('action_node', 'agent_node')
    workflow.add_edge('task_manager', 'agent_node')

    return workflow.compile(checkpointer=MemorySaver())


async def process_interrupt(interrupt_value: dict) -> str:
    """
    Handle workflow interrupts and collect user input.

    Args:
        interrupt_value: Dictionary containing interrupt details

    Returns:
        str: User input response

    Raises:
        ValueError: If interrupt_value lacks required 'prompt' field
    """
    if 'prompt' not in interrupt_value:
        raise ValueError('Unhandled interrupt case - missing prompt')

    loop = asyncio.get_running_loop()
    prompt = interrupt_value.get('prompt', 'User input:')
    user_input = await loop.run_in_executor(None, input, f"{prompt}\n")
    user_input = user_input.strip()

    if not user_input:
        print('No input provided, click enter again to quit...')
        user_input = await loop.run_in_executor(None, input)
        user_input = user_input.strip() or '/quit'

    return user_input

async def process_graph_updates(
    output: Dict[str, Any],
    app: CompiledStateGraph,
    app_config: dict,
) -> None:
    """
    Process streaming updates from workflow execution.

    Args:
        output: Workflow output data
        app: Compiled workflow graph
        app_config: Application configuration
    """
    output_processor = OutputProcessor()

    async def handle_interrupt(interrupt_value: dict) -> None:
        """Handle interrupt by getting user input and processing resulting stream."""
        user_input = await process_interrupt(interrupt_value)
        
        async for chunk in app.astream(
            ResumeCommand(resume=user_input), 
            app_config,
            stream_mode='updates'
        ):
            for node, updates in chunk.items():
                if node == '__interrupt__':
                    await handle_interrupt(updates[0].value)
                else:
                    output_processor.process_updates(node, updates)

    for key, value in output.items():
        if key == '__interrupt__':
            await handle_interrupt(value[0].value)
        else:
            updates = output_processor.process_updates(key, value)


def process_graph_values(output: GraphState) -> None:
    """
    Process complete graph state output.

    Args:
        output: Complete graph state
    """
    output_processor = OutputProcessor()
    output_processor.process_values(output)


async def main_async(stream_mode: str = 'updates') -> None:
    """
    Run the async workflow with streaming support.

    Args:
        stream_mode: Output processing mode ("updates" or "values")
    """
    app = create_workflow()

    # Initialize workflow state
    graph_state = get_initial_graph_state()
    graph_state['keys'] = get_default_state_dict() if not DEV_MODE else get_planning_state_dict()

    # Configure runtime settings
    app_config = {
        'recursion_limit': RECURSION_LIMIT,
        'configurable': {'thread_id': uuid4()},
    }

    # Process workflow outputs
    async for output in app.astream(graph_state, app_config, stream_mode=stream_mode):
        if stream_mode == 'updates':
            await process_graph_updates(output, app, app_config)
        else:
            process_graph_values(output)

    print('Workflow completed.')


def main() -> None:
    """Command-line entry point with configuration options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stream-mode',
        choices=['values', 'updates'],
        default='updates',
        help='Streaming mode for workflow output',
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.stream_mode))


def set_env_vars() -> None:
    """Load environment configuration from .env file."""
    from dotenv import load_dotenv

    load_dotenv()


if __name__ == '__main__':
    set_env_vars()
    main()
