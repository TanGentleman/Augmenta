"""
Node implementations for the workflow graph.

Each node processes GraphState and implements workflow logic. Nodes take a GraphState
input and return an updated state. These functions are meant to be used as StateGraph
nodes and should not be called directly.

See tan_graph.py for workflow architecture details.
"""

import logging
from typing import Literal, Optional

from langgraph.types import interrupt

from tangents.classes.actions import ActionType, Status
from tangents.classes.commands import Command
from tangents.classes.states import GraphState
from tangents.classes.tasks import Task
from tangents.core.handle_action import (
    execute_action,
    handle_action_result,
    start_action,
)
from tangents.core.handle_user_input import execute_command, handle_user_message
from tangents.utils.action_utils import (
    is_human_action_next,
    is_stash_action_next,
    save_action_data,
)
from tangents.utils.message_utils import user_input_is_command
from tangents.utils.task_utils import (
    delete_tasks,
    get_task,
    save_completed_tasks,
    save_failed_tasks,
    save_stashed_tasks,
    start_task,
)




def start_node(state: GraphState) -> GraphState:
    """Initialize graph state with config and default task."""

    def validate_start_node(state: GraphState) -> None:
        """Validate fields in state dictionary."""
        if state['is_done']:
            raise ValueError('GraphState is_done must be False before start node!')
        if state['mutation_count'] > 0:
            logging.warning(f'Mutation count is {state["mutation_count"]} at start node!')

        state_dict = state['keys']
        if not state_dict['config']:
            raise ValueError('Config is not set!')
        if not state_dict['task_dict']:
            raise ValueError('Agent must have at least one task!')

        # Add assertions for entering a graph
        return

    validate_start_node(state)
    state_dict = state['keys']

    return {
        'keys': state_dict,
        'mutation_count': state['mutation_count'] + 1,
        'is_done': False,
    }


def agent_node(state: GraphState) -> GraphState:
    """
    Manage agent state and handle failure conditions.

    Checks action and mutation counts against maximums and fails tasks if exceeded.
    """
    # NOTE: Can add validation function here
    state_dict = state['keys']
    config = state_dict['config']
    current_task = get_task(state_dict['task_dict'])
    if current_task:
        if state['is_done']:
            logging.warning('Early exit: task is still in progress!')
        elif state_dict['action_count'] > config.workflow_settings.max_actions:
            logging.error('Action count exceeded max actions, marking task as failed')
            current_task['status'] = Status.FAILED
        elif state['mutation_count'] > config.workflow_settings.max_mutations:
            logging.error('Mutation count exceeded, marking task as failed')
            current_task['status'] = Status.FAILED

    return {
        'keys': state_dict,
        'mutation_count': state['mutation_count'] + 1,
        'is_done': state['is_done'],
    }


def task_manager_node(state: GraphState) -> GraphState:
    """
    Manage task lifecycle transitions and cleanup.

    Handles:
    - Saving/removing completed and failed tasks
    - Starting next available task with proper initialization
    - Task stashing when requested
    - Workflow termination when no tasks remain

    Task States: NOT_STARTED -> IN_PROGRESS -> DONE/FAILED
    """

    # NOTE: Can add validation function here
    state_dict = state['keys']
    task_dict = state_dict['task_dict']
    config = state_dict['config']

    # Process completed/failed tasks
    # NOTE: Double check this implementation
    # NOTE: Not yet parallelized.

    # Process completed tasks
    completed_task_names = save_completed_tasks(task_dict)
    if completed_task_names:
        logging.info(f'Saved done tasks: {completed_task_names}')
        delete_tasks(task_dict, completed_task_names)

    # Process failed tasks
    failed_task_names = save_failed_tasks(task_dict)
    if failed_task_names:
        logging.warning(f'Saved failed tasks: {failed_task_names}')
        delete_tasks(task_dict, failed_task_names)

    # Get/start next task
    current_task = get_task(task_dict)
    if not current_task:
        unstarted_task = get_task(task_dict, status=Status.NOT_STARTED)

        if not unstarted_task:
            logging.info('No tasks remaining!')
            return {
                'keys': state_dict,
                'mutation_count': state['mutation_count'] + 1,
                'is_done': True,
            }

        current_task = start_task(unstarted_task, config)
        assert current_task['status'] == Status.IN_PROGRESS

    # Handle task stashing
    if is_stash_action_next(current_task['actions']):
        stashed_task_names = save_stashed_tasks(task_dict)
        if stashed_task_names:
            logging.info(f'Stashed tasks: {stashed_task_names}')
            for name in stashed_task_names:
                del task_dict[name]

    if not task_dict:
        logging.info('No tasks remaining!')
        is_done = True
    else:
        is_done = False

    return {
        'keys': state_dict,
        'mutation_count': state['mutation_count'] + 1,
        'is_done': is_done,
    }


def human_node(state: GraphState) -> GraphState:
    """
    Handle human input through interrupts or mock inputs.

    Uses langgraph interrupt mechanism to pause execution and get user input.
    Supports mock inputs for testing.
    """

    def validate_human_node(state: GraphState) -> Task:
        """Validate fields in state dictionary."""
        state_dict = state['keys']
        current_task = get_task(state_dict['task_dict'])
        if not current_task:
            raise ValueError('No in-progress task found')
        return current_task

    current_task = validate_human_node(state)
    state_dict = state['keys']
    mock_inputs = state_dict['mock_inputs']

    # Handle mock inputs for testing, otherwise get real user input
    if mock_inputs:
        print(f'Mock inputs: {mock_inputs}')
        user_input = mock_inputs.pop(0)
    else:
        # Initialize default interrupt context
        interrupt_context = {'prompt': 'Enter your message (or /help for commands):'}

        if is_human_action_next(current_task['actions']):
            custom_interrupt_prompt = current_task['actions'][0]['args'].get('prompt')
            if custom_interrupt_prompt:
                interrupt_context['prompt'] = custom_interrupt_prompt

        # Get user input via interrupt
        user_input = interrupt(interrupt_context)

    state_dict['user_input'] = user_input

    return {
        'keys': state_dict,
        'mutation_count': state['mutation_count'] + 1,
        'is_done': False,
    }


# TODO: Refactor this for best extensibility
# NOTE: May change user_inputs type to dict instead of str from the human node.
# This could include multiple updates from the Interrupt in the human node.
def processor_node(state: GraphState) -> GraphState:
    """
    Process user input and update task state.

    Handles:
    - Command vs message detection
    - Task-specific message processing
    - Action creation based on task type
    """

    def validate_processor_node(state: GraphState) -> tuple[Task, str]:
        """Validate fields in state dictionary."""
        state_dict = state['keys']
        current_task = get_task(state_dict['task_dict'])
        user_input = state_dict['user_input']

        if not current_task:
            raise ValueError('No in-progress task found')
        if not user_input:
            raise ValueError('No user input found')

        return current_task, user_input

    current_task, user_input = validate_processor_node(state)

    state_dict = state['keys']
    config = state_dict['config']

    if is_human_action_next(current_task['actions']):
        current_task['actions'].pop(0)

    # Catch command input
    if user_input_is_command(user_input):
        logging.info('Command detected!')
    else:
        handle_user_message(user_input, current_task, config)

    return {
        'keys': state_dict,
        'mutation_count': state['mutation_count'] + 1,
        'is_done': False,
    }


async def action_node(state: GraphState, config: Optional[dict] = None) -> GraphState:
    """
    Execute and manage task actions.

    Handles action lifecycle:
    - Validates task state
    - Executes next action
    - Processes results and updates state
    """

    def validate_action_node(state: GraphState) -> Task:
        """Validate task has queued non-stash actions."""
        state_dict = state['keys']
        current_task = get_task(state_dict['task_dict'])
        if not current_task:
            raise ValueError('No in-progress task found')
        if not current_task['actions']:
            raise ValueError('No actions found for in-progress task.')

        action = current_task['actions'][0]
        if action['type'] == ActionType.STASH:
            raise ValueError('Stash action should be handled in task manager')

        return current_task

    current_task = validate_action_node(state)

    # Get and validate state
    state_dict = state['keys']
    action = current_task['actions'][0]

    # Execute action
    if action['status'] == Status.NOT_STARTED:
        action = start_action(action, current_task)
    assert action['status'] == Status.IN_PROGRESS
    
    # Set up runtime stream config
    if config is not None and action['type'] == ActionType.GENERATE:
        stream_callback = config.get('configurable', {}).get('stream_callback')
        print(f"stream_callback: {bool(stream_callback)}")
        action['args']['stream_callback'] = stream_callback
        # exit()
    logging.info(f"Executing action: {action['type']}")
    result = await execute_action(action)
    logging.info(result)

    # Update state
    current_task = handle_action_result(current_task, result)
    if result['success']:
        state_dict['action_count'] += 1

    if action['status'] in [Status.DONE, Status.FAILED]:
        completed_action = current_task['actions'].pop(0)
        save_action_data(completed_action)

        if action['status'] == Status.FAILED:
            current_task['status'] = Status.FAILED
            logging.error('Action failed, marking task as failed')
    else:
        logging.warning('Action still in progress - may not be DB-serializable')

    return {
        'keys': state_dict,
        'mutation_count': state['mutation_count'] + 1,
        'is_done': False,
    }


async def execute_command_node(state: GraphState) -> GraphState:
    """
    Process command input and update state.

    Handles system commands like:
    - Task management (quit, save, load)
    - UI commands (help, clear, debug)
    - Mode switching and settings
    """

    def validate_command_node(state: GraphState) -> tuple[Task, Command]:
        """Validate fields in state dictionary."""
        state_dict = state['keys']
        current_task = get_task(state_dict['task_dict'])
        user_input = state_dict['user_input']

        if not current_task:
            raise ValueError('No in-progress task found')
        if not user_input:
            raise ValueError('No user input found')

        command = Command(user_input)
        if not command.is_valid:
            raise ValueError(f'Invalid command: {command.command}')

        return current_task, command

    current_task, command = validate_command_node(state)
    state_dict = state['keys']
    state_dict['user_input'] = None
    logging.info(f'Executing command: {command.command}')

    try:
        await execute_command(command, current_task, state_dict)
    except Exception as e:
        logging.error(f'Command execution failed: {e}')
        logging.critical('This should never happen!')
        # NOTE: Can set is_done to True here if we want to fail the graph

    return {'keys': state['keys'], 'mutation_count': state['mutation_count'] + 1, 'is_done': False}


def decide_from_agent(
    state: GraphState,
) -> Literal['human_node', 'task_manager', 'action_node', 'end_node']:
    """Route from agent node based on state conditions."""
    if state['is_done']:
        return 'end_node'

    state_dict = state['keys']
    task_dict = state_dict['task_dict']

    current_task = get_task(task_dict)
    if not current_task:
        return 'task_manager'
    if not current_task['actions']:
        return 'human_node'
    if is_stash_action_next(current_task['actions']):
        return 'task_manager'
    elif is_human_action_next(current_task['actions']):
        return 'human_node'
    else:
        return 'action_node'


def decide_from_processor(
    state: GraphState,
) -> Literal['execute_command', 'agent_node']:
    """Route from processor based on input type (command vs message)."""
    state_dict = state['keys']
    user_input = state_dict['user_input']
    if user_input is None:
        raise ValueError('No user input found')
    return 'execute_command' if user_input_is_command(user_input) else 'agent_node'
