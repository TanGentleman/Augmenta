"""
Node functions for the conversational AI workflow graph.

This module contains all the node functions used in the state machine graph that manages
conversational AI interactions. Each node represents a discrete processing step in the workflow.

Node Types:
- start_node: Initializes graph state and configuration
- agent_node: Central coordinator managing agent state and decisions
- task_manager_node: Handles task lifecycle (start, complete, fail)
- human_node: Processes user input
- processor_node: Routes input to appropriate handlers
- execute_command_node: Processes system commands
- action_node: Executes task-specific actions

Each node function:
1. Receives a GraphState object containing the current state
2. Processes the state according to its responsibility
3. Returns an updated GraphState with any modifications

Usage:
    These nodes are used by the StateGraph in tan_graph.py to construct the workflow.
    They should not be called directly, but rather through the graph execution.

Dependencies:
    - GraphState: Main state container class
    - Action: Action definition and execution
    - Command: System command processing
    - Various utility functions for tasks, messages, and actions
"""

from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
import logging

from tangents.template import INITIAL_STATE_DICT, PLANNING_STATE_DICT
from tangents.utils.chains import get_summary_chain, get_llm
from tangents.utils.task_utils import get_task, save_completed_tasks, save_failed_tasks, start_next_task
from augmenta.utils import read_sample

from .classes.tasks import Task, TaskType
from .classes.actions import Action, PlanActionType, Status, ActionType
from .classes.commands import Command, CommandType
from .classes.states import GraphState, PlanState

from .utils.message_utils import insert_system_message, remove_last_message, clear_messages
from .utils.action_utils import execute_action, save_action_data, create_action

# Constants
MAX_MUTATIONS = 50
MAX_ACTIONS = 5
DEFAULT_STATE_DICT = INITIAL_STATE_DICT

def start_node(state: GraphState) -> GraphState:
    """Initialize the graph state with configuration and default task"""
    state_dict = DEFAULT_STATE_DICT

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }


def agent_node(state: GraphState) -> GraphState:
    """Manages agent state and decision making"""
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Handle initialization
    if state["mutation_count"] == 1:
        if not state_dict["config"].chat_settings.disable_system_message:
            insert_system_message(
                state_dict["messages"],
                state_dict["config"].chat_settings.system_message
            )            
    
    # Check failure conditions
    if state["mutation_count"] > MAX_MUTATIONS:
        logging.warning("Mutation count exceeded max mutations!")
        current_task = get_task(task_dict, status=Status.IN_PROGRESS)
        if current_task:
            logging.error("Task is still in progress, marking as failed")
            current_task["status"] = Status.FAILED

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": state["is_done"]
    }

def task_manager_node(state: GraphState) -> GraphState:
    """
    Manages the lifecycle of tasks in the system.
    
    Responsibilities:
    1. Saves completed tasks to persistent storage
    2. Removes completed/failed tasks from active dictionary
    3. Starts next available task if no task is in progress
    4. Determines if workflow should terminate
    
    Args:
        state (GraphState): Current graph state containing task dictionary
        
    Returns:
        GraphState: Updated state with modified task dictionary and completion status
    """
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Save completed tasks
    completed_task_names = save_completed_tasks(task_dict)
    if completed_task_names:
        logging.info(f"Saved completed tasks: {completed_task_names}")
        for task_name in completed_task_names:
            del task_dict[task_name]

    # Check for failed tasks
    failed_task_names = save_failed_tasks(task_dict)
    if failed_task_names:
        logging.error(f"Failed tasks: {failed_task_names}")
        for task_name in failed_task_names:
            del task_dict[task_name]
    
    is_running = start_next_task(task_dict)
    # Check task completion
    if not is_running:
        logging.info("No tasks remaining!")
        # TODO: Add logic to fetch new tasks
        return {
            "keys": state_dict,
            "mutation_count": state["mutation_count"] + 1,
            "is_done": True
        }

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def human_node(state: GraphState) -> GraphState:
    """Handles human input"""
    # TODO: Implement Interrupt for HITL tasks
    # https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/#interrupt
    # This is useful for brief HITL confirmations too. Potentially useful preceding action node?
    state_dict = state["keys"]
    mock_inputs = state_dict["mock_inputs"]
    if mock_inputs:
        print(f"Mock inputs: {mock_inputs}")
        user_input = mock_inputs.pop(0)
    else:
        user_input = input("Enter your message: ").strip()
        if not user_input:
            print("No input provided, click enter again to quit...")
            user_input = input().strip()
            if not user_input:
                user_input = "/quit"
    
    state_dict["user_input"] = user_input
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def processor_node(state: GraphState) -> GraphState:
    """Processes input and updates state"""
    state_dict = state["keys"]
    user_input = state_dict["user_input"]
    
    # TODO: Add logic to determine if we are in a HITL task and should not append to messages
    if not user_input.startswith('/'):
        state_dict["messages"].append(HumanMessage(content=user_input))
        task_dict = state_dict["task_dict"]
        current_task = next((task for task in task_dict.values() if task["status"] == Status.IN_PROGRESS), None)
        if not current_task:
            raise ValueError("No in-progress task found")
        
        # UPDATE STATE BASED ON TASK TYPE
        if current_task["type"] == TaskType.CHAT:
            if state_dict["active_chain"] is None:
                llm = get_llm(state_dict["config"].chat_settings.primary_model)
                if llm is not None:
                    state_dict["active_chain"] = llm
                else:
                    raise ValueError("Chain not initialized!")
            generate_action = create_action(ActionType.GENERATE)
            # TODO: Add messages / state variables to action args
            # NOTE: Will there be edge cases where we need to update state variables downstream?
            # generate_action["args"]["messages"] = state_dict["messages"]
            # generate_action["args"]["chain"] = state_dict["active_chain"]
            # generate_action["args"]["stream"] = state_dict["config"].chat_settings.stream
            
            # NOTE: This must NOT override action args with non-empty values
            # For example, don't override the chain if it's already set
            # What about stream settings changing later in state_dict["config"]?
            # imo, this is a good thing. We should have control over each Action's args.
            current_task["actions"].append(generate_action)

        elif current_task["type"] == TaskType.RAG:
            # TODO: Add RAG logic here
            if state_dict["config"].rag_settings.enabled:
                print("RAG task. Doing nothing for now.")
                pass
            else:
                print("ERROR: RAG task is disabled!")
                raise ValueError("RAG task is disabled!")
        elif current_task["type"] == TaskType.PLANNING:
            # TODO: Add planning logic here
            if state_dict["plan_dict"]["proposed_plan"]:
                if state_dict["plan_dict"]["plan"] is not None:
                    raise ValueError("Plan is already created!")
                print("Using your revision!")
            pass
        else:
            raise ValueError("Current task is not a chat task")
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }



def action_node(state: GraphState) -> GraphState:
    """Executes in-progress actions in the current task"""
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Find current in-progress task
    current_task = next(
        (task for task in task_dict.values() if task["status"] == Status.IN_PROGRESS),
        None
    )
    if not current_task:
        raise ValueError("No in-progress task found")
    if not current_task["actions"]:
        raise ValueError("No actions found for in-progress task.")
    

    # Get next action and execute it
    action = current_task["actions"][0]
    action_type = action["type"]
    if action["status"] == Status.NOT_STARTED:
        action["status"] = Status.IN_PROGRESS
    assert action["status"] == Status.IN_PROGRESS

    ### Load required state variables into action args ###

    logging.info(f"Executing action: {action}")
    result = execute_action(action, state_dict)
    logging.info(result)
    
    if result["success"]:
        logging.info(f"Action succeeded: {result['data']}")
        state_dict["action_count"] += 1
        action["status"] = Status.DONE

        if action_type == ActionType.GENERATE:
            # TODO: Handle more complex chains here!
            response_string = result["data"]
            state_dict["messages"].append(AIMessage(content=response_string))

        elif action_type == PlanActionType.FETCH:
            # TODO: Handle
            # Fetch the data from the source
            print("Fetching data from the source...")
            state_dict["plan_dict"]["context"] = result["data"]

        elif action_type == PlanActionType.CREATE_PLAN:
            # TODO: Handle creating a plan here
            print("Proposing a plan...")
            state_dict["plan_dict"]["proposed_plan"] = result["data"]
        
        elif action_type == PlanActionType.REVISE_PLAN:
            # TODO: This could be the output of a separate Revision agent run as a graph. The plan graph should take in the plan_context and return a proposed plan.
            state_dict["plan_dict"]["plan"] = result["data"]
            print("Finalized plan!")
            print(f"Plan: {state_dict['plan_dict']['plan']}")
            current_task["status"] = Status.DONE
        
    else:
        if result["error"]:
            if action_type == ActionType.GENERATE:
                print("Popping human message")
                assert isinstance(state_dict["messages"][-1], HumanMessage)
                state_dict["messages"].pop()
            logging.error(f"Action failed: {result['error']}")
            action["status"] = Status.FAILED
        else:
            if action_type == PlanActionType.REVISE_PLAN:
                action_args = action["args"]
                action_args["revision_count"] += 1

            if state_dict["action_count"] > MAX_ACTIONS and current_task["status"] == Status.IN_PROGRESS:
                logging.error("Action count exceeded max actions, marking action as failed")
                action["status"] = Status.FAILED
            
    if action["status"] == Status.FAILED:
        logging.error("Action failed, marking task as failed")
        current_task["status"] = Status.FAILED

    if action["status"] in [Status.DONE, Status.FAILED]:
        completed_action = current_task["actions"].pop(0)
        save_action_data(completed_action)
    else:
        assert action["status"] == Status.IN_PROGRESS
        logging.warning("Action is still in progress!")
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }


def execute_command_node(state: GraphState) -> GraphState:
    """Process a command and update state"""
    state_dict = state["keys"]
    command = Command(state_dict["user_input"])
    state_dict["user_input"] = None
    
    if not command.is_valid:
        logging.warning(f"Invalid command: {command.command}")
        return {
            "keys": state_dict,
            "mutation_count": state["mutation_count"] + 1,
            "is_done": False
        }
    
    cmd_type = command.type
    
    if cmd_type == CommandType.QUIT:
        # TODO: Change the status here to Quit
        # Task manager will decide in-progress, done, or failed
        task_dict = state_dict["task_dict"]
        if not task_dict:
            raise ValueError("No tasks found!")
        current_task = get_task(task_dict, status=Status.IN_PROGRESS)
        if current_task and not current_task["actions"]:
            current_task["status"] = Status.DONE
    
    elif cmd_type == CommandType.HELP:
        print("\nAvailable Commands:")
        for cmd in CommandType:
            print(f"/{cmd.value}")
            
    elif cmd_type == CommandType.CLEAR:
        clear_messages(state_dict["messages"])
        
    elif cmd_type == CommandType.SETTINGS:
        print("\nCurrent Settings:")
        print(state_dict["config"])
        
    elif cmd_type == CommandType.SAVE:
        # Implement save state logic
        print("Saving state...")
        print("\nMessages:")
        for message in state_dict["messages"]:
            if isinstance(message, HumanMessage):
                print(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"AI: {message.content}")
        
    elif cmd_type == CommandType.LOAD:
        # Implement load state logic
        print("Loading state...")
        
    elif cmd_type == CommandType.DEBUG:
        print("\nCurrent State:")
        print(f"Messages: {len(state_dict['messages'])}")
        print(f"Tasks: {state_dict['task_dict']}")

        
    elif cmd_type == CommandType.UNDO:
        if state_dict["messages"]:
            remove_last_message(state_dict["messages"])

    elif cmd_type == CommandType.MODE:
        current_task = get_task(state_dict["task_dict"], status=Status.IN_PROGRESS)
        if current_task:
            if current_task["type"] == TaskType.CHAT:
                cmd_args = command.args
                if cmd_args == "summary":
                    print("CHANGING TO SUMMARY MODE")
                    new_chain = get_summary_chain(state_dict["config"].chat_settings.primary_model)
                    if new_chain is not None:
                        state_dict["active_chain"] = new_chain
                    else:
                        raise ValueError("Summary chain not initialized!")
                else:
                    print("\nCurrent Mode:")
                    print(current_task["type"])
        else:
            print("No in-progress task found")

    elif cmd_type == CommandType.READ:
        # TODO: Read file(s) from args
        print("Reading from file...")
        user_input = read_sample()
        if user_input:
            state_dict["mock_inputs"].insert(0, user_input)
        else:
            print("No text found in file!")
    else:
        print(f"Command not implemented: {command.command}")

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

# Decision functions
def decide_from_agent(state: GraphState) -> Literal["human_node", "task_manager", "action_node", "end_node"]:
    """Routes from agent node based on state"""
    if state["is_done"]:
        return "end_node"
    
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Check for pending actions
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    if current_task:
        if current_task["actions"]:
            return "action_node"
        else:
            return "human_node"
    
    return "task_manager"


def decide_from_processor(state: GraphState) -> Literal["execute_command", "agent_node"]:
    """Routes from processor based on input type"""
    return "execute_command" if state["keys"]["user_input"].startswith('/') else "agent_node"