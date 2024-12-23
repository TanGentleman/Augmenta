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
from tangents.utils.task_utils import get_task, save_completed_tasks, save_failed_tasks, save_stashed_tasks, start_next_task, stash_task
from augmenta.utils import read_sample

from .classes.tasks import TaskType
from .classes.actions import PlanActionType, Status, ActionType
from .classes.commands import Command, CommandType
from .classes.states import GraphState

from .utils.message_utils import insert_system_message, remove_last_message, clear_messages
from .utils.action_utils import is_stash_action_next, save_action_data, create_action
from .utils.execute_action import execute_action

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
    # NOTE: This should be moved to the task manager node when starting a new task
    if state["mutation_count"] == 1:
        pass
    
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
    
    This node handles task state transitions and cleanup:
    - Saves and removes completed tasks from active dictionary
    - Saves and removes failed tasks from active dictionary 
    - Starts the next available task if none are in progress
    - Handles task stashing when requested
    - Determines if workflow should terminate
    
    Task Lifecycle:
    1. NOT_STARTED -> IN_PROGRESS: When selected as next task
    2. IN_PROGRESS -> DONE: When task succeeds
    3. IN_PROGRESS -> FAILED: When task fails
    
    Args:
        state (GraphState): Current graph state containing task dictionary
        
    Returns:
        GraphState: Updated state with:
            - Modified task dictionary reflecting lifecycle changes
            - Updated mutation count
            - Updated completion status
    """
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    # Process completed tasks
    completed_task_names = save_completed_tasks(task_dict)
    if completed_task_names:
        logging.info(f"Saved completed tasks: {completed_task_names}")
        for task_name in completed_task_names:
            del task_dict[task_name]

    # Process failed tasks
    failed_task_names = save_failed_tasks(task_dict)
    if failed_task_names:
        logging.error(f"Failed tasks: {failed_task_names}")
        for task_name in failed_task_names:
            del task_dict[task_name]
    
    # Check for in-progress task or start next available
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    if not current_task:
        unstarted_task = get_task(task_dict, status=Status.NOT_STARTED)
        
        # No tasks remaining - mark workflow as complete
        if not unstarted_task:
            logging.info("No tasks remaining!")
            return {
                "keys": state_dict,
                "mutation_count": state["mutation_count"] + 1,
                "is_done": True
            }
        
        task_type = unstarted_task["type"]
        task_state = unstarted_task["state"]
            
        # Initialize chat-specific task state
        # TODO: Add assertions and modularity here
        if task_type == TaskType.CHAT:
            # Set up task state
            if task_state is None:
                # Can add optionally set up chain here.
                unstarted_task["state"] = {
                    "messages": [],
                    "active_chain": None,
                    "stream": state_dict["config"].chat_settings.stream
                }
            if not state_dict["config"].chat_settings.disable_system_message:
                insert_system_message(
                    unstarted_task["state"]["messages"],
                    state_dict["config"].chat_settings.system_message
                )
        elif task_type == TaskType.RAG:
            if not state_dict["config"].rag_settings.enabled:
                raise ValueError("RAG task is disabled!")
        elif task_type == TaskType.PLANNING:
            # TODO: Add assertions about task state and Actions for the task
            if task_state is None:
                unstarted_task["state"] = {
                    "context": None,
                    "proposed_plan": None,
                    "plan": None,
                    "revision_count": 0
                }
            else:
                # TODO: Reason about task state and Actions for the task
                pass
        else:
            raise ValueError("Invalid task type!")
        
        # Start the task
        unstarted_task["status"] = Status.IN_PROGRESS
        current_task = unstarted_task
        print(f"Started task: {current_task['type']}")
    
    # Handle task stashing if requested
    if is_stash_action_next(current_task["actions"]):
        stashed_task_names = save_stashed_tasks(task_dict)
        if stashed_task_names:
            logging.info(f"Stashed tasks: {stashed_task_names}")
            for task_name in stashed_task_names:
                del task_dict[task_name]
    
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
        task_dict = state_dict["task_dict"]
        current_task = get_task(task_dict, status=Status.IN_PROGRESS)
        if not current_task:
            raise ValueError("No in-progress task found")
        
        task_state = current_task["state"]
        # UPDATE STATE BASED ON TASK TYPE
        if current_task["type"] == TaskType.CHAT:
            task_state["messages"].append(HumanMessage(content=user_input))
            if task_state["active_chain"] is None:
                llm = get_llm(state_dict["config"].chat_settings.primary_model)
                if llm is None:
                    raise ValueError("Chain not initialized!")
                task_state["active_chain"] = llm
            generate_action = create_action(ActionType.GENERATE,
                                            args={
                                                "messages": task_state["messages"],
                                                "chain": task_state["active_chain"],
                                                "stream": task_state["stream"]
                                            }
            )
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
            if task_state["proposed_plan"]:
                if task_state["plan"] is not None:
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
    assert action_type != ActionType.STASH, "Stash action should be handled in task manager"
    if action["status"] == Status.NOT_STARTED:
        action["status"] = Status.IN_PROGRESS
    assert action["status"] == Status.IN_PROGRESS


    logging.info(f"Executing action: {action}")
    result = execute_action(action, current_task)
    logging.info(result)
    
    if result["success"]:
        task_state = current_task["state"]
        logging.info(f"Action succeeded: {result['data']}")
        # NOTE: Is action_count officially a state variable?
        state_dict["action_count"] += 1
        action["status"] = Status.DONE

        if action_type == ActionType.GENERATE:
            # TODO: Handle more complex chains here!
            response_string = result["data"]
            if current_task["type"] == TaskType.CHAT:
                task_state["messages"].append(AIMessage(content=response_string))
            else:
                raise ValueError("Task type not supported for generate action")

        elif action_type == PlanActionType.FETCH:
            # TODO: Handle
            # Fetch the data from the source
            print("Fetching data from the source...")
            task_state["context"] = result["data"]

        elif action_type == PlanActionType.CREATE_PLAN:
            # TODO: Handle creating a plan here
            print("Proposing a plan...")
            task_state["proposed_plan"] = result["data"]
        
        elif action_type == PlanActionType.REVISE_PLAN:
            # TODO: This could be the output of a separate Revision agent run as a graph. The plan graph should take in the plan_context and return a proposed plan.
            task_state["plan"] = result["data"]
            print("Finalized plan!")
            print(f"Plan: {task_state['plan']}")
            current_task["status"] = Status.DONE
        
    else:
        if result["error"]:
            if action_type == ActionType.GENERATE:
                if current_task["type"] == TaskType.CHAT:
                    # TODO: Make robust!
                    print("Popping human message")
                    task_messages = task_state["messages"]
                    assert isinstance(task_messages[-1], HumanMessage)
                    task_messages.pop()
                else:
                    raise ValueError("Task type not supported for generate action")
            logging.error(f"Action failed: {result['error']}")
            action["status"] = Status.FAILED
        else:
            if action_type == PlanActionType.REVISE_PLAN:
                action_args = action["args"]
                action_args["revision_count"] += 1
                # TODO: Update task state here rather than action args?

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

    task_dict = state_dict["task_dict"]
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    assert current_task, "No in-progress task found"
    task_state = current_task["state"]
    
    if cmd_type == CommandType.QUIT:
        if not current_task["actions"]:
            current_task["status"] = Status.DONE
        else:
            logging.warning("Found in-progress task with actions, stashing task")
            current_task["actions"].insert(0, create_action(ActionType.STASH))
    
    elif cmd_type == CommandType.HELP:
        print("\nAvailable Commands:")
        for cmd in CommandType:
            print(f"/{cmd.value}")
            
    elif cmd_type == CommandType.CLEAR:
        # TODO: Move to chat_task_state
        if current_task["type"] == TaskType.CHAT:
            clear_messages(task_state["messages"])
        else:
            logging.error("Clear command only supported in chat tasks.")
            pass
        
    elif cmd_type == CommandType.SETTINGS:
        print("\nCurrent Settings:")
        print(state_dict["config"])
        
    elif cmd_type == CommandType.SAVE:
        # Implement save state logic
        print("Saving state...")
        print("\nMessages:")
        # TODO: Move to chat_task_state
        # NOTE: We may abstract to a get_messages function
        if current_task["type"] == TaskType.CHAT:
            for message in task_state["messages"]:
                if isinstance(message, HumanMessage):
                    print(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    print(f"AI: {message.content}")
        else:
            logging.error("Save command only supported in chat tasks.")
            pass
        
    elif cmd_type == CommandType.LOAD:
        # Implement load state logic
        print("Loading state...")
        
    elif cmd_type == CommandType.DEBUG:
        print("\nCurrent State:")
        if current_task["type"] == TaskType.CHAT:
            print(f"Message Count: {len(task_state['messages'])}")
        print(f"Tasks: {state_dict['task_dict']}")

        
    elif cmd_type == CommandType.UNDO:
        if current_task["type"] == TaskType.CHAT:
            remove_last_message(task_state["messages"])
        else:
            logging.error("Undo command only supported in chat tasks.")
            pass

    elif cmd_type == CommandType.MODE:
        if current_task["type"] == TaskType.CHAT:
            cmd_args = command.args
            if cmd_args == "summary":
                print("CHANGING TO SUMMARY MODE")
                new_chain = get_summary_chain(state_dict["config"].chat_settings.primary_model)
                if new_chain is not None:
                    task_state["active_chain"] = new_chain
                else:
                    raise ValueError("Summary chain not initialized!")
            else:
                print("\nCurrent Mode:")
                print(current_task["type"])
        else:
            logging.error("Mode command only supported in chat tasks.")
            pass

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
            # If action is stash, we need to get to the task manager
            if is_stash_action_next(current_task["actions"]):
                return "task_manager"
            else:
                return "action_node"
        else:
            return "human_node"
    
    return "task_manager"


def decide_from_processor(state: GraphState) -> Literal["execute_command", "agent_node"]:
    """Routes from processor based on input type"""
    return "execute_command" if state["keys"]["user_input"].startswith('/') else "agent_node"