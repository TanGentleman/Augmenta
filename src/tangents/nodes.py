"""
Processing node implementations for the workflow graph.

Contains the core node functions that process GraphState and implement the workflow logic.
Each node takes a GraphState as input and returns an updated state. These functions are
meant to be used as StateGraph nodes and should not be called directly.

See tan_graph.py for the complete workflow architecture and node descriptions.
"""
import logging
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt


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

def validate_state(state: GraphState) -> GraphState:
    """Validate required fields in state dictionary."""
    state_dict = state["keys"]
    if not state_dict["config"]:
        raise ValueError("Config is not set!")
    if not state_dict["task_dict"]:
        raise ValueError("Agent must have at least one task!")
    return state

def start_node(state: GraphState) -> GraphState:
    """Initialize the graph state with configuration and default task."""
    validate_state(state)
    state_dict = state["keys"]
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": state["is_done"]
    }

def agent_node(state: GraphState) -> GraphState:
    """Manages agent state and decision making."""
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    
    if current_task:
        # Check failure conditions
        if state_dict["action_count"] > MAX_ACTIONS:
            logging.error("Action count exceeded max actions, marking task as failed")
            current_task["status"] = Status.FAILED
        elif state["mutation_count"] > MAX_MUTATIONS:
            logging.error("Mutation count exceeded, marking task as failed")
            current_task["status"] = Status.FAILED

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": state["is_done"]
    }

def task_manager_node(state: GraphState) -> GraphState:
    """
    Manages task lifecycle transitions and cleanup.

    Handles:
    - Saving/removing completed and failed tasks
    - Starting next available task
    - Task stashing
    - Workflow termination

    Task States:
    NOT_STARTED -> IN_PROGRESS -> DONE/FAILED

    Args:
        state: Current graph state

    Returns:
        Updated state with task changes and status
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
            
        # Initialize task-specific state
        match task_type:
            case TaskType.CHAT:
                # NOTE: Chain can be initialized here.
                if task_state is None:
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
            
            case TaskType.RAG:
                if not state_dict["config"].rag_settings.enabled:
                    raise ValueError("RAG task is disabled!")
            
            case TaskType.PLANNING:
                if task_state is None:
                    unstarted_task["state"] = {
                        "context": None,
                        "proposed_plan": None,
                        "plan": None,
                        "revision_count": 0
                    }
                # TODO: Add logic for pre-set task states.
            
            case _:
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
    """Handles human input."""
    # https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/#interrupt
    state_dict = state["keys"]
    mock_inputs = state_dict["mock_inputs"]
    if mock_inputs:
        print(f"Mock inputs: {mock_inputs}")
        user_input = mock_inputs.pop(0)
    else:
        # Get current task info for interrupt context
        task_dict = state_dict["task_dict"]
        current_task = get_task(task_dict, status=Status.IN_PROGRESS)
        assert current_task is not None, "No in-progress task found"
        
        # Use interrupt to pause graph and get input
        interrupt_context = {
            # "task_type": current_task["type"].value,
            # "task_state": current_task["state"],
            "prompt": "Enter your message (or /help for commands):"
        }
        
        user_input = interrupt(interrupt_context)
    
    state_dict["user_input"] = user_input
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def processor_node(state: GraphState) -> GraphState:
    """Processes input and updates state."""
    state_dict = state["keys"]
    user_input = state_dict["user_input"]
    
    # TODO: Add logic to determine if we are in a HITL task and should not append to messages
    if not user_input.startswith('/'):
        task_dict = state_dict["task_dict"]
        current_task = get_task(task_dict, status=Status.IN_PROGRESS)
        if not current_task:
            raise ValueError("No in-progress task found")
        
        task_state = current_task["state"]
        
        match current_task["type"]:
            case TaskType.CHAT:
                # TODO: Consider keeping this to the task initialization.
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

            case TaskType.RAG:
                if state_dict["config"].rag_settings.enabled:
                    print("RAG task. Doing nothing for now.")
                else:
                    raise ValueError("RAG task is disabled!")
                    
            case TaskType.PLANNING:
                # TODO: Add logic for pre-set task states.
                if task_state["proposed_plan"]:
                    if task_state["plan"] is not None:
                        raise ValueError("Plan is already created!")
                    print("Using your revision!")
                    
            case _:
                raise ValueError("Invalid task type")
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def action_node(state: GraphState) -> GraphState:
    """Executes in-progress actions in the current task."""
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

    task_state = current_task["state"]
    if result["success"]:
        logging.info(f"Action succeeded: {result['data']}")
        state_dict["action_count"] += 1
        action["status"] = Status.DONE

        match action_type:
            case ActionType.GENERATE:
                response_string = result["data"]
                if current_task["type"] == TaskType.CHAT:
                    task_state["messages"].append(AIMessage(content=response_string))
                else:
                    raise ValueError("Task type not supported for generate action")

            case PlanActionType.FETCH:
                print("Fetching data from the source...")
                task_state["context"] = result["data"]

            case PlanActionType.CREATE_PLAN:
                print("Proposing a plan...")
                task_state["proposed_plan"] = result["data"]
            
            case PlanActionType.REVISE_PLAN:
                # TODO: This may be the output of a revision subgraph flow.
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
                action["args"]["revision_count"] += 1
            
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
    """Process a command and update state."""
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
    
    task_dict = state_dict["task_dict"]
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    assert current_task, "No in-progress task found"
    task_state = current_task["state"]
    
    match command.type:
        case CommandType.QUIT:
            if not current_task["actions"]:
                current_task["status"] = Status.DONE
            else:
                logging.warning("Found in-progress task with actions, stashing task")
                current_task["actions"].insert(0, create_action(ActionType.STASH))
        
        case CommandType.HELP:
            print("\nAvailable Commands:")
            for cmd in CommandType:
                print(f"/{cmd.value}")
                
        case CommandType.CLEAR:
            if current_task["type"] == TaskType.CHAT:
                clear_messages(task_state["messages"])
            else:
                logging.error("Clear command only supported in chat tasks.")
            
        case CommandType.SETTINGS:
            print("\nCurrent Settings:")
            print(state_dict["config"])
            
        case CommandType.SAVE:
            print("Saving state...")
            if current_task["type"] == TaskType.CHAT:
                print("\nMessages:")
                for message in task_state["messages"]:
                    match message:
                        case HumanMessage():
                            print(f"Human: {message.content}")
                        case AIMessage():
                            print(f"AI: {message.content}")
            else:
                logging.error("Save command only supported in chat tasks.")
            
        case CommandType.LOAD:
            print("Loading state...")
            
        case CommandType.DEBUG:
            print("\nCurrent State:")
            if current_task["type"] == TaskType.CHAT:
                print(f"Message Count: {len(task_state['messages'])}")
            print(f"Tasks: {state_dict['task_dict']}")
            
        case CommandType.UNDO:
            if current_task["type"] == TaskType.CHAT:
                remove_last_message(task_state["messages"])
            else:
                logging.error("Undo command only supported in chat tasks.")

        case CommandType.MODE:
            if current_task["type"] == TaskType.CHAT:
                if command.args == "summary":
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

        case CommandType.READ:
            # TODO: Add args here.
            print("Reading from file...")
            user_input = read_sample()
            if user_input:
                state_dict["mock_inputs"].insert(0, user_input)
            else:
                print("No text found in file!")
                
        case _:
            print(f"Command not implemented: {command.command}")

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def decide_from_agent(state: GraphState) -> Literal["human_node", "task_manager", "action_node", "end_node"]:
    """Routes from agent node based on state."""
    if state["is_done"]:
        return "end_node"
    
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    if current_task:
        if current_task["actions"]:
            return "task_manager" if is_stash_action_next(current_task["actions"]) else "action_node"
        return "human_node"
    
    return "task_manager"

def decide_from_processor(state: GraphState) -> Literal["execute_command", "agent_node"]:
    """Routes from processor based on input type."""
    state_dict = state["keys"]
    user_input = state_dict["user_input"]
    return "execute_command" if user_input.startswith('/') else "agent_node"
