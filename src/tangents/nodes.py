"""
Node implementations for the workflow graph.

Each node processes GraphState and implements workflow logic. Nodes take a GraphState 
input and return an updated state. These functions are meant to be used as StateGraph 
nodes and should not be called directly.

See tan_graph.py for workflow architecture details.
"""
import logging
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt

from tangents.classes.settings import Config
from tangents.utils.chains import get_summary_chain, fast_get_llm
from tangents.utils.task_utils import (
    get_task, save_completed_tasks, save_failed_tasks,
    save_stashed_tasks
)
from augmenta.utils import read_sample

from .classes.tasks import Task, TaskType
from .classes.actions import Action, PlanActionType, Status, ActionType  
from .classes.commands import Command, CommandType
from .classes.states import GraphState

from .utils.message_utils import insert_system_message, remove_last_message, clear_messages
from .utils.action_utils import add_stash_action, is_human_action_next, is_stash_action_next, save_action_data, create_action
from .utils.execute_action import execute_action

# Constants
MAX_MUTATIONS = 50  # Maximum state mutations before failing
MAX_ACTIONS = 5     # Maximum actions per task before failing

def validate_state(state: GraphState) -> GraphState:
    """Validate fields in state dictionary."""
    state_dict = state["keys"]
    if not state_dict["config"]:
        raise ValueError("Config is not set!")
    if not state_dict["task_dict"]:
        raise ValueError("Agent must have at least one task!")
    # Assign task_id to each task?
    return state

def start_node(state: GraphState) -> GraphState:
    """Initialize graph state with config and default task."""
    validate_state(state)
    state_dict = state["keys"]
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": state["is_done"]
    }

def agent_node(state: GraphState) -> GraphState:
    """
    Manage agent state and handle failure conditions.
    
    Checks action and mutation counts against maximums and fails tasks if exceeded.
    """
    state_dict = state["keys"]
    current_task = get_task(state_dict["task_dict"], status=Status.IN_PROGRESS)
    is_done = state["is_done"]
    if current_task:
        if is_done:
            logging.error("CRITICAL: Early exit: task is still in progress!")
        elif state_dict["action_count"] > MAX_ACTIONS:
            logging.error("Action count exceeded max actions, marking task as failed")
            current_task["status"] = Status.FAILED
        elif state["mutation_count"] > MAX_MUTATIONS:
            logging.error("Mutation count exceeded, marking task as failed") 
            current_task["status"] = Status.FAILED

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": is_done
    }

def start_task(task: Task, config: Config) -> Task:
    """Initialize a new task with proper state based on task type."""
    if task["status"] != Status.NOT_STARTED:
        raise ValueError("Task must have NOT_STARTED status to start!")
    
    task_type = task["type"]
    task_state = task["state"]
    match task_type:
        case TaskType.CHAT:
            if task_state is None:
                task["state"] = {
                    "messages": [],
                    "active_chain": None,
                    "stream": config.chat_settings.stream
                }
            if not config.chat_settings.disable_system_message:
                insert_system_message(
                    task["state"]["messages"],
                    config.chat_settings.system_message
                )
        
        case TaskType.RAG:
            if not config.rag_settings.enabled:
                raise ValueError("RAG task is disabled!")
        
        case TaskType.PLANNING:
            if task_state is None:
                task["state"] = {
                    "context": None,
                    "proposed_plan": None,
                    "plan": None,
                    "revision_count": 0
                }
        
        case _:
            raise ValueError("Invalid task type!")
    
    task["status"] = Status.IN_PROGRESS
    print(f"Started task: {task['type']}")
    return task

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
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    config = state_dict["config"]
    
    # Process completed/failed tasks
    # NOTE: Double check this implementation
    logging.info("Processing completed/failed tasks. Not yet parallelized.")
    for task_names, save_func in [
        (save_completed_tasks(task_dict), logging.info),
        (save_failed_tasks(task_dict), logging.warning)
    ]:
        if task_names:
            save_func(f"Saved tasks: {task_names}")
            for name in task_names:
                del task_dict[name]
    
    # Get/start next task
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    if not current_task:
        unstarted_task = get_task(task_dict, status=Status.NOT_STARTED)
        
        if not unstarted_task:
            logging.info("No tasks remaining!")
            return {
                "keys": state_dict,
                "mutation_count": state["mutation_count"] + 1,
                "is_done": True
            }
            
        current_task = start_task(unstarted_task, config)
        assert current_task["status"] == Status.IN_PROGRESS
    
    # Handle task stashing
    if is_stash_action_next(current_task["actions"]):
        stashed_task_names = save_stashed_tasks(task_dict)
        if stashed_task_names:
            logging.info(f"Stashed tasks: {stashed_task_names}")
            for name in stashed_task_names:
                del task_dict[name]
    
    if not task_dict:
        logging.info("No tasks remaining!")
        is_done = True
    else:
        is_done = False

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": is_done
    }

def human_node(state: GraphState) -> GraphState:
    """
    Handle human input through interrupts or mock inputs.
    
    Uses langgraph interrupt mechanism to pause execution and get user input.
    Supports mock inputs for testing.
    """
    state_dict = state["keys"]
    mock_inputs = state_dict["mock_inputs"]
    current_task = get_task(state_dict["task_dict"], status=Status.IN_PROGRESS)
    assert current_task is not None, "No in-progress task found"

    if is_human_action_next(current_task["actions"]):
        # TODO: Handle interrupt_context based on task
        pass

    if mock_inputs:
        print(f"Mock inputs: {mock_inputs}")
        user_input = mock_inputs.pop(0)
    else:
        # TODO: Handle interrupt_context based on task
        # NOTE: Three different cases: 1. Get user input string, 2. Validate with bool, 3. Updates to state
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
    """
    Process user input and update task state.
    
    Handles:
    - Command vs message detection
    - Task-specific message processing
    - Action creation based on task type
    """
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    config = state_dict["config"]
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    assert current_task is not None, "No in-progress task found"
    
    # TODO: HITL behavior is set by ActionType.HUMAN_INPUT
    if is_human_action_next(current_task["actions"]):
        # TODO: Handle interrupt_context based on task
        current_task["actions"].pop(0)

    user_input = state_dict["user_input"]
    if not user_input.startswith('/'):
        if not current_task:
            raise ValueError("No in-progress task found")
        
        task_state = current_task["state"]
        
        match current_task["type"]:
            case TaskType.CHAT:
                task_state["messages"].append(HumanMessage(content=user_input))
                if task_state["active_chain"] is None:
                    llm = fast_get_llm(config.chat_settings.primary_model)
                    if llm is None:
                        raise ValueError("Chain not initialized!")
                    task_state["active_chain"] = llm
                
                generate_action = create_action(
                    ActionType.GENERATE,
                    args={
                        "messages": task_state["messages"],
                        "chain": task_state["active_chain"],
                        "stream": task_state["stream"]
                    }
                )
                current_task["actions"].append(generate_action)
                assert len(current_task["actions"]) == 1 or is_stash_action_next(current_task["actions"]), "Only one action should be queued for chat!"
                # Can stash here if needed

            case TaskType.RAG:
                if config.rag_settings.enabled:
                    print("RAG task. Doing nothing for now.")
                else:
                    raise ValueError("RAG task is disabled!")
                    
            case TaskType.PLANNING:
                # Handle cases during PLANNING task
                if task_state["proposed_plan"]:
                    if task_state["plan"] is not None:
                        raise ValueError("Plan is already created!")
                    print("Using your revision!")
                    
            case _:
                raise ValueError("Invalid task type")
    
    assert not is_human_action_next(current_task["actions"]), "Human action should be removed by the end of the node!"
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def start_action(action: Action, task: Task) -> Action:
    """Initialize an action with required state before execution."""
    action["status"] = Status.IN_PROGRESS
    
    # Initialize action args from task state if needed
    task_state = task["state"]
    action_args = action["args"]

    if action["type"] == ActionType.GENERATE:
        if task["type"] == TaskType.CHAT:
            if action_args.get("active_chain") is None:
                action_args["active_chain"] = task_state["active_chain"]
            if action_args.get("messages") is None:
                action_args["messages"] = task_state["messages"]
            if action_args.get("stream") is None:
                action_args["stream"] = task_state["stream"]
        else:
            raise ValueError("Missing support in start_action for ActionType.GENERATE!")

    elif action["type"] == PlanActionType.PROPOSE_PLAN:
        if action_args.get("plan_context") is None:
            action_args["plan_context"] = task_state["context"]

    elif action["type"] == PlanActionType.REVISE_PLAN:
        if action_args.get("proposed_plan") is None:
            action_args["proposed_plan"] = task_state["proposed_plan"]
        if action_args.get("revision_count") is None:
            action_args["revision_count"] = 0

    return action

def action_node(state: GraphState) -> GraphState:
    """
    Execute and manage task actions.
    
    Handles:
    - Action execution and status tracking
    - Result processing based on action type
    - Error handling and task failure conditions
    """
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    current_task = next(
        (task for task in task_dict.values() if task["status"] == Status.IN_PROGRESS),
        None
    )
    if not current_task:
        raise ValueError("No in-progress task found")
    if not current_task["actions"]:
        raise ValueError("No actions found for in-progress task.")
    
    action = current_task["actions"][0]
    action_type = action["type"]
    assert action_type != ActionType.STASH, "Stash action should be handled in task manager"
    
    if action["status"] == Status.NOT_STARTED:
        action["status"] = Status.IN_PROGRESS
    assert action["status"] == Status.IN_PROGRESS

    action = start_action(action, current_task)
    logging.info(f"Executing action: {action}")
    result = execute_action(action)
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

            case PlanActionType.PROPOSE_PLAN:
                print("Proposing a plan...")
                task_state["proposed_plan"] = result["data"]
            
            case PlanActionType.REVISE_PLAN:
                # NOTE: Consider moving to revision subgraph flow
                task_state["plan"] = result["data"]
                print("Finalized plan!")
                print(f"Plan: {task_state['plan']}")
                current_task["status"] = Status.DONE
        
    else:
        if result["error"]:
            if action_type == ActionType.GENERATE:
                if current_task["type"] == TaskType.CHAT:
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
    """
    Process command input and update state.
    
    Handles system commands like:
    - Task management (quit, save, load)
    - UI commands (help, clear, debug)
    - Mode switching and settings
    """
    state_dict = state["keys"]
    config = state_dict["config"]
    
    user_input = state_dict["user_input"]
    if not user_input:
        raise ValueError("No user command to execute!")
    command = Command(user_input)
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
                add_stash_action(current_task["actions"])
        
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
            print(config)
            
        case CommandType.SAVE:
            # add args for file path
            if current_task["type"] == TaskType.CHAT:
                print("\nMessages:")
                for msg in task_state["messages"]:
                    if isinstance(msg, HumanMessage):
                        prefix = "Human:"
                    elif isinstance(msg, AIMessage): 
                        prefix = "AI:"
                    else:
                        prefix = "System:" # Handle system/tool messages
                    print(f"{prefix} {msg.content}")
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
                    new_chain = get_summary_chain(config.chat_settings.primary_model)
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
            # NOTE: Add file path argument support
            print("Reading from file...")
            user_input = read_sample()
            if user_input:
                state_dict["mock_inputs"].insert(0, user_input)
            else:
                print("No text found in file!")
                
        case CommandType.STASH:
            print("Stashing current state...")
            add_stash_action(current_task["actions"])
            
        case _:
            print(f"Command not implemented: {command.command}")

    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

def decide_from_agent(state: GraphState) -> Literal["human_node", "task_manager", "action_node", "end_node"]:
    """Route from agent node based on state conditions."""
    if state["is_done"]:
        return "end_node"
    
    state_dict = state["keys"]
    task_dict = state_dict["task_dict"]
    
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    if current_task:
        if current_task["actions"]:
            if is_stash_action_next(current_task["actions"]):
                return "task_manager"
            elif is_human_action_next(current_task["actions"]):
                return "human_node"
            else:
                return "action_node"
        return "human_node"
    
    return "task_manager"

def decide_from_processor(state: GraphState) -> Literal["execute_command", "agent_node"]:
    """Route from processor based on input type (command vs message)."""
    state_dict = state["keys"]
    user_input = state_dict["user_input"]
    return "execute_command" if user_input.startswith('/') else "agent_node"

"""
Future Improvements:
1. Move chain initialization to processor_node for better separation of concerns
2. Add support for pre-set task states in PLANNING tasks
3. Implement revision subgraph flow for plan revisions
4. Add file path argument support for READ command
5. Add logic for HITL tasks that shouldn't append messages
6. Consider breaking out task type handlers into separate modules
7. Add more robust error handling and recovery mechanisms
8. Implement proper state persistence for SAVE/LOAD commands
"""
