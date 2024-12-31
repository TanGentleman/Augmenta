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

from tangents.utils.chains import fast_get_llm
from tangents.utils.task_utils import (
    get_task, save_completed_tasks, save_failed_tasks,
    save_stashed_tasks, start_task
)
from tangents.utils.command_utils import read_sample

from .classes.tasks import Task, TaskType
from .classes.actions import Action, ActionResult, PlanActionType, Status, ActionType  
from .classes.commands import Command, CommandType
from .classes.states import GraphState

from .utils.message_utils import remove_last_message, clear_messages
from .utils.action_utils import add_human_action, add_stash_action, is_human_action_next, is_stash_action_next, save_action_data, create_action
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

    
    # Handle mock inputs for testing, otherwise get real user input
    if mock_inputs:
        print(f"Mock inputs: {mock_inputs}")
        user_input = mock_inputs.pop(0)
    else:
        # Initialize default interrupt context
        interrupt_context = {
            "prompt": "Enter your message (or /help for commands):"
        }
        
        if is_human_action_next(current_task["actions"]):
            custom_interrupt_prompt = current_task["actions"][0]["args"].get("prompt")
            if custom_interrupt_prompt:
                interrupt_context["prompt"] = custom_interrupt_prompt
        
        # Get user input via interrupt
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
    current_task = get_task(task_dict, status=Status.IN_PROGRESS)
    user_input = state_dict["user_input"]

    assert current_task is not None, "No in-progress task found"
    assert user_input is not None, "No user input found"

    config = state_dict["config"]


    def handle_human_input(user_input: str, task: Task) -> None:
        assert is_human_action_next(task["actions"]), "Human action should be next!"
        human_action = task["actions"].pop(0)
        # TODO: Handle cases with custom interrupt logic
        # Example: Update task state + tweak the interrupt prompt string
        return None

    # Logic should go here
    if is_human_action_next(current_task["actions"]):
        handle_human_input(user_input, current_task)
    

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
                if user_input == "y":
                    print("Plan is confirmed!")
                    current_task["actions"][0]["args"]["is_done"] = True
                else:
                    print("Using your revision!")
                    
            case _:
                raise ValueError("Invalid task type")
    
    # NOTE: This will be removed as logic is put in place for repeated human input handling
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

    return action

def handle_action_result(task: Task, action_result: ActionResult) -> Task:
    """Handle the result of an action based on task and action types."""
    action = task["actions"][0]
    task_state = task["state"]

    if action_result["success"]:
        action["status"] = Status.DONE
        
        match (task["type"], action["type"]):
            case (TaskType.CHAT, ActionType.GENERATE):
                task_state["messages"].append(AIMessage(content=action_result["data"]))
                
            case (TaskType.PLANNING, PlanActionType.FETCH):
                task_state["context"] = action_result["data"]
                
            case (TaskType.PLANNING, PlanActionType.PROPOSE_PLAN):
                task_state["proposed_plan"] = action_result["data"]
                
            case (TaskType.PLANNING, PlanActionType.REVISE_PLAN):
                task_state["plan"] = action_result["data"]
                task["status"] = Status.DONE
                
            case _:
                raise ValueError(f"Unsupported task/action combination: {task['type']}/{action['type']}")
    
    elif action_result["error"]:
        action["status"] = Status.FAILED
        # Handle error cleanup
        match (task["type"], action["type"]):
            case (TaskType.CHAT, ActionType.GENERATE):
                if isinstance(task_state["messages"][-1], HumanMessage):
                    task_state["messages"].pop()
            case _:
                pass  # Other types may not need cleanup
        
    else:
        # Retry cases
        match (task["type"], action["type"]):
            case (TaskType.PLANNING, PlanActionType.REVISE_PLAN):
                if "revision_count" not in task_state:
                    task_state["revision_count"] = 0
                task_state["revision_count"] += 1
                action["args"]["revision_context"] = f"Revision {task_state['revision_count']}"

                DEFAULT_MAX_REVISIONS = 3
                max_revisions = action["args"].get("max_revisions", DEFAULT_MAX_REVISIONS)
                # TODO: Move this value to current_task["conditions"]
                if task_state["revision_count"] >= max_revisions:
                    action["args"]["is_done"] = True
                else:
                    add_human_action(task["actions"], prompt=f"Review revision #{task_state['revision_count']}. Enter 'y' to finalize or any other input to revise again.")
            case _:
                pass

    return task

async def action_node(state: GraphState) -> GraphState:
    """Execute and manage task actions."""
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
    assert action["type"] != ActionType.STASH, "Stash action should be handled in task manager"
    
    # Initialize action if needed
    if action["status"] == Status.NOT_STARTED:
        action = start_action(action, current_task)
    assert action["status"] == Status.IN_PROGRESS

    # Execute action and process result
    logging.info(f"Executing action: {action}")
    result = await execute_action(action)
    logging.info(result)

    # Update state based on result
    current_task = handle_action_result(current_task, result)
    if result["success"]:
        state_dict["action_count"] += 1

    # Handle action completion
    if action["status"] in [Status.DONE, Status.FAILED]:
        completed_action = current_task["actions"].pop(0)
        save_action_data(completed_action)
        
        # Mark task as failed if action failed
        if action["status"] == Status.FAILED:
            logging.error("Action failed, marking task as failed")
            current_task["status"] = Status.FAILED
    else:
        logging.warning("Action is still in progress!")
    
    return {
        "keys": state_dict,
        "mutation_count": state["mutation_count"] + 1,
        "is_done": False
    }

async def execute_command_node(state: GraphState) -> GraphState:
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
    
    # TODO: Make a better interface for commands with custom logic for task state
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
        
        # NOTE: Logic is based on task type/state
        case CommandType.CLEAR:
            if current_task["type"] == TaskType.CHAT:
                clear_messages(task_state["messages"])
            else:
                logging.error("Clear command only supported in chat tasks.")
            
        case CommandType.SETTINGS:
            print("\nCurrent Settings:")
            print(config)
            
        # NOTE: Logic is based on task type/state
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
            
        # NOTE: Logic is based on task type/state
        case CommandType.UNDO:
            if current_task["type"] == TaskType.CHAT:
                remove_last_message(task_state["messages"])
            else:
                logging.error("Undo command only supported in chat tasks.")


        case CommandType.MODE:
            if current_task["type"] == TaskType.CHAT:
                if command.args == "summary":
                    print("CHANGING TO SUMMARY MODE")
                    # TODO: Implement a change in active chain system message and messages context
                else:
                    print("\nCurrent Mode:")
                    print(current_task["type"])
            else:
                logging.error("Mode command only supported in chat tasks.")

        case CommandType.READ:
            # NOTE: Add file path argument support
            print("Reading from file...")
            user_input = await read_sample()
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
    if not current_task:
        return "task_manager"
    if not current_task["actions"]:
        return "human_node"
    if is_stash_action_next(current_task["actions"]):
        return "task_manager"
    elif is_human_action_next(current_task["actions"]):
        return "human_node"
    else:
        return "action_node"

def decide_from_processor(state: GraphState) -> Literal["execute_command", "agent_node"]:
    """Route from processor based on input type (command vs message)."""
    state_dict = state["keys"]
    user_input = state_dict["user_input"]
    return "execute_command" if user_input.startswith('/') else "agent_node"
