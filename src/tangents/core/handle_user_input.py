import logging
from typing import Callable

from langchain_core.messages import HumanMessage, AIMessage

from paths import TEXT_FILE_DIR

from tangents.classes.actions import ActionType, PlanActionType, Status
from tangents.classes.settings import Config
from tangents.classes.tasks import Task, TaskType
from tangents.utils.action_utils import create_action, is_stash_action_next
from tangents.utils.chains import fast_get_llm
from langchain_core.messages import HumanMessage
from tangents.classes.commands import Command, CommandType
from tangents.utils.message_utils import clear_messages, remove_last_message
from tangents.utils.action_utils import add_stash_action
from tangents.utils.file_utils import read_text_file

def handle_user_message(user_input: str, current_task: Task, config: Config) -> None:
    assert not user_input.startswith('/'), "Command should not be passed as human input!"
    assert current_task["status"] == Status.IN_PROGRESS, "Task is not in progress!"
    task_state = current_task["state"]
    
    match current_task["type"]:
        case TaskType.CHAT:
            task_state["messages"].append(HumanMessage(content=user_input))
            if task_state["active_chain"] is None:
                logging.warning("No active chain found, initializing new chain!")
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
            # assert that there is an in-progress revise action queued next
            action_list = current_task["actions"]
            if len(action_list) == 0:
                raise ValueError("No revise action found!")
            
            if action_list[0]["type"] != PlanActionType.REVISE_PLAN or action_list[0]["status"] != Status.IN_PROGRESS:
                raise ValueError("Next action must be an in-progress revise action!")
            
            if user_input == "y":
                print("Plan is confirmed!")
                current_task["actions"][0]["args"]["is_done"] = True
            else:
                print("Using your revision!")
                # The next human action is added in handle_action.handle_action_result
                # That logic can alternatively be added here
        case _:
            raise ValueError("Invalid task type")
        
    return None

async def execute_command(command: Command, current_task: Task, state_dict: dict) -> None:
    """Execute a command and update state accordingly."""
    if not command.is_valid:
        logging.warning(f"Invalid command: {command.command}")
        return

    handlers = {
        # Handlers that need task only
        CommandType.QUIT: lambda: _handle_quit_command(current_task),
        CommandType.STASH: lambda: _handle_stash_command(current_task),
        CommandType.HELP: _print_help_command,
        
        # Handlers that need task + state_dict
        CommandType.CLEAR: lambda: _handle_clear_command(current_task, state_dict),
        CommandType.SAVE: lambda: _handle_save_command(current_task),
        CommandType.UNDO: lambda: _handle_undo_command(current_task),
        CommandType.MODE: lambda: _handle_mode_command(current_task, command.args),
        CommandType.DEBUG: lambda: _print_debug_info(current_task, state_dict),
        
        # Handlers that need state_dict only
        CommandType.SETTINGS: lambda: print("\nCurrent Settings:", state_dict["config"]),
        CommandType.READ: lambda: _handle_read_command(state_dict, command.args),
        
        # Handlers that need nothing
        CommandType.LOAD: lambda: print("Loading state..."),
    }
    
    handler = handlers.get(command.type)
    if handler:
        result = handler()
        if _is_async_handler(result):
            await result
        else:
            result
    else:
        print(f"Command not implemented: {command.command}")

def _is_async_handler(handler: Callable) -> bool:
    """Check if a handler is async."""
    return hasattr(handler, '__await__')

def _handle_quit_command(task: Task) -> None:
    """Handle quit command logic."""
    if not task["actions"]:
        task["status"] = Status.DONE
    else:
        logging.warning("Found in-progress task with actions, stashing task")
        add_stash_action(task["actions"])

def _print_help_command() -> None:
    """Print available commands."""
    print("\nAvailable Commands:")
    for cmd in CommandType:
        print(f"/{cmd.value}")

def _handle_clear_command(task: Task, state_dict: dict) -> None:
    """Handle clearing messages in chat tasks."""
    if task["type"] == TaskType.CHAT:
        clear_messages(task["state"]["messages"])
    else:
        logging.error("Clear command only supported in chat tasks.")

def _handle_save_command(task: Task) -> None:
    """Handle saving chat messages."""
    if task["type"] == TaskType.CHAT:
        print("\nMessages:")
        for msg in task["state"]["messages"]:
            prefix = "Human:" if isinstance(msg, HumanMessage) else "AI:" if isinstance(msg, AIMessage) else "System:"
            print(f"{prefix} {msg.content}")
    else:
        logging.error("Save command only supported in chat tasks.")

def _handle_undo_command(task: Task) -> None:
    """Handle undoing last message in chat tasks."""
    if task["type"] == TaskType.CHAT:
        # Remove both AI and human messages
        remove_last_message(task["state"]["messages"])
        remove_last_message(task["state"]["messages"])
    else:
        logging.error("Undo command only supported in chat tasks.")

def _print_debug_info(task: Task, state_dict: dict) -> None:
    """Print debug information."""
    print("\nCurrent State:")
    if task["type"] == TaskType.CHAT:
        print(f"Message Count: {len(task['state']['messages'])}")
    print(f"Tasks: {state_dict['task_dict']}")

def _handle_mode_command(task: Task, mode_arg: str) -> None:
    """Handle mode switching command."""
    if task["type"] == TaskType.CHAT:
        if mode_arg == "summary":
            print("CHANGING TO SUMMARY MODE")
        else:
            print("\nCurrent Mode:")
            print(task["type"])
    else:
        logging.error("Mode command only supported in chat tasks.")

def _handle_read_command(state_dict: dict, filename_arg: str) -> None:
    """Handle file reading command."""
    if not filename_arg:
        filename_arg = "sample.txt"
    filepath = TEXT_FILE_DIR / filename_arg
    user_input = read_text_file(filepath)
    if user_input:
        state_dict["mock_inputs"].insert(0, user_input)
        print(f"Successfully read {filepath}!")
    else:
        print(f"No text found in file {filepath}!")
    
def _handle_stash_command(task: Task) -> None:
    """Handle stashing a task."""
    if task["status"] == Status.IN_PROGRESS:
        add_stash_action(task["actions"])
    else:
        logging.error("Stash command only supported for in-progress tasks.")
