from typing import Callable
import logging
from langchain_core.messages import HumanMessage, AIMessage

from paths import TEXT_FILE_DIR
from tangents.classes.actions import Status
from tangents.classes.commands import Command, CommandType
from tangents.classes.tasks import Task, TaskType
from tangents.utils.message_utils import clear_messages, remove_last_message
from tangents.utils.action_utils import add_stash_action
from tangents.utils.file_utils import read_text_file

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
