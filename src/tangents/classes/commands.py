from typing import Optional
from enum import Enum


class CommandType(Enum):
    """System command types.

    Values:
        QUIT: Exit the application
        HELP: Show help information
        CLEAR: Clear conversation history
        SETTINGS: Modify configuration
        SAVE: Save current state
        LOAD: Load saved state
        DEBUG: Toggle debug mode
        UNDO: Revert last action
        MODE: Change operation mode
        READ: Load external content
        STASH: Save current state
    """

    QUIT = 'quit'
    HELP = 'help'
    CLEAR = 'clear'
    SETTINGS = 'settings'
    SAVE = 'save'
    LOAD = 'load'
    DEBUG = 'debug'
    UNDO = 'undo'
    MODE = 'mode'
    READ = 'read'
    STASH = 'stash'


class Command:
    """System command parser and validator.

    Parses raw command strings into structured command objects.
    Commands must be prefixed with '/' and may include arguments.

    Attributes:
        command: The base command name
        args: Optional command arguments

    Properties:
        is_valid: Whether command is recognized
        type: Parsed command type
    """

    def __init__(self, raw_input: str):
        parts = raw_input.lstrip('/').split(' ', 1)
        self.command = parts[0].lower()
        self.args = parts[1] if len(parts) > 1 else ''

    @property
    def is_valid(self) -> bool:
        try:
            CommandType(self.command)
            return True
        except ValueError:
            return False

    @property
    def type(self) -> Optional[CommandType]:
        try:
            return CommandType(self.command)
        except ValueError:
            return None
