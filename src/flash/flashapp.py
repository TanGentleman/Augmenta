import json
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt as RichPrompt
from rich.layout import Layout
from rich.table import Table
from typing import Callable, Dict, Literal, Tuple, Optional, Union
from enum import Enum
import random
import logging
import shutil
from typeguard import typechecked
from abc import abstractmethod
from typing_extensions import Protocol

from paths import FLASHCARD_FILEPATH
from .flashcard import Flashcard
from .manager import Flashcards

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()


class FlashcardError(Exception):
    """Base exception class for Flashcard-related errors."""
    pass


class FlashcardLoadError(FlashcardError):
    """Exception raised when there's an error loading flashcards."""
    pass


class FlashcardSaveError(FlashcardError):
    """Exception raised when there's an error saving flashcards."""
    pass


class FlashcardNotFoundError(FlashcardError):
    """Exception raised when a requested flashcard is not found."""
    pass


class MenuChoice(Enum):
    """Enum representing the main menu choices."""
    STUDY = "1"
    SEARCH = "2"
    CONFIGURE = "3"
    QUIT = "4"


class Panels(Enum):
    """Enum representing the panels."""
    CARD_COUNT = "Card Count"
    FLASHCARD = "Flashcard"
    ANSWER = "Answer"
    MENU = "Menu"

class TypedPrompt:
    """Utility class for prompting user input with type checking."""

    @staticmethod
    def ask(prompt: str,
            type_: type,
            choices: Optional[list[str]] = None,
            default: Optional[str] = None) -> Union[str,
                                                    int]:
        """
        Prompt the user for input and validate the type.

        Args:
            prompt (str): The prompt to display to the user.
            type_ (type): The expected type of the input.
            choices (Optional[list[str]]): A list of valid choices, if applicable.

        Returns:
            Union[str, int]: The user's input, converted to the specified type.

        Raises:
            ValueError: If the input cannot be converted to the specified type.
        """
        assert type_ in (str, int), "Only str and int types are supported"
        assert choices is None or isinstance(
            choices, list), "Choices must be a list or None"
        assert default is None or isinstance(
            default, str), "Default value must be a string or None"
        while True:
            if default is not None:
                result = RichPrompt.ask(
                    choices=choices, default=default, show_default=True)
            else:
                result = RichPrompt.ask(prompt, choices=choices)
            try:
                return type_(result)
            except ValueError:
                console.print(
                    f"[bold red]Invalid input. Please enter a {type_.__name__}.[/bold red]")
                logger.warning(f"Invalid input received for prompt: {prompt}")

    # NOTE: These can be combined into a single method with a default value
    @staticmethod
    def get_input_with_default(
            prompt: str, type_: type, default: str) -> Union[str, int]:
        """
        Prompt the user for input with a default value.

        Args:
            prompt (str): The prompt to display to the user.
            type_ (type): The expected type of the input.
            default (str): The default value to use if the user input is empty.

        Returns:
            Union[str, int]: The user's input or the default value, converted to the specified type.
        """
        while True:
            result = RichPrompt.ask(f"{prompt} [default: {default}]")
            if result == "":
                return type_(default)
            try:
                return type_(result)
            except ValueError:
                console.print(
                    f"[bold red]Invalid input. Please enter a {type_.__name__}.[/bold red]")
                logger.warning(f"Invalid input received for prompt: {prompt}")


class LayoutConfig:
    def __init__(self):
        self.show_card_count = True
        self.show_flashcard = True
        self.show_answer = True
        self.show_menu = True

    @typechecked
    def get_ratios(self) -> dict[Literal[Panels.CARD_COUNT,
                                         Panels.FLASHCARD,
                                         Panels.ANSWER,
                                         Panels.MENU], int]:
        ratios = {
            Panels.CARD_COUNT: 1,
            Panels.FLASHCARD: 2,
            Panels.ANSWER: 3,
            Panels.MENU: 2
        }
        if not self.show_card_count:
            ratios[Panels.CARD_COUNT] = 0

        if not self.show_flashcard:
            ratios[Panels.FLASHCARD] = 0

        if not self.show_answer:
            ratios[Panels.ANSWER] = 0

        if not self.show_menu:
            ratios[Panels.MENU] = 0

        return ratios


class TerminalLayout:
    """Manages the terminal layout for displaying flashcards."""

    def __init__(self) -> None:
        """Initialize the TerminalLayout and update terminal size."""
        self.update_terminal_size()

    def update_terminal_size(self) -> None:
        """Update the stored terminal size."""
        self.terminal_width, self.terminal_height = shutil.get_terminal_size()

    def render_screen(self, app: 'FlashcardApp') -> None:
        """
        Render the current flashcard in the terminal.

        Args:
            app (FlashcardApp): The FlashcardApp instance.
        """
        self.update_terminal_size()

        card = app.flashcard_manager.get_current_card()
        if card is None:
            console.print("No flashcard to display.")
            return

        flashcard_panel = card.create_flashcard(include_answer=app.show_answer)
        answer_panel = flashcard_panel if app.show_answer else Panel(
            "Press 'S' to show answer", title="Answer", border_style="green")

        menu_panel = Panel(
            app.create_menu(),
            title="Menu",
            border_style="blue")

        # TODO: Perform ratio calculations based on LayoutConfig and terminal
        # size

        layout_elements = []
        ratios = app.layout_config.get_ratios()
        for panel, ratio in ratios.items():
            if panel == Panels.CARD_COUNT:
                if ratio:
                    layout_elements.append(
                        Layout(
                            Panel(f"Card {app.flashcard_manager.get_current_index() + 1} of {app.flashcard_manager.get_card_count()}"),
                            name="card_count",
                            ratio=ratio))
            elif panel == Panels.FLASHCARD:
                if ratio and not app.show_answer:
                    layout_elements.append(Layout(
                        flashcard_panel,
                        name="flashcard",
                        ratio=ratio))
            elif panel == Panels.ANSWER:
                if ratio:
                    layout_elements.append(Layout(
                        answer_panel,
                        name="answer",
                        ratio=ratio))
            elif panel == Panels.MENU:
                if ratio:
                    layout_elements.append(Layout(
                        menu_panel,
                        name="menu",
                        ratio=ratio))

        if layout_elements:
            layout = Layout()
            layout.split_column(*layout_elements)
        else:
            layout = Layout(
                Panel("No elements to display. Please check your layout configuration."))

        console.print(layout)


class FlashcardManagerProtocol(Protocol):
    """Protocol defining the interface for FlashcardManager."""

    # TODO: Look into using @property and setter methods for read-only and writable attributes
    # Seems like I won't be able to use abstract properties, I'll have to
    # include the logic in the FlashcardManagerProtocol class.
    @abstractmethod
    def get_key_names(self) -> list[str]:
        """Get the keys for the flashcards."""
        pass

    @abstractmethod
    def get_current_index(self) -> int:
        """Get the current index of the flashcard."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the flashcard deck is empty."""
        pass

    @abstractmethod
    def get_card_count(self) -> int:
        """Get the total number of flashcards."""
        pass

    @abstractmethod
    def get_flashcards(self, index: int) -> list[Flashcard]:
        """Get the flashcards."""
        pass

    @abstractmethod
    def load_flashcards(self, file_path: str) -> None:
        """Load flashcards from a file."""
        pass

    @abstractmethod
    def save_flashcards(self, file_path: str) -> None:
        """Save flashcards to a file."""
        pass

    @abstractmethod
    def get_current_card(self) -> Optional[Flashcard]:
        """Get the current flashcard."""
        pass

    @abstractmethod
    def next_card(self) -> None:
        """Move to the next flashcard."""
        pass

    @abstractmethod
    def previous_card(self) -> None:
        """Move to the previous flashcard."""
        pass

    @abstractmethod
    def jump_to_card(self, index: int) -> None:
        """Jump to a specific flashcard."""
        pass

    @abstractmethod
    def add_flashcard(self, card_data: Dict[str, str]) -> None:
        """Add a new flashcard."""
        pass

    @abstractmethod
    def edit_flashcard(self, index: int, card_data: Dict[str, str]) -> None:
        """Edit an existing flashcard."""
        pass

    @abstractmethod
    def delete_flashcard(self, index: int) -> bool:
        """Delete a flashcard."""
        pass

    @abstractmethod
    def study_random(self) -> None:
        """Randomize the order of flashcards for studying."""
        pass

    @abstractmethod
    def search_flashcards(self, keyword: str) -> list[Flashcard]:
        """Search for flashcards containing a specific keyword."""
        pass


class FlashcardManager(FlashcardManagerProtocol):
    """
    Manages the flashcard deck, including loading, saving, and manipulating flashcards.
    """

    def __init__(self) -> None:
        """Initialize the FlashcardManager."""
        self.manager = Flashcards()
        self.flashcards = self.manager.flashcards
        self.keys = self.manager.keys
        self.styles = self.manager.styles
        self.current_index: int = 0

    def get_key_names(self) -> list[str]:
        """Get the keys for the flashcards."""
        return list(self.manager.keys.values())

    def get_current_index(self) -> int:
        """Get the current index of the flashcard."""
        if self.current_index < 0 or self.current_index >= len(
                self.flashcards):
            print(f"Invalid index retrieval attempted: {self.current_index}")
            raise FlashcardNotFoundError(
                f"Invalid card number: {self.current_index + 1}")
        return self.current_index

    def is_empty(self) -> bool:
        """Check if the flashcard deck is empty."""
        return self.get_card_count() == 0

    def get_card_count(self) -> int:
        """Get the total number of flashcards."""
        return len(self.flashcards)

    @typechecked
    def get_flashcards(self, index: int | None = None) -> list[Flashcard]:
        """Get the flashcards. Optional Index starts at 0."""
        if index is not None:
            if 0 <= index < self.get_card_count():
                return [self.flashcards[index]]
            else:
                raise FlashcardNotFoundError(
                    f"Invalid card number: {index + 1}")
        return self.flashcards

    @typechecked
    def load_flashcards(self, file_path: str) -> None:
        """
        Load flashcards from a JSON file.

        This method reads the flashcard data from the specified file,
        populates the flashcards list, and sets up the keys and styles.

        Args:
            file_path (str): Path to the JSON file containing flashcard data.

        Raises:
            FlashcardLoadError: If there's an error loading the flashcards.
        """
        try:
            self.manager.load_from_json(
                file_path)
            self.flashcards, self.keys, self.styles = self.manager.flashcards, self.manager.keys, self.manager.styles
            logger.info(
                f"Loaded {len(self.flashcards)} flashcards from {file_path}")
        except FileNotFoundError:
            logger.error(f"Flashcard file not found: {file_path}")
            raise FlashcardLoadError(f"Flashcard file not found: {file_path}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {file_path}")
            raise FlashcardLoadError(
                f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load flashcards: {str(e)}")
            raise FlashcardLoadError(f"Failed to load flashcards: {str(e)}")

    @typechecked
    def save_flashcards(self, file_path: str) -> None:
        """
        Save flashcards to a JSON file.

        Args:
            file_path (str): Path to save the JSON file.

        Raises:
            FlashcardSaveError: If there's an error saving the flashcards.
        """
        try:
            self.manager.save_to_json(file_path)
        except Exception as e:
            logger.error(f"Failed to save flashcards: {str(e)}")
            raise FlashcardSaveError(f"Failed to save flashcards: {str(e)}")

    def get_current_card(self) -> Optional[Flashcard]:
        """
        Get the current flashcard.

        Returns:
            Optional[Flashcard]: The current flashcard, or None if no flashcards are available.
        """
        if self.flashcards and 0 <= self.current_index < len(self.flashcards):
            return self.flashcards[self.current_index]
        return None

    def next_card(self) -> None:
        """Move to the next flashcard."""
        if self.current_index < len(self.flashcards) - 1:
            self.current_index += 1
            logger.debug(f"Moved to next card: {self.current_index + 1}")
        else:
            console.print("You've reached the end of the deck.")
            logger.info("Attempted to move past the last card")

    def previous_card(self) -> None:
        """Move to the previous flashcard."""
        if self.current_index > 0:
            self.current_index -= 1
            logger.debug(f"Moved to previous card: {self.current_index + 1}")
        else:
            console.print("You're at the beginning of the deck.")
            logger.info("Attempted to move before the first card")

    @typechecked
    def jump_to_card(self, index: int) -> None:
        """
        Jump to a specific flashcard.

        Args:
            index (int): The index of the flashcard to jump to.

        Raises:
            FlashcardNotFoundError: If the specified index is invalid.
        """
        if 0 <= index < len(self.flashcards):
            self.current_index = index
            logger.debug(f"Jumped to card: {index + 1}")
        else:
            logger.warning(f"Attempted to jump to invalid card index: {index}")
            raise FlashcardNotFoundError(f"Invalid card number: {index + 1}")

    @typechecked
    def add_flashcard(self, card_data: Dict[str, str]) -> None:
        """
        Add a new flashcard to the deck.

        Args:
            card_data (Dict[str, str]): The data for the new flashcard.
        """
        try:
            new_card = Flashcard(card_data, self.keys, self.styles)
        except Exception as e:
            logger.error(f"Failed to create new flashcard: {str(e)}")
            raise FlashcardError(f"Failed to create new flashcard: {str(e)}")
        self.flashcards.append(new_card)
        logger.info(f"Added new flashcard: {card_data}")

    @typechecked
    def edit_flashcard(self, index: int, card_data: Dict[str, str]) -> None:
        """
        Edit an existing flashcard.

        Args:
            index (int): The index of the flashcard to edit.
            card_data (Dict[str, str]): The updated data for the flashcard.

        Raises:
            FlashcardNotFoundError: If the specified index is invalid.
        """
        if 0 <= index < len(self.flashcards):
            for key, value in card_data.items():
                self.flashcards[index].card_data[key] = value
            logger.info(f"Edited flashcard at index {index}: {card_data}")
        else:
            logger.warning(f"Attempted to edit invalid card index: {index}")
            raise FlashcardNotFoundError(f"Invalid card number: {index + 1}")

    @typechecked
    def delete_flashcard(self, index: int) -> bool:
        """
        Delete a flashcard from the deck.

        Args:
            index (int): The index of the flashcard to delete.

        Returns:
            bool: True if the flashcard was successfully deleted, False otherwise.

        Raises:
            FlashcardNotFoundError: If the specified index is invalid.
        """
        if 0 <= index < len(self.flashcards):
            del self.flashcards[index]
            if self.current_index >= len(self.flashcards):
                self.current_index = max(0, len(self.flashcards) - 1)
            logger.info(f"Deleted flashcard at index {index}")
            return True
        else:
            logger.warning(f"Attempted to delete invalid card index: {index}")
            raise FlashcardNotFoundError(f"Invalid card number: {index + 1}")

    def study_random(self) -> None:
        """Randomize the order of flashcards for studying."""
        random.shuffle(self.flashcards)
        self.current_index = 0
        logger.info("Shuffled flashcards for random study")

    @typechecked
    def search_flashcards(self, keyword: str) -> list[Flashcard]:
        """
        Search for flashcards containing a specific keyword.

        Args:
            keyword (str): The keyword to search for.

        Returns:
            list[Flashcard]: A list of flashcards that match the search criteria.
        """
        results = [card for card in self.flashcards if any(
            keyword.lower() in str(value).lower() for value in card.card_data.values())]
        logger.info(
            f"Searched for keyword '{keyword}', found {len(results)} results")
        return results


class KeyboardHandler:
    def __init__(self, app: 'FlashcardApp'):
        self.app = app
        # NOTE: Key bindings MUST be in all caps
        self.key_bindings: Dict[str, Tuple[Callable[[], bool], str]] = {
            'HELP': (self.app.toggle_menu_bar, "Toggle Menu Bar"),
            'H': (self.app.print_help, "Help"),
            'N': (self.app.next_card, "Next Card"),
            'P': (self.app.previous_card, "Previous Card"),
            'S': (self.app.toggle_answer, "Show/Hide Answer"),
            'J': (self.app.jump_to_card, "Jump to Card"),
            'R': (self.app.study_random, "Study Random"),
            'A': (self.app.add_flashcard, "Add Flashcard"),
            'E': (self.app.edit_flashcard, "Edit Flashcard"),
            'D': (self.app.delete_flashcard, "Delete Flashcard"),
            'F': (self.app.search_flashcards, "Search Flashcards"),
            # M and Q keys should be handled beforehand. They both end study.
            'M': (lambda: True, "Return to Main Menu"),
            'Q': (lambda: True, "Quit"),
        }

    @typechecked
    def handle_input(self, choice: str) -> bool:
        action, _ = self.key_bindings.get(choice.upper(), (None, None))
        if action:
            return action()
        return False

    def get_menu_options(self) -> list[str]:
        return [f"{key}: {description}" for key,
                (_, description) in self.key_bindings.items()]


class FlashcardApp:
    def __init__(self) -> None:
        self.transparent = False
        self.force_refresh = True

        self.flashcard_manager: FlashcardManagerProtocol = FlashcardManager()
        self.show_answer: bool = False
        self.terminal_layout = TerminalLayout()
        self.keyboard_handler = KeyboardHandler(self)
        self.layout_config = LayoutConfig()

    # TODO: Make show answer a property that has side effects

    @property
    def show_answer(self) -> bool:
        return self._show_answer

    @typechecked
    @show_answer.setter
    def show_answer(self, value: bool) -> None:
        if self.transparent:
            value = True
        self._show_answer = value

    @typechecked
    def load_flashcards(self, file_path: str) -> None:
        self.flashcard_manager.load_flashcards(file_path)

    @typechecked
    def save_flashcards(self, file_path: str) -> None:
        self.flashcard_manager.save_flashcards(file_path)

    @typechecked
    def toggle_layout_element(self, element: str) -> None:
        if hasattr(self.layout_config, element):
            setattr(
                self.layout_config,
                element,
                not getattr(
                    self.layout_config,
                    element))
            console.print(
                f"[bold green]{element.replace('_', ' ').title()} {'shown' if getattr(self.layout_config, element) else 'hidden'}[/bold green]")
        else:
            console.print(
                f"[bold red]Invalid layout element: {element}[/bold red]")

    def toggle_menu_bar(self) -> bool:
        self.layout_config.show_menu = not self.layout_config.show_menu  # Toggle menu bar
        self.force_refresh = True
        return False

    def configure_layout(self) -> None:
        while True:
            console.clear()
            console.print(Panel("Layout Configuration", style="bold magenta"))
            console.print(
                "0. Enable Transparency (Answers always shown)")
            console.print(
                f"1. Card Count: {'Shown' if self.layout_config.show_card_count else 'Hidden'}")
            console.print(
                f"2. Flashcard: {'Shown' if self.layout_config.show_flashcard else 'Hidden'}")
            console.print(
                f"3. Answer: {'Shown' if self.layout_config.show_answer else 'Hidden'}")
            console.print(
                f"4. Menu: {'Shown' if self.layout_config.show_menu else 'Hidden'}")
            console.print("5. Return to main menu")

            choice = TypedPrompt.ask(
                "Choose an option to toggle", str, choices=[
                    "0", "1", "2", "3", "4", "5"])

            if choice == "0":
                self.transparent = not self.transparent
                if self.transparent:
                    self.show_answer = True
                console.print(
                    f"[bold green]Transparency toggled to {'enabled' if self.transparent else 'disabled'}[/bold green]")
            elif choice == "1":
                self.toggle_layout_element("show_card_count")
                console.print(
                    f"[bold green]Card count {'shown' if self.layout_config.show_card_count else 'hidden'}[/bold green]")
            elif choice == "2":
                self.toggle_layout_element("show_flashcard")
                console.print(
                    f"[bold green]Flashcard {'shown' if self.layout_config.show_flashcard else 'hidden'}[/bold green]")
            elif choice == "3":
                self.toggle_layout_element("show_answer")
                console.print(
                    f"[bold green]Answer {'shown' if self.layout_config.show_answer else 'hidden'}[/bold green]")
            elif choice == "4":
                self.toggle_layout_element("show_menu")
                console.print(
                    f"[bold green]Menu {'shown' if self.layout_config.show_menu else 'hidden'}[/bold green]")
            elif choice == "5":
                break

            TypedPrompt.ask("Press any key to continue", str)

    def print_help(self) -> bool:
        console.print(
            Panel(
                self.create_menu(
                    condense=False),
                title="Keyboard Shortcuts",
                border_style="blue"))
        return False

    def next_card(self) -> bool:
        self.flashcard_manager.next_card()
        self.show_answer = False
        self.display_current_card()
        logger.debug("Moved to next card")
        return False  # Continue studying

    def previous_card(self) -> bool:
        self.flashcard_manager.previous_card()
        self.show_answer = False
        self.display_current_card()
        logger.debug("Moved to previous card")
        return False  # Continue studying

    def toggle_answer(self) -> bool:
        self.show_answer = not self.show_answer
        self.display_current_card()
        logger.debug(f"Answer {'shown' if self.show_answer else 'hidden'}")
        return False  # Continue studying

    def jump_to_card(self) -> bool:
        # NOTE: This check will be deprecated in favor of is_empty
        if self.flashcard_manager.is_empty():
            console.print(
                "[bold yellow]No flashcards available.[/bold yellow]")
            return False

        card_number = TypedPrompt.ask("Enter card number", int)
        if 1 <= card_number <= self.flashcard_manager.get_card_count():
            self.flashcard_manager.jump_to_card(card_number - 1)
            self.show_answer = False
            self.display_current_card()
        else:
            console.print(
                f"[bold red]Error: Please enter a number between 1 and {self.flashcard_manager.get_card_count()}.[/bold red]")
        return False  # Continue studying

    def add_flashcard(self) -> bool:
        new_card_data: Dict[str, str] = {}
        for key in self.flashcard_manager.get_key_names():
            while True:
                value = TypedPrompt.ask(f"Enter the {key}", str)
                if value.strip():
                    new_card_data[key] = value
                    break
                else:
                    console.print(
                        f"[bold red]Error: {key} cannot be empty. Please try again.[/bold red]")
        self.flashcard_manager.add_flashcard(new_card_data)
        console.print("[bold green]Flashcard added successfully![/bold green]")
        self.force_refresh = True
        return False  # Continue studying

    def edit_flashcard(self) -> bool:
        if self.flashcard_manager.is_empty():
            console.print(
                "[bold yellow]No flashcards available to edit.[/bold yellow]")
            return False

        index = TypedPrompt.ask("Enter the card number to edit", int) - 1
        try:
            card: Flashcard = self.flashcard_manager.get_flashcards(index=index)[
                0]
            updated_data: Dict[str, str] = {}
            for key in self.flashcard_manager.get_key_names():
                current_value = str(card.card_data.get(key, ""))
                # TODO: In the future, reconsider forcing a string type?
                value = str(
                    TypedPrompt.get_input_with_default(
                        prompt=f"Enter the new {key}",
                        type_=str,
                        default=current_value)
                )
                if value.strip():
                    updated_data[key] = value
                else:
                    console.print(
                        f"[bold red]Error: {key} cannot be empty. Using previous value.[/bold red]")
                    updated_data[key] = current_value
            self.flashcard_manager.edit_flashcard(index, updated_data)
            console.print(
                "[bold green]Flashcard updated successfully![/bold green]")
        except FlashcardNotFoundError:
            console.print(
                f"[bold red]Error: Please enter a number between 1 and {self.flashcard_manager.get_card_count()}.[/bold red]")
        except Exception as e:
            console.print(
                f"[bold red]An error occurred while editing the flashcard: {str(e)}[/bold red]")
            logger.error(f"Error editing flashcard: {str(e)}")
        self.force_refresh = True
        return False  # Continue studying

    def delete_flashcard(self) -> bool:
        index = TypedPrompt.ask("Enter the card number to delete", int) - 1
        if 0 <= index < self.flashcard_manager.get_card_count():
            confirm = TypedPrompt.ask(
                "Are you sure you want to delete this flashcard? (y/n)",
                str,
                choices=[
                    "y",
                    "n"])
            if confirm.lower() == "y":
                self.flashcard_manager.delete_flashcard(index)
                console.print(f"Flashcard {index + 1} deleted successfully!")
            else:
                console.print("Deletion cancelled.")
        else:
            console.print("Invalid card number.")
        self.force_refresh = True
        return False  # Continue studying

    def study_random(self) -> bool:
        self.flashcard_manager.study_random()
        self.show_answer = False
        self.display_current_card()
        return False  # Continue studying

    def search_flashcards(self) -> bool:
        """Searches for flashcards matching a user-provided keyword.

        Displays search results in a formatted table, or an "not found" message if no matches are found.

        Returns:
            bool: False (indicating the study session should continue).
        """

        keyword = TypedPrompt.ask(
            "Enter search keyword ('.' for all)",
            str)  # Prompt for keyword input

        if keyword.strip():  # Ensure keyword is not empty after whitespace removal
            if keyword == ".":
                results = self.flashcard_manager.get_flashcards()
            else:
                results = self.flashcard_manager.search_flashcards(keyword)

            if results:  # If matching flashcards are found
                table = Table(title=f"Search Results for '{keyword}'")
                # Add a numbered column
                table.add_column("Card #", style="cyan")

                # Dynamically add columns based on flashcard keys (e.g.,
                # "Term", "Definition")
                key_list = self.flashcard_manager.get_key_names()
                for key in key_list:
                    table.add_column(key.capitalize(), style="magenta")

                for i, card in enumerate(
                        results):  # Populate table rows with card data
                    table.add_row(str(i + 1), *[str(card.card_data.get(key, "N/A"))
                                  for key in key_list])
                # Iterate over the results, with each card and its index
                # for i, card in enumerate(results):
                #     # Initialize an empty list to hold the card data for the current row
                #     row_data = []
                #     # For each key in the list of keys, retrieve the corresponding value from the card's data
                #     # If the key doesn't exist in the card's data, use "N/A" as the value
                #     for key in key_list:
                #         cell_value = str(card.card_data.get(key, "N/A"))
                #         row_data.append(cell_value)

                #     # Add a new row to the table
                #     # The row starts with the card's index (i + 1 to start counting from 1 instead of 0)
                #     # followed by the card data for each key
                #     table.add_row(str(i + 1), *row_data)

                console.print(table)
            else:
                # Inform user if no matches
                console.print("No matching flashcards found.")
        else:
            console.print(
                "[bold red]Error: Search keyword cannot be empty. Please try again.[/bold red]")

        return False  # Continue studying after search

    def display_current_card(self) -> None:
        current_card = self.flashcard_manager.get_current_card()
        if current_card:
            self.terminal_layout.render_screen(self)
        else:
            console.print("No flashcards to display.")

    def create_menu(self, condense: bool = False) -> str:
        options = self.keyboard_handler.get_menu_options()

        if condense:
            return " | ".join(options)

        max_width = max(len(option) for option in options)
        columns = max(1, 80 // (max_width + 2))  # Assuming 80 characters width

        formatted_options = []
        for i in range(0, len(options), columns):
            row = options[i:i + columns]
            formatted_row = "  ".join(option.ljust(max_width)
                                      for option in row)
            formatted_options.append(formatted_row)

        return "\n".join(formatted_options)

    def study_flashcards(self) -> bool:
        """
        Start an interactive flashcard study session.

        This method displays flashcards, handles user input, and manages
        the study flow. It returns True if the user chooses to quit the
        application, and False if they want to return to the main menu.

        Returns:
            bool: True if the user wants to quit, False otherwise.
        """
        if not self.flashcard_manager.flashcards:
            console.print(
                "[bold yellow]No flashcards available. Please add some flashcards first.[/bold yellow]")
            TypedPrompt.ask("Press Enter to continue", str)
            return False

        HIDE_MENU_OPTIONS = True
        while True:
            if self.force_refresh:
                self.display_current_card()
                self.force_refresh = False
            menu_options = self.keyboard_handler.get_menu_options()
            if not HIDE_MENU_OPTIONS:
                for option in menu_options:
                    console.print(option)
            choice = TypedPrompt.ask(
                "Press a key (h: Help, q: Quit):", str, choices=[
                    key.lower() for key in self.keyboard_handler.key_bindings.keys()])
            choice = choice.upper()
            if choice == 'Q':  # Quit
                return True
            elif choice == 'M':  # Return to Main Menu
                self.force_refresh = True
                return False
            self.keyboard_handler.handle_input(choice)
        return False


@click.command()
@click.option('--file', default=FLASHCARD_FILEPATH,
              help='Path to the flashcards JSON file.')
def main(file: str) -> None:
    """
    Main entry point for the Flashcard Study App.

    Args:
        file (str): Path to the flashcards JSON file.
    """
    logger.info("Starting Flashcard Study App")
    app = FlashcardApp()
    try:
        app.load_flashcards(file)
    except FlashcardLoadError as e:
        console.print(f"[bold red]Error loading flashcards: {str(e)}")
        logger.error(f"Failed to load flashcards: {str(e)}")
        return

    while True:
        console.clear()
        console.print(Panel("Flashcard Study App", style="bold magenta"))
        console.print(
            Panel(
                "1. Study Flashcards: Study the flashcards interactively.",
                style="bold green"))
        console.print(
            Panel(
                "2. Search Flashcards: Search for flashcards containing a specific keyword.",
                style="bold green"))
        console.print(
            Panel(
                "3. Configure Layout: Customize the visible elements of the layout.",
                style="bold green"))
        console.print(
            Panel(
                "4. Save and Quit: Save the flashcards and quit the application.",
                style="bold green"))

        choice = TypedPrompt.ask(
            "Choose an option", str, choices=[
                choice.value for choice in MenuChoice])
        logger.debug(f"User selected menu option: {choice}")

        if choice == MenuChoice.STUDY.value:
            logger.info("Starting flashcard study session")
            if app.study_flashcards():
                break
        elif choice == MenuChoice.SEARCH.value:
            logger.info("Starting flashcard search")
            app.search_flashcards()
            TypedPrompt.ask("Press Enter to continue", str)

        elif choice == MenuChoice.CONFIGURE.value:
            logger.info("Configuring layout")
            app.configure_layout()

        elif choice == MenuChoice.QUIT.value:
            logger.info("User requested to save and quit")
            try:
                app.save_flashcards(file)
                console.print("Flashcards saved. Goodbye!")
                logger.info("Flashcards saved successfully")
                break
            except FlashcardSaveError as e:
                console.print(f"[bold red]Error saving flashcards: {str(e)}")
                logger.error(f"Failed to save flashcards: {str(e)}")

    logger.info("Flashcard Study App closed")


if __name__ == "__main__":
    main()
