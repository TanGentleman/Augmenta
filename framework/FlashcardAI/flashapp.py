import json
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt as RichPrompt
from rich.layout import Layout
from rich.table import Table
from typing import Callable, List, Dict, Tuple, Optional, Union
from enum import Enum
import random
import logging
import shutil
from typeguard import typechecked
from abc import ABC, abstractmethod
from typing_extensions import Protocol

from .flashcards import load_flashcards_from_json, Flashcard, save_flashcards_to_json
ROOT = Path(__file__).parent
FLASHCARD_FILEPATH = ROOT / "flashcards.json"
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


class TypedPrompt:
    """Utility class for prompting user input with type checking."""

    @staticmethod
    def ask(prompt: str,
            type_: type,
            choices: Optional[List[str]] = None,
            default: Optional[str] = None) -> Union[str,
                                                    int]:
        """
        Prompt the user for input and validate the type.

        Args:
            prompt (str): The prompt to display to the user.
            type_ (type): The expected type of the input.
            choices (Optional[List[str]]): A list of valid choices, if applicable.

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
                    prompt + f" [default: {default}]", choices=choices)
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


class TerminalLayout:
    """Manages the terminal layout for displaying flashcards."""

    def __init__(self) -> None:
        """Initialize the TerminalLayout and update terminal size."""
        self.update_terminal_size()

    def update_terminal_size(self) -> None:
        """Update the stored terminal size."""
        self.terminal_width, self.terminal_height = shutil.get_terminal_size()

    def render_flashcard(self, app: 'FlashcardApp') -> None:
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

        flashcard_panel = card.create_flashcard()
        answer_panel = card.create_answer() if app.show_answer else Panel(
            "Press 'S' to show answer", title="Answer", border_style="green")
        menu_panel = Panel(
            app.create_menu(),
            title="Menu",
            border_style="blue")

        layout_elements = [
            Layout(
                Panel(f"Card {app.flashcard_manager.current_index + 1} of {len(app.flashcard_manager.flashcards)}"),
                name="card_count",
                ratio=3) if app.layout_config.show_card_count else None,
            Layout(
                flashcard_panel,
                name="flashcard",
                ratio=5) if app.layout_config.show_flashcard else None,
            Layout(
                answer_panel,
                name="answer",
                ratio=5) if app.layout_config.show_answer else None,
            Layout(
                menu_panel,
                name="menu",
                ratio=4) if app.layout_config.show_menu else None]

        layout_elements = [
            element for element in layout_elements if element is not None]

        if layout_elements:
            layout = Layout()
            layout.split_column(*layout_elements)
        else:
            layout = Layout(
                Panel("No elements to display. Please check your layout configuration."))

        console.print(layout)


class FlashcardManagerProtocol(Protocol):
    """Protocol defining the interface for FlashcardManager."""

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


class FlashcardManager(FlashcardManagerProtocol):
    """
    Manages the flashcard deck, including loading, saving, and manipulating flashcards.
    """

    def __init__(self) -> None:
        """Initialize the FlashcardManager."""
        self.flashcards: List[Flashcard] = []
        self.current_index: int = 0
        self.keys: Dict[str, str] = {}
        self.styles: List[Tuple[str, str]] = []

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
            self.flashcards, self.keys, self.styles = load_flashcards_from_json(
                file_path)
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
            save_flashcards_to_json(self.flashcards, file_path)
        except Exception as e:
            logger.error(f"Failed to save flashcards: {str(e)}")
            raise FlashcardSaveError(f"Failed to save flashcards: {str(e)}")

    def ensure_valid_index(self) -> None:
        """Ensure that the current index is within the valid range."""
        if self.flashcards:
            self.current_index = max(
                0, min(
                    self.current_index, len(
                        self.flashcards) - 1))
            logger.warning(
                f"Safely adjusted current index to {self.current_index + 1}")
        else:
            self.current_index = 0

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
        new_card = Flashcard(card_data, self.keys, self.styles)
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
    def search_flashcards(self, keyword: str) -> List[Flashcard]:
        """
        Search for flashcards containing a specific keyword.

        Args:
            keyword (str): The keyword to search for.

        Returns:
            List[Flashcard]: A list of flashcards that match the search criteria.
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
            # M-key handled beforehand.
            'M': (lambda: True, "Return to Main Menu"),
            'Q': (lambda: True, "Quit"),  # Q-key handled beforehand.
        }

    @typechecked
    def handle_input(self, choice: str) -> bool:
        action, _ = self.key_bindings.get(choice.upper(), (None, None))
        if action:
            return action()
        return False

    def get_menu_options(self) -> List[str]:
        return [f"{key}: {description}" for key,
                (_, description) in self.key_bindings.items()]


class LayoutConfig:
    def __init__(self):
        self.show_card_count = True
        self.show_flashcard = True
        self.show_answer = True
        self.show_menu = False


class FlashcardApp:
    def __init__(self) -> None:
        self.flashcard_manager: FlashcardManagerProtocol = FlashcardManager()
        self.show_answer: bool = False
        self.terminal_layout = TerminalLayout()
        self.keyboard_handler = KeyboardHandler(self)
        self.layout_config = LayoutConfig()

        self.force_refresh = True

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
        self.layout_config.show_menu = not self.layout_config.show_menu # Toggle menu bar
        self.force_refresh = True
        return False

    def configure_layout(self) -> None:
        while True:
            console.clear()
            console.print(Panel("Layout Configuration", style="bold magenta"))
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
                    "1", "2", "3", "4", "5"])
            if choice == "1":
                self.toggle_layout_element("show_card_count")
            elif choice == "2":
                self.toggle_layout_element("show_flashcard")
            elif choice == "3":
                self.toggle_layout_element("show_answer")
            elif choice == "4":
                self.toggle_layout_element("show_menu")
            elif choice == "5":
                break

            TypedPrompt.ask("Press Enter to continue", str)

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
        if not self.flashcard_manager.flashcards:
            console.print(
                "[bold yellow]No flashcards available.[/bold yellow]")
            return False

        card_number = TypedPrompt.ask("Enter card number", int)
        if 1 <= card_number <= len(self.flashcard_manager.flashcards):
            self.flashcard_manager.jump_to_card(card_number - 1)
            self.show_answer = False
            self.display_current_card()
        else:
            console.print(
                f"[bold red]Error: Please enter a number between 1 and {len(self.flashcard_manager.flashcards)}.[/bold red]")
        return False  # Continue studying

    def add_flashcard(self) -> bool:
        new_card_data: Dict[str, str] = {}
        for key in self.flashcard_manager.keys.values():
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
        if not self.flashcard_manager.flashcards:
            console.print(
                "[bold yellow]No flashcards available to edit.[/bold yellow]")
            return False

        index = TypedPrompt.ask("Enter the card number to edit", int) - 1
        try:
            card = self.flashcard_manager.flashcards[index]
            updated_data: Dict[str, str] = {}
            for key in self.flashcard_manager.keys.values():
                current_value = str(card.card_data.get(key, ""))
                value = str(
                    TypedPrompt.get_input_with_default(
                        f"Enter the new {key}", str, current_value))
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
                f"[bold red]Error: Please enter a number between 1 and {len(self.flashcard_manager.flashcards)}.[/bold red]")
        except Exception as e:
            console.print(
                f"[bold red]An error occurred while editing the flashcard: {str(e)}[/bold red]")
            logger.error(f"Error editing flashcard: {str(e)}")
        self.force_refresh = True
        return False  # Continue studying

    def delete_flashcard(self) -> bool:
        index = TypedPrompt.ask("Enter the card number to delete", int) - 1
        if 0 <= index < len(self.flashcard_manager.flashcards):
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
            "Enter search keyword",
            str)  # Prompt for keyword input

        if keyword.strip():  # Ensure keyword is not empty after whitespace removal
            results = self.flashcard_manager.search_flashcards(keyword)

            if results:  # If matching flashcards are found
                table = Table(title=f"Search Results for '{keyword}'")
                # Add a numbered column
                table.add_column("Card #", style="cyan")

                # Dynamically add columns based on flashcard keys (e.g.,
                # "Term", "Definition")
                for key in self.flashcard_manager.keys.values():
                    table.add_column(key.capitalize(), style="magenta")

                for i, card in enumerate(
                        results):  # Populate table rows with card data
                    table.add_row(str(i + 1), *[str(card.card_data.get(key, "N/A"))
                                  for key in self.flashcard_manager.keys.values()])

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
            self.terminal_layout.render_flashcard(self)
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
