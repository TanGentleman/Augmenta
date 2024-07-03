import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.layout import Layout
from rich.table import Table
from typing import List, Dict, Tuple
import random
import logging
import shutil

from flashcards import load_flashcards_from_json, Flashcard, save_flashcards_to_json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

class TerminalLayout:
    """Manages the terminal layout for the flashcard application."""

    def __init__(self) -> None:
        """Initialize the TerminalLayout instance."""
        self.update_terminal_size()

    def update_terminal_size(self) -> None:
        """Update the terminal size."""
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
        answer_panel = card.create_answer() if app.show_answer else Panel("Press 'S' to show answer", title="Answer", border_style="green")
        
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"Card {app.flashcard_manager.current_index + 1} of {len(app.flashcard_manager.flashcards)}")),
            Layout(flashcard_panel, name="question"),
            Layout(answer_panel, name="answer"),
            Layout(Panel(app.create_menu(), title="Menu", border_style="blue"))
        )
        layout["question"].ratio = 2
        layout["answer"].ratio = 2

        console.print(layout)

class FlashcardManager:
    """Manages flashcard operations."""

    def __init__(self) -> None:
        """Initialize the FlashcardManager instance."""
        self.flashcards: List[Flashcard] = []
        self.current_index: int = 0
        self.keys: Dict[str, str] = {}
        self.styles: List[Tuple[str, str]] = []

    def load_flashcards(self, file_path: str) -> None:
        """
        Load flashcards from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing flashcard data.

        Raises:
            Exception: If there's an error loading the flashcards.
        """
        try:
            self.flashcards, self.keys, self.styles = load_flashcards_from_json(file_path)
            logger.info(f"Loaded {len(self.flashcards)} flashcards from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load flashcards: {str(e)}")
            raise

    def save_flashcards(self, file_path: str) -> None:
        """
        Save flashcards to a JSON file.

        Args:
            file_path (str): Path to save the JSON file containing flashcard data.

        Raises:
            Exception: If there's an error saving the flashcards.
        """
        save_flashcards_to_json(self.flashcards, file_path)

    def ensure_valid_index(self) -> None:
        """Ensure that the current_index is valid."""
        if self.flashcards:
            self.current_index = max(0, min(self.current_index, len(self.flashcards) - 1))
        else:
            self.current_index = 0
    
    def get_current_card(self) -> Flashcard | None:
        """Get the current flashcard."""
        if self.flashcards and 0 <= self.current_index < len(self.flashcards):
            card = self.flashcards[self.current_index]
            return card
        return None

    def next_card(self) -> None:
        """Move to the next card in the deck."""
        if self.current_index < len(self.flashcards) - 1:
            self.current_index += 1
        else:
            console.print("You've reached the end of the deck.")

    def previous_card(self) -> None:
        """Move to the previous card in the deck."""
        if self.current_index > 0:
            self.current_index -= 1
        else:
            console.print("You're at the beginning of the deck.")

    def jump_to_card(self, index: int) -> None:
        """
        Jump to a specific card in the deck.

        Args:
            index (int): The index of the card to jump to (0-based).
        """
        if 0 <= index < len(self.flashcards):
            self.current_index = index
        else:
            console.print("Invalid card number.")

    def add_flashcard(self, card_data: Dict[str, str]) -> None:
        """
        Add a new flashcard to the deck.

        Args:
            card_data (Dict[str, str]): The data for the new flashcard.
        """
        new_card = Flashcard(card_data, self.keys, self.styles)
        self.flashcards.append(new_card)

    def edit_flashcard(self, index: int, card_data: Dict[str, str]) -> None:
        """
        Edit an existing flashcard in the deck.

        Args:
            index (int): The index of the card to edit.
            card_data (Dict[str, str]): The updated data for the flashcard.
        """
        if 0 <= index < len(self.flashcards):
            self.flashcards[index].card_data.update(card_data)
        else:
            console.print("Invalid card number.")

    def delete_flashcard(self, index: int) -> None:
        """
        Delete a flashcard from the deck.

        Args:
            index (int): The index of the card to delete.
        """
        if 0 <= index < len(self.flashcards):
            del self.flashcards[index]
            if self.current_index >= len(self.flashcards):
                self.current_index = max(0, len(self.flashcards) - 1)
        else:
            console.print("Invalid card number.")

    def study_random(self) -> None:
        """Shuffle the flashcards and start studying from the beginning."""
        random.shuffle(self.flashcards)
        self.current_index = 0

    def search_flashcards(self, keyword: str) -> List[Flashcard]:
        """
        Search for flashcards containing a specific keyword.

        Args:
            keyword (str): The keyword to search for in flashcards.

        Returns:
            List[Flashcard]: A list of matching flashcards.
        """
        return [card for card in self.flashcards if any(keyword.lower() in str(value).lower() for value in card.card_data.values())]

class KeyboardHandler:
    """Handles keyboard input and manages key bindings."""

    def __init__(self, app: 'FlashcardApp'):
        """
        Initialize the KeyboardHandler instance.

        Args:
            app (FlashcardApp): The FlashcardApp instance.
        """
        self.app = app
        self.key_bindings = {
            'N': self.app.next_card,
            'P': self.app.previous_card,
            'S': self.app.toggle_answer,
            'J': self.app.jump_to_card,
            'R': self.app.study_random,
            'A': self.app.add_flashcard,
            'E': self.app.edit_flashcard,
            'D': self.app.delete_flashcard,
            'F': self.app.search_flashcards,
            'M': lambda: None,  # Return to main menu
            'Q': lambda: True,  # Save and quit
        }

    def handle_input(self, choice: str) -> bool:
        """
        Handle user input based on key bindings.

        Args:
            choice (str): The user's input choice.

        Returns:
            bool: True if the user chooses to save and quit, False otherwise.
        """
        action = self.key_bindings.get(choice.upper())
        if action:
            return action()
        return False

class FlashcardApp:
    """Represents the flashcard application."""

    def __init__(self) -> None:
        """Initialize the FlashcardApp instance."""
        self.flashcard_manager = FlashcardManager()
        self.show_answer: bool = False
        self.terminal_layout = TerminalLayout()
        self.keyboard_handler = KeyboardHandler(self)

    def load_flashcards(self, file_path: str) -> None:
        """Load flashcards from a JSON file."""
        self.flashcard_manager.load_flashcards(file_path)

    def save_flashcards(self, file_path: str) -> None:
        """Save flashcards to a JSON file."""
        self.flashcard_manager.save_flashcards(file_path)

    def next_card(self) -> None:
        """Move to the next card in the deck."""
        self.flashcard_manager.next_card()
        self.show_answer = False
        self.display_current_card()

    def previous_card(self) -> None:
        """Move to the previous card in the deck."""
        self.flashcard_manager.previous_card()
        self.show_answer = False
        self.display_current_card()

    def toggle_answer(self) -> None:
        """Toggle the visibility of the answer."""
        self.show_answer = not self.show_answer
        self.display_current_card()

    def jump_to_card(self) -> None:
        """Jump to a specific card in the deck."""
        card_number = int(Prompt.ask("Enter card number", default="1"))
        self.flashcard_manager.jump_to_card(card_number - 1)
        self.show_answer = False
        self.display_current_card()

    def add_flashcard(self) -> None:
        """Add a new flashcard to the deck."""
        new_card_data = {key: Prompt.ask(f"Enter the {key}") for key in self.flashcard_manager.keys.values()}
        self.flashcard_manager.add_flashcard(new_card_data)
        console.print("Flashcard added successfully!")

    def edit_flashcard(self) -> None:
        """Edit an existing flashcard in the deck."""
        index = int(Prompt.ask("Enter the card number to edit")) - 1
        if 0 <= index < len(self.flashcard_manager.flashcards):
            card = self.flashcard_manager.flashcards[index]
            updated_data = {}
            for key in self.flashcard_manager.keys.values():
                updated_data[key] = Prompt.ask(f"Enter the new {key}", default=card.card_data.get(key, ""))
            self.flashcard_manager.edit_flashcard(index, updated_data)
            console.print("Flashcard updated successfully!")
        else:
            console.print("Invalid card number.")

    def delete_flashcard(self) -> None:
        """Delete a flashcard from the deck."""
        index = int(Prompt.ask("Enter the card number to delete")) - 1
        self.flashcard_manager.delete_flashcard(index)
        console.print("Flashcard deleted successfully!")

    def study_random(self) -> None:
        """Shuffle the flashcards and start studying from the beginning."""
        self.flashcard_manager.study_random()
        self.show_answer = False
        self.display_current_card()

    def search_flashcards(self) -> None:
        """Search for flashcards containing a specific keyword."""
        keyword = Prompt.ask("Enter search keyword")
        results = self.flashcard_manager.search_flashcards(keyword)
        if results:
            table = Table(title=f"Search Results for '{keyword}'")
            table.add_column("Card #", style="cyan")
            for key in self.flashcard_manager.keys.values():
                table.add_column(key.capitalize(), style="magenta")
            for i, card in enumerate(results):
                table.add_row(str(i+1), *[str(card.card_data.get(key, "N/A")) for key in self.flashcard_manager.keys.values()])
            console.print(table)
        else:
            console.print("No matching flashcards found.")
    
    def display_current_card(self) -> None:
        """Display the current flashcard."""
        self.flashcard_manager.ensure_valid_index()
        current_card = self.flashcard_manager.get_current_card()
        if current_card:
            self.terminal_layout.render_flashcard(self)
        else:
            console.print("No flashcards to display.")

    def create_menu(self) -> str:
        """
        Create a string representation of the menu options.

        Returns:
            str: A string containing the menu options.
        """
        menu_options = [
            "Next Card (N)",
            "Previous Card (P)",
            "Show/Hide Answer (S)",
            "Jump to Card (J)",
            "Study Random (R)",
            "Add Flashcard (A)",
            "Edit Flashcard (E)",
            "Delete Flashcard (D)",
            "Search Flashcards (F)",
            "Return to Main Menu (M)",
            "Save and Quit (Q)"
        ]
        return "\n".join(menu_options)
    
    def study_flashcards(self) -> bool:
        """
        Study the flashcards interactively.

        Returns:
            bool: True if the user chooses to save and quit, False otherwise.
        """
        while True:
            self.display_current_card()
            choice = Prompt.ask("Choose an option", choices=list(self.keyboard_handler.key_bindings.keys()))
            if self.keyboard_handler.handle_input(choice):
                return True
            if choice.upper() == 'M':
                break
        return False

def display_menu(title: str, options: List[str]) -> Panel:
    """
    Create a panel displaying a menu with numbered options.

    Args:
        title (str): The title of the menu.
        options (List[str]): A list of menu options.

    Returns:
        Panel: A Rich Panel object containing the formatted menu.
    """
    menu_text = "\n".join(f"[{i+1}] {option}" for i, option in enumerate(options))
    return Panel(menu_text, title=title, expand=False)

@click.command()
@click.option('--file', default='flashcards.json', help='Path to the flashcards JSON file.')
def main(file: str) -> None:
    """
    Main function to run the flashcard application.

    Args:
        file (str): Path to the flashcards JSON file.
    """
    app = FlashcardApp()
    try:
        app.load_flashcards(file)
    except Exception as e:
        console.print(f"[bold red]Error loading flashcards: {str(e)}")
        return

    while True:
        console.clear()
        console.print(Panel("Flashcard Study App", style="bold magenta"))
        console.print(Panel("1. Study Flashcards: Study the flashcards interactively.", style="bold green"))
        console.print(Panel("2. Search Flashcards: Search for flashcards containing a specific keyword.", style="bold green"))
        console.print(Panel("3. Save and Quit: Save the flashcards and quit the application.", style="bold green"))

        choice = Prompt.ask("Choose an option", choices=["1", "2", "3"])

        if choice == "1":  # Study Flashcards
            if app.study_flashcards():
                break
        elif choice == "2":  # Search Flashcards
            app.search_flashcards()
            Prompt.ask("Press Enter to continue")
        elif choice == "3":  # Save and Quit
            app.save_flashcards(file)
            console.print("Flashcards saved. Goodbye!")
            break

if __name__ == "__main__":
    main()