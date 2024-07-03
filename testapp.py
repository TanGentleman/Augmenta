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

from flashcards import load_flashcards_from_json, Flashcard, save_flashcards_to_json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

class MenuChoice(Enum):
    STUDY = "1"
    SEARCH = "2"
    QUIT = "3"

class TypedPrompt:
    @staticmethod
    def ask(prompt: str, type_: type, choices: Optional[List[str]] = None) -> Union[str, int]:
        while True:
            result = RichPrompt.ask(prompt, choices=choices)
            try:
                return type_(result)
            except ValueError:
                console.print(f"[bold red]Invalid input. Please enter a {type_.__name__}.[/bold red]")

class TerminalLayout:
    def __init__(self) -> None:
        self.update_terminal_size()

    def update_terminal_size(self) -> None:
        self.terminal_width, self.terminal_height = shutil.get_terminal_size()

    def render_flashcard(self, app: 'FlashcardApp') -> None:
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

class FlashcardManagerProtocol(Protocol):
    @abstractmethod
    def load_flashcards(self, file_path: str) -> None:
        pass

    @abstractmethod
    def save_flashcards(self, file_path: str) -> None:
        pass

    @abstractmethod
    def get_current_card(self) -> Optional[Flashcard]:
        pass

    @abstractmethod
    def next_card(self) -> None:
        pass

    @abstractmethod
    def previous_card(self) -> None:
        pass

class FlashcardManager(FlashcardManagerProtocol):
    def __init__(self) -> None:
        self.flashcards: List[Flashcard] = []
        self.current_index: int = 0
        self.keys: Dict[str, str] = {}
        self.styles: List[Tuple[str, str]] = []

    @typechecked
    def load_flashcards(self, file_path: str) -> None:
        try:
            self.flashcards, self.keys, self.styles = load_flashcards_from_json(file_path)
            logger.info(f"Loaded {len(self.flashcards)} flashcards from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load flashcards: {str(e)}")
            raise

    @typechecked
    def save_flashcards(self, file_path: str) -> None:
        save_flashcards_to_json(self.flashcards, file_path)

    def ensure_valid_index(self) -> None:
        if self.flashcards:
            self.current_index = max(0, min(self.current_index, len(self.flashcards) - 1))
        else:
            self.current_index = 0
    
    def get_current_card(self) -> Optional[Flashcard]:
        if self.flashcards and 0 <= self.current_index < len(self.flashcards):
            return self.flashcards[self.current_index]
        return None

    def next_card(self) -> None:
        if self.current_index < len(self.flashcards) - 1:
            self.current_index += 1
        else:
            console.print("You've reached the end of the deck.")

    def previous_card(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
        else:
            console.print("You're at the beginning of the deck.")

    @typechecked
    def jump_to_card(self, index: int) -> None:
        if 0 <= index < len(self.flashcards):
            self.current_index = index
        else:
            console.print("Invalid card number.")

    @typechecked
    def add_flashcard(self, card_data: Dict[str, str]) -> None:
        new_card = Flashcard(card_data, self.keys, self.styles)
        self.flashcards.append(new_card)

    @typechecked
    def edit_flashcard(self, index: int, card_data: Dict[str, str]) -> None:
        if 0 <= index < len(self.flashcards):
            for key, value in card_data.items():
                self.flashcards[index].card_data[key] = value
        else:
            console.print("Invalid card number.")

    @typechecked
    def delete_flashcard(self, index: int) -> bool:
        if 0 <= index < len(self.flashcards):
            del self.flashcards[index]
            if self.current_index >= len(self.flashcards):
                self.current_index = max(0, len(self.flashcards) - 1)
            return True
        else:
            console.print("Invalid card number.")
            return False

    def study_random(self) -> None:
        random.shuffle(self.flashcards)
        self.current_index = 0

    @typechecked
    def search_flashcards(self, keyword: str) -> List[Flashcard]:
        return [card for card in self.flashcards if any(keyword.lower() in str(value).lower() for value in card.card_data.values())]

class KeyboardHandler:
    def __init__(self, app: 'FlashcardApp'):
        self.app = app
        self.key_bindings: Dict[str, Callable[[], bool]] = {
            'N': self.app.next_card,
            'P': self.app.previous_card,
            'S': self.app.toggle_answer,
            'J': self.app.jump_to_card,
            'R': self.app.study_random,
            'A': self.app.add_flashcard,
            'E': self.app.edit_flashcard,
            'D': self.app.delete_flashcard,
            'F': self.app.search_flashcards,
            'M': lambda: True,  # Return to main menu
            'Q': lambda: True,  # Save and quit
        }

    @typechecked
    def handle_input(self, choice: str) -> bool:
        action = self.key_bindings.get(choice.upper())
        if action:
            return action()
        return False

class FlashcardApp:
    def __init__(self) -> None:
        self.flashcard_manager: FlashcardManagerProtocol = FlashcardManager()
        self.show_answer: bool = False
        self.terminal_layout = TerminalLayout()
        self.keyboard_handler = KeyboardHandler(self)

    @typechecked
    def load_flashcards(self, file_path: str) -> None:
        self.flashcard_manager.load_flashcards(file_path)

    @typechecked
    def save_flashcards(self, file_path: str) -> None:
        self.flashcard_manager.save_flashcards(file_path)

    def next_card(self) -> bool:
        self.flashcard_manager.next_card()
        self.show_answer = False
        self.display_current_card()
        return False  # Continue studying

    def previous_card(self) -> bool:
        self.flashcard_manager.previous_card()
        self.show_answer = False
        self.display_current_card()
        return False  # Continue studying

    def toggle_answer(self) -> bool:
        self.show_answer = not self.show_answer
        self.display_current_card()
        return False  # Continue studying

    def jump_to_card(self) -> bool:
        if not self.flashcard_manager.flashcards:
            console.print("[bold yellow]No flashcards available.[/bold yellow]")
            return False

        card_number = TypedPrompt.ask("Enter card number", int)
        if 1 <= card_number <= len(self.flashcard_manager.flashcards):
            self.flashcard_manager.jump_to_card(card_number - 1)
            self.show_answer = False
            self.display_current_card()
        else:
            console.print(f"[bold red]Error: Please enter a number between 1 and {len(self.flashcard_manager.flashcards)}.[/bold red]")
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
                    console.print(f"[bold red]Error: {key} cannot be empty. Please try again.[/bold red]")
        self.flashcard_manager.add_flashcard(new_card_data)
        console.print("[bold green]Flashcard added successfully![/bold green]")
        return False  # Continue studying

    def edit_flashcard(self) -> bool:
        if not self.flashcard_manager.flashcards:
            console.print("[bold yellow]No flashcards available to edit.[/bold yellow]")
            return False

        index = TypedPrompt.ask("Enter the card number to edit", int) - 1
        if 0 <= index < len(self.flashcard_manager.flashcards):
            card = self.flashcard_manager.flashcards[index]
            updated_data: Dict[str, str] = {}
            for key in self.flashcard_manager.keys.values():
                while True:
                    value = TypedPrompt.ask(f"Enter the new {key}", str, default=str(card.card_data.get(key, "")))
                    if value.strip():
                        updated_data[key] = value
                        break
                    else:
                        console.print(f"[bold red]Error: {key} cannot be empty. Please try again.[/bold red]")
            self.flashcard_manager.edit_flashcard(index, updated_data)
            console.print("[bold green]Flashcard updated successfully![/bold green]")
        else:
            console.print(f"[bold red]Error: Please enter a number between 1 and {len(self.flashcard_manager.flashcards)}.[/bold red]")
        return False  # Continue studying

    def delete_flashcard(self) -> bool:
        index = TypedPrompt.ask("Enter the card number to delete", int) - 1
        if 0 <= index < len(self.flashcard_manager.flashcards):
            confirm = TypedPrompt.ask("Are you sure you want to delete this flashcard? (y/n)", str, choices=["y", "n"])
            if confirm.lower() == "y":
                self.flashcard_manager.delete_flashcard(index)
                console.print(f"Flashcard {index + 1} deleted successfully!")
            else:
                console.print("Deletion cancelled.")
        else:
            console.print("Invalid card number.")
        return False  # Continue studying

    def study_random(self) -> bool:
        self.flashcard_manager.study_random()
        self.show_answer = False
        self.display_current_card()
        return False  # Continue studying

    def search_flashcards(self) -> bool:
        keyword = TypedPrompt.ask("Enter search keyword", str)
        if keyword.strip():
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
        else:
            console.print("[bold red]Error: Search keyword cannot be empty. Please try again.[/bold red]")
        return False  # Continue studying
    
    def display_current_card(self) -> None:
        current_card = self.flashcard_manager.get_current_card()
        if current_card:
            self.terminal_layout.render_flashcard(self)
        else:
            console.print("No flashcards to display.")

    def create_menu(self) -> str:
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
        if not self.flashcard_manager.flashcards:
            console.print("[bold yellow]No flashcards available. Please add some flashcards first.[/bold yellow]")
            TypedPrompt.ask("Press Enter to continue", str)
            return False

        while True:
            self.display_current_card()
            choice_list = list(self.keyboard_handler.key_bindings.keys())
            # allow lowercase input as well
            choice_list.extend([choice.lower() for choice in choice_list])
            choice = TypedPrompt.ask("Choose an option", str, choices=choice_list)
            assert choice in choice_list
            choice = choice.upper()
            if choice == 'Q':  # Save and Quit
                return True
            elif choice == 'M':  # Return to Main Menu
                return False
            self.keyboard_handler.handle_input(choice)
        return False

def display_menu(title: str, options: List[str]) -> Panel:
    menu_text = "\n".join(f"[{i+1}] {option}" for i, option in enumerate(options))
    return Panel(menu_text, title=title, expand=False)

@click.command()
@click.option('--file', default='flashcards.json', help='Path to the flashcards JSON file.')
def main(file: str) -> None:
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

        choice = TypedPrompt.ask("Choose an option", str, choices=[choice.value for choice in MenuChoice])

        if choice == MenuChoice.STUDY.value:
            if app.study_flashcards():
                break
        elif choice == MenuChoice.SEARCH.value:
            app.search_flashcards()
            TypedPrompt.ask("Press Enter to continue", str)
        elif choice == MenuChoice.QUIT.value:
            app.save_flashcards(file)
            console.print("Flashcards saved. Goodbye!")
            break

if __name__ == "__main__":
    main()