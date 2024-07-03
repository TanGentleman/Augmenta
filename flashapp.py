import click
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.prompt import Prompt
from rich.layout import Layout
from rich.table import Table
from typing import List, Dict, Tuple
import random
import json
import logging

from flashcards import load_flashcards_from_json, Flashcard

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

class FlashcardApp:
    """
    A class representing the flashcard application.
    """

    def __init__(self) -> None:
        """
        Initialize the FlashcardApp instance.
        """
        self.flashcards: List[Flashcard] = []
        self.current_index: int = 0
        self.show_answer: bool = False
        self.keys: Dict[str, str] = {}
        self.styles: List[Tuple[str, str]] = []

    def load_flashcards(self, file_path: str) -> None:
        """
        Load flashcards from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing flashcard data.
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
        """
        try:
            data = [card.card_data for card in self.flashcards]
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.flashcards)} flashcards to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save flashcards: {str(e)}")
            raise

    def display_current_card(self) -> None:
        """
        Display the current flashcard.
        """
        if 0 <= self.current_index < len(self.flashcards):
            card = self.flashcards[self.current_index]
            
            layout = Layout()
            layout.split_column(
                Layout(Panel(f"Card {self.current_index + 1} of {len(self.flashcards)}", box=box.ROUNDED)),
                Layout(card.create_flashcard()),
                Layout(card.create_answer() if self.show_answer else Panel("", title="Answer", border_style="green"))
            )
            console.print(layout)
        else:
            console.print("No flashcards to display.")

    def next_card(self) -> None:
        """
        Move to the next card in the deck.
        """
        if self.current_index < len(self.flashcards) - 1:
            self.current_index += 1
            self.show_answer = False
            self.display_current_card()
        else:
            console.print("You've reached the end of the deck.")

    def previous_card(self) -> None:
        """
        Move to the previous card in the deck.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.show_answer = False
            self.display_current_card()
        else:
            console.print("You're at the beginning of the deck.")

    def toggle_answer(self) -> None:
        """
        Toggle the visibility of the answer.
        """
        self.show_answer = not self.show_answer
        self.display_current_card()

    def jump_to_card(self, index: int) -> None:
        """
        Jump to a specific card in the deck.

        Args:
            index (int): The index of the card to jump to (0-based).
        """
        if 0 <= index < len(self.flashcards):
            self.current_index = index
            self.show_answer = False
            self.display_current_card()
        else:
            console.print("Invalid card number.")

    def add_flashcard(self) -> None:
        """
        Add a new flashcard to the deck.
        """
        new_card_data = {}
        for key in self.keys.values():
            value = Prompt.ask(f"Enter the {key}")
            new_card_data[key] = value
        new_card = Flashcard(new_card_data, self.keys, self.styles)
        self.flashcards.append(new_card)
        console.print("Flashcard added successfully!")

    def edit_flashcard(self) -> None:
        """
        Edit an existing flashcard in the deck.
        """
        index = int(Prompt.ask("Enter the card number to edit")) - 1
        if 0 <= index < len(self.flashcards):
            card = self.flashcards[index]
            for key in self.keys.values():
                card.card_data[key] = Prompt.ask(f"Enter the new {key}", default=card.card_data.get(key, ""))
            console.print("Flashcard updated successfully!")
        else:
            console.print("Invalid card number.")

    def delete_flashcard(self) -> None:
        """
        Delete a flashcard from the deck.
        """
        index = int(Prompt.ask("Enter the card number to delete")) - 1
        if 0 <= index < len(self.flashcards):
            del self.flashcards[index]
            console.print("Flashcard deleted successfully!")
        else:
            console.print("Invalid card number.")

    def study_random(self) -> None:
        """
        Shuffle the flashcards and start studying from the beginning.
        """
        random.shuffle(self.flashcards)
        self.current_index = 0
        self.show_answer = False
        self.display_current_card()

    def search_flashcards(self, keyword: str) -> None:
        """
        Search for flashcards containing a specific keyword.

        Args:
            keyword (str): The keyword to search for in flashcards.
        """
        results = [card for card in self.flashcards if any(keyword.lower() in str(value).lower() for value in card.card_data.values())]
        if results:
            table = Table(title=f"Search Results for '{keyword}'")
            table.add_column("Card #", style="cyan")
            for key in self.keys.values():
                table.add_column(key.capitalize(), style="magenta")
            for i, card in enumerate(results):
                table.add_row(str(i+1), *[str(card.card_data.get(key, "N/A")) for key in self.keys.values()])
            console.print(table)
        else:
            console.print("No matching flashcards found.")

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

    main_menu_options = ["Study Flashcards", "Manage Flashcards", "Search Flashcards", "Save and Quit"]
    study_menu_options = ["Next Card", "Previous Card", "Show/Hide Answer", "Jump to Card", "Study in Random Order", "Return to Main Menu"]
    manage_menu_options = ["Add Flashcard", "Edit Flashcard", "Delete Flashcard", "Return to Main Menu"]

    while True:
        console.clear()
        console.print(display_menu("Main Menu", main_menu_options))
        choice = Prompt.ask("Choose an option", choices=[str(i) for i in range(1, len(main_menu_options) + 1)])
        
        if choice == "1":  # Study Flashcards
            while True:
                console.clear()
                app.display_current_card()
                console.print(display_menu("Study Menu", study_menu_options))
                study_choice = Prompt.ask("Choose an option", choices=[str(i) for i in range(1, len(study_menu_options) + 1)])
                if study_choice == "1":
                    app.next_card()
                elif study_choice == "2":
                    app.previous_card()
                elif study_choice == "3":
                    app.toggle_answer()
                elif study_choice == "4":
                    card_number = int(Prompt.ask("Enter card number", default="1"))
                    app.jump_to_card(card_number - 1)
                elif study_choice == "5":
                    app.study_random()
                elif study_choice == "6":
                    break
        
        elif choice == "2":  # Manage Flashcards
            while True:
                console.clear()
                console.print(display_menu("Manage Flashcards", manage_menu_options))
                manage_choice = Prompt.ask("Choose an option", choices=[str(i) for i in range(1, len(manage_menu_options) + 1)])
                if manage_choice == "1":
                    app.add_flashcard()
                elif manage_choice == "2":
                    app.edit_flashcard()
                elif manage_choice == "3":
                    app.delete_flashcard()
                elif manage_choice == "4":
                    break
        
        elif choice == "3":  # Search Flashcards
            keyword = Prompt.ask("Enter search keyword")
            app.search_flashcards(keyword)
            Prompt.ask("Press Enter to continue")
        
        elif choice == "4":  # Save and Quit
            app.save_flashcards(file)
            console.print("Flashcards saved. Goodbye!")
            break

if __name__ == "__main__":
    main()
