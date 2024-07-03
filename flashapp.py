import click
from rich.console import Console
from rich.prompt import Prompt

from flashcard import load_flashcards_from_json

console = Console()

class FlashcardApp:
    def __init__(self):
        self.flashcards = []
        self.current_index = 0

    def load_flashcards(self, file_path):
        self.flashcards, self.keys, self.styles = load_flashcards_from_json(file_path)

    def display_current_card(self):
        if 0 <= self.current_index < len(self.flashcards):
            console.print(self.flashcards[self.current_index].create_flashcard())
        else:
            console.print("No flashcards to display.")

    def next_card(self):
        if self.current_index < len(self.flashcards) - 1:
            self.current_index += 1
            self.display_current_card()
        else:
            console.print("You've reached the end of the deck.")

    def previous_card(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_card()
        else:
            console.print("You're at the beginning of the deck.")

@click.command()
@click.option('--file', default='flashcards.json', help='Path to the flashcards JSON file.')
def main(file):
    app = FlashcardApp()
    app.load_flashcards(file)
    is_first_card = True
    while True:
        if is_first_card:
            app.display_current_card()
            is_first_card = False
        action = Prompt.ask("Action", choices=["next", "prev", "quit"])
        if action == "next":
            app.next_card()
        elif action == "prev":
            app.previous_card()
        elif action == "quit":
            break

if __name__ == "__main__":
    main()
