import json
from rich import print
from rich.panel import Panel
from rich.text import Text
from typing import List, Dict, Any, Tuple
from rich.color import ANSI_COLOR_NAMES
import logging
from time import sleep

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_mapping_and_styles(data: List[Dict[str, Any]]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Generate field mapping and styles for flashcards based on the first item in the data.

    Args:
        data (List[Dict[str, Any]]): List of dictionaries containing flashcard data.

    Returns:
        Tuple[Dict[str, str], List[Tuple[str, str]]]: A tuple containing the field mapping and styles.

    Raises:
        ValueError: If the input data is empty or not a list of dictionaries.
    """
    if not data:
        raise ValueError("Input data is empty")
    
    first_item = data[0]
    if not isinstance(first_item, dict):
        raise ValueError("Data must be a list of dictionaries")
    
    mapping = {f"field_{i}": key for i, key in enumerate(first_item.keys())}
    colors = list(ANSI_COLOR_NAMES.keys())[1:]  # Exclude 'default' color
    styles = [(f"field_{i}", f"bold {colors[i % len(colors)]}") for i in range(len(mapping))]
    
    return mapping, styles

class Flashcard:
    """
    Represents a flashcard with its data, field keys, and styles.
    """

    def __init__(self, card_data: Dict[str, Any], keys: Dict[str, str], styles: List[Tuple[str, str]]):
        """
        Initialize a Flashcard instance.

        Args:
            card_data (Dict[str, Any]): The data for this flashcard.
            keys (Dict[str, str]): Mapping of field names to their corresponding keys.
            styles (List[Tuple[str, str]]): List of styles for each field.
        """
        self.card_data = card_data
        self.keys = keys
        self.styles = styles

    # def create_flashcard(self) -> Panel:
    #     """
    #     Create a Rich Panel representing the flashcard.

    #     Returns:
    #         Panel: A Rich Panel object representing the flashcard.
    #     """
    #     flashcard_text = Text()
    #     for i, (field, style) in enumerate(self.styles):
    #         key_name = self.keys.get(field)
    #         if key_name:
    #             value = self.card_data.get(key_name, 'N/A')
    #             flashcard_text.append(f"{key_name}: {value}", style=style)
    #             if i < len(self.styles) - 1:
    #                 flashcard_text.append("\n")
    #     return Panel(flashcard_text, title="Flashcard", border_style="red")
    def create_flashcard(self) -> Panel:
        return self._create_panel("Question", include_answer=False)
    
    def create_answer(self) -> Panel:
        return self._create_panel("Answer", include_answer=True)

    def _create_panel(self, title: str, include_answer: bool) -> Panel:
        flashcard_text = Text()
        for i, (field, style) in enumerate(self.styles):
            key_name = self.keys.get(field)
            if key_name:
                if include_answer or key_name.lower() != 'answer':
                    value = self.card_data.get(key_name, 'N/A')
                    flashcard_text.append(f"{key_name}: {value}", style=style)
                    if i < len(self.styles) - 1:
                        flashcard_text.append("\n")
        return Panel(flashcard_text, title=title, border_style="cyan" if title == "Question" else "green")

def display_flashcards(flashcards: List[Flashcard], delay: float = 0.1):
    """
    Display a list of flashcards with a delay between each.

    Args:
        flashcards (List[Flashcard]): List of Flashcard objects to display.
        delay (float, optional): Delay in seconds between displaying each flashcard. Defaults to 0.1.
    """
    for flashcard in flashcards:
        print(flashcard.create_flashcard())
        sleep(delay)

def load_flashcards_from_json(file_path: str) -> Tuple[List[Flashcard], Dict[str, str], List[Tuple[str, str]]]:
    """
    Load flashcards from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing flashcard data.

    Returns:
        Tuple[List[Flashcard], Dict[str, str], List[Tuple[str, str]]]: 
        A tuple containing the list of Flashcard objects, field mapping, and styles.

    Raises:
        FileNotFoundError: If the specified file is not found.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the JSON data is not in the expected format.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        keys, styles = generate_mapping_and_styles(data)
        flashcards = [Flashcard(card, keys, styles) for card in data]
        return flashcards, keys, styles
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        raise
    except ValueError as e:
        logger.error(f"Error processing data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        flashcards, keys, styles = load_flashcards_from_json('flashcards.json')
        logger.info(f"Generated mapping: {keys}")
        logger.info(f"Generated styles: {styles}")
        display_flashcards(flashcards)
    except Exception as e:
        logger.error(f"Failed to load or display flashcards: {str(e)}")