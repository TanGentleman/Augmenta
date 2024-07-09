"""
flashcards.py
-------------
This module provides classes and functions for managing flashcards.
It includes functionality for loading flashcards from JSON files,
creating flashcard objects, and displaying them.
"""

import json
from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.color import ANSI_COLOR_NAMES
from typing import List, Dict, Any, Tuple
import logging
from time import sleep

from pathlib import Path

from utils import MANIFEST_FILEPATH
FLASH_DIR = Path(__file__).resolve().parent
FLASHCARD_FILEPATH = FLASH_DIR / "flashcards.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_mapping_and_styles(
        data: List[Dict[str, Any]]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
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
    styles = [(f"field_{i}", f"bold {colors[i % len(colors)]}")
              for i in range(len(mapping))]

    return mapping, styles


class Flashcard:
    """Represents a flashcard with its data, field keys, and styles."""

    def __init__(self, card_data: Dict[str, Any],
                 keys: Dict[str, str], styles: List[Tuple[str, str]]):
        """
        Initialize a Flashcard instance.

        Args:
            card_data (Dict[str, Any]): The data for this flashcard.
            keys (Dict[str, str]): Mapping of field names to their corresponding keys.
            styles (List[Tuple[str, str]]): List of styles for each field.
        """
        self.original_card_data: Dict[str, Any] = card_data
        self.card_data: Dict[str, str] = {
            key: str(value) for key, value in card_data.items()}
        self.keys: Dict[str, str] = keys
        self.styles: List[Tuple[str, str]] = styles
        self.cache_flashcard_types()

    def cache_flashcard_types(self):
        """
        Cache the types of flashcard fields.
        """
        self.field_types = {key: type(value)
                            for key, value in self.card_data.items()}
        # Convert types to string representations
        new_card_data = {key: str(value)
                         for key, value in self.card_data.items()}
        self.original_card_data = self.card_data
        self.card_data = new_card_data
        # logger.warning(f"Field types have been converted. This means values are now compared to str(value). This may lead to unexpected behavior.")

    def create_flashcard(
            self,
            panel_name: str = "Question",
            include_answer=False) -> Panel:
        """
        Create a panel representing the question side of the flashcard.

        Args:
            panel_name (str, optional): The name of the panel. Defaults to "Question".

        Returns:
            Panel: A Rich Panel object containing the formatted question.
        """
        return self._create_panel(panel_name, include_answer=include_answer)

    def create_answer(self) -> Panel:
        """
        Create a panel representing the answer side of the flashcard.

        Returns:
            Panel: A Rich Panel object containing the formatted answer.
        """
        logger.debug(f"Answer card created: {self.card_data}")
        return self._create_panel("Answer", include_answer=True)

    def _create_panel(self, title: str, include_answer: bool) -> Panel:
        """
        Create a panel with the flashcard content.

        Args:
            title (str): The title of the panel.
            include_answer (bool): Whether to include the answer in the panel.

        Returns:
            Panel: A Rich Panel object containing the formatted flashcard content.
        """
        flashcard_text = Text()
        for i, (field, style) in enumerate(self.styles):
            key_name = self.keys.get(field)
            if key_name:
                if include_answer or key_name.lower() != 'answer':
                    if key_name == 'answer':
                        logger.info(
                            f"Exposed Answer: {self.card_data.get(key_name, 'N/A')}")
                    value = self.card_data.get(key_name, 'N/A')
                    flashcard_text.append(f"{key_name}: {value}", style=style)
                    if i < len(self.styles) - 1:
                        flashcard_text.append("\n")
        return Panel(
            flashcard_text,
            title=title,
            border_style="cyan" if title == "Question" else "green",
            expand=False)


def display_flashcards(
        flashcards: List[Flashcard],
        panel_name: str = "Question",
        delay: float = 0.1,
        include_answer=False) -> None:
    """
    Display a list of flashcards with a delay between each.

    Args:
        flashcards (List[Flashcard]): List of Flashcard objects to display.
        delay (float, optional): Delay in seconds between displaying each flashcard. Defaults to 0.1.
    """
    for flashcard in flashcards:
        print(flashcard.create_flashcard(panel_name, include_answer))
        sleep(delay)


def construct_flashcards(
        data: dict) -> Tuple[List[Flashcard], Dict[str, str], List[Tuple[str, str]]]:
    """
    Construct Flashcard objects from a dictionary.

    Args:
        data (dict): Dictionary containing flashcard data.

    Returns:
        List[Flashcard]: List of Flashcard objects.
        Dict[str, str]: Field mapping.
        List[Tuple[str, str]]: List of field styles.
    """
    keys, styles = generate_mapping_and_styles(data)
    flashcards = [Flashcard(card, keys, styles) for card in data]
    return flashcards, keys, styles


def load_flashcards_from_json(
        file_path: str) -> Tuple[List[Flashcard], Dict[str, str], List[Tuple[str, str]]]:
    """
    Load flashcards from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing flashcard data.

    Returns:
        Tuple[List[Flashcard], Dict[str, str], List[Tuple[str, str]]]: A tuple containing the list of Flashcard objects,
        field mapping, and styles.

    Raises:
        FileNotFoundError: If the specified file is not found.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the JSON data is not in the expected format.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            flashcards, keys, styles = construct_flashcards(data)
            logger.info(
                f"Loaded {len(flashcards)} flashcards from {file_path}")
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


def save_flashcards_to_json(
        flashcards: List[Flashcard],
        file_path: str) -> None:
    """
    Save flashcards to a JSON file.

    Args:
        flashcards (List[Flashcard]): List of Flashcard objects to save.
        file_path (str): Path to save the JSON file.

    Raises:
        IOError: If there's an error writing to the file.
    """
    try:
        data = [card.card_data for card in flashcards]
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        logger.info(f"Saved {len(flashcards)} flashcards to {file_path}")
    except IOError as e:
        logger.error(f"Error saving flashcards to {file_path}: {str(e)}")
        raise


def get_flashcards_from_manifest():
    """
    Get flashcards from the manifest file.
    """
    try:
        with open(MANIFEST_FILEPATH, 'r') as file:
            manifest = json.load(file)
            data = manifest['databases']
            flashcards, keys, styles = construct_flashcards(data)
            return flashcards, keys, styles
    except Exception as e:
        logger.error(f"Failed to load or display flashcards: {str(e)}")
        return None, None, None

def main():
    METHOD = "manifest"
    try:
        if METHOD == "manifest":
            flashcards, keys, styles = get_flashcards_from_manifest()
        elif METHOD == "flashcards":
            file_path = FLASHCARD_FILEPATH
            flashcards, keys, styles = load_flashcards_from_json(file_path)
        else:
            raise ValueError(
                "Invalid method specified. Please choose 'manifest' or 'flashcards'.")
        logger.info(f"Generated mapping: {keys}")
        logger.info(f"Generated styles: {styles}")
        flashcards = flashcards[:10]
        display_flashcards(flashcards, include_answer=True)
    except Exception as e:
        logger.error(f"Failed to load or display flashcards: {str(e)}")

if __name__ == "__main__":
    main()
