"""Manages flashcard operations."""
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from time import sleep
from rich import print
from paths import MANIFEST_FILEPATH
from .flashcard import Flashcard
from .styles import StyleManager
from .storage import StorageManager

# Logging setup
logger = logging.getLogger(__name__)

class Flashcards:
    def __init__(self, front_panel_name: Optional[str] = None,
                 back_panel_name: Optional[str] = None):
        self.flashcards: List[Flashcard] = []
        self.keys: Dict[str, str] = {}
        self.styles: List[Tuple[str, str]] = []
        self.front_panel_name: str = front_panel_name
        self.back_panel_name: str = back_panel_name

    def load_from_manifest(self) -> None:
        """Load flashcards from manifest file."""
        try:
            with open(MANIFEST_FILEPATH, 'r') as file:
                manifest = json.load(file)
                self.construct_flashcards(manifest['databases'])
        except Exception as e:
            logger.error(f"Failed to load flashcards from manifest: {str(e)}")
            raise

    def load_from_json(self, file_path: str) -> None:
        """Load flashcards from JSON file."""
        data = StorageManager.load_from_json(file_path)
        self.construct_flashcards(data)

    def save_to_json(self, file_path: str) -> None:
        """Save flashcards to JSON file."""
        StorageManager.save_to_json(self.flashcards, file_path)

    def construct_flashcards(self, data: List[Dict[str, Any]]) -> None:
        """Construct Flashcard objects from data."""
        self.keys, self.styles = StyleManager.generate_mapping_and_styles(data)
        self.flashcards = [Flashcard(card, self.keys, self.styles,
                                     self.front_panel_name, self.back_panel_name)
                           for card in data]
        logger.info(f"Constructed {len(self.flashcards)} flashcards")

    def display_flashcards(self, delay: float = 0.1, include_answer: bool = False) -> None:
        """Display all flashcards with optional delay."""
        for flashcard in self.flashcards:
            print(flashcard.create_flashcard(include_answer=include_answer))
            sleep(delay) 