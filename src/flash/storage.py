"""Handles storage operations for flashcards."""
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StorageManager:
    @staticmethod
    def load_from_json(file_path: str) -> List[Dict[str, Any]]:
        """Load raw flashcard data from JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            raise

    @staticmethod
    def save_to_json(flashcards: List[Any], file_path: str) -> None:
        """Save flashcards to JSON file.

        Args:
            flashcards (List[Any]): List of flashcard objects to save
            file_path (str): Path to JSON file to save to

        Raises:
            IOError: If there is an error writing to the file
        """
        try:
            data = [card.card_data for card in flashcards]
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=2)
            logger.info(f"Saved {len(flashcards)} flashcards to {file_path}")
        except IOError as e:
            logger.error(f"Error saving flashcards to {file_path}: {str(e)}")
            raise 