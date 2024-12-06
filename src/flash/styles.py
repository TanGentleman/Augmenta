"""Manages styling for flashcards by providing consistent color schemes and field mappings."""
from typing import Dict, List, Tuple, Any
from rich.color import ANSI_COLOR_NAMES

class StyleManager:
    @staticmethod
    def generate_mapping_and_styles(data: List[Dict[str, Any]]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
        """
        Generate field mapping and styles for flashcards based on the first item in the data.

        This function takes a list of flashcard dictionaries and creates:
        1. A mapping from generic field names (field_0, field_1, etc) to actual field names
        2. A list of styles that assigns a unique bold color to each field

        Args:
            data: List of dictionaries containing flashcard data, where each dictionary
                 represents a flashcard with keys like 'term', 'definition', etc.

        Returns:
            A tuple containing:
            - Dict mapping generic field names to actual field names (e.g. {'field_0': 'term'})
            - List of tuples pairing generic field names with rich text styles (e.g. [('field_0', 'bold red')])

        Raises:
            ValueError: If data is empty or first item is not a dictionary
        """
        # Validate input data
        if not data:
            raise ValueError("Input data is empty")

        first_item = data[0]
        if not isinstance(first_item, dict):
            raise ValueError("Data must be a list of dictionaries")

        # Create mapping from generic field_N to actual field names from first flashcard
        mapping = {f"field_{i}": key for i, key in enumerate(first_item.keys())}

        # Get list of ANSI colors, excluding 'default'
        colors = list(ANSI_COLOR_NAMES.keys())[1:]  

        # Create styles list, cycling through colors if we have more fields than colors
        styles = [(f"field_{i}", f"bold {colors[i % len(colors)]}")
                  for i in range(len(mapping))]

        return mapping, styles