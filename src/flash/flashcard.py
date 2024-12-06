"""Flashcard class definition."""
from typing import Dict, Any, List, Optional, Tuple
from rich.panel import Panel
from rich.text import Text

import logging
logger = logging.getLogger(__name__)

FRONT_PANEL_NAME = "Question"
BACK_PANEL_NAME = "Answer"

class Flashcard:
    """Represents a flashcard with its data, field keys, and styles."""

    def __init__(self, card_data: Dict[str, Any],
                 keys: Dict[str, str], styles: List[Tuple[str, str]],
                 front_panel_name: Optional[str] = None,
                 back_panel_name: Optional[str] = None):
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
        self.front_panel_name: str = front_panel_name or FRONT_PANEL_NAME
        self.back_panel_name: str = back_panel_name or BACK_PANEL_NAME
        # TODO: Check that panel names are valid (Can't be too long)
        self.cache_flashcard_types()

    def cache_flashcard_types(self):
        """Cache and validate flashcard field types and values.
        
        This method performs two main functions:
        1. Records the original data types of all fields, including nested structures
        2. Converts all values to strings with length validation
        
        The field_types attribute will contain either:
        - A Python type object (e.g. str, int)
        - A list of nested types
        - A dict of nested types
        """
        MAX_STR_LENGTH = 10000
        
        def get_nested_type(value, depth=0, max_depth=100):
            """Recursively get types of nested data structures up to max_depth."""
            if depth >= max_depth:
                return type(value)
                
            if isinstance(value, (list, dict)):
                if isinstance(value, list):
                    return [get_nested_type(item, depth + 1) for item in value]
                return {k: get_nested_type(v, depth + 1) for k, v in value.items()}
            return type(value)
            
        # Store original types
        self.field_types = {
            key: get_nested_type(value) 
            for key, value in self.card_data.items()
        }
        
        # Convert all values to length-validated strings
        new_card_data = {}
        for key, value in self.card_data.items():
            str_value = str(value)
            if len(str_value) > MAX_STR_LENGTH:
                logger.warning(f"Value for {key} truncated to {MAX_STR_LENGTH} characters.")
                str_value = str_value[:MAX_STR_LENGTH] + "..."
            new_card_data[key] = str_value
            
        self.original_card_data = self.card_data
        self.card_data = new_card_data

    def create_flashcard(self, panel_name: Optional[str] = None,
                         include_answer=False) -> Panel:
        """
        Create a panel representing the question side of the flashcard.

        Args:
            panel_name (str, optional): The name of the panel. Defaults to "Question".
            include_answer (bool, optional): Whether to include the answer. Defaults to False.

        Returns:
            Panel: A Rich Panel object containing the formatted question.
        """
        panel_name = panel_name or self.front_panel_name
        return self.create_panel(panel_name, include_answer=include_answer)

    def create_panel(self, title: str, include_answer: bool) -> Panel:
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
                        logger.debug(
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
    