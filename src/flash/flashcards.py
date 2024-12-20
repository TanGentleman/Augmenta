"""Main entry point for flashcards module.

This script provides a command-line interface for loading and displaying flashcards
from JSON files.

Example usage:
    python flashcards.py  # Uses default flashcards.json
    python flashcards.py --filename cards.json
    python flashcards.py --limit 5 --show-answers
"""
import argparse
from paths import FLASHCARD_DATA_DIR
from .manager import Flashcards

import logging
logger = logging.getLogger(__name__)

# Seconds to wait between displaying flashcards
DEFAULT_DELAY = 0.05

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and display flashcards from JSON files"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="flashcards.json",
        help="Name of JSON file in FLASHCARD_DATA_DIR (default: flashcards.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of flashcards to display (default: 10)"
    )
    parser.add_argument(
        "--show-answers",
        action="store_true",
        help="Show answers when displaying cards"
    )
    return parser.parse_args()

def main():
    """Main entry point for flashcards script."""
    args = parse_args()
    manager = Flashcards()
    
    try:
        filepath = FLASHCARD_DATA_DIR / args.filename
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        manager.load_from_json(filepath)
            
        # Limit number of flashcards if specified
        if args.limit > 0:
            manager.flashcards = manager.flashcards[:args.limit]
            
        manager.display_flashcards(include_answer=args.show_answers, delay=DEFAULT_DELAY)
        
    except Exception as e:
        logger.error(f"Failed to load or display flashcards: {str(e)}")
        raise

if __name__ == "__main__":
    main()
