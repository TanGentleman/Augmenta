"""Main entry point for flashcards module.

This script provides a command-line interface for managing flashcards from JSON files.
It allows loading flashcards from any location and saving them to the flashcards data directory.

Example usage:
    python flashcards.py display  # Uses default flashcards.json
    python flashcards.py display -f cards.json --limit 5 --show-answers
    python flashcards.py import path/to/cards.json  # Import cards from any JSON file
"""
import argparse
import json
from pathlib import Path
from paths import FLASHCARD_DATA_DIR
from .manager import Flashcards

import logging
logger = logging.getLogger(__name__)

class Display:
    """Class to handle flashcard display and CLI functionality."""
    
    def __init__(self):
        """Initialize Display object with default values."""
        self.delay = 0.05  # Default delay in seconds
        self.manager = Flashcards()
        
    def parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Manage and display flashcards from JSON files",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s display                     Display cards from default flashcards.json
  %(prog)s display -f custom.json      Display cards from custom.json in data dir
  %(prog)s import /path/to/cards.json  Import cards from external JSON file
            """
        )
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Display command
        display_parser = subparsers.add_parser('display', help='Display flashcards')
        display_parser.add_argument(
            "-f", "--filename",
            type=str,
            default="flashcards.json",
            help="Name of JSON file in flashcards data directory (default: flashcards.json)"
        )
        display_parser.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Number of flashcards to display (default: 50)"
        )
        display_parser.add_argument(
            "--show-answers",
            action="store_true",
            help="Show answers when displaying cards"
        )
        display_parser.add_argument(
            "--delay",
            type=float,
            default=0.05,
            help="Delay in seconds between displaying cards (default: 0.05)"
        )

        # Import command
        import_parser = subparsers.add_parser('import', help='Import flashcards from any JSON file')
        import_parser.add_argument(
            'source',
            type=str,
            help='Source JSON file path to import'
        )
        import_parser.add_argument(
            '-n', '--name',
            type=str,
            help='Optional name for saved file (default: source filename)'
        )

        return parser.parse_args()

    def validate_json_file(self, filepath: Path) -> bool:
        """
        Validate that file exists and contains valid JSON list.
        
        Args:
            filepath: Path to JSON file to validate
            
        Returns:
            bool: True if file is valid
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file does not contain a JSON list
        """
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        try:    
            with open(filepath) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"JSON file must contain a list: {filepath}")
        except json.decoder.JSONDecodeError as e:
            raise ValueError(f"Broken JSON file: {e}")
        return True

    def display_command(self, args):
        """
        Handle display command.
        
        Args:
            args: Parsed command line arguments
        """
        filepath = FLASHCARD_DATA_DIR / args.filename
        self.validate_json_file(filepath)
        self.manager.load_from_json(filepath)
        
        if args.limit > 0:
            self.manager.flashcards = self.manager.flashcards[:args.limit]
            
        self.manager.display_flashcards(
            include_answer=args.show_answers,
            delay=args.delay  # Use the delay from command line args
        )

    def import_command(self, args):
        """
        Handle import command. Imports cards from source and saves to data directory.
        
        Args:
            args: Parsed command line arguments containing source path and optional name
        """
        source_path = Path(args.source)
        self.validate_json_file(source_path)
        
        # Determine target filename 
        target_name = args.name if args.name else source_path.name
        target_path = FLASHCARD_DATA_DIR / target_name
        
        # Load flashcards from source
        self.manager.load_from_json(source_path)
        
        # Save to target location
        self.manager.save_to_json(target_path)
        
        logger.info(
            f"Successfully imported {len(self.manager.flashcards)} flashcards to {target_path}"
        )

    def run(self):
        """Main entry point for flashcards script."""
        args = self.parse_args()
        
        if not args.command:
            logger.error("No command specified. Use --help for usage information.")
            return

        try:
            command_handlers = {
                'display': self.display_command,
                'import': self.import_command
            }
            
            handler = command_handlers.get(args.command)
            if handler:
                handler(args)
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise

def main():
    """Create and run Display object."""
    display = Display()
    display.run()

if __name__ == "__main__":
    main()
