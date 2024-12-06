"""Main entry point for flashcards module."""
from paths import FLASHCARD_FILEPATH
from .manager import Flashcards

import logging
logger = logging.getLogger(__name__) 

def main():
    METHOD = "manifest"
    manager = Flashcards()
    
    try:
        if METHOD == "manifest":
            manager.load_from_manifest()
        elif METHOD == "flashcards":
            manager.load_from_json(FLASHCARD_FILEPATH)
        else:
            raise ValueError("Invalid method specified. Choose 'manifest' or 'flashcards'.")
            
        # Display first 10 flashcards
        manager.flashcards = manager.flashcards[:10]
        manager.display_flashcards(include_answer=True)
        
    except Exception as e:
        logger.error(f"Failed to load or display flashcards: {str(e)}")

if __name__ == "__main__":
    main()
