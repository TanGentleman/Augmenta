"""Central configuration of paths for the project."""
from pathlib import Path

# Root paths
ROOT_DIR = Path(__file__).resolve().parent
FLASH_DIR = ROOT_DIR / "flash"
AUGMENTA_DIR = ROOT_DIR / "augmenta"

# Flash module paths
FLASHCARD_FILEPATH = FLASH_DIR / "flashcards.json"
MANIFEST_FILEPATH = FLASH_DIR / "manifest.json"

# Augmenta module paths
CONFIG_DIR = AUGMENTA_DIR / "config"

MODELS_YAML_PATH = AUGMENTA_DIR / "models" / "models.yaml"

# Data subdirectories
DOCUMENTS_DIR = ROOT_DIR / "documents"
DATA_DIR = ROOT_DIR / "data"
LLM_OUTPUTS_PATH = DATA_DIR / "llm-outputs"
LLM_RESPONSE_PATH = LLM_OUTPUTS_PATH / "markdown"
TEXT_FILE_DIR = DATA_DIR / "txt"
DB_DIR = DATA_DIR / "databases"
CHROMA_FOLDER_PATH = DB_DIR / "chroma-vector-dbs"
FAISS_FOLDER_PATH = DB_DIR / "faiss-vector-dbs"

MANIFEST_FILEPATH = DATA_DIR / "manifest.json"

INITIAL_MANIFEST_CONTENTS = '{"databases": []}'

def ensure_valid_framework():
    """
    Validate and create necessary folders and files for the project structure.
    Creates all required directories and initializes manifest.json if needed.
    """
    assert ROOT_DIR.exists(), "Root path does not exist"

    # Create data directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    DB_DIR.mkdir(exist_ok=True)
    LLM_OUTPUTS_PATH.mkdir(exist_ok=True)
    LLM_RESPONSE_PATH.mkdir(exist_ok=True)
    TEXT_FILE_DIR.mkdir(exist_ok=True)
    CHROMA_FOLDER_PATH.mkdir(exist_ok=True)
    FAISS_FOLDER_PATH.mkdir(exist_ok=True)

    # Create manifest.json if it doesn't exist
    if not MANIFEST_FILEPATH.exists():
        with open(MANIFEST_FILEPATH, "w") as f:
            f.write(INITIAL_MANIFEST_CONTENTS)

# TODO: This should only be called once at repo initialization
ensure_valid_framework()