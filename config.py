# This file is for setting up the configuration variables for both chat and RAG scripts.

from models import get_together_quen, get_together_nous_mix
from models import get_local_model, get_together_coder, get_claude_opus, get_openai_gpt4

# User config
# ROOT_DIR goes here

# Rag config
EMBEDDINGS_STEP = True
PDF_FILENAME = "yang.pdf"
EMBEDDING_CONTEXT_SIZE = '2k'
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
USE_ADVANCED = False # This should be False unless I troubleshoot the errors
RAG_SYSTEM_MESSAGE = "You are a helpful AI assistant. Respond being clear, concise, and comprehensive."

# chat.py config
CHOSEN_MODEL = get_claude_opus
# BACKUP_MODEL = get_openai_gpt4
BACKUP_MODEL = get_local_model
PERSISTENCE_ENABLED = True
ENABLE_SYSTEM_MESSAGE = True

TOGETHER_API_ENABLED = True
ACTIVE_MODEL_TYPE = "together"

ALLOWED_MODEL_TYPES = ["local", "together"]
assert ACTIVE_MODEL_TYPE in ALLOWED_MODEL_TYPES, f"ACTIVE_MODEL_TYPE must be one of {ALLOWED_MODEL_TYPES}"

DEFAULT_TO_SAMPLE = True # When set to true, prompt defaults to sample.txt

SAVE_ONESHOT_RESPONSE = True