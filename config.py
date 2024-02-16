from models import get_together_mix, get_together_quen
from helpers import read_sample

# Rag config
EMBEDDINGS_STEP = False
PDF_FILENAME = "unit6-4k.pdf"
EMBEDDING_CONTEXT_SIZE = '8k'
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
USE_ADVANCED = False
CHOSEN_MODEL = get_together_quen
RAG_SYSTEM_MESSAGE = "You are a helpful AI assistant. Respond being clear, concise, and comprehensive."

# chat.py config
PERSISTENCE_ENABLED = True
ENABLE_SYSTEM_MESSAGE = True

TOGETHER_API_ENABLED = True
ACTIVE_MODEL_TYPE = "together"

ALLOWED_MODEL_TYPES = ["local", "together"]
assert ACTIVE_MODEL_TYPE in ALLOWED_MODEL_TYPES, f"ACTIVE_MODEL_TYPE must be one of {ALLOWED_MODEL_TYPES}"

DISABLE_DEFAULT_PROMPT = True # When set to true, prompt defaults to sample.txt

SAVE_ONESHOT_RESPONSE = True