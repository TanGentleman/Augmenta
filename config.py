from models import get_together_quen, get_together_mix
from models import get_mistral, get_together_coder

# Rag config
EMBEDDINGS_STEP = True
PDF_FILENAME = "yang.pdf"
EMBEDDING_CONTEXT_SIZE = '8k'
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
USE_ADVANCED = False # This should be False unless I troubleshoot the errors
RAG_SYSTEM_MESSAGE = "You are a helpful AI assistant. Respond being clear, concise, and comprehensive."

# chat.py config
CHOSEN_MODEL = get_together_quen
BACKUP_MODEL = get_together_mix
# BACKUP_MODEL = get_mistral
PERSISTENCE_ENABLED = True
ENABLE_SYSTEM_MESSAGE = True

TOGETHER_API_ENABLED = True
ACTIVE_MODEL_TYPE = "together"

ALLOWED_MODEL_TYPES = ["local", "together"]
assert ACTIVE_MODEL_TYPE in ALLOWED_MODEL_TYPES, f"ACTIVE_MODEL_TYPE must be one of {ALLOWED_MODEL_TYPES}"

DISABLE_DEFAULT_PROMPT = False # When set to true, prompt defaults to sample.txt

SAVE_ONESHOT_RESPONSE = True