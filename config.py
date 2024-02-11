# Rag config
PDF_FILENAME = "chromosome.pdf"

# chat.py config
PERSISTENCE_ENABLED = False
ENABLE_SYSTEM_MESSAGE = True

TOGETHER_API_ENABLED = False
ACTIVE_MODEL_TYPE = "local"

ALLOWED_MODEL_TYPES = ["local", "together"]
assert ACTIVE_MODEL_TYPE in ALLOWED_MODEL_TYPES, f"ACTIVE_MODEL_TYPE must be one of {ALLOWED_MODEL_TYPES}"

DISABLE_DEFAULT_PROMPT = True # When set to true, prompt defaults to sample.txt

SAVE_ONESHOT_RESPONSE = True