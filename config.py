# This file is for setting up the configuration variables for both chat and RAG scripts.
# Configuration is moving in favor of the settings.json file.

# For chat.py
DEFAULT_TO_SAMPLE = True  # When set to true, prompt defaults to sample.txt
SAVE_ONESHOT_RESPONSE = True
LOCAL_MODEL_ONLY = False
# If set to true, -np will format sample.txt with FORMATTED_PROMPT
EXPLAIN_EXCERPT = False

VECTOR_DB_SUFFIX = "-vector-dbs"
CHROMA_FOLDER = "chroma" + VECTOR_DB_SUFFIX
FAISS_FOLDER = "faiss" + VECTOR_DB_SUFFIX
