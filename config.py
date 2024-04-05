# This file is for special configuration variables affecting chat.py
# Use settings.json for chat and RAG settings

### chat.py ###
DEFAULT_TO_SAMPLE = True  # When set to true, prompt defaults to sample.txt
SAVE_ONESHOT_RESPONSE = True  # Save the response to a non-persistent query
LOCAL_MODEL_ONLY = False
EXPLAIN_EXCERPT = False  # if true, -np will format sample.txt
# Formatted using constants.FORMATTED_PROMPT

### MISC ###
VECTOR_DB_SUFFIX = "-vector-dbs"
CHROMA_FOLDER = "chroma" + VECTOR_DB_SUFFIX
FAISS_FOLDER = "faiss" + VECTOR_DB_SUFFIX
