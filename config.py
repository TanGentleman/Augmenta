# This file is for special configuration variables affecting chat.py
# Use settings.json for chat and RAG settings

### chat.py ###
DEFAULT_TO_SAMPLE = True  # When set to true, prompt defaults to sample.txt
SAVE_ONESHOT_RESPONSE = True  # Save the response to a non-persistent query
LOCAL_MODEL_ONLY = False
EXPLAIN_EXCERPT = False  # if true, -np will format sample.txt


# TODO: Add friendly interface for setting system messages.
# SYSTEM_MSG_MAP = {
#     "eval": "You are an AI assistant that evaluates text excerpts to determine if it meets specified criteria. Respond ONLY with a valid JSON output with 2 keys: index: int, and meetsCriteria: bool.",
#     "code": "You are an expert programmer that helps to review Python code and provide optimizations.",
#     "chat": "You are a helpful AI."
# }
SYSTEM_MSG_MAP = {"deepseek-ai/deepseek-llm-67b-chat":
                  "You are an expert programmer. Review the Python code and provide optimizations.", }
# Formatted using constants.SUMMARY_TEMPLATE

### MISC ###
VECTOR_DB_SUFFIX = "-vector-dbs"
CHROMA_FOLDER = "chroma" + VECTOR_DB_SUFFIX
FAISS_FOLDER = "faiss" + VECTOR_DB_SUFFIX

### RAG ###
MAX_PARENT_DOCS = 8  # This is only relevant for the multiretrieval mode with child docs
MAX_CHARACTERS_IN_PARENT_DOC = 20000

ALLOW_MULTI_VECTOR = True # Langchain deprecations cause an annoying warning atm
EXPERIMENTAL_UNSTRUCTURED = False
METADATA_MAP = None
FILTER_TOPIC = None

# METADATA_MAP = {
#     "example.pdf": "Examples",
#     "example_1.txt": "Examples",

#     "SciencePaper.pdf": "Papers",
#     "https://arxiv.org/1234": "Papers",

#     "RelevantFileForThisCollection.txt": "Misc",
# }
# FILTER_TOPIC = "Papers"
