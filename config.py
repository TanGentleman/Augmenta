# This file is for special configuration variables affecting chat.py
# Use settings.json for chat and RAG settings

### chat.py ###
DEFAULT_TO_SAMPLE = True  # When set to true, prompt defaults to sample.txt
SAVE_ONESHOT_RESPONSE = True  # Save the response to a non-persistent query
LOCAL_MODEL_ONLY = False
EXPLAIN_EXCERPT = False  # if true, -np will format sample.txt


# TODO: Adjust model: system message mapping to be more intuitive.
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
