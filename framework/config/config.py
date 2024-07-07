# This file is for special configuration variables affecting chat.py
# Use settings.json for chat and RAG settings

### chat.py ###
DEFAULT_TO_SAMPLE = True  # When set to true, prompt defaults to sample.txt
SAVE_ONESHOT_RESPONSE = True  # Save the response to a non-persistent query
LOCAL_MODEL_ONLY = False
EXPLAIN_EXCERPT = False  # if true, -np will format sample.txt

### RAG ###
MAX_PARENT_DOCS = 10  # This is only relevant for the multiretrieval mode with child docs
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
