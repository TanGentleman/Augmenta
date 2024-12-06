# Constants for chat.py
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage


HISTORY_CHAPTER_TO_PAGE_RANGE = {
    1: (0, 25),
    2: (25, 43),
    3: (43, 66),
    4: (66, 84),
    5: (84, 106),
    6: (106, 122),
    7: (122, 142),
    8: (142, 165),
    9: (165, 191),
    10: (191, 212),
    11: (212, 234),
    12: (234, 257),
    13: (257, 288),
    14: (288, 321),
    15: (321, 349),
    16: (349, 371),
    17: (371, 391),
    18: (391, 410),
    19: (410, 435),
    20: (435, 452),
    21: (452, 478),
    22: (478, 501),
    23: (501, 529),
    24: (529, 558),
    25: (558, 591),
    26: (591, 624),
    27: (624, 645),
    28: (645, 665),
    29: (665, 688),
    30: (688, 706),
    31: (706, 729),
    32: (729, 754),
    33: (754, 779),
    34: (779, 808),
    35: (808, 829),
    36: (829, 858),
    37: (858, 889),
    38: (889, 918),
    39: (918, 948),
    40: (948, 978),
    41: (978, 1016),
    42: (1016, 1036),
}

#TODO: Find a better home for this function
def get_page_range_from_prompt(prompt: str) -> tuple[int, int]:
    raw_prompt_string = prompt.lower()
    
    # Check for direct page range pattern (e.g., "page 5-10" or "pages 5-10")
    page_pattern = r'(?:page|pages)\s*(\d+)-(\d+)'
    page_match = re.search(page_pattern, raw_prompt_string)
    if page_match:
        start_page = int(page_match.group(1))
        if start_page == 0:
            start_page = 1
        end_page = int(page_match.group(2))
        return start_page - 1, end_page  # Convert to 0-based indexing
        
    # Check for chapter patterns (e.g., "ch 5", "ch. 5", "chapter 5")
    chapter_pattern = r'(?:chapter|ch\.?)\s*(\d+)'
    chapter_match = re.search(chapter_pattern, raw_prompt_string)
    if chapter_match:
        # Get the first non-None group (either group 1 or 2)
        chapter_number = int(next(g for g in chapter_match.groups() if g is not None))
        if chapter_number in HISTORY_CHAPTER_TO_PAGE_RANGE:
            return HISTORY_CHAPTER_TO_PAGE_RANGE[chapter_number]
        raise ValueError(f"Could not find valid chapter number in: {prompt}")
        
    print("WARNING: No page range or chapter number found in prompt. Using all pages!")
    return 0, -1

### MISC ###
VECTOR_DB_SUFFIX = "-vector-dbs"
CHROMA_FOLDER = "chroma" + VECTOR_DB_SUFFIX
FAISS_FOLDER = "faiss" + VECTOR_DB_SUFFIX

CURRENT_SYSTEM_MESSAGE = "You are working on Augmenta AI. This is a framework for LLM and RAG pipelines. You are currently iterating on chat.py"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI."
CODE_SYSTEM_MESSAGE = "You are an expert programmer. Review the Python code and provide optimizations."
RAG_SYSTEM_MESSAGE = "You are a helpful AI. Use the document excerpts to respond to the best of your ability."
PROMPT_CHOOSER_SYSTEM_MESSAGE = "Use the context from Anthropic's example prompting guides to create a sample system message and user message template for the given task."
# EVAL_EXCERPT_SYSTEM_MESSAGE = "You are an AI assistant that evaluates text excerpts to determine if it meets specified criteria. Respond ONLY with a valid JSON output with 2 keys: index: int, and meetsCriteria: bool."
EVAL_EXCERPT_SYSTEM_MESSAGE = 'Evaluate a text excerpt to determine if it meets specified criteria. Always include a boolean key meetsCriteria. Respond ONLY with a valid JSON output.'
MUSIC_SYSTEM_MESSAGE = 'You will be provided with unstructured data, and your task is to parse it into JSON format. You should output a list of dictionaries, where each dictionary contains the keys "title", "artist", and "album".'
FLASHCARD_SYSTEM_MESSAGE = """You are Flashcard AI. Use the document excerpts to generate a list of JSON flashcards. Example output:
[
    {
        "term": "Python",
        "definition": "A high-level, interpreted programming language with a focus on code readability.",
        "example": "print('Hello, World!')"
    },
    {
        "term": "JavaScript",
        "definition": "A high-level, dynamic, and interpreted programming language that is primarily used for building web applications and adding interactive elements to websites.",
        "example": "console.log('Hello, World!');"
    }
]"""

FLASHCARD_SIMPLE_SYSTEM_MESSAGE = "Generate a list of JSON flashcards with consistent keys appropriate for the given request. Only output a list[dict] with valid JSON."
# "Output a list[dict]. Each JSON object should stay consistent to the schema. Use lowercase for boolean values."

MODEL_CODES = {
    "gpt4": "get_openai_gpt4",
    "bigmix": "get_together_bigmix",
    "mix": "get_together_fn_mix",
    "code": "get_deepseek_coder",
    "dbrx": "get_together_dbrx",
    "arctic": "get_together_arctic",
    # "llama": "get_together_llama3",
    "llama": "get_together_new_llama",
    "llama400": "get_together_llama_400b",
    "deepseek": "get_together_deepseek_4k",
    "qwen": "get_together_qwen",
    "sonnet": "get_openrouter_sonnet",
    "ollama": "get_ollama_llama3",
    "flash": "get_gemini_litellm",
}

SYSTEM_MESSAGE_CODES = {
    "default": DEFAULT_SYSTEM_MESSAGE,
    "code": CODE_SYSTEM_MESSAGE,
    "eval": EVAL_EXCERPT_SYSTEM_MESSAGE,
    "rag": RAG_SYSTEM_MESSAGE,
    "prompting": PROMPT_CHOOSER_SYSTEM_MESSAGE,
    "flashcard": FLASHCARD_SIMPLE_SYSTEM_MESSAGE,
}

MODEL_TO_SYSTEM_MSG = {
    # "meta-llama/Llama-3-70b-chat-hf": FLASHCARD_SIMPLE_SYSTEM_MESSAGE,
}

RAG_COLLECTION_TO_SYSTEM_MESSAGE = {
    "default": RAG_SYSTEM_MESSAGE,
    "best_reference_prompt_collection": PROMPT_CHOOSER_SYSTEM_MESSAGE,
    "reference_prompt_collection": PROMPT_CHOOSER_SYSTEM_MESSAGE,
    "fixed_reference_nomic_prompt_collection": PROMPT_CHOOSER_SYSTEM_MESSAGE,
}

LOCAL_MODELS = [
    "llama3",
    "mistral:7b-instruct-v0.3-q6_K",
    "nomic-embed-text",
    "lmstudio-embedding-model",
]

MAX_CHARS_IN_PROMPT = 200000
MAX_CHAT_EXCHANGES = 20

# This is the template for the RAG prompt
RAG_CONTEXT_TEMPLATE = """Use the following context to answer the question:
<context>
{context}
</context>

Question: {question}
"""

# This is a basic summary template
SUMMARY_TEMPLATE = """Summarize the following text, retaining the main keywords:
<excerpt>
{excerpt}
</excerpt>
"""

EVAL_TEMPLATE = """Evaluate the following text excerpt(s) based on the given criteria:
<excerpt>
{excerpt}
</excerpt>

Criteria: {criteria}
"""

MUSIC_TEMPLATE = """Please convert the following string into a JSON output (a list of dictionaries) with the keys "title", "artist", and "album". Each entry should obey the given SearchSchema.
class SearchSchema(BaseModel):
    title: str
    artist: str
    album: int

input:
{input}

ONLY output the list[dict]. Do not include any other information.

output:
"""

ALT_MUSIC_TEMPLATE = """Convert the string into JSON output (a list of dictionaries). Each dictionary should obey the given SearchSchema.
class SearchSchema(BaseModel):
    title: str
    artist: str
    album: int

ONLY output the list[dict]. No preamble.
{few_shot_examples}
input:
{input}
output:
"""


def get_rag_template(system_message=None):
    """
    Fetches the RAG template for the prompt.
    This template expects to be passed values for both context and question.
    """
    if system_message is None:
        system_message = RAG_COLLECTION_TO_SYSTEM_MESSAGE["default"]
    template = RAG_CONTEXT_TEMPLATE
    rag_prompt_template = ChatPromptTemplate.from_template(template)
    # Make sure this is the correct message for the task
    rag_prompt_template.messages.insert(
        0, SystemMessage(content=system_message))
    return rag_prompt_template


def get_summary_template():
    """
    Fetches the template for summarization.
    """
    template = SUMMARY_TEMPLATE
    summary_template = ChatPromptTemplate.from_template(template)
    summary_template.messages.insert(
        0, SystemMessage(
            content="You are a helpful AI."))
    return summary_template


def get_eval_template():
    """
    Fetches the template for evaluation.
    """
    system_message = EVAL_EXCERPT_SYSTEM_MESSAGE
    template = EVAL_TEMPLATE
    eval_template = ChatPromptTemplate.from_template(template)
    eval_template.messages.insert(
        0, SystemMessage(content=system_message))
    return eval_template


def get_music_template():
    """
    Fetches the template for the music pipeline.

    Args:
    few_shot_data: list[dict] - A list of dictionaries containing few-shot examples for the task.
    Each dictionary should be of the form {"input": str, "output": str}
    """
    system_message = MUSIC_SYSTEM_MESSAGE
    # template = MUSIC_TEMPLATE
    template = ALT_MUSIC_TEMPLATE
    music_template = ChatPromptTemplate.from_template(template)
    music_template.messages.insert(
        0, SystemMessage(content=system_message))

    return music_template
