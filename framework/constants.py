# Constants for chat.py
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage

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

MODEL_CODES = {
    "gpt4": "get_openai_gpt4",
    "bigmix": "get_together_bigmix",
    "code": "get_deepseek_coder",
    "dbrx": "get_together_dbrx",
    "arctic": "get_together_arctic",
    "llama": "get_together_llama3",
    "deepseek": "get_together_deepseek_4k",
    "qwen": "get_together_qwen",
    "opus": "get_claude_opus",
    "local": "get_local_model",
    "ollama3": "get_ollama_llama3",
    "ollama": "get_ollama_mistral",
}

SYSTEM_MESSAGE_CODES = {
    "default": DEFAULT_SYSTEM_MESSAGE,
    "code": CODE_SYSTEM_MESSAGE,
    "eval": EVAL_EXCERPT_SYSTEM_MESSAGE,
    "rag": RAG_SYSTEM_MESSAGE,
    "prompting": PROMPT_CHOOSER_SYSTEM_MESSAGE,
    "flashcard": FLASHCARD_SYSTEM_MESSAGE,
}

RAG_COLLECTION_TO_SYSTEM_MESSAGE = {
    "default": RAG_SYSTEM_MESSAGE,
    "best_reference_prompt_collection": PROMPT_CHOOSER_SYSTEM_MESSAGE,
    "reference_prompt_collection": PROMPT_CHOOSER_SYSTEM_MESSAGE,
    "fixed_reference_nomic_prompt_collection": PROMPT_CHOOSER_SYSTEM_MESSAGE,
}

LOCAL_MODELS = [
    "local-ollama3",
    "mistral:7b-instruct-v0.3-q6_K",
    "local-model",
    "nomic-embed-text",
    "lmstudio-embedding-model",
    "local-hermes"
]


DEFAULT_QUERY = '''Name 5 strange vegetables that I am unlikely to see in Western countries.'''
# DEFAULT_SYSTEM_MESSAGE = "You are a domain expert AI for a graduate class. Be articulate, clear, and concise and your response."
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI."
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
