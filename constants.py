# Constants for chat.py
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage

RAG_SYSTEM_MESSAGE = "You are a helpful AI. Use the document excerpts to respond to the best of your ability."
PROMPT_CHOOSER_SYSTEM_MESSAGE = "Use the context from Anthropic's example prompting guides to create a sample system message and user message template for the given task."
EVAL_EACH_EXCERPT_SYSTEM_MESSAGE = "You are an AI assistant that evaluates text excerpts to determine if they meet specified criteria. When given text excerpt(s) and criteria, respond only with valid JSON output containing a single boolean field for each excerpt indicating if the text meets the given criteria."
RAG_COLLECTION_TO_SYSTEM_MESSAGE = {
    "default": RAG_SYSTEM_MESSAGE,
    "best_reference_prompt_collection": PROMPT_CHOOSER_SYSTEM_MESSAGE,
    "hw18_collection": EVAL_EACH_EXCERPT_SYSTEM_MESSAGE,
    "metacognition_collection": EVAL_EACH_EXCERPT_SYSTEM_MESSAGE,
    "metacognition_nomic_collection": EVAL_EACH_EXCERPT_SYSTEM_MESSAGE,
    "metacognition_together_collection": EVAL_EACH_EXCERPT_SYSTEM_MESSAGE,
}

DEFAULT_QUERY = '''Name 5 strange vegetables that I am unlikely to see in Western countries.'''
# DEFAULT_SYSTEM_MESSAGE = "You are a domain expert AI for a graduate class. Be articulate, clear, and concise and your response."
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI."
MAX_CHARS_IN_PROMPT = 200000
MAX_CHAT_EXCHANGES = 20

CODE_SYSTEM_MESSAGE = "You are an expert programmer that helps to review Python code and provide optimizations."

# This is the template for the RAG prompt
RAG_CONTEXT_TEMPLATE = """Use the following context to answer the question:
<context>
{context}
</context>

Question: {question}

AI: """

# This is a basic summary template
SUMMARY_TEMPLATE = """Summarize the following text, retaining the main keywords:
<excerpt>
{excerpt}
</excerpt>"""


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
