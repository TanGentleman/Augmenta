# Constants for chat.py
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage

DEFAULT_QUERY = '''Name 5 strange vegetables that I am unlikely to see in Western countries.'''
# DEFAULT_SYSTEM_MESSAGE = "You are a domain expert AI for a graduate class. Be articulate, clear, and concise and your response."
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI."
MAX_CHARS_IN_PROMPT = 50000
MAX_CHAT_EXCHANGES = 20

CODE_SYSTEM_MESSAGE = "You are an expert programmer that helps to review Python code and provide optimizations."

EXPLANATION_TEMPLATE = '''Explain the following text using comprehensive bulletpoints:
"""
{excerpt}
"""
'''

def get_rag_template():
    """
    Fetches the RAG template for the prompt.
    This template expects to be passed values for both context and question.
    """
    template = """Answer the question based only on the following context:
    <context>
    {context}
    </context>

    Question: {question}
    """
    rag_prompt_template = ChatPromptTemplate.from_template(template)
    rag_prompt_template.messages.insert(0, 
        SystemMessage(
            content="You are a helpful AI. Use the document excerpts to respond to the best of your ability."
        )
    )
    return rag_prompt_template

RAG_TEMPLATE = get_rag_template()