import logging
from agents.utils.message_utils import get_last_user_message
from augmenta.chains import SimpleChain
from augmenta.constants import get_summary_template
from augmenta.models.models import LLM, LLM_FN
from langchain_core.output_parsers import StrOutputParser#, JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

def get_llm(model_name: str) -> LLM:
    try:
        llm = LLM(LLM_FN(model_name))
        return llm
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        return None

def get_summary_chain(model_name: str, system_prompt: str = 'You are a helpful AI.') -> SimpleChain | None:
    """
    Returns a chain for summarization only.
    Can be invoked, like `chain.invoke({"excerpt": "Excerpt of long reading:..."})` to get a response.
    """
    try:
        llm = get_llm(model_name)
        if llm is None:
            raise ValueError("LLM not initialized!")
        
        chain = SimpleChain(
            {"excerpt": lambda x: get_last_user_message(x)}
            | get_summary_template(system_prompt)
            | llm.llm
            # | StrOutputParser()
        )
        return chain
    except Exception as e:
        logging.error(f"Error creating summary chain: {e}")
        return None