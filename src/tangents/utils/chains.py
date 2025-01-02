import logging
import os
from typing import Any

# from augmenta.models.models import LLM, LLM_FN
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

# from augmenta.chains import SimpleChain
from augmenta.constants import get_summary_template
from tangents.utils.message_utils import get_last_user_message

ALLOWED_MODELS = ['free', 'Llama-3.3-70B']
LITELLM_BASE_URL = 'http://localhost:4000/v1'


# TODO: Support callbacks
def fast_get_llm(model_name: str, config: dict = {}) -> ChatOpenAI | None:
    """Uses LiteLLM endpoint."""
    if model_name not in ALLOWED_MODELS:
        logging.warning(f'Model {model_name} not in allowed models list')
    try:
        # TODO: Unpack config
        # This is for callbacks, hyperparams, etc.
        llm = ChatOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=os.getenv('LITELLM_API_KEY'),
            model=model_name,
        )
        return llm
    except Exception as e:
        logging.error(f'Error initializing LLM: {e}')
        return None


# def get_llm(model_name: str) -> ChatOpenAI | None:
#     try:
#         llm = LLM(LLM_FN(model_name))
#         return llm.llm
#     except Exception as e:
#         logging.error(f"Error initializing LLM: {e}")
#         return None


def get_summary_chain(
    model_name: str, system_prompt: str = 'You are a helpful AI.'
) -> RunnableSerializable[Any, str] | None:
    """
    Returns a chain for summarization only.

    Expects a list of messages. (Or a runnableLambda that returns a list of messages.)
    Can be invoked, like `chain.invoke(messages)` to get a response.
    """
    try:
        llm = fast_get_llm(model_name)
        if llm is None:
            raise ValueError('LLM not initialized!')

        chain = (
            {'excerpt': lambda x: get_last_user_message(x)}
            | get_summary_template(system_prompt)
            | llm
            | StrOutputParser()
        )
        # chain = SimpleChain(raw_chain, description="Summary Chain")
        return chain
    except Exception as e:
        logging.error(f'Error creating summary chain: {e}')
        return None
