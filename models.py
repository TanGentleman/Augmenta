# This file defines functions for API calls to different models
# TODO: Allow model hyperparameters to be passed as arguments to the functions
# TODO: Implement embedding model context size checks, potentially issues
# during vectorstore steps?

### NOTE: IF DOTENV NOT LOADED, API KEYS WILL NEED TO BE SET MANUALLY
# from dotenv import load_dotenv
# load_dotenv()

from os import getenv
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage

### FOR DEBUG
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.callbacks.manager import CallbackManager

### DEPRECATED
# from langchain_anthropic import ChatAnthropic
# from langchain_together import TogetherEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms.ollama import Ollama

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
TOGETHER_BASE_URL = "https://api.together.xyz"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
LOCAL_BASE_URL = "http://localhost:1234/v1" # This is for LMSTUDIO, but I set llamacpp to same port
logger = logging.getLogger(__name__)

TOGETHER_BASE_URL = "https://api.together.xyz"


def get_openai_gpt4(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in .env file"
    return ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_dolphin(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_qwen(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="Qwen/Qwen2-72B-Instruct",
        temperature=0,
        max_tokens=2000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_nous_mix(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_bigmix(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        temperature=0,
        max_tokens=4000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_dbrx(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="databricks/dbrx-instruct",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_arctic(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="Snowflake/snowflake-arctic-instruct",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_llama3(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="meta-llama/Llama-3-70b-chat-hf",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_fn_mix(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_fn_mistral(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_deepseek_4k(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="deepseek-ai/deepseek-llm-67b-chat",
        temperature=0,
        max_tokens=800,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_deepseek_32k(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="deepseek-ai/deepseek-coder-33b-instruct",
        temperature=0,
        max_tokens=2000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_openai_embedder_large(hyperparameters=None) -> OpenAIEmbeddings:
    api_key = getenv("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in .env file"
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key
    )


def get_together_embedder_large(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        model="BAAI/bge-large-en-v1.5",
    )

def get_deepseek_coder(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("DEEPSEEK_API_KEY")
    assert api_key, "Please set DEEPSEEK_API_KEY in .env file"
    return ChatOpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        model="deepseek-coder",
        max_tokens=3000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_local_model(hyperparameters=None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LOCAL_BASE_URL,
        api_key='lm-studio',
        model="lmstudio-model",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_ollama_llama3(hyperparameters=None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key='LOCAL-API-KEY',
        model="llama3",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_ollama_mistral(hyperparameters=None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key='LOCAL-API-KEY',
        model="mistral:7b-instruct-v0.3-q6_K",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_local_hermes(hyperparameters=None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key='LOCAL-API-KEY',
        model="local-hermes",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_ollama_local_embedder(hyperparameters=None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text",
        api_key="LOCAL-API-KEY"
    )


def get_lmstudio_local_embedder(hyperparameters=None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",
        model="lmstudio-embedding-model",
        api_key="lm-studio"
    )


MODEL_DICT = {
    "get_openai_gpt4": {
        "function": get_openai_gpt4,
        "context_size": 128000,
        "model_name": "gpt-4o",
        "model_type": "llm"
    },
    "get_together_dolphin": {
        "function": get_together_dolphin,
        "context_size": 32768,
        "model_name": "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
        "model_type": "llm"
    },
    "get_together_qwen": {
        "function": get_together_qwen,
        "context_size": 4096,
        "model_name": "Qwen/Qwen2-72B-Instruct",
        "model_type": "llm"
    },
    "get_together_nous_mix": {
        "function": get_together_nous_mix,
        "context_size": 32768,
        "model_name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "model_type": "llm"
    },
    "get_together_fn_mix": {
        "function": get_together_fn_mix,
        "context_size": 32768,
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "model_type": "llm"
    },
    "get_together_bigmix": {
        "function": get_together_bigmix,
        "context_size": 65536,
        "model_name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "model_type": "llm"
    },
    "get_together_dbrx": {
        "function": get_together_dbrx,
        "context_size": 32768,
        "model_name": "databricks/dbrx-instruct",
        "model_type": "llm"
    },
    "get_together_arctic": {
        "function": get_together_arctic,
        "context_size": 4096,
        "model_name": "Snowflake/snowflake-arctic-instruct",
        "model_type": "llm"
    },
    "get_together_llama3": {
        "function": get_together_llama3,
        "context_size": 8000,
        "model_name": "meta-llama/Llama-3-70b-chat-hf",
        "model_type": "llm"
    },
    "get_together_deepseek_4k": {
        "function": get_together_deepseek_4k,
        "context_size": 4096,
        "model_name": "deepseek-ai/deepseek-llm-67b-chat",
        "model_type": "llm"
    },
    "get_together_deepseek_32k": {
        "function": get_together_deepseek_32k,
        "context_size": 32768,
        "model_name": "deepseek-ai/deepseek-coder-33b-instruct",
        "model_type": "llm"
    },
    "get_deepseek_coder": {
        "function": get_deepseek_coder,
        "context_size": 128000,
        "model_name": "deepseek-coder",
        "model_type": "llm"
    },
    # Note: Local model has an undefined context size
    "get_local_model": {
        "function": get_local_model,
        "context_size": 32768,
        "model_name": "lmstudio-model",
        "model_type": "llm"
    },
    "get_ollama_llama3": {
        "function": get_ollama_llama3,
        "context_size": 4096,
        "model_name": "local-ollama3",  # Should this be llama3?
        "model_type": "llm"
    },
    "get_ollama_mistral": {
        "function": get_ollama_mistral,
        "context_size": 4096,
        "model_name": "mistral:7b-instruct-v0.3-q6_K",
        "model_type": "llm"
    },
    "get_local_hermes": {
        "function": get_local_hermes,
        "context_size": 4096,
        "model_name": "local-hermes",
        "model_type": "llm"
    },
    "get_openai_embedder_large": {
        "function": get_openai_embedder_large,
        "context_size": 128000,
        "model_name": "text-embedding-3-large",
        "model_type": "embedder"
    },
    "get_together_embedder_large": {
        "function": get_together_embedder_large,
        "context_size": 8192,
        "model_name": "BAAI/bge-large-en-v1.5",
        "model_type": "embedder"
    },
    "get_ollama_local_embedder": {
        "function": get_ollama_local_embedder,
        "context_size": 8192,
        "model_name": "nomic-embed-text",
        "model_type": "embedder"
    },
    "get_lmstudio_local_embedder": {
        "function": get_lmstudio_local_embedder,
        "context_size": 8192,
        "model_name": "lmstudio-embedding-model",
        "model_type": "embedder"
    }
}

EMBEDDING_CONTEXT_SIZE_DICT = {
    "get_openai_embedder_large": 128000,
    "get_together_embedder_large": 8192,
    "get_ollama_local_embedder": 8192
}


# Create class LLM_FN that takes a function that is a value in MODEL_DICT
class LLM_FN:
    def __init__(self, model_fn, hyperparameters=None):
        # If it's not a value in MODEL_DICT, raise an error
        # This means embedding models pass here (for now)
        self.model_name = ""
        self.context_size = 0
        for model in MODEL_DICT.values():
            if model["function"] == model_fn:
                self.model_name = str(model["model_name"])
                self.context_size = int(model["context_size"])

                if not self.model_name:
                    raise ValueError("Model name not found")
                if self.context_size <= 0:
                    raise ValueError(
                        "Context size must be a positive integer")
                break
        else:
            raise ValueError("Model function not found in MODEL_DICT")
        self.model_fn = model_fn
        self.hyperparameters = None
        if hyperparameters is not None:
            # assert isinstance(hyperparameters, dict), "This can be any check to make sure hyperparams are valid"
            self.hyperparameters = hyperparameters

    def get_llm(self, hyperparameters=None):
        if hyperparameters is not None:
            return self.model_fn(hyperparameters)
        else:
            return self.model_fn(self.hyperparameters)

    def __str__(self):
        return f"LLM: model_name={self.model_name}, context_size={self.context_size}"

    def __repr__(self):
        return f"LLM(model_name={self.model_name}, context_size={self.context_size})"


class LLM:
    def __init__(self, llm_fn: LLM_FN, hyperparameters=None):
        if isinstance(llm_fn, LLM):
            raise ValueError("LLM object passed to LLM constructor")
        assert isinstance(llm_fn, LLM_FN), "llm_fn must be an LLM_FN object"
        found = False
        for model in MODEL_DICT.values():
            if model["function"] == llm_fn.model_fn:
                assert model["model_name"] == llm_fn.model_name, "Model name does not match"
                found = True
                break
        assert found, "Model function not found in MODEL_DICT"
        # TODO: Filter out embedding models

        self.model_name = llm_fn.model_name
        self.context_size = llm_fn.context_size
        # replace the hyperparameters with the new ones
        self.llm = llm_fn.get_llm(hyperparameters)

        if not self.confirm_model_name():
            logger.error(
                "Critical error. Model name failed in LLM.confirm_model_name. Exiting.")
            raise SystemExit
        self.is_ollama = self.is_ollama_model()

    def confirm_model_name(self) -> bool:
        """
        Raise an error if the model name from the API does not match the MODEL_DICT
        """
        model_name = getattr(
            self.llm, 'model_name', getattr(
                self.llm, 'model', None))
        if model_name is None:
            logger.error("Model name not found in model object")
            return False

        # override-llama3 name
        if model_name == "llama3":
            logger.info("Model name override: llama3 -> local-ollama3")
            model_name = "local-ollama3"

        if model_name != self.model_name:
            logger.error(
                f"Model name from API: {model_name} does not match expected model name: {self.model_name}")
            return False
        return True
    pass

    def is_ollama_model(self) -> bool:
        # Ollama refactoring is now deprecated
        # return self.model_name in ["mistral:7b-instruct-v0.3-q6_K"]
        return False

    def invoke(self, query: str | list, tools=None, tool_choice=None) -> list[BaseMessage]:
        """
        Generate a response from the model
        """
        # This will break embedding models if they don't have an invoke method
        return self.llm.invoke(query)

    def stream(self, query):
        """
        Generate a response from the model
        """
        return self.llm.stream(query)

    def __str__(self):
        return f"LLM: model_name={self.model_name}, context_size={self.context_size}"

    def __repr__(self):
        return f"LLM(model_name={self.model_name}, context_size={self.context_size})"

### DEPRECATED
# def get_claude_sonnet(hyperparameters=None) -> ChatAnthropic:
#     api_key = getenv("ANTHROPIC_API_KEY")
#     assert api_key, "Please set ANTHROPIC_API_KEY in .env file"
#     return ChatAnthropic(
#         model_name="claude-3-sonnet-20240229",
#         anthropic_api_key=api_key,
#         temperature=0,
#         streaming=True,
#         # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#     )


# def get_claude_opus(hyperparameters=None) -> ChatAnthropic:
#     api_key = getenv("ANTHROPIC_API_KEY")
#     assert api_key, "Please set ANTHROPIC_API_KEY in .env file"
#     return ChatAnthropic(
#         temperature=0,
#         anthropic_api_key=api_key,
#         model_name="claude-3-opus-20240229",
#         max_tokens_to_sample=4000,
#         streaming=True,
#         # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#     )