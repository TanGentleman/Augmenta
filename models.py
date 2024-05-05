# This file defines functions for API calls to different models
# TODO: Allow model hyperparameters to be passed as arguments to the functions
# TODO: Implement embedding model context size checks, potentially issues
# during vectorstore steps?

from typing import Union
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from os import getenv
from os.path import join, dirname
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

TOGETHER_API_KEY = getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
assert TOGETHER_API_KEY and ANTHROPIC_API_KEY, "Please set API keys in .env file"


# This model is the flagship for improved instruction following,
# JSON mode, reproducible outputs, parallel function calling (training data up to Dec 2023)
# 128K model context size

def get_openai_gpt4(hyperparameters=None) -> ChatOpenAI:
    assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in .env file"
    return ChatOpenAI(
        model="gpt-4-turbo-2024-04-09",
        api_key=OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_dolphin(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
        temperature=0,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_quen(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="Qwen/Qwen1.5-72B-Chat",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_nous_mix(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_bigmix(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_dbrx(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="databricks/dbrx-instruct",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_arctic(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="Snowflake/snowflake-arctic-instruct",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_llama3(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="meta-llama/Llama-3-70b-chat-hf",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_fn_mix(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_fn_mistral(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_deepseek_4k(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="deepseek-ai/deepseek-llm-67b-chat",
        temperature=0,
        max_tokens=800,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_deepseek_32k(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="deepseek-ai/deepseek-coder-33b-instruct",
        temperature=0,
        max_tokens=2000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_claude_sonnet(hyperparameters=None) -> ChatAnthropic:
    return ChatAnthropic(
        model_name="claude-3-sonnet-20240229",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_claude_opus(hyperparameters=None) -> ChatAnthropic:
    assert ANTHROPIC_API_KEY, "Please set ANTHROPIC_API_KEY in .env file"
    return ChatAnthropic(
        temperature=0,
        anthropic_api_key=ANTHROPIC_API_KEY,
        model_name="claude-3-opus-20240229",
        max_tokens_to_sample=4000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_openai_embedder_large(hyperparameters=None) -> OpenAIEmbeddings:
    assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in .env file"
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY
    )


def get_together_embedder_large(hyperparameters=None) -> TogetherEmbeddings:
    # assert False, "This model is not available yet. Please use get_openai_embedder_large() instead."
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return TogetherEmbeddings(
        model="BAAI/bge-large-en-v1.5",
        together_api_key=TOGETHER_API_KEY
    )


def get_local_model(hyperparameters=None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key='lm-studio',
        model="local-model",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
def get_ollama_local_model(hyperparameters=None) -> Ollama:
    return Ollama(
        model="llama3",
    )


def get_nomic_local_embedder(hyperparameters=None) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )

def get_lmstudio_local_embedder(hyperparameters=None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",
        model="local-embedding-model",
        api_key="lm-studio"
    )

MODEL_DICT = {
    "get_openai_gpt4": {
        "function": get_openai_gpt4,
        "context_size": 128000,
        "model_name": "gpt-4-turbo-2024-04-09",
        "model_type": "llm"
    },
    "get_together_dolphin": {
        "function": get_together_dolphin,
        "context_size": 32768,
        "model_name": "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
        "model_type": "llm"
    },
    "get_together_quen": {
        "function": get_together_quen,
        "context_size": 4096,
        "model_name": "Qwen/Qwen1.5-72B-Chat",
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
        "context_size": 4096,
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
    "get_claude_sonnet": {
        "function": get_claude_sonnet,
        "context_size": 200000,
        "model_name": "claude-3-sonnet-20240229",
        "model_type": "llm"
    },
    "get_claude_opus": {
        "function": get_claude_opus,
        "context_size": 200000,
        "model_name": "claude-3-opus-20240229",
        "model_type": "llm"
    },
    # Note: Local model has an undefined context size
    "get_local_model": {
        "function": get_local_model,
        "context_size": 32768,
        "model_name": "local-model",
        "model_type": "llm"
    },
    "get_ollama_local_model": {
        "function": get_ollama_local_model,
        "context_size": 4096,
        "model_name": "local-ollama3", # Should this be llama3?
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
    "get_nomic_local_embedder": {
        "function": get_nomic_local_embedder,
        "context_size": 8192,
        "model_name": "nomic-embed-text",
        "model_type": "embedder"
    },
    "get_lmstudio_local_embedder": {
        "function": get_lmstudio_local_embedder,
        "context_size": 8192,
        "model_name": "local-embedding-model",
        "model_type": "embedder"
    }
}

EMBEDDING_CONTEXT_SIZE_DICT = {
    "get_openai_embedder_large": 128000,
    "get_together_embedder_large": 8192,
    "get_nomic_local_embedder": 8192
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

        self.confirm_model_name()

    def confirm_model_name(self) -> str:
        """
        Get the name of the model
        """
        if hasattr(self.llm, 'model_name'):
            model_name = self.llm.model_name
        elif hasattr(self.llm, 'model'):
            model_name = self.llm.model
        else:
            raise ValueError("Model name not found in model object")
        
        if model_name == "llama3":
            model_name = "local-ollama3"
        if model_name != self.model_name:
            raise ValueError(
                f"Model name from API: {model_name} does not match expected model name: {self.model_name}")
        return model_name
    pass

    def invoke(self, query):
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


Embedder = Union[OpenAIEmbeddings, TogetherEmbeddings, OllamaEmbeddings]
