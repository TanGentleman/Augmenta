# This file defines functions for API calls to different models
# TODO: Allow model hyperparameters to be passed as arguments to the functions
# TODO: Implement embedding model context size checks, potentially issues
# during vectorstore steps?

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
        model="gpt-4-0125-preview",
        api_key=OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_together_deepseek_4k(hyperparameters=None) -> ChatOpenAI:
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url="https://api.together.xyz",
        api_key=TOGETHER_API_KEY,
        model="deepseek-ai/deepseek-llm-67b-chat",
        temperature=0,
        max_tokens=2000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_claude_sonnet(hyperparameters=None) -> ChatAnthropic:
    return ChatAnthropic(
        model_name="claude-3-sonnet-20240229",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_claude_opus(hyperparameters=None) -> ChatAnthropic:
    assert ANTHROPIC_API_KEY, "Please set ANTHROPIC_API_KEY in .env file"
    return ChatAnthropic(
        temperature=0,
        anthropic_api_key=ANTHROPIC_API_KEY,
        model_name="claude-3-opus-20240229",
        max_tokens_to_sample=4000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_nomic_local_embedder(hyperparameters=None) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )

MODEL_DICT = {
    "get_openai_gpt4": get_openai_gpt4,
    "get_together_dolphin": get_together_dolphin,
    "get_together_quen": get_together_quen,
    "get_together_nous_mix": get_together_nous_mix,
    "get_together_fn_mix": get_together_fn_mix,
    "get_together_deepseek_4k": get_together_deepseek_4k,
    "get_together_deepseek_32k": get_together_deepseek_32k,
    "get_claude_sonnet": get_claude_sonnet,
    "get_claude_opus": get_claude_opus,
    "get_local_model": get_local_model,
    "get_openai_embedder_large": get_openai_embedder_large,
    "get_together_embedder_large": get_together_embedder_large,
    "get_nomic_local_embedder": get_nomic_local_embedder
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
        # This technically means embedding models would pass here, but fine for now
        if model_fn not in MODEL_DICT.values():
            raise ValueError(f"Model function {model_fn} not found in MODEL_DICT")
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
    

class LLM:
    def __init__(self, llm_fn: LLM_FN, hyperparameters=None):
        assert llm_fn.model_fn  in MODEL_DICT.values(), "Model function not found in MODEL_DICT"
        # I will split these into LLM_DICT and EMBEDDING_DICT to filter out embedding models
        if hyperparameters is not None:
            # replace the hyperparameters with the new ones
            self.llm = llm_fn.get_llm(hyperparameters)
        else:
            self.llm = llm_fn.get_llm()
        
        self.model_name = self.get_model_name()
    
    def get_model_name(self) -> str:
        name = 'Unknown'
        if hasattr(self.llm, 'model_name'):
            name = self.llm.model_name
        elif hasattr(self.llm, 'model'):
            name = self.llm.model
        assert isinstance(name, str), "Model name must be a string"
        return name
    pass

    def invoke(self, query):
        # This will break embedding models if they don't have an invoke method
        return self.llm.invoke(query)


class Embedder(OpenAIEmbeddings, TogetherEmbeddings, OllamaEmbeddings):
    pass
