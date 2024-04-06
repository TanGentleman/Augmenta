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

def get_openai_gpt4() -> ChatOpenAI:
    assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in .env file"
    return ChatOpenAI(
        model="gpt-4-0125-preview",
        api_key=OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_dolphin() -> ChatOpenAI:
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

def get_together_quen() -> ChatOpenAI:
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


def get_together_nous_mix() -> ChatOpenAI:
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


def get_together_fn_mix() -> ChatOpenAI:
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


def get_together_fn_mistral() -> ChatOpenAI:
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


def get_together_coder() -> ChatOpenAI:
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


def get_claude_sonnet() -> ChatAnthropic:
    return ChatAnthropic(
        model_name="claude-3-sonnet-20240229",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_claude_opus() -> ChatAnthropic:
    assert ANTHROPIC_API_KEY, "Please set ANTHROPIC_API_KEY in .env file"
    return ChatAnthropic(
        temperature=0,
        anthropic_api_key=ANTHROPIC_API_KEY,
        model_name="claude-3-opus-20240229",
        max_tokens_to_sample=4000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_openai_embedder_large() -> OpenAIEmbeddings:
    assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in .env file"
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY
    )


def get_together_embedder_large() -> TogetherEmbeddings:
    # assert False, "This model is not available yet. Please use get_openai_embedder_large() instead."
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return TogetherEmbeddings(
        model="BAAI/bge-large-en-v1.5",
        together_api_key=TOGETHER_API_KEY
    )


def get_local_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key='lm-studio',
        model="local-model",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )


def get_nomic_local_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


class LLM(ChatOpenAI, ChatAnthropic):
    pass


class Embedder(OpenAIEmbeddings, TogetherEmbeddings, OllamaEmbeddings):
    pass


MODEL_DICT = {
    "get_openai_gpt4": get_openai_gpt4,
    "get_together_dolphin": get_together_dolphin,
    "get_together_quen": get_together_quen,
    "get_together_nous_mix": get_together_nous_mix,
    "get_together_fn_mix": get_together_fn_mix,
    "get_together_coder": get_together_coder,
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
