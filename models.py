# This file defines functions for API calls to different models

from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

from os import getenv
TOGETHER_API_KEY = getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
assert TOGETHER_API_KEY and ANTHROPIC_API_KEY, "Please set API keys in .env file"

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import OpenAIEmbeddings

# This model is the flagship for improved instruction following,
# JSON mode, reproducible outputs, parallel function calling (training data up to Dec 2023)
# 128K model context size
def get_openai_gpt4():
    assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in .env file"
    return ChatOpenAI(
        model="gpt-4-0125-preview",
        api_key=OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_quen():
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url = "https://api.together.xyz",
        api_key = TOGETHER_API_KEY,
        model = "Qwen/Qwen1.5-72B-Chat",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_nous_mix():
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url = "https://api.together.xyz",
        api_key = TOGETHER_API_KEY,
        model = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
def get_together_fn_mix():
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url = "https://api.together.xyz",
        api_key = TOGETHER_API_KEY,
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_fn_mistral():
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url = "https://api.together.xyz",
        api_key = TOGETHER_API_KEY,
        model = "mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_coder():
    assert TOGETHER_API_KEY, "Please set TOGETHER_API_KEY in .env file"
    return ChatOpenAI(
        base_url = "https://api.together.xyz",
        api_key = TOGETHER_API_KEY,
        model = "deepseek-ai/deepseek-coder-33b-instruct",
        temperature=0,
        max_tokens=2000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_claude_opus():
    assert ANTHROPIC_API_KEY, "Please set ANTHROPIC_API_KEY in .env file"
    return ChatAnthropic(
        temperature=0, 
        anthropic_api_key=ANTHROPIC_API_KEY, 
        model_name="claude-3-opus-20240229",
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_openai_embedder_large():
    assert OPENAI_API_KEY, "Please set OPENAI_API_KEY in .env file"
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY
    )

def get_local_model():
    return ChatOpenAI(
        base_url = "http://localhost:1234/v1",
        api_key='None',
        model = "local-model",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )