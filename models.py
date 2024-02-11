from dotenv import load_dotenv
load_dotenv()
from os import getenv
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
TOGETHER_API_KEY = getenv("TOGETHER_API_KEY")
assert TOGETHER_API_KEY is not None
def get_together_quen():
    return ChatOpenAI(
        base_url = "https://api.together.xyz",
        api_key = TOGETHER_API_KEY,
        model = "Qwen/Qwen1.5-72B-Chat",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_together_mix():
    return ChatOpenAI(
        base_url = "https://api.together.xyz",
        api_key = TOGETHER_API_KEY,
        model = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_beagle():
    return ChatOpenAI(
        base_url = "http://localhost:1234/v1",
        api_key='None',
        model = "local-beagle",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def get_mistral():
    return ChatOpenAI(
        base_url = "http://localhost:1234/v1",
        api_key='None',
        model = "local-mistral",
        temperature=0.1,
        max_tokens=1000,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
