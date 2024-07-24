# This file defines functions for API calls to different models
# TODO: Allow model hyperparameters to be passed as arguments to the functions
# TODO: Implement embedding model context size checks, potentially issues
# during vectorstore steps?

# TODO: Implement a YAML file to store model names and their corresponding values

from os import getenv
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
# from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage

from constants import LOCAL_MODELS, MODEL_CODES

# FOR DEBUG OR PIPING OUTPUT
# NOTE: Is this usable in any use cases like asynchronously populating convex tables?
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import yaml

# DEPRECATED
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms.ollama import Ollama

## NEW ##
def get_model_config_from_yaml(filename: str):
    import os
    this_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(this_directory, filename)
    with open(file_path, 'r') as f:
        models_config = yaml.safe_load(f)
    # Make assertions about the structure of the yaml file
    if 'models' not in models_config:
        raise ValueError("models key not found in models.yaml")
    return models_config

MODEL_CONFIG = get_model_config_from_yaml('models.yaml')

def get_model_dict():
    model_dict = {}
    for model in MODEL_CONFIG['models']:
        model_dict[model['key']] = {
            'provider': model['provider'],
            'model_name': model['model'],
            'context_size': model['context_size'],
            'model_type': model['type']
        }
    return model_dict

MODEL_DICT = get_model_dict()

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
TOGETHER_BASE_URL = "https://api.together.xyz"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
# This is for LMSTUDIO, but I set llamacpp to same port
LOCAL_BASE_URL = "http://localhost:1234/v1"
logger = logging.getLogger(__name__)

TOGETHER_BASE_URL = "https://api.together.xyz"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ALL_MODELS = {
    "providers": {
        "openai": MODEL_CONFIG['valid_openai_models'],
        "together": MODEL_CONFIG['valid_together_models'],
        "deepseek": MODEL_CONFIG['valid_deepseek_models'],
        "local": MODEL_CONFIG['valid_local_models'],
        "ollama": MODEL_CONFIG['valid_ollama_models']
    }
}

VALID_TOGETHER_MODELS = ALL_MODELS["providers"]["together"]
VALID_OPENAI_MODELS = ALL_MODELS["providers"]["openai"]
VALID_DEEPSEEK_MODELS = ALL_MODELS["providers"]["deepseek"]
VALID_LOCAL_MODELS = ALL_MODELS["providers"]["local"]
VALID_OLLAMA_MODELS = ALL_MODELS["providers"]["ollama"]

DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2000

def get_temp_and_tokens(hyperparameters: dict | None = None):
    if hyperparameters:
        temperature = hyperparameters.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = hyperparameters.get("max_tokens", DEFAULT_MAX_TOKENS)
    else:
        temperature = DEFAULT_TEMPERATURE
        max_tokens = DEFAULT_MAX_TOKENS
    return temperature, max_tokens

def get_together_wrapper(model_name: str, hyperparameters=None):
    assert model_name in VALID_TOGETHER_MODELS, f"Invalid model name: {model_name}"
    
    def wrapped_function(hyperparameters=hyperparameters):
        api_key = getenv("TOGETHER_API_KEY")
        assert api_key, "Please set TOGETHER_API_KEY in .env file"
        temperature, max_tokens = get_temp_and_tokens(hyperparameters)
        
        return ChatOpenAI(
            base_url=TOGETHER_BASE_URL,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    return wrapped_function

def get_together_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    wrapped_function = get_together_wrapper(model_name, hyperparameters)
    return wrapped_function()

def get_openai_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    assert model_name in VALID_OPENAI_MODELS
    api_key = getenv("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in .env file"
    temperature, max_tokens = get_temp_and_tokens(hyperparameters)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
        

def get_openai_gpt4(hyperparameters=None) -> ChatOpenAI:
    model="gpt-4o"
    return get_openai_model(model, hyperparameters)

def get_openai_gpt3(hyperparameters=None) -> ChatOpenAI:
    model="gpt-3.5-turbo"
    return get_openai_model(model, hyperparameters)

def get_together_dolphin(hyperparameters=None) -> ChatOpenAI:
    model="cognitivecomputations/dolphin-2.5-mixtral-8x7b"
    return get_together_model(model, hyperparameters)

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
        max_tokens=2000,
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

def get_claude_sonnet(hyperparameters=None) -> ChatOpenAI:
    api_key = getenv("OPENROUTER_API_KEY")
    assert api_key, "Please set OPENROUTER_API_KEY in .env file"
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        model="anthropic/claude-3.5-sonnet",
        max_tokens=1000,
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

# def get_together_embedder_large(hyperparameters=None) -> OpenAIEmbeddings:
#     api_key = getenv("TOGETHER_API_KEY")
#     assert api_key, "Please set TOGETHER_API_KEY in .env file"
#     return OpenAIEmbeddings(
#         base_url=TOGETHER_BASE_URL,
#         api_key=api_key,
#         model="BAAI/bge-large-en-v1.5",
#     )


def get_together_embedder_large(hyperparameters=None) -> TogetherEmbeddings:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return TogetherEmbeddings(
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
        model="local-model",
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

# This maps the model keys to the functions
FUNCTION_MAP = {
    "get_openai_gpt4": get_openai_gpt4,
    "get_openai_gpt3": get_openai_gpt3,
    "get_together_dolphin": get_together_dolphin,
    "get_together_qwen": get_together_qwen,
    "get_together_nous_mix": get_together_nous_mix,
    "get_together_fn_mix": get_together_fn_mix,
    "get_together_bigmix": get_together_bigmix,
    "get_together_dbrx": get_together_dbrx,
    "get_together_arctic": get_together_arctic,
    "get_together_llama3": get_together_llama3,
    "get_together_new_llama": get_together_wrapper("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    "get_together_llama_400b": get_together_wrapper("meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
    "get_together_deepseek_4k": get_together_deepseek_4k,
    "get_together_deepseek_32k": get_together_deepseek_32k,
    "get_deepseek_coder": get_deepseek_coder,
    "get_claude_sonnet": get_claude_sonnet,
    "get_local_model": get_local_model,
    "get_ollama_llama3": get_ollama_llama3,
    "get_ollama_mistral": get_ollama_mistral,
    "get_local_hermes": get_local_hermes,
    "get_openai_embedder_large": get_openai_embedder_large,
    "get_together_embedder_large": get_together_embedder_large,
    "get_ollama_local_embedder": get_ollama_local_embedder,
    "get_lmstudio_local_embedder": get_lmstudio_local_embedder
}
MODEL_NAMES = [model["model_name"] for model in MODEL_DICT.values()]

def model_key_from_name(model_name: str) -> str:
    """Get the model key from the model name"""
    for key, model in MODEL_DICT.items():
        if model["model_name"] == model_name:
            return key
    return None


def model_name_from_key(model_key: str) -> str | None:
    """Get the model name from the model key"""
    model_name = MODEL_DICT.get(model_key, {}).get("model_name", None)
    return model_name


# Create class LLM_FN that takes a function that is a value in MODEL_DICT
class LLM_FN:
    def __init__(self, model_fn=None, hyperparameters=None, model_experimental: str | None = None):
        # If it's not a value in MODEL_DICT, raise an error
        # This means embedding models pass here (for now)
        if hyperparameters is not None and not isinstance(hyperparameters, dict):
            raise ValueError("Hyperparameters must be a dictionary")
        
        if model_experimental is not None:
            assert model_fn is None, "model_fn must be None if model_experimental is not None"
            if model_experimental in MODEL_CODES:
                model_experimental = MODEL_CODES[model_experimental]
        for key, info in MODEL_DICT.items():
            if model_fn is not None:
                if model_fn == FUNCTION_MAP[key]:
                    self.model_name = str(info["model_name"])
                    self.context_size = int(info["context_size"])
                    self.model_fn = model_fn
                    break
            else:
                if model_experimental is not None:
                    if key == model_experimental:
                        self.model_fn = FUNCTION_MAP[key]
                        self.model_name = str(info["model_name"])
                        self.context_size = int(info["context_size"])
                        break
                    if info["model_name"] == model_experimental:
                        self.model_fn = FUNCTION_MAP[key]
                        self.model_name = model_experimental
                        self.context_size = int(info["context_size"])
                        break
        else:
            raise ValueError("Model not found in MODEL_DICT")
        
        
        
        self.hyperparameters = hyperparameters
        assert self.model_fn in FUNCTION_MAP.values()
        assert self.model_name in MODEL_NAMES
        assert self.context_size > 0

    def get_llm(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return self.model_fn(hyperparameters)

    def __str__(self):
        return f"LLM: model_name={self.model_name}, context_size={self.context_size}"

    def __repr__(self):
        return f"LLM(model_name={self.model_name}, context_size={self.context_size})"


class LLM:
    def __init__(self, llm_fn: LLM_FN, hyperparameters=None):
        if isinstance(llm_fn, LLM):
            raise ValueError("LLM object passed to LLM constructor")
        assert isinstance(llm_fn, LLM_FN), "llm_fn must be an LLM_FN object"
        # TODO: Filter out embedding models

        self.model_name = llm_fn.model_name
        self.context_size = llm_fn.context_size
        # replace the hyperparameters with the new ones
        self.llm = llm_fn.get_llm(hyperparameters)

        if not self.confirm_model_name():
            logger.error(
                "Critical error. Model name failed in LLM.confirm_model_name. Exiting.")
            raise SystemExit
        self.is_local = self.model_name in LOCAL_MODELS
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

    def invoke(self, query: str | list, tools=None,
               tool_choice=None) -> list[BaseMessage]:
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
