# This file defines functions for API calls to different models
# TODO: Allow model hyperparameters to be passed as arguments to the functions
# TODO: Implement embedding model context size checks, potentially issues
# during vectorstore steps?

# TODO: Implement a YAML file to store model names and their corresponding values

from os import getenv, path
import logging
from typing import Callable

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
# from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage

from ..constants import LOCAL_MODELS, MODEL_CODES
from paths import MODELS_YAML_PATH

# FOR DEBUG OR PIPING OUTPUT
# NOTE: Is this usable in any use cases like asynchronously populating convex tables?
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import yaml
logger = logging.getLogger(__name__)


def get_model_config_from_yaml(filename: str):
    """Load and validate model configuration from YAML file."""
    with open(MODELS_YAML_PATH, 'r') as f:
        models_config = yaml.safe_load(f)
    
    required_keys = ['models', 'valid_together_models', 'valid_openai_models', 
                    'valid_deepseek_models', 'valid_openrouter_models',
                    'valid_ollama_models', 'valid_litellm_models']
    
    for key in required_keys:
        if key not in models_config:
            raise ValueError(f"{key} not found in models.yaml")
            
    return models_config

MODEL_CONFIG = get_model_config_from_yaml('models.yaml')

# Build provider-to-models mapping
ALL_MODELS = {
    "providers": {
        "openai": MODEL_CONFIG['valid_openai_models'],
        "together": MODEL_CONFIG['valid_together_models'], 
        "deepseek": MODEL_CONFIG['valid_deepseek_models'],
        "openrouter": MODEL_CONFIG['valid_openrouter_models'],
        "ollama": MODEL_CONFIG['valid_ollama_models'],
        "litellm": MODEL_CONFIG['valid_litellm_models']
    }
}

def get_model_dict():
    """Create a comprehensive model dictionary from valid models and model configs."""
    model_dict = {}
    
    # First populate from valid provider models
    for provider, models in ALL_MODELS["providers"].items():
        for model_name in models:
            model_dict[model_name] = {
                'provider': provider,
                'model_name': model_name,
                'context_size': 4096,  # Default context size
                'model_type': 'llm'    # Default type
            }
    
    # Then update/override with specific configurations from models list
    for model_config in MODEL_CONFIG['models']:
        model_name = model_config['model']
        key = model_config['key']
        
        # Only update if model name matches key
        if model_name in model_dict:
            model_dict[key] = {
                'provider': model_config['provider'],
                'model_name': model_name,
                'context_size': model_config.get('context_size', 4096),
                'model_type': model_config.get('type', 'llm')
            }
    return model_dict

MODEL_DICT = get_model_dict()

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
TOGETHER_BASE_URL = "https://api.together.xyz"
OPENAI_BASE_URL = "https://api.openai.com/v1"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
# This is for LMSTUDIO, but I set llamacpp to same port
LOCAL_BASE_URL = "http://localhost:1234/v1"
LLAMA_CPP_BASE_URL = LOCAL_BASE_URL # This can be changed to a different port


TOGETHER_BASE_URL = "https://api.together.xyz"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LITELLM_BASE_URL = "http://localhost:4000/v1"



VALID_TOGETHER_MODELS = ALL_MODELS["providers"]["together"]
VALID_OPENAI_MODELS = ALL_MODELS["providers"]["openai"]
VALID_DEEPSEEK_MODELS = ALL_MODELS["providers"]["deepseek"]
VALID_OPENROUTER_MODELS = ALL_MODELS["providers"]["openrouter"]
VALID_OLLAMA_MODELS = ALL_MODELS["providers"]["ollama"]
VALID_LITELLM_MODELS = ALL_MODELS["providers"]["litellm"]
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2000

def get_temp_and_tokens(hyperparameters: dict | None = None) -> tuple[int, int]:
    # TODO: This will later be converted to unpack_hyperparameters
    if hyperparameters:
        temperature = hyperparameters.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = hyperparameters.get("max_tokens", DEFAULT_MAX_TOKENS)
    else:
        temperature = DEFAULT_TEMPERATURE
        max_tokens = DEFAULT_MAX_TOKENS
    return int(temperature), int(max_tokens)

def get_api_key(provider: str):
    api_key = ""
    if provider == "openai":
        api_key = getenv("OPENAI_API_KEY")
    elif provider == "together":
        api_key = getenv("TOGETHER_API_KEY")
    elif provider == "deepseek":
        api_key = getenv("DEEPSEEK_API_KEY")
    elif provider == "openrouter":
        api_key = getenv("OPENROUTER_API_KEY")
    elif provider == "ollama":
        api_key = "LOCAL-API-KEY"
    elif provider == "local":
        api_key = "LOCAL-API-KEY"
    elif provider == "litellm":
        api_key = getenv("LITELLM_API_KEY")
    else:
        raise ValueError("Invalid provider")
    if not api_key:
        raise ValueError(f"Please set {provider.upper()}_API_KEY in .env file")
    return api_key

def get_base_url(provider: str):
    base_url = ""
    if provider == "together":
        base_url = TOGETHER_BASE_URL
    elif provider == "openai":
        base_url = OPENAI_BASE_URL
    elif provider == "deepseek":
        base_url = DEEPSEEK_BASE_URL
    elif provider == "openrouter":
        base_url = OPENROUTER_BASE_URL
    elif provider == "local":
        base_url = LOCAL_BASE_URL
    elif provider == "ollama":
        base_url = OLLAMA_BASE_URL
    elif provider == "litellm":
        base_url = LITELLM_BASE_URL
    else:
        raise ValueError("Invalid provider")
    assert base_url, "Base URL not set"
    return base_url

def validate_model_name(provider: str, model_name: str):
    if provider == "openai":
        assert model_name in VALID_OPENAI_MODELS, f"Invalid model name: {model_name}"
    elif provider == "together":
        assert model_name in VALID_TOGETHER_MODELS, f"Invalid model name: {model_name}"
    elif provider == "deepseek":
        assert model_name in VALID_DEEPSEEK_MODELS, f"Invalid model name: {model_name}"
    elif provider == "openrouter":
        assert model_name in VALID_OPENROUTER_MODELS, f"Invalid model name: {model_name}"
    elif provider == "ollama":
        assert model_name in VALID_OLLAMA_MODELS, f"Invalid model name: {model_name}"
    elif provider == "litellm":
        assert model_name in VALID_LITELLM_MODELS, f"Invalid model name: {model_name}"
    else:
        raise ValueError("Invalid provider")

def get_model_wrapper(provider: str, model_name: str, hyperparameters=None, validate: bool = True, callback_printer: bool = False):
    if validate:
        validate_model_name(provider, model_name)
    def wrapped_function(hyperparameters=hyperparameters):
        temperature, max_tokens = get_temp_and_tokens(hyperparameters)
        return ChatOpenAI(
            base_url=get_base_url(provider),
            api_key=get_api_key(provider),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()] if callback_printer else None
        )
    return wrapped_function

def get_together_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    provider = "together"
    wrapped_function = get_model_wrapper(provider, model_name, hyperparameters)
    return wrapped_function()

def get_openai_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    provider = "openai"
    wrapped_function = get_model_wrapper(provider, model_name, hyperparameters)
    return wrapped_function()

def get_deepseek_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    provider = "deepseek"
    wrapped_function = get_model_wrapper(provider, model_name, hyperparameters)
    return wrapped_function()

def get_openrouter_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    provider = "openrouter"
    wrapped_function = get_model_wrapper(provider, model_name, hyperparameters)
    return wrapped_function()

def get_local_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    # TODO: Add port as a parameter
    provider = "local"
    wrapped_function = get_model_wrapper(provider, model_name, hyperparameters)
    return wrapped_function()

def get_ollama_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    provider = "ollama"
    wrapped_function = get_model_wrapper(provider, model_name, hyperparameters)
    return wrapped_function()

def get_litellm_model(model_name: str, hyperparameters=None) -> ChatOpenAI:
    provider = "litellm"
    wrapped_function = get_model_wrapper(provider, model_name, hyperparameters)
    return wrapped_function()

def get_openai_embedder_large(hyperparameters=None) -> OpenAIEmbeddings:
    api_key = getenv("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in .env file"
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key
    )

def get_together_embedder_large(hyperparameters=None) -> TogetherEmbeddings:
    api_key = getenv("TOGETHER_API_KEY")
    assert api_key, "Please set TOGETHER_API_KEY in .env file"
    return TogetherEmbeddings(
        api_key=api_key,
        model="BAAI/bge-large-en-v1.5",
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
        api_key="LOCAL-API-KEY"
    )



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
    def __init__(self, model_experimental: str | None = None, model_fn: Callable[[], ChatOpenAI] | None = None, hyperparameters=None):
        if hyperparameters is not None and not isinstance(hyperparameters, dict):
            raise ValueError("Hyperparameters must be a dictionary")

        
        # Handle experimental model string
        if model_experimental is not None:
            if model_fn is not None:
                raise ValueError("Cannot specify both model_fn and model_experimental")
                
            if model_experimental in MODEL_CODES:
                model_experimental = MODEL_CODES[model_experimental]
                
            # Look up model by key or model name
            for key, info in MODEL_DICT.items():
                if key == model_experimental or info["model_name"] == model_experimental:
                    provider = info['provider']
                    model_name = info['model_name']
                    self.model_fn = get_model_wrapper(provider, model_name)
                    self.model_name = model_name
                    self.context_size = info["context_size"]
                    break
            else:
                raise ValueError(f"Model '{model_experimental}' not found in MODEL_DICT")
                
        # Handle model_fn (deprecated method)
        elif model_fn is not None:
            # Try to find matching model in MODEL_DICT
            for info in MODEL_DICT.values():
                provider = info['provider']
                model_name = info['model_name']
                wrapper = get_model_wrapper(provider, model_name)
                if model_fn == wrapper:
                    self.model_fn = model_fn
                    self.model_name = model_name
                    self.context_size = info["context_size"]
                    break
            else:
                raise ValueError("Provided model_fn does not match any model in MODEL_DICT")
        else:
            raise ValueError("Must specify either model_fn or model_experimental")
            
        self.hyperparameters = hyperparameters
        MODEL_NAMES = [model["model_name"] for model in MODEL_DICT.values()]
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
