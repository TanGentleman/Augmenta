# This file defines functions for API calls to different models
# TODO: Allow model hyperparameters to be passed as arguments to the functions
# TODO: Implement embedding model context size checks, potentially issues
# during vectorstore steps?

# TODO: Implement a YAML file to store model names and their corresponding values

from os import getenv, path
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
# from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage

from ..constants import LOCAL_MODELS, MODEL_CODES

# FOR DEBUG OR PIPING OUTPUT
# NOTE: Is this usable in any use cases like asynchronously populating convex tables?
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
import yaml
logger = logging.getLogger(__name__)

# DEPRECATED
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms.ollama import Ollama

## NEW ##
def get_model_config_from_yaml(filename: str):
    models_dir = path.dirname(path.realpath(__file__))
    file_path = path.join(models_dir, filename)
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
OPENAI_BASE_URL = "https://api.openai.com/v1"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
# This is for LMSTUDIO, but I set llamacpp to same port
LOCAL_BASE_URL = "http://localhost:1234/v1"
LLAMA_CPP_BASE_URL = LOCAL_BASE_URL # This can be changed to a different port


TOGETHER_BASE_URL = "https://api.together.xyz"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LITELLM_BASE_URL = "http://localhost:4000/v1"

ALL_MODELS = {
    "providers": {
        "openai": MODEL_CONFIG['valid_openai_models'],
        "together": MODEL_CONFIG['valid_together_models'],
        "deepseek": MODEL_CONFIG['valid_deepseek_models'],
        "openrouter": MODEL_CONFIG['valid_openrouter_models'],
        "local": MODEL_CONFIG['valid_local_models'],
        "ollama": MODEL_CONFIG['valid_ollama_models'],
        "litellm": MODEL_CONFIG['valid_litellm_models']
    }
}

VALID_TOGETHER_MODELS = ALL_MODELS["providers"]["together"]
VALID_OPENAI_MODELS = ALL_MODELS["providers"]["openai"]
VALID_DEEPSEEK_MODELS = ALL_MODELS["providers"]["deepseek"]
VALID_OPENROUTER_MODELS = ALL_MODELS["providers"]["openrouter"]
VALID_LOCAL_MODELS = ALL_MODELS["providers"]["local"]
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
        api_key = getenv("LLM_API_KEY")
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
    elif provider == "local":
        assert model_name in VALID_LOCAL_MODELS, f"Invalid model name: {model_name}"
    elif provider == "ollama":
        assert model_name in VALID_OLLAMA_MODELS, f"Invalid model name: {model_name}"
    elif provider == "litellm":
        assert model_name in VALID_LITELLM_MODELS, f"Invalid model name: {model_name}"
    else:
        raise ValueError("Invalid provider")

def get_model_wrapper(provider: str, model_name: str, hyperparameters=None, validate: bool = True):
    ENFORCE_STREAMING_CALLBACK = False
    if ENFORCE_STREAMING_CALLBACK:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    else:
        callback_manager = None
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
            callback_manager=callback_manager
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

def get_openai_gpt4(hyperparameters=None) -> ChatOpenAI:
    model="gpt-4o"
    return get_openai_model(model, hyperparameters)

def get_openai_gpt4_mini(hyperparameters=None) -> ChatOpenAI:
    model="gpt-4o-mini"
    return get_openai_model(model, hyperparameters)

def get_together_dolphin(hyperparameters=None) -> ChatOpenAI:
    model="cognitivecomputations/dolphin-2.5-mixtral-8x7b"
    return get_together_model(model, hyperparameters)

def get_together_qwen(hyperparameters=None) -> ChatOpenAI:
    model="Qwen/Qwen2-72B-Instruct"
    return get_together_model(model, hyperparameters)

def get_together_nous_mix(hyperparameters=None) -> ChatOpenAI:
    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    return get_together_model(model, hyperparameters)

def get_together_bigmix(hyperparameters=None) -> ChatOpenAI:
    model="mistralai/Mixtral-8x22B-Instruct-v0.1"
    return get_together_model(model, hyperparameters)


def get_together_dbrx(hyperparameters=None) -> ChatOpenAI:
    model="databricks/dbrx-instruct"
    return get_together_model(model, hyperparameters)

def get_together_arctic(hyperparameters=None) -> ChatOpenAI:
    model="Snowflake/snowflake-arctic-instruct"
    return get_together_model(model, hyperparameters)


def get_together_llama3(hyperparameters=None) -> ChatOpenAI:
    model="meta-llama/Llama-3-70b-chat-hf"
    return get_together_model(model, hyperparameters)

def get_together_fn_mix(hyperparameters=None) -> ChatOpenAI:
    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    return get_together_model(model, hyperparameters)

def get_together_deepseek_4k(hyperparameters=None) -> ChatOpenAI:
    model="deepseek-ai/deepseek-llm-67b-chat"
    return get_together_model(model, hyperparameters)

def get_together_deepseek_32k(hyperparameters=None) -> ChatOpenAI:
    model="deepseek-ai/deepseek-llm-405b-chat"
    return get_together_model(model, hyperparameters)

def get_openrouter_sonnet(hyperparameters=None) -> ChatOpenAI:
    model="anthropic/claude-3.5-sonnet"
    return get_openrouter_model(model, hyperparameters)

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

def get_deepseek_coder(hyperparameters=None) -> ChatOpenAI:
    model="deepseek-coder"
    return get_deepseek_model(model, hyperparameters)

def get_deepseek_chat(hyperparameters=None) -> ChatOpenAI:
    model="deepseek-chat"
    return get_deepseek_model(model, hyperparameters)

def get_local_model(hyperparameters=None) -> ChatOpenAI:
    model="local-model"
    return get_local_model(model, hyperparameters)

def get_ollama_llama3(hyperparameters=None) -> ChatOpenAI:
    model="llama3"
    return get_ollama_model(model, hyperparameters)

def get_ollama_mistral(hyperparameters=None) -> ChatOpenAI:
    model="mistral:7b-instruct-v0.3-q6_K"
    return get_ollama_model(model, hyperparameters)

def get_local_llama_cpp(hyperparameters=None) -> ChatOpenAI:
    # This will be deprecated when port parameter is implemented
    return ChatOpenAI(
        base_url=LLAMA_CPP_BASE_URL,
        api_key='LOCAL-API-KEY',
        model="local-model",
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
        api_key="LOCAL-API-KEY"
    )

# This maps the model keys to the functions
FUNCTION_MAP = {
    "get_gemini_flash": get_model_wrapper("litellm", "openrouter/google/gemini-flash-1.5"),
    "gemini": get_model_wrapper("openrouter", "google/gemini-flash-1.5"),
    "get_openai_gpt4": get_openai_gpt4,
    "get_openai_gpt4_mini": get_openai_gpt4_mini,
    "get_together_dolphin": get_together_dolphin,
    "get_together_qwen": get_together_qwen,
    "get_together_nous_mix": get_together_nous_mix,
    "get_together_fn_mix": get_together_fn_mix,
    "get_together_bigmix": get_together_bigmix,
    "get_together_dbrx": get_together_dbrx,
    "get_together_arctic": get_together_arctic,
    "get_together_llama3": get_together_llama3,
    "get_together_new_llama": get_model_wrapper("together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    "get_together_llama_400b": get_model_wrapper("together", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
    "get_together_deepseek_4k": get_together_deepseek_4k,
    "get_together_deepseek_32k": get_together_deepseek_32k,
    "get_deepseek_coder": get_deepseek_coder,
    "get_deepseek_chat": get_deepseek_chat,
    "get_openrouter_sonnet": get_openrouter_sonnet,
    "get_local_model": get_local_model,
    "get_ollama_llama3": get_ollama_llama3,
    "get_ollama_mistral": get_ollama_mistral,
    "get_local_llama_cpp": get_local_llama_cpp,
    "get_openai_embedder_large": get_openai_embedder_large,
    "get_together_embedder_large": get_together_embedder_large,
    "get_ollama_local_embedder": get_ollama_local_embedder,
    "get_lmstudio_local_embedder": get_lmstudio_local_embedder
}

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

### DEPRECATED ###

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

# def get_together_embedder_large(hyperparameters=None) -> OpenAIEmbeddings:
#     api_key = getenv("TOGETHER_API_KEY")
#     assert api_key, "Please set TOGETHER_API_KEY in .env file"
#     return OpenAIEmbeddings(
#         base_url=TOGETHER_BASE_URL,
#         api_key=api_key,
#         model="BAAI/bge-large-en-v1.5",
#     )