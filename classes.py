from config import LOCAL_MODEL_ONLY
from helpers import read_settings
# use pydantic to enforce Config schema
from typing import Literal
from pydantic import BaseModel, Field

from models import LLM_FN, MODEL_DICT

VALID_LLM = Literal[
    "get_openai_gpt4",
    "get_together_dolphin",
    "get_together_quen",
    "get_together_nous_mix",
    "get_together_fn_mix",
    "get_together_bigmix",
    "get_together_dbrx",
    "get_together_deepseek_4k",
    "get_together_deepseek_32k",
    "get_claude_opus",
    "get_claude_sonnet",
    "get_local_model",
]
VALID_EMBEDDER = Literal["get_openai_embedder_large",
                         "get_together_embedder_large",
                         "get_nomic_local_embedder"]


class RagSchema(BaseModel):
    """
    Configuration for RAG
    """
    collection_name: str
    embedding_model: VALID_EMBEDDER
    method: Literal["faiss", "chroma"]
    chunk_size: int = Field(ge=0)
    chunk_overlap: int = Field(ge=0)
    k_excerpts: int = Field(ge=0, le=8)
    rag_llm: VALID_LLM
    inputs: list[str]
    multivector_enabled: bool
    multivector_method: Literal["summary", "qa"]
    pass


class ChatSchema(BaseModel):
    """
    Configuration for Chat
    """
    primary_model: VALID_LLM
    backup_model: VALID_LLM
    enable_system_message: bool
    system_message: str
    rag_mode: bool


class HyperparameterSchema(BaseModel):
    """
    Hyperparameters for LLM
    """
    max_tokens: int
    temperature: float


class Config:
    """
    Configuration class
    """

    def __init__(self, config_file="settings.json"):
        config = read_settings(config_file)
        self.rag_config = config["rag_config"]
        self.chat_config = config["chat_config"]
        self.hyperparameters = config["hyperparameters"]
        self.__validate_configs()

    def __check_context_max(self):
        """
        Validate the context maxes
        """
        rag_llm = self.rag_config["rag_llm"]
        assert isinstance(rag_llm, LLM_FN), "Rag LLM must be type LLM_FN"
        context_max = rag_llm.context_size
        chunk_size = self.rag_config["chunk_size"]
        k = self.rag_config["k_excerpts"]
        # get context length of the model. For now, 128K to not throw errors
        max_chars = context_max * 4
        if chunk_size * k > max_chars:
            raise ValueError(
                f"Chunk size * k exceeds model limit: {chunk_size * k} > {max_chars}")

    def __validate_rag_config(self):
        """
        Validate the RAG settings
        """
        if self.chat_config["rag_mode"]:
            if not self.rag_config["inputs"]:
                raise ValueError("RAG mode requires inputs")
            for i in range(len(self.rag_config["inputs"])):
                if not self.rag_config["inputs"][i]:
                    raise ValueError(f"Input {i} is empty")

        self.__check_context_max()

    def __validate_configs(self):
        """
        Validate the configuration
        """
        RagSchema(**self.rag_config)
        ChatSchema(**self.chat_config)
        HyperparameterSchema(**self.hyperparameters)
        # Set LLMs
        rag_llm_fn = MODEL_DICT[self.rag_config["rag_llm"]]["function"]
        embedder_fn = MODEL_DICT[self.rag_config["embedding_model"]]["function"]

        primary_model_fn = MODEL_DICT[self.chat_config["primary_model"]]["function"]
        backup_model_fn = MODEL_DICT[self.chat_config["backup_model"]]["function"]

        self.rag_config["rag_llm"] = LLM_FN(rag_llm_fn)
        self.rag_config["embedding_model"] = LLM_FN(embedder_fn)

        self.chat_config["primary_model"] = LLM_FN(primary_model_fn)
        self.chat_config["backup_model"] = LLM_FN(backup_model_fn)

        if LOCAL_MODEL_ONLY:
            llm_models = [self.chat_config["primary_model"],
                          self.chat_config["backup_model"],
                          self.rag_config["rag_llm"],
                          ]
            for model in llm_models:
                if model.model_name != "local-model":
                    raise ValueError(
                        f"LOCAL_MODEL_ONLY is set to True. {model.model_name}, is non-local.")
            assert self.rag_config["embedding_model"].model_name == "nomic-embed-text", "Only local embedder supported is get_nomic_local_embedder"
            if self.chat_config["rag_mode"]:
                for input in self.rag_config["inputs"]:
                    if "https" in input:
                        raise ValueError(
                            f"LOCAL_MODEL_ONLY is set to True. {input}, is non-local.")
        self.__validate_rag_config()

    def __str__(self):
        return self.props()

    def props(self):
        """
        Return the keys and values for each item in rag_config and chat_config
        """
        rag_config_str = "\n".join(
            f"{k}: {v}" for k,
            v in self.rag_config.items())
        chat_config_str = "\n".join(
            f"{k}: {v}" for k,
            v in self.chat_config.items())
        hyperparameters_str = "\n".join(
            f"{k}: {v}" for k, v in self.hyperparameters.items())

        return f"Rag Config:\n{rag_config_str}\n\nChat Config:\n{chat_config_str}\n\nHyperparameters:\n{hyperparameters_str}"
