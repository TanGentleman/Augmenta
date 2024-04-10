from helpers import read_settings
# use pydantic to enforce Config schema
from typing import Literal
from pydantic import BaseModel, Field

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
    rag_llm: str
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

    def __check_context_max(self) -> bool:
        """
        Validate the context maxes
        """
        chunk_size = self.rag_config["chunk_size"]
        k = self.rag_config["k_excerpts"]
        # get context length of the model. For now, 128K to not throw errors
        context_max = 128000
        max_chars = context_max * 5
        if chunk_size * k > max_chars:
            return False
        return True

    def __validate_configs(self):
        """
        Validate the configuration
        """
        RagSchema(**self.rag_config)
        ChatSchema(**self.chat_config)
        HyperparameterSchema(**self.hyperparameters)
        # TODO:
        # Enforce chunk size check against the embedder here
        if not self.__check_context_max():
            raise ValueError("Context max exceeds model limit")
        
        if self.chat_config["rag_mode"]:
            if not self.rag_config["inputs"] or not self.rag_config["inputs"][0]:
                raise ValueError("RAG mode requires inputs")
        pass

    def __str__(self):
        return self.props()
    
    def props(self):
        """
        Return the keys and values for each item in rag_config and chat_config
        """
        rag_config_str = "\n".join(f"{k}: {v}" for k, v in self.rag_config.items())
        chat_config_str = "\n".join(f"{k}: {v}" for k, v in self.chat_config.items())
        hyperparameters_str = "\n".join(f"{k}: {v}" for k, v in self.hyperparameters.items())
        
        return f"Rag Config:\n{rag_config_str}\n\nChat Config:\n{chat_config_str}\n\nHyperparameters:\n{hyperparameters_str}"
        
