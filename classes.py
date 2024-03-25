from helpers import read_settings
# use pydantic to enforce Config schema
from typing import Literal
from pydantic import BaseModel, Field
# use typing to define types for the Config class
class LLM(BaseModel):
    """
    Configuration for LLM
    """
    model: Literal[
        "get_openai_gpt4",
        "get_together_quen",
        "get_together_nous_mix",
        "get_together_fn_mix",
        "get_together_coder",
        "get_claude_opus",
        "get_local_model",
        ]
    pass

class Embedder(BaseModel):
    """
    Configuration for EMBEDDER
    """
    model: Literal["get_openai_embedder_large"]
    pass

class RagSchema(BaseModel):
    """
    Configuration for RAG
    """
    collectionName: str
    embedding_model: Embedder
    method: Literal["faiss", "chroma"]
    chunkSize: int = Field(ge=0)
    chunkOverlap: int = Field(ge=0)
    rag_model: str
    inputs: list[str]
    pass

class ChatSchema(BaseModel):
    """
    Configuration for Chat
    """
    primary_model: LLM
    backup_model: LLM
    persistence_enabled: bool
    enable_system_message: bool
    system_message: str

class Config():
    """
    Configuration class
    """
    def __init__(self, config_file = "settings.json"):
        config = read_settings(config_file)
        self.__rag_config = config["rag_config"]
        self.__chat_config = config["chat_config"]
        self.__validate_configs()

    def validate_configs(self):
        """
        Validate the configuration
        """
        RagSchema(**self.__rag_config)
        ChatSchema(**self.__chat_config)
        pass