from helpers import read_settings
# use pydantic to enforce Config schema
from typing import Literal
from pydantic import BaseModel, Field

LLM = Literal[
        "get_openai_gpt4",
        "get_together_quen",
        "get_together_nous_mix",
        "get_together_fn_mix",
        "get_together_coder",
        "get_claude_opus",
        "get_claude_sonnet",
        "get_local_model",
        ]
Embedder = Literal["get_openai_embedder_large",
                   "get_together_embedder_large",
                   "get_nomic_local_embedder"]

class RagSchema(BaseModel):
    """
    Configuration for RAG
    """
    collection_name: str
    embedding_model: Embedder
    method: Literal["faiss", "chroma"]
    chunk_size: int = Field(ge=0)
    chunk_overlap: int = Field(ge=0)
    rag_llm: str
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
    rag_mode: bool

class Config():
    """
    Configuration class
    """
    def __init__(self, config_file = "settings.json"):
        config = read_settings(config_file)
        self.rag_config = config["rag_config"]
        self.chat_config = config["chat_config"]
        self.__validate_configs()

    def __validate_configs(self):
        """
        Validate the configuration
        """
        RagSchema(**self.rag_config)
        ChatSchema(**self.chat_config)
        # TODO:
        # Enforce chunk size check against the embedder here
        pass