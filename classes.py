from config import LOCAL_MODEL_ONLY
from helpers import database_exists, read_settings
# use pydantic to enforce Config schema
from typing import Literal
from pydantic import BaseModel, Field

from models import LLM_FN, MODEL_DICT
from os import path, mkdir
from json import load as json_load, dump as json_dump

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
    k_excerpts: int = Field(ge=0, le=15)
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

    def __init__(self, config_file="settings.json", rag_mode=False):
        config = read_settings(config_file)

        self.database_exists = None
        self.rag_mode = rag_mode
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
            print("Warning: Chunk size * k exceeds model limit. This is expected to cause errors.")
            # # In the future, throw an error here.
            # raise ValueError(
            #     f"Chunk size * k exceeds model limit: {chunk_size * k} > {max_chars}")

    def __validate_rag_config(self):
        """
        Validate the RAG settings
        """
        if self.rag_mode:
            self.chat_config["rag_mode"] = True
        if self.chat_config["rag_mode"]:
            self.rag_mode = True
            self.__check_rag_files()
        self.__check_context_max()

    def __adjust_rag_config(self, metadata: dict):
        """
        Adjust the rag_config based on the metadata from manifest.json
        """
        assert self.rag_mode is True, "RAG mode must be enabled"
        if metadata["embedding_model"] != self.rag_config["embedding_model"].model_name:
            self.rag_config["embedding_model"] = metadata["embedding_model"]
            print(f"Warning: Embedding model in settings.json switched to {self.rag_config['embedding_model'].model_name} from manifest.json.")
        if metadata["method"] != self.rag_config["method"]:
            print(f"Warning: Method in settings.json has been switched to {self.rag_config['method']} from manifest.json.")
            self.rag_config["method"] = metadata["method"]
        
        manifest_chunk_size = int(metadata["chunk_size"])
        if manifest_chunk_size != self.rag_config["chunk_size"]:
            self.rag_config["chunk_size"] = manifest_chunk_size
            print(f"Warning: Chunk size in settings.json has been switched to {self.rag_config['chunk_size']} from manifest.json.")
        
        manifest_chunk_overlap = int(metadata["chunk_overlap"])
        if manifest_chunk_overlap != self.rag_config["chunk_overlap"]:
            self.rag_config["chunk_overlap"] = manifest_chunk_overlap
            print(f"Warning: Chunk overlap in settings.json has been switched to {self.rag_config['chunk_overlap']} from manifest.json.")
        if metadata["inputs"] != self.rag_config["inputs"]:
            self.rag_config["inputs"] = metadata["inputs"]
            print("Warning: Inputs loaded from manifest.json.")
        if self.rag_config["multivector_enabled"]:
            if not metadata["doc_ids"]:
                raise ValueError("Multivector enabled but no doc_ids in manifest")
        if metadata["doc_ids"]:
            print("Multivector disabled but doc_ids in manifest")
            print("Warning: Enabling multivector.")
            self.rag_config["multivector_enabled"] = True
                
    def __check_rag_files(self):
        """
        Make sure documents folder and manifest.json initialized correctly.
        """
        # Make a folder called documents if it doesn't exist
        if not path.exists("documents"):
            mkdir("documents")
        if not path.exists("manifest.json"):
            self.database_exists = False
            with open("manifest.json", "w") as f:
                # Make databases key
                f.write('{"databases": []}')
        else:
            # Check if the collection exists in manifest.json/vector DB
            # Adjust rag_config with correct fields (include print statements)

            with open("manifest.json", "r") as f:
                data = json_load(f)
                assert "databases" in data, "databases key not found in manifest.json"
                collection_in_manifest = False
                if self.chat_config["rag_mode"]:
                    for item in data["databases"]:
                        if item["collection_name"] == self.rag_config["collection_name"]:
                            collection_in_manifest = True
                    if collection_in_manifest:
                        if database_exists(self.rag_config["collection_name"], self.rag_config["method"]):
                            self.database_exists = True
                            print("Collection found in vector DB")
                            # Adjust rag config to match the collection
                            self.__adjust_rag_config(item["metadata"])
                        
                        else:
                            if not any(i for i in self.rag_config["inputs"]):
                                raise ValueError("RAG mode requires inputs")
                            self.database_exists = False
                            print("Vector DB does not exist, removing from manifest.json")
                            data["databases"] = [item for item in data["databases"] if item["collection_name"] != self.rag_config["collection_name"]]
                            with open("manifest.json", "w") as f:
                                json_dump(data, f, indent=2)
                    else:
                        self.database_exists = False
                        print("Setting database_exists to False. Can this change later?")
                    
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

        
            
        self.__validate_rag_config()
        if LOCAL_MODEL_ONLY:
            if self.rag_mode:
                if self.rag_config["rag_llm"].model_name != "local-model":
                    raise ValueError(
                        f"LOCAL_MODEL_ONLY is set to True. {self.rag_config['rag_llm'].model_name}, is non-local.")
                ALLOWED_EMBEDDERS = ["nomic-embed-text"]
                if self.rag_config["embedding_model"].model_name not in ALLOWED_EMBEDDERS:
                    print(self.rag_config["embedding_model"].model_name)
                    raise ValueError("Only local embedder supported is get_nomic_local_embedder")
            else:
                for model in [self.chat_config["primary_model"], self.chat_config["backup_model"]]:
                    if model.model_name != "local-model":
                        raise ValueError(
                            f"LOCAL_MODEL_ONLY is set to True. {model.model_name}, is non-local.")
            if self.chat_config["rag_mode"] and self.database_exists is False:
                for input in self.rag_config["inputs"]:
                    if input.startswith("http"):
                        raise ValueError(
                            f"LOCAL_MODEL_ONLY is set to True. '{input}' is non-local.")

    def __str__(self):
        return self.props()
    
    def __repr__(self):
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
