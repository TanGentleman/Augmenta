from config import LOCAL_MODEL_ONLY, SYSTEM_MSG_MAP
from constants import SYSTEM_MESSAGE_CODES
from helpers import database_exists, read_settings
# use pydantic to enforce Config schema
from typing import Literal
from pydantic import BaseModel, Field

from models import LLM_FN, MODEL_DICT
from os import path, mkdir
from json import load as json_load, dump as json_dump

ACTIVE_JSON_FILE = "active.json"

VALID_LLM = Literal[
    "get_openai_gpt4",
    "get_together_dolphin",
    "get_together_quen",
    "get_together_nous_mix",
    "get_together_fn_mix",
    "get_together_bigmix",
    "get_together_dbrx",
    "get_together_arctic",
    "get_together_llama3",
    "get_together_deepseek_4k",
    "get_together_deepseek_32k",
    "get_claude_opus",
    "get_claude_sonnet",
    "get_local_model",
    "get_ollama_local_model"
]

VALID_EMBEDDER = Literal["get_openai_embedder_large",
                         "get_together_embedder_large",
                         "get_ollama_local_embedder",
                         "get_lmstudio_local_embedder"]


class ManifestSchema(BaseModel):
    """
    Manifest schema
    """
    embedding_model: VALID_EMBEDDER
    method: Literal["faiss", "chroma"]
    chunk_size: int
    chunk_overlap: int
    inputs: list[str]
    doc_ids: list[str]


class ChatSchema(BaseModel):
    """
    Configuration for Chat
    """
    primary_model: VALID_LLM
    backup_model: VALID_LLM
    enable_system_message: bool
    system_message: str


class HyperparameterSchema(BaseModel):
    """
    Hyperparameters for LLM
    """
    max_tokens: int
    temperature: float


class RagSchema(BaseModel):
    """
    Configuration for RAG
    """
    rag_mode: bool
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


class RagSettings:
    """
    Configuration for RAG
    """

    def __init__(self, **kwargs):
        self.__config = RagSchema(**kwargs)
        self._rag_mode = False
        # TODO: Expose more attributes

        if self.__config.rag_mode:
            self.__rag_mode = True
            self.__enable_rag_mode(
                self.__config.collection_name,
                self.__config.embedding_model,
                self.__config.method,
                self.__config.chunk_size,
                self.__config.chunk_overlap,
                self.__config.k_excerpts,
                self.__config.rag_llm,
                self.__config.inputs,
                self.__config.multivector_enabled,
                self.__config.multivector_method)

    def __enable_rag_mode(
            self,
            collection_name,
            embedding_model,
            method,
            chunk_size,
            chunk_overlap,
            k_excerpts,
            rag_llm,
            inputs,
            multivector_enabled,
            multivector_method):
        """
        Enable RAG mode
        """
        assert self.rag_mode, "RAG mode must be enabled"
        self.database_exists = database_exists(collection_name, method)

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_excerpts = k_excerpts
        self.rag_llm = rag_llm
        self.inputs = inputs
        self.multivector_enabled = multivector_enabled
        self.multivector_method = multivector_method

    def adjust_rag_settings(self, metadata: dict):
        """
        Adjust the rag settings based on the metadata from manifest.json

        Metadata:
        - embedding_model
        - method
        - chunk_size
        - chunk_overlap
        - inputs
        - doc_ids ([] unless multivector_enabled is True)
        """
        assert "embedding_model" in metadata, "Embedding model not found"
        for model in MODEL_DICT:
            if MODEL_DICT[model]["model_name"] == metadata["embedding_model"]:
                metadata["embedding_model"] = model
                break
        else:
            raise ValueError("Embedding model not found")
        ManifestSchema(**metadata)
        model = MODEL_DICT.get(metadata["embedding_model"])
        if model is None:
            raise ValueError("Embedding model not found")
        metadata["embedding_model"] = model["model_name"]
        if metadata["embedding_model"] != self.embedding_model.model_name:
            self.embedding_model = metadata["embedding_model"]
            assert isinstance(
                self.embedding_model, LLM_FN), "Embedding model must be type LLM_FN"
            print(
                f"Warning: Embedding model in settings.json switched to {self.embedding_model.model_name} from manifest.json.")
        if metadata["method"] != self.method:
            self.method = metadata["method"]
            print(
                f"Warning: Method in settings.json has been switched to {self.method} from manifest.json.")

        manifest_chunk_size = int(metadata["chunk_size"])
        if manifest_chunk_size != self.chunk_size:
            self.chunk_size = manifest_chunk_size
            print(
                f"Warning: Chunk size in settings.json has been switched to {manifest_chunk_size} from manifest.json.")

        manifest_chunk_overlap = int(metadata["chunk_overlap"])
        if manifest_chunk_overlap != self.chunk_overlap:
            self.chunk_overlap = manifest_chunk_overlap
            print(
                f"Warning: Chunk overlap in settings.json has been switched to {manifest_chunk_overlap} from manifest.json.")
        if metadata["inputs"] != self.inputs:
            self.inputs = metadata["inputs"]
            print("Warning: Inputs loaded from manifest.json.")
        if self.multivector_enabled is True:
            if not metadata["doc_ids"]:
                raise ValueError(
                    "Multivector enabled but no doc_ids in manifest")
        else:
            if metadata["doc_ids"]:
                print("Multivector disabled but doc_ids in manifest")
                print("Warning: Enabling multivector.")
                self.multivector_enabled = True

    @property
    def rag_mode(self):
        return self.__rag_mode

    @rag_mode.setter
    def rag_mode(self, value):
        if not isinstance(value, bool):
            raise ValueError("Rag mode must be a boolean")
        # TODO: Should this have side effects?
        self.__rag_mode = value
        print(f"Rag mode set to {self.__rag_mode}")
        if value is True:
            print("Enabling RAG mode. Doing some validation steps")
            self.__enable_rag_mode(
                self.__config.collection_name,
                self.__config.embedding_model,
                self.__config.method,
                self.__config.chunk_size,
                self.__config.chunk_overlap,
                self.__config.k_excerpts,
                self.__config.rag_llm,
                self.__config.inputs,
                self.__config.multivector_enabled,
                self.__config.multivector_method)

    def set_database_exists(self):
        """
        Check if the collection exists in vector DB
        """
        if database_exists(self.collection_name, self.method):
            self.database_exists = True
        else:
            self.database_exists = False

    @property
    def collection_name(self):
        return self.__collection_name

    @collection_name.setter
    def collection_name(self, value):
        if not value:
            raise ValueError("Collection name cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Collection name must be a string")
        self.__collection_name = value
        # TODO: Check if collection exists in manifest.json
        # TODO: Check if collection exists as vector DB

    @property
    def embedding_model(self) -> LLM_FN:
        return self.__embedding_model

    @embedding_model.setter
    def embedding_model(self, value):
        if not value:
            raise ValueError("Embedding model cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Embedding model must be a string")
        for embedder_fn_name in MODEL_DICT.keys():
            if value == embedder_fn_name:
                if MODEL_DICT[embedder_fn_name]["model_type"] != "embedder":
                    raise ValueError("Embedding model must be type embedder")
                embedding_fn = MODEL_DICT[embedder_fn_name]["function"]
                self.__embedding_model = LLM_FN(embedding_fn)
                print(
                    f"Embedding model set to {self.__embedding_model.model_name}")
                break
        else:
            raise ValueError("Embedding model not found")

        if LOCAL_MODEL_ONLY:
            ALLOWED_EMBEDDERS = ["nomic-embed-text", "local-embedding-model"]
            if self.__embedding_model.model_name not in ALLOWED_EMBEDDERS:
                raise ValueError(
                    "LOCAL_MODEL_ONLY is set to True. Only local embedders are supported.")
        # TODO: Check if embedding model matches loaded collection

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, value):
        if value not in ["faiss", "chroma"]:
            raise ValueError("Method must be 'faiss' or 'chroma'")
        self.__method = value

    @property
    def chunk_size(self):
        return self.__chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        if not isinstance(value, int):
            raise ValueError("Chunk size must be an integer")
        if value <= 0 or value > 100000:
            raise ValueError("Chunk size must be 1-100000")
        self.__chunk_size = value

    @property
    def chunk_overlap(self):
        return self.__chunk_overlap

    @chunk_overlap.setter
    def chunk_overlap(self, value):
        if not isinstance(value, int):
            raise ValueError("Chunk overlap must be an integer")
        if value < 0 or value >= self.chunk_size:
            raise ValueError("Chunk overlap must be 0 to chunk_size")
        self.__chunk_overlap = value

    @property
    def k_excerpts(self):
        return self.__k_excerpts

    @k_excerpts.setter
    def k_excerpts(self, value):
        if not isinstance(value, int):
            raise ValueError("k_excerpts must be an integer")
        if value < 0 or value > 15:
            raise ValueError("k_excerpts must be 0-25")
        self.__k_excerpts = value

    @property
    def rag_llm(self):
        return self.__rag_llm

    @rag_llm.setter
    def rag_llm(self, value):
        if not value:
            raise ValueError("RAG LLM cannot be empty")
        if not isinstance(value, str):
            raise ValueError("RAG LLM must be a string")
        for llm_fn_name in MODEL_DICT.keys():
            if value == llm_fn_name:
                if MODEL_DICT[llm_fn_name]["model_type"] != "llm":
                    raise ValueError("RAG LLM must be type llm")
                rag_llm_fn = MODEL_DICT[llm_fn_name]["function"]
                self.__rag_llm = LLM_FN(rag_llm_fn)
                print(f"RAG LLM set to {self.__rag_llm.model_name}")
                # Check context max
                context_max = self.__rag_llm.context_size
                max_chars = context_max * 4
                if self.chunk_size * self.k_excerpts > max_chars:
                    print("Warning: Chunk size * k exceeds model limit.")
                break
        else:
            raise ValueError("RAG LLM not found")

        if LOCAL_MODEL_ONLY:
            if self.__rag_llm.model_name != "local-model":
                raise ValueError(
                    f"LOCAL_MODEL_ONLY is set to True. {self.__rag_llm.model_name}, is non-local.")

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, value):
        if not isinstance(value, list):
            raise ValueError("Inputs must be a list")

        if self.database_exists:
            print("Claiming that db exists!")
            print("Skipping input validation")
        else:
            print("Claiming that db does not exist!")
            if not any(i for i in value):
                raise ValueError("RAG mode requires valid string inputs")

        if LOCAL_MODEL_ONLY and self.database_exists is False:
            for input in value:
                if input.startswith("http"):
                    raise ValueError(
                        f"LOCAL_MODEL_ONLY is set to True. '{input}' is non-local.")
        self.__inputs = value

    @property
    def multivector_enabled(self):
        return self.__multivector_enabled

    @multivector_enabled.setter
    def multivector_enabled(self, value):
        if not isinstance(value, bool):
            raise ValueError("Multivector enabled must be a boolean")
        self.__multivector_enabled = value

    @property
    def multivector_method(self):
        return self.__multivector_method

    @multivector_method.setter
    def multivector_method(self, value):
        if value not in ["summary", "qa"]:
            raise ValueError("Multivector method must be 'summary' or 'qa'")
        self.__multivector_method = value

    def to_dict(self):
        if not self.rag_mode:
            return {"rag_mode": False}
        data = {
            "rag_mode": self.rag_mode,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model.model_name,
            "method": self.method,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "k_excerpts": self.k_excerpts,
            "rag_llm": self.rag_llm.model_name,
            "inputs": self.inputs,
            "multivector_enabled": self.multivector_enabled,
            "multivector_method": self.multivector_method
        }
        return data

    def props(self):
        return self.to_dict()

    def __str__(self):
        return str(self.props())


class ChatSettings:
    """
    Configuration for Chat
    """

    def __init__(
            self,
            primary_model,
            backup_model,
            enable_system_message,
            system_message):
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.enable_system_message = enable_system_message
        self.system_message = system_message

        assert self.primary_model, "ChatSettings.primary_model exists must be set"
        assert self.backup_model, "ChatSettings.backup_model exists must be set"
        assert self.system_message, "ChatSettings.system_message exists must be set"

    @property
    def primary_model(self):
        return self.__primary_model

    @primary_model.setter
    def primary_model(self, value):
        if not value:
            raise ValueError("Primary model cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Primary model must be a string")
        for primary_model_fn_name in MODEL_DICT.keys():
            if value == primary_model_fn_name:
                if MODEL_DICT[primary_model_fn_name]["model_type"] != "llm":
                    raise ValueError("Primary model must be type llm")
                self.__primary_model = LLM_FN(
                    MODEL_DICT[primary_model_fn_name]["function"])
                print(
                    f"Primary model set to {self.__primary_model.model_name}")
                break
        else:
            raise ValueError("Primary model not found")

        if LOCAL_MODEL_ONLY:
            if self.__primary_model.model_name != "local-model":
                raise ValueError(
                    f"LOCAL_MODEL_ONLY is set to True. {self.__primary_model.model_name}, is non-local.")

    @property
    def backup_model(self):
        return self.__backup_model

    @backup_model.setter
    def backup_model(self, value):
        if not value:
            raise ValueError("Backup model cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Backup model must be a string")
        for backup_model_fn_name in MODEL_DICT.keys():
            if value == backup_model_fn_name:
                if MODEL_DICT[backup_model_fn_name]["model_type"] != "llm":
                    raise ValueError("Backup model must be type llm")
                self.__backup_model = LLM_FN(
                    MODEL_DICT[backup_model_fn_name]["function"])
                print(f"Backup model set to {self.__backup_model.model_name}")
                break
        else:
            raise ValueError("Backup model not found")

        if LOCAL_MODEL_ONLY:
            if self.__backup_model.model_name != "local-model":
                raise ValueError(
                    f"LOCAL_MODEL_ONLY is set to True. {self.__backup_model.model_name}, is non-local.")

    @property
    def enable_system_message(self):
        return self.__enable_system_message

    @enable_system_message.setter
    def enable_system_message(self, value):
        if not isinstance(value, bool):
            raise ValueError("Enable system message must be a boolean")
        self.__enable_system_message = value

    @property
    def system_message(self):
        return self.__system_message

    @system_message.setter
    def system_message(self, value):
        if not value:
            raise ValueError("System message cannot be empty")
        if not isinstance(value, str):
            raise ValueError("System message must be a string")
        if value in SYSTEM_MESSAGE_CODES:
            value = SYSTEM_MESSAGE_CODES[value]
        print(f"Config INFO: System message set to {value}")
        self.__system_message = value

    def to_dict(self):
        data = {
            "primary_model": self.primary_model.model_name,
            "backup_model": self.backup_model.model_name,
            "enable_system_message": self.enable_system_message,
            "system_message": self.system_message
        }
        return data

    def props(self):
        return self.to_dict()

    def __str__(self):
        return str(self.props())


class Config:
    """
    Configuration class
    """

    def __init__(
            self,
            config_file="settings.json",
            config_override=None | dict):
        # assert config file exists
        assert path.exists(config_file), "Config file not found"
        config = read_settings(config_file)

        rag_mode = None
        if config_override is not None:
            override_rag_mode = config_override.get("rag_mode")
            if override_rag_mode is not None:
                assert isinstance(
                    override_rag_mode, bool), "rag_mode must be a boolean"
                rag_mode = override_rag_mode
                # print(f"Rag mode overridden to {rag_mode}")
            override_inputs = config_override.get("inputs")
            if override_inputs is not None:
                assert isinstance(
                    override_inputs, list), "inputs must be a list"
                config["rag_config"]["inputs"] = override_inputs
                # print(f"Inputs overridden to {override_inputs}")
            # Not yet implemented
            # override_rag = config_override.get("rag_config")
            # if override_rag is not None:
            #     RagSchema(**override_rag)
            #     config["rag_config"] = override_rag
            # override_chat = config_override.get("chat_config")
            # if override_chat is not None:
            #     ChatSchema(**override_chat)
            #     config["chat_config"] = override_chat

        self.rag_settings = RagSettings(**config["rag_config"])
        self.chat_settings = ChatSettings(**config["chat_config"])

        HyperparameterSchema(**config["hyperparameters"])
        self.hyperparameters = config["hyperparameters"]

        if rag_mode is not None:
            self.rag_settings.rag_mode = rag_mode

        if self.chat_settings.enable_system_message:
            model_name = self.chat_settings.primary_model.model_name
            if model_name in SYSTEM_MSG_MAP:
                self.chat_settings.system_message = SYSTEM_MSG_MAP[model_name]
                print(f"System message adjusted for model {model_name}.")

        self.__validate_rag_config()
        # print(self)
        self.save_to_json()
        print(f"Config initialized and set in {ACTIVE_JSON_FILE}.")

    def __validate_rag_config(self):
        """
        Validate the RAG settings
        """
        if not path.exists("documents"):
            mkdir("documents")
        if not path.exists("manifest.json"):
            with open("manifest.json", "w") as f:
                # Make databases key
                f.write('{"databases": []}')

        if self.rag_settings.rag_mode:
            self.__check_rag_files()

    def __check_rag_files(self):
        """
        Make sure documents folder and manifest.json initialized correctly.
        """
        # Check if the collection exists in manifest.json/vector DB
        # Adjust rag_config with correct fields (include print statements)

        with open("manifest.json", "r") as f:
            data = json_load(f)
            assert "databases" in data, "databases key not found in manifest.json"
            collection_in_manifest = False
            if self.rag_settings.rag_mode:
                for item in data["databases"]:
                    if item["collection_name"] == self.rag_settings.collection_name:
                        collection_in_manifest = True
                if self.rag_settings.database_exists:
                    if collection_in_manifest:
                        print("Collection found in vector DB")
                        # Adjust rag config to match the collection
                        self.rag_settings.adjust_rag_settings(item["metadata"])
                    else:
                        print(
                            "Collection found in vector DB, but not in manifest.json")
                else:
                    # NOTE: Above is already handled in RagSettings
                    if collection_in_manifest:
                        print("Vector DB does not exist, removing from manifest.json")
                        data["databases"] = [item for item in data["databases"]
                                             # if item["collection_name"] !=
                                             # self.rag_config["collection_name"]]
                                             if item["collection_name"] != self.rag_settings.collection_name]
                        with open("manifest.json", "w") as f:
                            json_dump(data, f, indent=2)

    def save_to_json(self, filename=ACTIVE_JSON_FILE):
        """
        Save the current config to a JSON file
        """
        data = {
            "rag_config": self.rag_settings.to_dict(),
            "chat_config": self.chat_settings.to_dict(),
            "hyperparameters": self.hyperparameters
        }
        with open(filename, "w") as f:
            json_dump(data, f, indent=2)

    def __str__(self):
        return self.props()

    def __repr__(self):
        return self.props()

    def props(self) -> str:
        """
        Return the keys and values for each item in rag_config and chat_config
        """
        rag_config_str = "\n".join(
            f"{k}: {v}" for k,
            v in self.rag_settings.props().items())
        chat_config_str = "\n".join(
            f"{k}: {v}" for k,
            v in self.chat_settings.props().items())
        hyperparameters_str = "\n".join(
            f"{k}: {v}" for k, v in self.hyperparameters.items())
        return f"Rag Config:\n{rag_config_str}\n\nChat Config:\n{chat_config_str}\n\nHyperparameters:\n{hyperparameters_str}"
