from config.config import LOCAL_MODEL_ONLY
from constants import LOCAL_MODELS, MODEL_CODES, MODEL_TO_SYSTEM_MSG, SYSTEM_MESSAGE_CODES
# use pydantic to enforce Config schema
from typing import Literal, Union
from pydantic import BaseModel, Field

from models.models import LLM_FN, MODEL_DICT
from os import path, mkdir
from json import load as json_load, dump as json_dump

import logging

import utils
logger = logging.getLogger(__name__)

ACTIVE_JSON_FILE = "active.json"

VALID_LLM = Literal[
    "get_openai_gpt4",
    "get_together_dolphin",
    "get_together_qwen",
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
    "get_deepseek_coder",
    "get_local_model",
    "get_ollama_llama3",
    "get_ollama_mistral",
    "get_local_hermes"
]

VALID_EMBEDDER = Literal["get_openai_embedder_large",
                         "get_together_embedder_large",
                         "get_ollama_local_embedder",
                         "get_lmstudio_local_embedder"]

configValue = Union[str, int, float, bool]


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
    doc_ids: list[str] = []


class OptionalSchema(BaseModel):
    """
    Optional configuration
    """
    prompt_prefix: str
    prompt_suffix: str
    amnesia: bool
    display_flashcards: bool


def get_llm_fn(model_name: str,
               model_type: Literal["llm",
                                   "embedder"]) -> LLM_FN:
    """
    Get the LLM function from the name
    """
    assert model_type in [
        "llm", "embedder"], "Model type must be llm or embedder"
    model_function = MODEL_DICT.get(model_name)
    if model_function is None:
        raise ValueError(f"Model {model_name} not found")
    if model_function["model_type"] != model_type:
        raise ValueError(f"Model {model_name} must be type {model_type}")

    llm_fn = LLM_FN(model_function["function"])
    if LOCAL_MODEL_ONLY:
        if llm_fn.model_name not in LOCAL_MODELS:
            logger.warning(f"Local model must be in {LOCAL_MODELS}")
            logger.error(f"Model {llm_fn.model_name} is not local")
            raise ValueError(
                f"LOCAL_MODEL_ONLY is set to True. Change this {model_type}.")
    return llm_fn


class RagSettings:
    """
    Configuration for RAG
    """

    def __init__(self, **kwargs):
        self.__config = RagSchema(**kwargs)
        # TODO: Expose more attributes
        self.database_exists = None
        if self.__config.rag_mode:
            self.rag_mode = True
            # The other attributes are ONLY set if RAG mode is enabled
        else:
            self.rag_mode = False

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
        if self.database_exists is not None:
            logger.warning(
                "Found value for rag_settings.database_exists! Resetting.")
        self.database_exists = utils.database_exists(collection_name, method)
        # NOTE: Metadata:
        # - embedding_model
        # - method
        # - chunk_size
        # - chunk_overlap
        # - inputs
        # - doc_ids ([] unless multivector_enabled is True)
        # TODO: Decide the best way to set these values
        self.collection_name = collection_name
        self.k_excerpts = k_excerpts
        self.multivector_method = multivector_method
        self.multivector_enabled = multivector_enabled

        if self.database_exists:
            logger.info("DB exists! Replacing the manifest.json settings.")
            self.update_rag_settings(override_all=True)
            # Passing override_all will ensure that the model, chunk size,
            # overlap, inputs, and multivector are updated
        else:
            self.method = method
            self.embedding_model = embedding_model
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.inputs = inputs

        self.rag_llm = rag_llm
        print(self.props())

    def adjust_rag_settings(
            self, metadata: dict[str, configValue], override_all=False):
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
        assert "embedding_model" in metadata, "Embedding model key not found"
        # Replace the manifest "embedKding model" value from the model name to
        # the MODEL_DICT key
        for model, model_info in MODEL_DICT.items():
            if model_info["model_name"] == metadata["embedding_model"]:
                metadata["embedding_model"] = model
                break
        else:
            raise ValueError("Model not found in MODEL_DICT")
        ManifestSchema(**metadata)

        manifest_chunk_size = int(metadata["chunk_size"])
        if override_all or (manifest_chunk_size != self.chunk_size):
            self.chunk_size = manifest_chunk_size
            logger.warning(
                f"Switched chunk size to {manifest_chunk_size} from manifest.json.")

        manifest_chunk_overlap = int(metadata["chunk_overlap"])
        if override_all or (manifest_chunk_overlap != self.chunk_overlap):
            self.chunk_overlap = manifest_chunk_overlap
            logger.warning(
                f"Switched chunk overlap to {manifest_chunk_overlap} from manifest.json.")

        if override_all or (
                metadata["embedding_model"] != self.embedding_model.model_name):
            self.embedding_model = metadata["embedding_model"]
            logger.warning(
                f"Switched embedding model to {self.embedding_model.model_name} from manifest.json.")
        if override_all or (metadata["method"] != self.method):
            self.method = metadata["method"]
            logger.warning(
                f"Switched method to {self.method} from manifest.json.")
        if override_all or (metadata["inputs"] != self.inputs):
            self.inputs = metadata["inputs"]
            logger.warning(
                f"Switched to {len(metadata['inputs'])} inputs from manifest.json.")
            logger.info(f"Inputs: {self.inputs}")

        # Check if metadata["doc_ids"] exists

        if metadata.get("doc_ids"):
            if override_all:
                self.multivector_enabled = True
            else:
                # If multivector is not enabled and doc_ids exist
                if not self.multivector_enabled:
                    logger.warning(
                        "Multivector switched to True from manifest.json")
            self.multivector_enabled = True
            logger.warning(
                "Multivector is enabled! Make sure to get the doc IDS for RAG needs!")
        else:
            # No doc ids found
            self.multivector_enabled = False
            logger.warning(
                "Multivector forcibly disabled. Could this be problematic?")
            # if self.multivector_enabled:
            #     logger.error("Multivector should not be enabled with no doc ids! Aborting.")
            #     raise ValueError("Error in manifest.json")

    def update_rag_settings(self, override_all=False):
        """
        Make sure documents folder and manifest.json initialized correctly.
        """
        # Check if the collection exists in manifest.json/vector DB
        # Adjust RAG settings with correct fields (include print statements)
        if override_all:
            assert self.database_exists is True, "DB must exist to override all settings"
        filepath = utils.DATA_DIR / "manifest.json"
        with open(filepath, "r") as f:
            data = json_load(f)
        assert "databases" in data, "databases key not found in manifest.json"
        if self.rag_mode:
            prune_collection = False
            if self.database_exists is None:
                logger.error(
                    "This check should not ever be called! Set DB exists first!")
                self.set_database_exists()

            for item in data["databases"]:
                if item["collection_name"] == self.collection_name:
                    if self.database_exists:
                        logger.info("Collection found in vector DB")
                        # Adjust rag config to match the collection
                        self.adjust_rag_settings(
                            item["metadata"], override_all=override_all)
                        logger.info("RAG settings adjusted.")
                        break
                    else:
                        logger.warning(
                            "Collection found in manifest.json but not in vector DB")
                        prune_collection = True
                        break
            else:
                if self.database_exists:
                    logger.warning(
                        "Collection from DB not found in manifest.json")

            if prune_collection:
                # Happens when DB is deleted but collection in manifest
                logger.warning(
                    "Removing this collection from manifest.json")
                # Remove by collection name
                data["databases"] = [item for item in data["databases"]
                                     if item["collection_name"] != self.collection_name]
                with open(filepath, "w") as f:
                    json_dump(data, f, indent=2)
                # NOTE: What happens to the values in the config?

    @property
    def rag_mode(self):
        return self.__rag_mode

    @rag_mode.setter
    def rag_mode(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("Rag mode must be a boolean")
        # TODO: Should this have side effects?
        self.__rag_mode = value
        logger.info(f"Rag mode set to {self.__rag_mode}")
        if value is True:
            logger.info("Enabling RAG mode. Doing some validation steps")
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

    def set_database_exists(self, collection_name: str, method: str):
        """
        Check if the collection exists in vector DB
        """
        logger.warning(
            "This is deprecated. Directly call database_exists, sparingly.")
        self.database_exists = utils.database_exists(collection_name, method)

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
    def embedding_model(self, model_name: str):
        if not isinstance(model_name, str):
            raise ValueError("Embedding model must be a string")

        self.__embedding_model = get_llm_fn(model_name, "embedder")

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, vector_db_method):
        if vector_db_method not in ["faiss", "chroma"]:
            raise ValueError("Method must be 'faiss' or 'chroma'")
        self.__method = vector_db_method

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
    def rag_llm(self, model_name: str):
        if not model_name:
            raise ValueError("RAG LLM cannot be empty")
        if not isinstance(model_name, str):
            raise ValueError("RAG LLM must be a string")

        rag_llm_fn = get_llm_fn(model_name, "llm")
        self.__rag_llm = rag_llm_fn
        context_max = rag_llm_fn.context_size
        max_chars = context_max * 4
        if self.chunk_size * self.k_excerpts > max_chars:
            logger.warning("Warning: Chunk size * k exceeds model limit.")

        if LOCAL_MODEL_ONLY:
            if self.__rag_llm.model_name not in LOCAL_MODELS:
                raise ValueError(
                    f"LOCAL_MODEL_ONLY is set to True. {self.__rag_llm.model_name}, is non-local.")

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, value):
        if not isinstance(value, list):
            raise ValueError("Inputs must be a list")

        logger.info(
            f"Checking if DB exists for inputs. Make sure this is after settings are finalized!")
        if self.database_exists is False:
            logger.info(
                "Vector DB does not exist! (Or self.database_exists is None)")
            if not any(i for i in value):
                # TODO: Check if the inputs are valid
                raise ValueError("RAG mode requires valid string inputs")
            if LOCAL_MODEL_ONLY:
                for input in value:
                    if input.startswith("http"):
                        raise ValueError(
                            f"LOCAL_MODEL_ONLY is set to True. '{input}' is non-local.")
        logger.info(f"Set {len(value)} inputs")
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
        # Can check if multivector is enabled
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
            system_message,
            stream):
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.enable_system_message = enable_system_message
        self.system_message = system_message
        self.stream = stream
        assert self.primary_model, "ChatSettings.primary_model exists must be set"
        assert self.backup_model, "ChatSettings.backup_model exists must be set"

    @property
    def primary_model(self) -> LLM_FN:
        return self.__primary_model

    @primary_model.setter
    def primary_model(self, value):
        if not isinstance(value, str):
            raise ValueError("Primary model must be a string")
        primary_model_llm_fn = get_llm_fn(value, "llm")
        logger.info(f"Primary model set to {primary_model_llm_fn.model_name}")
        self.__primary_model = primary_model_llm_fn

    @property
    def backup_model(self) -> LLM_FN:
        return self.__backup_model

    @backup_model.setter
    def backup_model(self, value):
        if not isinstance(value, str):
            raise ValueError("Backup model must be a string")
        backup_model_llm_fn = get_llm_fn(value, "llm")
        logger.info(f"Backup model set to {backup_model_llm_fn.model_name}")
        self.__backup_model = backup_model_llm_fn

    @property
    def enable_system_message(self) -> bool:
        return self.__enable_system_message

    @enable_system_message.setter
    def enable_system_message(self, value):
        if not isinstance(value, bool):
            raise ValueError("Enable system message must be a boolean")
        self.__enable_system_message = value

    @property
    def system_message(self) -> str:
        return self.__system_message

    @system_message.setter
    def system_message(self, value):
        if not value:
            raise ValueError("System message cannot be empty")
        if value in SYSTEM_MESSAGE_CODES:
            value = SYSTEM_MESSAGE_CODES[value]
        if not isinstance(value, str):
            raise ValueError("System message must be a string")
        logger.info(f"System message set to {value}")
        self.__system_message = value

    @property
    def stream(self) -> bool:
        return self.__stream

    @stream.setter
    def stream(self, value):
        if not isinstance(value, bool):
            raise ValueError("Stream must be a vool")
        self.__stream = value

    def to_dict(self):
        data = {
            "primary_model": self.primary_model.model_name,
            "backup_model": self.backup_model.model_name,
            "enable_system_message": self.enable_system_message,
            "system_message": self.system_message,
            "stream": self.stream,
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
            config_override: dict | None = None,
            config_file="settings.json",):

        config = utils.read_settings(config_file)

        if config_override is not None:
            if "RAG" in config_override:
                for key in config_override["RAG"]:
                    if key in config["RAG"]:
                        config["RAG"][key] = config_override["RAG"][key]
                        logger.info(f"Rag config key {key} overridden")
                    else:
                        raise ValueError(
                            f"Key {key} not found in RAG settings")
            if "chat" in config_override:
                for key in config_override["chat"]:
                    if key in config["chat"]:
                        config["chat"][key] = config_override["chat"][key]
                        logger.info(f"Chat config key {key} overridden")
                    else:
                        raise ValueError(
                            f"Key {key} not found in chat settings")
            if "optional" in config_override:
                for key in config_override["optional"]:
                    if key in config["optional"]:
                        config["optional"][key] = config_override["optional"][key]
                        logger.info(f"Optional config key {key} overridden")
                    else:
                        print("aaaah")
                        print(f"This key from object: {key}")
                        raise ValueError((f"Key {key} not found in config"))
        # Replace the LLM codes with the function name
        if config["chat"]["primary_model"] in MODEL_CODES:
            config["chat"]["primary_model"] = MODEL_CODES[config["chat"]
                                                          ["primary_model"]]
        if config["chat"]["backup_model"] in MODEL_CODES:
            config["chat"]["backup_model"] = MODEL_CODES[config["chat"]
                                                         ["backup_model"]]
        if config["RAG"]["rag_llm"] in MODEL_CODES:
            config["RAG"]["rag_llm"] = MODEL_CODES[config["RAG"]["rag_llm"]]

        self.rag_settings = RagSettings(**config["RAG"])
        self.chat_settings = ChatSettings(**config["chat"])

        # Validate the hyperparameters and optional settings
        self.hyperparameters = HyperparameterSchema(
            **config["hyperparameters"])

        # Validate the optional settings
        # # NOTE: Due to weird finickiness, the amnesia key does not fail if it is not a boolean
        # Exceptions are the strings "True" and "False"
        amnesia_value = config["optional"]["amnesia"]
        if isinstance(amnesia_value, str):
            logger.error("Amnesia value must be a boolean. Converting.")
            # Convert to boolean
            config["optional"]["amnesia"] = amnesia_value.lower() == "true"

        self.optional = OptionalSchema(**config["optional"])

        if self.chat_settings.enable_system_message:
            model_name = self.chat_settings.primary_model.model_name
            if model_name in MODEL_TO_SYSTEM_MSG:
                self.chat_settings.system_message = MODEL_TO_SYSTEM_MSG[model_name]
                print(f"Forced custom system message for model {model_name}.")

        self.__validate_rag_settings()
        self.save_active_settings()
        logger.info(f"Config initialized and set in {ACTIVE_JSON_FILE}.")

    def __validate_rag_settings(self):
        """
        Validate the RAG settings
        """
        if not path.exists(utils.DOCUMENTS_DIR):
            mkdir(utils.DOCUMENTS_DIR)

        filepath = utils.DATA_DIR / "manifest.json"
        if not path.exists(filepath):
            with open(filepath, "w") as f:
                # Make databases key
                f.write('{"databases": []}')

    def save_active_settings(self, filename=ACTIVE_JSON_FILE):
        """
        Save the current config to a JSON file
        """
        data = {
            "RAG": self.rag_settings.to_dict(),
            "chat": self.chat_settings.to_dict(),
            "optional": self.optional.model_dump(),
        }
        utils.save_config_as_json(data, filename)

    def __str__(self):
        return self.print_props()

    def __repr__(self):
        return self.print_props()

    def print_props(self) -> str:
        """
        Return the keys and values for each item in RAG settings and chat settings
        """
        def format_dict(d: dict):
            return "\n".join(f"{k}: {v}" for k, v in d.items())
        rag_settings = format_dict(self.rag_settings.to_dict())
        chat_settings = format_dict(self.chat_settings.to_dict())
        hyperparameters = format_dict(self.hyperparameters.model_dump())
        optional_settings = format_dict(self.optional.model_dump())

        return f"""Rag Config:
{rag_settings}

Chat Config:
{chat_settings}

Hyperparameters:
{hyperparameters}

Optional:
{optional_settings}
"""
