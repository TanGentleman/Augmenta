import json
from classes import Config
from chains import get_rag_chain
from models.models import LLM
from chat import Chatbot
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from .flashcards import load_flashcards_from_json, display_flashcards, FLASH_DIR
from constants import FLASHCARD_SIMPLE_SYSTEM_MESSAGE
from pyperclip import copy, paste
from pathlib import Path


MODEL = "llama"
ENABLE_RAG = False
USE_SYSTEM = False
FLASHCARD_FILEPATH = FLASH_DIR / "flashcards.json"

RUSSIAN_PROMPT = "The context is notes from a Russian class. Create comprehensive flashcards with the keys term, definition, and example. term, definition in russian, example in English."
# prompt = "The context is notes from a Russian class. Create comprehensive flashcards with the keys term, definition, and example. term, definition in russian, example in English."
FINAL_PROMPT_TEMPLATE = "{goal}. Include keys {required} and ONLY output valid JSON, no preamble.\n\n```json"

# DEFAULT_GOAL = "Create Q/A pairs that comprehensively cover the main ideas of the paper's excerpts. The answers should be supported by the text."
DEFAULT_GOAL = "Create witty facts about penguins"
DEFAULT_REQUIRED_KEYS = ["question", "answer"]
DEFAULT_INPUTS = (DEFAULT_GOAL, DEFAULT_REQUIRED_KEYS)


class FlashcardSchema(BaseModel):
    term: str
    definition: str
    example: str


def unpack_inputs(inputs: tuple[str, list[str]]) -> tuple[str, str]:
    goal, required_keys = inputs
    required_string = ", ".join(required_keys)
    return goal, required_string


def get_config(inputs: tuple = DEFAULT_INPUTS) -> Config:
    goal, required_string = unpack_inputs(inputs)

    if USE_SYSTEM:
        chat_settings = {
            "primary_model": MODEL,
            "system_message": "flashcard",
            "enable_system_message": True
        }
        optional_settings = {
            "prompt_prefix": "",
            "prompt_suffix": "",
            "amnesia": True
        }
    else:
        print("Not using system message")
        chat_settings = {
            "primary_model": MODEL,
            "enable_system_message": False
        }
        optional_settings = {
            "prompt_prefix": FLASHCARD_SIMPLE_SYSTEM_MESSAGE + "\n\n",
            "prompt_suffix": "",
            "amnesia": True}
    config_override = {
        "chat": chat_settings,
        "optional": optional_settings
    }

    if ENABLE_RAG:
        # use the dict notation
        rag_settings = {
            "rag_mode": True,
            "rag_llm": "local",
            "collection_name": "flashrag_collection"
        }
        inputs = []
        inputs = ["discord.txt"]
        # inputs = ["russian-notes.pdf"]
        # inputs = ["russian.txt"]
        if inputs:
            print('Found inputs. RAG mode enabled')
            assert isinstance(inputs, list)
            assert all(isinstance(i, str) for i in inputs)
            rag_settings["inputs"] = inputs
        config_override["RAG"] = rag_settings

    config = Config(config_override=config_override)
    return config


def is_output_valid(response_object):
    if not isinstance(response_object, list):
        return False
    # NOTE: The schema is not validated for the JSON response, only ensuring
    # valid JSON.
    for item in response_object:
        try:
            assert isinstance(item, dict)
            continue
            FlashcardSchema(**item)
        except Exception as e:
            print(f"Error: {e}")
            return False
    return True


def main():
    AUTOMATIC = True
    MAX_COUNT = 3

    goal = ""
    required_keys = []

    if not ENABLE_RAG:
        user_input = input("Enter a goal for the flashcards (leave empty for default inputs): ")
        
        # if empty response, use default inputs
        if not user_input.strip():
            goal, required_keys = DEFAULT_INPUTS
        else:
            print("Goal set!")
            goal = user_input
            while True:
                print("Enter a required key for the flashcards (leave empty to finish): ")
                key = input()
                if not key.strip():
                    break
                required_keys.append(key)
    assert goal and required_keys
    inputs = (goal, required_keys)
    config = get_config(inputs=inputs)
    chatbot = Chatbot(config)
    if ENABLE_RAG:
        chatbot.rag_chain = get_rag_chain(
            retriever=chatbot.retriever,
            llm=LLM(chatbot.config.rag_settings.rag_llm),
            format_fn=lambda docs: [doc.page_content for doc in docs],
            system_message=FLASHCARD_SIMPLE_SYSTEM_MESSAGE
        )
        # prompt = RUSSIAN_PROMPT
        chatbot = Chatbot()
        # Do rag stuff here
        print("Implement Chatbot.chat() to use RAG mode!")
        print("Exiting!")
        raise SystemExit
    else:
        count = 0
        print("Automatic is true! Empty input => clipboard paste") if AUTOMATIC else None
        while count < MAX_COUNT:
            try:
                prompt = input("Type a prompt for the flashcards!: ")
            except KeyboardInterrupt:
                print("Exiting!")
                exit()
            if AUTOMATIC and not prompt.strip():
                print("Reading from clipboard!")
                prompt = paste().strip()
            response = chatbot.chat(prompt)
            if response is None:
                print("No response, error!")
                exit()
            response_string = response.content


if __name__ == "__main__":
    main()
    # get the keys as a dictionary from the FlashcardSchema
    # print(keys)
