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

FLASHCARD_FILEPATH = FLASH_DIR / "flashcards.json"

# DEFAULT_PROMPT = "Create a list of JSON flashcards with keys: term, definition, example for the Russian terms. The term and example sentence should appropriately be in Russian."
DEFAULT_PROMPT = "The context is notes from a Russian class. Create comprehensive flashcards with the keys term, definition, and example. term, definition in russian, example in English."
# prompt = "The context is notes from a Russian class. Create comprehensive flashcards with the keys term, definition, and example. term, definition in russian, example in English."
FLASHCARD_SYSTEM_MESSAGE = """You are Flashcard AI. Use the document excerpts to generate a list of JSON flashcards. Example output:
[
    {
        "term": "Python",
        "definition": "A high-level, interpreted programming language with a focus on code readability.",
        "example": "print('Hello, World!')"
    },
    {
        "term": "JavaScript",
        "definition": "A high-level, dynamic, and interpreted programming language that is primarily used for building web applications and adding interactive elements to websites.",
        "example": "console.log('Hello, World!');"
    }
]"""

ENABLE_RAG = False

SUFFIX_INSTRUCTION_TEMPLATE = "\nCreate a {NOUN} with the above context. Include keys {REQUIRED} and ONLY output valid JSON, no preamble.\n\n```json"

class FlashcardSchema(BaseModel):
    term: str
    definition: str
    example: str

USE_SYSTEM = False
def get_config() -> Config:
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
            "prompt_suffix": SUFFIX_INSTRUCTION_TEMPLATE.format(NOUN="flashcard", REQUIRED="(all)"),
            "amnesia": True
        }
    config_override = {
        "chat": chat_settings,
        "optional": optional_settings
    }
        
    if ENABLE_RAG:
        # use the dict notation
        rag_settings = {
            "rag_mode": True,
            "rag_llm": "get_local_model",
            "collection_name": "russian-flashcards-text_collection"
        }
        inputs = []
        # inputs = ["discord.txt"]
        # inputs = ["russian-notes.pdf"]
        inputs = ["russian.txt"]
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
    config = get_config()
    chatbot = Chatbot(config)
    if ENABLE_RAG:
        chatbot.rag_chain = get_rag_chain(
            retriever=chatbot.retriever,
            llm=LLM(chatbot.config.rag_settings.rag_llm).llm,
            format_fn=lambda docs: [doc.page_content for doc in docs],
            system_message=FLASHCARD_SYSTEM_MESSAGE
        )
        # prompt = input("Enter the prompt: ")
        prompt = DEFAULT_PROMPT
        chatbot = Chatbot()
        # Do rag stuff here
        return
    max_count = 3
    count = 0
    AUTOMATIC = True
    print("Automatic is true! Empty input => clipboard paste") if AUTOMATIC else None
    while count < max_count:
        try:
            prompt = input("Type a prompt for the flashcards!: ")
        except KeyboardInterrupt:
            print("Exiting!")
            exit()
        if AUTOMATIC and not prompt.strip():
            print("Reading from clipboard!")
            prompt = paste().strip()
        response = chatbot.get_chat_response(prompt)
        if response is None:
            print("No response, error!")
            exit()
        response_string = response.content
        # Check if the JSON output is valid
        # response_object = JsonOutputParser().parse(response_string)
        # if not is_output_valid(response_object):
        #     raise ValueError("JSON output is not valid")

        # Save the response to flashcards.json
        # with open(FLASHCARD_FILEPATH, 'w') as file:
        #     json.dump(response_object, file, indent=4)

        # # Load the flashcards and display them
        # flashcards, keys, styles = load_flashcards_from_json(
        #     FLASHCARD_FILEPATH)
        # #
        # display_flashcards(flashcards, "Flashcard")
        # count += 1
        # prompt = None
    # from pyperclip import copy
    # copy(str(response_object))


if __name__ == "__main__":
    main()
    # get the keys as a dictionary from the FlashcardSchema
    # print(keys)
