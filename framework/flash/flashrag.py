import json

from typeguard import typechecked
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

FINAL_PROMPT_TEMPLATE = "{goal}. Include keys {required} and ONLY output valid JSON, no preamble.\n\n```json"

# DEFAULT_GOAL = "Create Q/A pairs that comprehensively cover the main ideas of the paper's excerpts. The answers should be supported by the text."
DEFAULT_GOAL = "Create witty facts about penguins"
DEFAULT_REQUIRED_KEYS = ["question", "answer"]
DEFAULT_INPUTS = (DEFAULT_GOAL, DEFAULT_REQUIRED_KEYS)


class FlashcardSchema(BaseModel):
    term: str
    definition: str
    example: str

@typechecked
def unpack_inputs(inputs: tuple[str, list[str]]) -> tuple[str, str]:
    goal, required_keys = inputs
    required_string = ", ".join(required_keys)
    return goal, required_string

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

@typechecked
def is_output_valid(response_object: list[dict]) -> bool:
    """
    Check if the response object is valid
    """
    try:
        for obj in response_object:
            continue
            FlashcardSchema(**obj)
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True

def get_goal_and_keys() -> tuple[str, list[str]]:
    """
    Get user input for goal and required keys
    """
    goal_input = ""
    required_keys = []
    finished = False
    while not finished:
        goal_input = input("Enter a goal for the flashcards (leave empty for default inputs): ")
        # if empty response, use default inputs
        if not goal_input.strip():
            goal, required_keys = DEFAULT_INPUTS
            print("Using default inputs!")
            return goal, required_keys
        else:
            print("Goal set!")
            goal = goal_input
            while not finished:
                print("Enter a required key for the flashcards (leave empty to finish): ")
                key = input()
                if not key.strip():
                    finished = True
                    break
                required_keys.append(key)
    return goal, required_keys

@typechecked
def convert_inputs_to_prompt(inputs: tuple[str, list[str]]) -> str:
    """
    Convert the inputs to a prompt string
    """
    goal_string, required_keys = inputs
    required_string = ", ".join(required_keys or ["None"])
    prompt = FINAL_PROMPT_TEMPLATE.format(goal=goal_string, required=required_string)
    return prompt

def run_rag_chain(inputs: tuple[str, list[str]]) -> None:
    """
    Run the RAG chain
    """
    config = get_config()
    chatbot = Chatbot(config)
    chatbot.rag_chain = get_rag_chain(
        retriever=chatbot.retriever,
        llm=LLM(chatbot.config.rag_settings.rag_llm),
        format_fn=lambda docs: [doc.page_content for doc in docs],
        system_message=FLASHCARD_SIMPLE_SYSTEM_MESSAGE
    )
    chatbot = Chatbot()
    print("Implement Chatbot.chat() to use RAG mode!")
    print("Exiting!")
    raise SystemExit

def run_flashrag(set_goal = True) -> None:
    """
    Run the flashrag program
    """
    config = get_config()
    chatbot = Chatbot(config)
    inputs = (goal, required_keys)
    print("First, let's construct a prompt fitting for this task.")
    goal, required_keys = get_goal_and_keys()
    prompt = convert_inputs_to_prompt(inputs)
    messages = chatbot.chat(prompt)
    if not messages:
        exit()
    response = messages[0]
    response_string = response.content
    return response_string

def main():
    
    run_flashrag()

if __name__ == "__main__":
    main()
