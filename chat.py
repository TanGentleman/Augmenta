from typing import Any
from langchain.schema import HumanMessage, SystemMessage
from helpers import save_response_to_markdown_file, save_history_to_markdown_file, read_sample, update_manifest
from constants import DEFAULT_QUERY, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES, EXPLANATION_TEMPLATE
from config import SAVE_ONESHOT_RESPONSE, DEFAULT_TO_SAMPLE, LOCAL_MODEL_ONLY, EXPLAIN_EXCERPT
from classes import Config
from models import MODEL_DICT, LLM, Embedder
from rag import vectorstore_from_inputs, get_rag_chain

DEFAULT_DOCS_USED = 6  # This will be moved to a value in settings.json
REFORMATTING_PROMPTS = ["paste", "read"]
COMMAND_LIST = [
    "del",
    "quit",
    "exit",
    "switch",
    "refresh",
    "save",
    "saveall",
    "info",
    "rag",
    "reg"]
# Current implemented commands


def get_chat_settings(config: Config):
    # Assert that all the keys are present in the chat_config
    assert "primary_model" in config.chat_config and config.chat_config[
        "primary_model"] in MODEL_DICT, "set valid primary_model in settings.json"
    assert "backup_model" in config.chat_config and config.chat_config[
        "backup_model"] in MODEL_DICT, "set valid backup_model in settings.json"

    assert "enable_system_message" in config.chat_config and isinstance(
        config.chat_config["enable_system_message"], bool), "set valid enable_system_message in settings.json"
    assert "system_message" in config.chat_config, "system_message key not found in chat_config"
    assert "rag_mode" in config.chat_config, "rag_mode key not found in chat_config"

    if LOCAL_MODEL_ONLY:
        assert config.chat_config["primary_model"] == "get_local_model", "LOCAL_MODEL_ONLY is set to True"
    # Note these models are uncalled get llm functions
    chat_settings = {
        "primary_model": MODEL_DICT[config.chat_config["primary_model"]],
        "backup_model": MODEL_DICT[config.chat_config["backup_model"]],
        "enable_system_message": config.chat_config["enable_system_message"],
        "system_message": config.chat_config["system_message"],
        "rag_mode": config.chat_config["rag_mode"]
    }
    return chat_settings


def get_rag_settings(config: Config) -> dict[str, Any]:
    # Note that these models are called and initialized here
    rag_settings = {
        "collection_name": config.rag_config["collection_name"],
        "embedding_model": MODEL_DICT[config.rag_config["embedding_model"]](),
        "method": config.rag_config["method"],
        "chunk_size": config.rag_config["chunk_size"],
        "chunk_overlap": config.rag_config["chunk_overlap"],
        "rag_llm": MODEL_DICT[config.rag_config["rag_llm"]](),
        "inputs": config.rag_config["inputs"]
    }
    return rag_settings


def get_retriever_from_settings(
        rag_settings: dict[str, Any], retriever_settings=None):
    # TODO:
    # Add error handling
    # Implement retriever_settings (like k value or score_threshold)
    try:
        vectorstore = vectorstore_from_inputs(rag_settings["inputs"],
                                              rag_settings["method"],
                                              rag_settings["embedding_model"],
                                              rag_settings["collection_name"],
                                              rag_settings["chunk_size"],
                                              rag_settings["chunk_overlap"])
        # TODO: Add excerpt_count to settings.json
    except Exception as e:
        print(f'Error: {e}\n')
        print(f'Error creating vectorstore, check RAG settings in settings.json!')
        raise SystemExit
    # retrieval settings
    search_kwargs = {}
    if retriever_settings is not None:
        search_kwargs["k"] = retriever_settings["document_count"]
        if retriever_settings["filter"] is not None:
            search_kwargs["filter"] = retriever_settings["filter_metadata"]
            # 'filter': {'paper_title':'GPT-4 Technical Report'}
    else:
        search_kwargs["k"] = DEFAULT_DOCS_USED
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever


def messages_to_strings(messages):
    messages = [msg.type.upper() + ": " + msg.content for msg in messages]
    return messages


def main(prompt=None, config=Config, persistence_enabled=True):
    # Note that by default with -np flag, main function reads prompt from sample.txt
    # TODO:
    # Add comments to explain the flow of the main function
    settings = get_chat_settings(config)
    rag_settings = get_rag_settings(config)
    chat_model: LLM = settings["primary_model"]()
    backup_model = None
    rag_chain = None
    retriever = None
    rag_mode = settings["rag_mode"]
    # assert rag_mode is False, "RAG mode must initialize ingestion first"
    if rag_mode:
        # pass
        print("RAG mode activated, using vectorstore and solo responses")
        retriever = get_retriever_from_settings(rag_settings)
        # Save to manifest.json
        rag_chain = get_rag_chain(retriever, rag_settings["rag_llm"])
        update_manifest(rag_settings)
    messages = []
    if settings["enable_system_message"]:
        messages.append(SystemMessage(content=settings["system_message"]))
    else:
        print('System message is disabled')
    save_response = False
    count = 0
    max_exchanges = MAX_CHAT_EXCHANGES
    force_prompt = False
    forced_prompt = ""
    # if not persistent:
    if not persistence_enabled:
        if prompt is None:
            prompt = DEFAULT_QUERY
        max_exchanges = 1
        force_prompt = True
        forced_prompt = prompt
        if SAVE_ONESHOT_RESPONSE:
            save_response = True
    elif prompt is not None:
        # This is a temporary solution
        force_prompt = True
        forced_prompt = prompt

    while count < max_exchanges:
        # get input
        if force_prompt:
            prompt = forced_prompt
            force_prompt = False
            forced_prompt = ""
        else:
            prompt = input("Enter your query: ")

        if prompt == "paste":
            # paste from clipboard
            try:
                from pyperclip import paste
            except ImportError:
                print('pyperclip not installed, try pip install pyperclip')
                continue
            from reformatter import reformat
            prompt = paste().strip()
            prompt = reformat(prompt)
            # NOTE: Now reformatting the prompt from the clipboard

        elif prompt == "read":
            # read from sample.txt
            prompt = read_sample()
            # No continue here

        if len(prompt) > MAX_CHARS_IN_PROMPT:
            print(f'Input too long, max characters is {MAX_CHARS_IN_PROMPT}')
            continue
        if not prompt.strip():
            print('No input given, try again')
            continue
        elif prompt == "del":
            # delete last message
            if len(messages) < 2:
                print('No messages to delete')
                continue
            messages.pop()
            messages.pop()
            print('Deleted last exchange')
            count -= 1
            continue
        elif prompt in ["quit", "exit"]:
            print('Exiting.')
            return

        elif prompt == "switch":
            # Switch to backup LLM
            if backup_model is None:
                backup_model = chat_model
                print('Switching to backup model')
                # chat_model = BACKUP_MODEL()
                chat_model = settings["backup_model"]()
            else:
                print('Switching back to primary model')
                # Switch chat model and backup model
                chat_model, backup_model = backup_model, chat_model
            continue
        elif prompt == "refresh":
            # Refresh chat model, system message, and RAG settings
            try:
                config = Config()
            except BaseException:
                print('Error reading settings.json')
                raise SystemExit
            settings = get_chat_settings(config)
            rag_settings = get_rag_settings(config)
            chat_model = settings["primary_model"]()
            backup_model = None
            if rag_mode:
                assert retriever is not None, "Retriever must be initialized"
                rag_chain = get_rag_chain(retriever, rag_settings["rag_llm"])
            continue
        elif prompt == "save":
            if len(messages) < 2:
                print('No responses to save')
                continue
            save_response_to_markdown_file(messages[-1].content)
            print('Saved response to response.md')
            continue
        elif prompt == "saveall":
            # Save full chat history
            # Iterate through messages array and add to history.md
            message_strings = messages_to_strings(messages)
            save_history_to_markdown_file(message_strings)
            print(f'Saved {count} exchanges to history.md')
            continue
        elif prompt == "info":
            # Print info about the model, # of exchanges, system message, etc.
            # Get name of the function
            model_name = "Unknown model"
            print(f'RAG mode: {rag_mode}')
            if rag_mode is False:
                print(f'Exchanges: {count}')
                if settings["enable_system_message"]:
                    print(f'System message: {settings["system_message"]}')
                try:
                    model_name = chat_model.model_name if hasattr(
                        chat_model, 'model_name') else chat_model.model
                    print(f'LLM: {model_name}')
                except AttributeError:
                    print('Could not get model name from chat model')
                continue
            else:
                try:
                    rag_model = rag_settings["rag_llm"]
                    model_name = rag_model.model_name if hasattr(
                        rag_model, 'model_name') else rag_model.model
                    print(f'LLM: {model_name}')
                except AttributeError:
                    print('Could not get model name from chat model')
                # RAG mode
                print(f'Using vectorstore: {rag_settings["collection_name"]}')
                print(f'Inputs: {rag_settings["inputs"]}')
                print(
                    f'Embedding model: {rag_settings["embedding_model"].model}')
                print(f'Method: {rag_settings["method"]}')
                print(f'Chunk size: {rag_settings["chunk_size"]}')
                print(f'Chunk overlap: {rag_settings["chunk_overlap"]}')
                continue
        elif prompt == "rag":
            # Ingest documents to vectorstore
            if retriever is not None:
                print('Retriever already exists, use "reg" to clear and reset first')
                continue
            rag_mode = True
            # get rag settings
            print('Now using vectorstore and solo responses')
            retriever = get_retriever_from_settings(rag_settings)
            # Save to manifest.json
            rag_chain = get_rag_chain(retriever, rag_settings["rag_llm"])
            update_manifest(rag_settings)
            continue
        elif prompt == "reg":
            # Return to regular chat
            rag_mode = False
            retriever = None
            rag_chain = None
            print('Returning to regular chat, resetting message history')
            messages = messages[:1]
            continue
        # add input to messages list and get response

        if rag_mode:
            # RAG mode
            assert rag_chain is not None, "Set rag_chain after rag command"
            try:
                response = rag_chain.invoke(prompt)
            except KeyboardInterrupt:
                print('Keyboard interrupt, aborting generation.')
                continue
            except Exception as e:
                print(f'Error!: {e}')
                continue
        else:
            messages.append(HumanMessage(content=prompt))
            count += 1
            print(f'Fetching response #{count}!')
            try:
                response = chat_model.invoke(messages)
                messages.append(response)
            except KeyboardInterrupt:
                print('Keyboard interrupt, aborting generation.')
                continue
            except Exception as e:
                print(f'Error!: {e}')
                continue
        print()

    if save_response:
        save_response_to_markdown_file(messages[-1].content)
        print('Saved response to response.md')
    print('Reached max exchanges, exiting.')
    return


class Chatbot:
    def __init__(self, config):
        self.config = config
        self.settings = self.get_chat_settings()
        self.rag_settings = self.get_rag_settings()
        self.chat_model = self.settings["primary_model"]()
        self.backup_model = None
        self.rag_chain = None
        self.retriever = None
        self.messages = []
        self.save_response = False
        self.count = 0
        self.rag_mode = self.settings["rag_mode"]
        self.exit = False

    def refresh_config(self, config: Config = None):
        if config is None:
            config = self.config
        self.config = config
        self.settings = self.get_chat_settings()
        self.rag_settings = self.get_rag_settings()
        self.chat_model = self.settings["primary_model"]()
        self.backup_model = None
        if self.rag_mode:
            # TODO: Implement re-ingesting documents here, for now use same docs
            # Maybe just get a new retriever at this step
            assert self.retriever is not None, "Retriever must be initialized"
            self.rag_chain = get_rag_chain(
                self.retriever, self.rag_settings["rag_llm"])

    def get_chat_settings(self):
        assert "primary_model" in self.config.chat_config and self.config.chat_config[
            "primary_model"] in MODEL_DICT, "set valid primary_model in settings.json"
        assert "backup_model" in self.config.chat_config and self.config.chat_config[
            "backup_model"] in MODEL_DICT, "set valid backup_model in settings.json"
        assert "enable_system_message" in self.config.chat_config and isinstance(
            self.config.chat_config["enable_system_message"], bool), "set valid enable_system_message in settings.json"
        assert "system_message" in self.config.chat_config, "system_message key not found in chat_config"
        assert "rag_mode" in self.config.chat_config, "rag_mode key not found in chat_config"
        if LOCAL_MODEL_ONLY:
            assert self.config.chat_config["primary_model"] == "get_local_model", "LOCAL_MODEL_ONLY is set to True"
        chat_settings = {
            "primary_model": MODEL_DICT[self.config.chat_config["primary_model"]],
            "backup_model": MODEL_DICT[self.config.chat_config["backup_model"]],
            "enable_system_message": self.config.chat_config["enable_system_message"],
            "system_message": self.config.chat_config["system_message"],
            "rag_mode": self.config.chat_config["rag_mode"]
        }
        return chat_settings

    def get_rag_settings(self):
        rag_settings = {
            "collection_name": self.config.rag_config["collection_name"],
            "embedding_model": MODEL_DICT[self.config.rag_config["embedding_model"]](),
            "method": self.config.rag_config["method"],
            "chunk_size": self.config.rag_config["chunk_size"],
            "chunk_overlap": self.config.rag_config["chunk_overlap"],
            "rag_llm": MODEL_DICT[self.config.rag_config["rag_llm"]](),
            "inputs": self.config.rag_config["inputs"]
        }
        return rag_settings

    def get_retriever_from_settings(self, retriever_settings=None):
        try:
            vectorstore = vectorstore_from_inputs(
                self.rag_settings["inputs"],
                self.rag_settings["method"],
                self.rag_settings["embedding_model"],
                self.rag_settings["collection_name"],
                self.rag_settings["chunk_size"],
                self.rag_settings["chunk_overlap"])
        except Exception as e:
            print(f'Error: {e}\n')
            print(f'Error creating vectorstore, check RAG settings in settings.json!')
            raise SystemExit
        search_kwargs = {}
        if retriever_settings is not None:
            search_kwargs["k"] = retriever_settings["document_count"]
            if retriever_settings["filter"] is not None:
                search_kwargs["filter"] = retriever_settings["filter_metadata"]
        else:
            search_kwargs["k"] = DEFAULT_DOCS_USED
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        return retriever

    def messages_to_strings(self, messages):
        messages = [msg.type.upper() + ": " + msg.content for msg in messages]
        return messages

    def prompt_from_clipboard(self):
        try:
            from pyperclip import paste
        except ImportError:
            print('pyperclip not installed, try pip install pyperclip')
            return None
        prompt = paste().strip()
        return prompt

    def command_handler(self, prompt):
        assert prompt in COMMAND_LIST, "Invalid command"
        if prompt == "del":
            if len(self.messages) < 2:
                print('No messages to delete')
                return
            self.messages.pop()
            self.messages.pop()
            print('Deleted last exchange')
            self.count -= 1
            return
        elif prompt == "quit" or prompt == "exit":
            print('Exiting.')
            self.exit = True
            return
        elif prompt == "switch":
            # TODO: Create a method for switching models
            if self.backup_model is None:
                self.backup_model = self.chat_model
                print('Switching to backup model')
                try:
                    self.chat_model = self.settings["backup_model"]()
                except BaseException:
                    print('Error switching to backup model')
                    raise SystemExit
        elif prompt == "refresh":
            self.refresh_config()
            return
        elif prompt == "save":
            if len(self.messages) < 2:
                print('No responses to save')
                return
            save_response_to_markdown_file(self.messages[-1].content)
            # TODO: Modify this to work with RAG mode
            print('Saved response to response.md')
            return
        elif prompt == "saveall":
            message_strings = self.messages_to_strings(self.messages)
            save_history_to_markdown_file(message_strings)
            print(f'Saved {self.count} exchanges to history.md')
            return
        elif prompt == "info":
            model_name = "Unknown model"
            print(f'RAG mode: {self.rag_mode}')
            if self.rag_mode is False:
                print(f'Exchanges: {self.count}')
                if self.settings["enable_system_message"]:
                    print(f'System message: {self.settings["system_message"]}')
                try:
                    # TODO: Implement better model name retrieval logic
                    # Such attributes should be consistent in LLM interface
                    model_name = self.chat_model.model_name if hasattr(
                        self.chat_model, 'model_name') else self.chat_model.model
                    print(f'LLM: {model_name}')
                except AttributeError:
                    print('Could not get model name from chat model')
                return
            else:
                try:
                    rag_model = self.rag_settings["rag_llm"]
                    model_name = rag_model.model_name if hasattr(
                        rag_model, 'model_name') else rag_model.model
                    print(f'LLM: {model_name}')
                except AttributeError:
                    print('Could not get model name from chat model')
                print(
                    f'Using vectorstore: {self.rag_settings["collection_name"]}')
                print(f'Inputs: {self.rag_settings["inputs"]}')
                print(
                    f'Embedding model: {self.rag_settings["embedding_model"].model}')
                print(f'Method: {self.rag_settings["method"]}')
                print(f'Chunk size: {self.rag_settings["chunk_size"]}')
                print(f'Chunk overlap: {self.rag_settings["chunk_overlap"]}')
                return
        elif prompt == "rag":
            if self.retriever is not None:
                print('Retriever already exists, use "reg" to clear and reset first')
                return
            self.rag_mode = True
            self.retriever = self.get_retriever_from_settings()
            self.rag_chain = get_rag_chain(
                self.retriever, self.rag_settings["rag_llm"])
            update_manifest(self.rag_settings)
            return
        elif prompt == "reg":
            self.rag_mode = False
            self.retriever = None
            self.rag_chain = None
            print('Returning to regular chat, resetting message history')
            self.messages = self.messages[:1]
            return
        else:
            print('Invalid command: ', prompt)
            return

    def get_chat_response(self, prompt: str):
        self.messages.append(HumanMessage(content=prompt))
        print(f'Fetching response #{self.count + 1}!')
        try:
            response = self.chat_model.invoke(self.messages)
            self.messages.append(response)
        except KeyboardInterrupt:
            print('Keyboard interrupt, aborting generation.')
            self.messages.pop()
            return
        except Exception as e:
            print(f'Error!: {e}')
            self.messages.pop()
            return
        self.count += 1
        return response

    def get_rag_response(self, prompt: str):
        try:
            response = self.rag_chain.invoke(prompt)
        except KeyboardInterrupt:
            print('Keyboard interrupt, aborting generation.')
            return
        except Exception as e:
            print(f'Error!: {e}')
            return
        return response

    def chat(self, prompt=None, persistence_enabled=True):
        force_prompt = False
        forced_prompt = ""
        save_response = False
        max_exchanges = MAX_CHAT_EXCHANGES
        if prompt is not None:
            force_prompt = True
            forced_prompt = prompt
        if persistence_enabled is False:
            if prompt is None:
                prompt = DEFAULT_QUERY
            max_exchanges = 1
            if SAVE_ONESHOT_RESPONSE:
                save_response = True
        while self.exit is False:
            if self.count >= max_exchanges:
                print(f'Max exchanges reached: {self.count}')
                self.exit = True
                continue
            if force_prompt:
                prompt = forced_prompt
                force_prompt = False
                forced_prompt = ""
            else:
                prompt = input("Enter your query: ")
            if prompt in REFORMATTING_PROMPTS:
                if prompt == "paste":
                    prompt = self.prompt_from_clipboard()
                    if not prompt:
                        print('No text in clipboard! Try again.')
                        continue
                elif prompt == "read":
                    # TODO: Add some checks for string content of sample.txt
                    prompt = read_sample()
                # Do not continue here
            if not prompt.strip():
                print('No input given, try again')
                continue
            if len(prompt) > MAX_CHARS_IN_PROMPT:
                print(
                    f'Input too long, max characters is {MAX_CHARS_IN_PROMPT}')
                continue

            if prompt in COMMAND_LIST:
                self.command_handler(prompt)
                continue
            # Generate response
            # TODO: Move this to a separate method
            if self.rag_mode:
                response = self.get_rag_response(prompt)
            else:
                response = self.get_chat_response(prompt)
            print()
        if save_response:
            save_response_to_markdown_file(self.messages[-1].content)
            print('Saved response to response.md')


# Argparse implementation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Interactive chat')
    parser.add_argument(
        'prompt',
        type=str,
        nargs='?',
        help='Prompt for the LLM')
    parser.add_argument(
        '-np',
        '--not-persistent',
        action='store_true',
        help='Disable persistent chat mode')
    parser.add_argument(
        '-rag',
        '--rag-mode',
        action='store_true',
        help='Enable RAG mode')
    args = parser.parse_args()
    config = Config()
    prompt = args.prompt

    persistence_enabled = not args.not_persistent
    if prompt is None and persistence_enabled is False:
        if DEFAULT_TO_SAMPLE:
            excerpt_as_prompt = read_sample()
            if EXPLAIN_EXCERPT:
                excerpt_as_prompt = EXPLANATION_TEMPLATE.format(
                    excerpt=excerpt_as_prompt)
            prompt = excerpt_as_prompt

    if args.rag_mode:
        config.chat_config["rag_mode"] = True

    try:
        # main(prompt, config=config, persistence_enabled=persistence_enabled)
        chatbot = Chatbot(config)
        chatbot.chat(prompt, persistence_enabled=persistence_enabled)
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting.')
        raise SystemExit
