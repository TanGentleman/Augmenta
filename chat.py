from langchain.schema import HumanMessage, SystemMessage
from helpers import save_response_to_markdown_file, save_history_to_markdown_file, read_sample
from constants import DEFAULT_QUERY, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES, EXPLANATION_TEMPLATE
from config import SAVE_ONESHOT_RESPONSE, DEFAULT_TO_SAMPLE, LOCAL_MODEL_ONLY, EXPLAIN_EXCERPT
from classes import Config
from models import MODEL_DICT
from rag import vectorstore_from_inputs, get_rag_chain
from json import dump as json_dump
from json import load as json_load
from uuid import uuid4
# Get current time from datatime
from datetime import datetime

def get_current_time() -> str:
    return str(datetime.now().strftime("%Y-%m-%d"))

def get_chat_settings(config: Config):
    if LOCAL_MODEL_ONLY:
        assert config.chat_config["primary_model"] == "get_local_model", "LOCAL_MODEL_ONLY is set to True"
    # Note these models are uncalled get llm functions
    chat_settings = {
        "primary_model": MODEL_DICT[config.chat_config["primary_model"]],
        "backup_model": MODEL_DICT[config.chat_config["backup_model"]],
        "persistence_enabled": config.chat_config["persistence_enabled"],
        "enable_system_message": config.chat_config["enable_system_message"],
        "system_message": config.chat_config["system_message"],
        "rag_mode": config.chat_config["rag_mode"]
    }
    return chat_settings

def get_rag_settings(config: Config):
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

def get_retriever_from_settings(rag_settings, retriever_settings = None):
    # TODO:
    # Add error handling
    # Implement retriever_settings (like k value or score_threshold)
    try:
        vectorstore = vectorstore_from_inputs(rag_settings["inputs"], 
                                            rag_settings["method"], 
                                            rag_settings["embedding_model"], 
                                            rag_settings["collection_name"])
    except Exception as e:
        print(f'Error: {e}\n')
        print(f'Error creating vectorstore, check RAG settings in settings.json!')
        raise SystemExit
    retriever = vectorstore.as_retriever()
    return retriever

def update_manifest(rag_settings):
    data = {}
    with open('manifest.json', 'r') as f:
        data = json_load(f) 
    assert isinstance(data, list), "manifest.json is not a list"
    # assert that the id is unique
    for item in data:
        if item["collection_name"] == rag_settings["collection_name"]:
            print("No need to update manifest.json")
            return
    # get unique id
    unique_id = str(uuid4())
    print()
    try:
        model_name = str(rag_settings["embedding_model"].model)
    except:
        print("Could not get model name from embedding model")
        model_name = "Unknown model"
    manifest = {
        "id": unique_id,
        "collection_name": rag_settings["collection_name"],
        "metadata": {
            "embedding_model": model_name,
            "method": rag_settings["method"],
            "chunk_size": str(rag_settings["chunk_size"]),
            "chunk_overlap": str(rag_settings["chunk_overlap"]),
            "inputs": rag_settings["inputs"],
            "timestamp": get_current_time()
        }
    }
    data.append(manifest)
    with open('manifest.json', 'w') as f:
        json_dump(data, f, indent=4)
    print("Updated manifest.json")

def messages_to_strings(messages):
    messages = [msg.type.upper() + ": " + msg.content for msg in messages]
    return messages

def main(prompt=None, config=Config):
    # Note that by default with -np flag, main function reads prompt from sample.txt
    # TODO:
    # Add comments to explain the flow of the main function
    settings = get_chat_settings(config)
    rag_settings = get_rag_settings(config)
    chat_model = settings["primary_model"]()
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
    if not settings["persistence_enabled"]:
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
            except:
                print('pyperclip not installed, try pip install pyperclip')
                continue
            prompt = paste().strip()
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
            except:
                print('Error reading settings.json')
                raise SystemExit
            settings = get_chat_settings(config)
            rag_settings = get_rag_settings(config)
            chat_model = settings["primary_model"]()
            backup_model = None
            continue
        elif prompt == "save":
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
        # TODO:
        elif prompt == "info":
            # Print info about the model, # of exchanges, system message, etc.
            # Get name of the function
            print(f'Chat model name: {chat_model.model_name}')
            print(f'RAG mode: {rag_mode}')
            if rag_mode is False:
                print(f'Exchanges: {count}')
                if settings["enable_system_message"]:
                    print(f'System message: {settings["system_message"]}')
            continue
        elif prompt == "ingest":
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
            assert rag_chain is not None, "Set rag_chain after ingest command"
            response = rag_chain.invoke(prompt)
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

# Argparse implementation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Interactive chat')
    parser.add_argument('prompt', type=str, nargs='?', help='Prompt for the LLM')
    parser.add_argument('-np', '--not-persistent', action='store_true', help='Disable persistent chat mode')
    parser.add_argument('-rag', '--rag-mode', action='store_true', help='Enable RAG mode')
    args = parser.parse_args()
    config = Config()
    prompt = args.prompt
    chat_is_persistent = config.chat_config["persistence_enabled"]
    if args.not_persistent:
        chat_is_persistent = False
        config.chat_config["persistence_enabled"] = chat_is_persistent

    if prompt is None and chat_is_persistent is False:
        if DEFAULT_TO_SAMPLE:
            excerpt_as_prompt = read_sample()
            if EXPLAIN_EXCERPT:
                excerpt_as_prompt = EXPLANATION_TEMPLATE.format(excerpt=excerpt_as_prompt)
            prompt = excerpt_as_prompt
    
    if args.rag_mode:
        config.chat_config["rag_mode"] = True
    try:
        main(prompt, config=config)
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting.')
        raise SystemExit
