from langchain.schema import HumanMessage, SystemMessage
from helpers import save_response_to_markdown_file, save_history_to_markdown_file, read_sample, read_settings
from constants import DEFAULT_QUERY, DEFAULT_SYSTEM_MESSAGE, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES, CODE_SYSTEM_MESSAGE
from config import PERSISTENCE_ENABLED, ENABLE_SYSTEM_MESSAGE, ACTIVE_MODEL_TYPE, TOGETHER_API_ENABLED, SAVE_ONESHOT_RESPONSE, DEFAULT_TO_SAMPLE
from config import CHOSEN_MODEL, BACKUP_MODEL
from classes import Config
from models import MODEL_DICT
from rag import vectorstore_from_inputs, get_rag_chain

def get_chat_settings(config: Config):
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
    rag_settings = {
        "collection_name": config.rag_config["collection_name"],
        "embedding_model": MODEL_DICT[config.rag_config["embedding_model"]],
        "method": config.rag_config["method"],
        "chunk_size": config.rag_config["chunk_size"],
        "chunk_overlap": config.rag_config["chunk_overlap"],
        "rag_llm": MODEL_DICT[config.rag_config["rag_llm"]],
        "inputs": config.rag_config["inputs"]
    }
    return rag_settings

FORMATTED_PROMPT = '''Explain the following text using comprehensive bulletpoints:
"""
{excerpt}
"""
'''
EXPLAIN_EXCERPT = False # If set to true, -np on sample.txt will format prompt like above

SYSTEM_MESSAGE = DEFAULT_SYSTEM_MESSAGE
# SYSTEM_MESSAGE = CODE_SYSTEM_MESSAGE

if TOGETHER_API_ENABLED:
    assert ACTIVE_MODEL_TYPE == "together", "Set ACTIVE_MODEL_TYPE to 'together' in config.py"
    TOGETHER_MODEL = CHOSEN_MODEL()
    LOCAL_MODEL = None
else:
    assert ACTIVE_MODEL_TYPE == "local", "Set ACTIVE_MODEL_TYPE to 'local' in config.py"
    TOGETHER_MODEL = None
    LOCAL_MODEL = CHOSEN_MODEL()

assert TOGETHER_MODEL is None or LOCAL_MODEL is None, "TOGETHER_MODEL and LOCAL_MODEL cannot both be enabled"

def main(prompt=None, config=Config):
    # Note that by default, main function reads prompt from sample.txt
    # model_type = ACTIVE_MODEL_TYPE
    # assert model_type in ["local", "together"]
    # chat_model = None
    # backup_model = None
    # if model_type == "together":
    #     assert TOGETHER_API_ENABLED, "Set TOGETHER_API_ENABLED in config.py"
    #     assert TOGETHER_MODEL is not None, "TOGETHER_MODEL is None"
    #     chat_model = TOGETHER_MODEL
    # elif model_type == "local":
    #     assert LOCAL_MODEL is not None, "LOCAL_MODEL is None"
    #     chat_model = LOCAL_MODEL
    settings = get_chat_settings(config)
    rag_settings = get_rag_settings(config)
    chat_model = settings["primary_model"]
    backup_model = None
    
    rag_mode = settings["rag_mode"]
    assert rag_mode is False, "RAG mode must initialize ingestion first"
    rag_chain = None
    retriever = None

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

    messages = []
    # if ENABLE_SYSTEM_MESSAGE:
    if settings["enable_system_message"]:
        messages.append(SystemMessage(content=settings["system_message"]))
    else:
        print('System message is disabled')
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
            prompt = paste()
        elif prompt == "read":
            # read from sample.txt
            prompt = read_sample()
        
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
            continue
        elif prompt in ["quit", "exit"]:
            print('Exiting.')
            return
        elif prompt == "save":
            save_response_to_markdown_file(messages[-1].content)
            print('Saved response to response.md')
            continue
        
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
            # Refresh chat model
            # NOTE:
            # This currently only adjusts LLM choices
            try:
                new_config = Config()
            except:
                print('Error reading settings.json')
                raise SystemExit
            settings = get_chat_settings(new_config)
            chat_model = settings["primary_model"]()
            backup_model = None
            # Other changes go here...reload if needed?
            rag_settings = get_rag_settings(new_config)
            continue
        # TODO:
        elif prompt == "saveall":
            # Save full chat history
            # Iterate through messages array and add to history.md
            save_history_to_markdown_file([msg.type + ": " + msg.content for msg in messages])
            continue
        # TODO:
        elif prompt == "info":
            # Print info about the model, # of exchanges, system message, etc.
            continue
        elif prompt == "ingest":
            # Ingest documents to vectorstore
            if retriever is not None:
                print('Vectorstore already exists, use "reg" to clear and reset')
                continue
            rag_mode = True
            # get rag settings
            print('Now using vectorstore and solo responses')
            vectorstore = vectorstore_from_inputs(rag_settings["inputs"], 
                                                  rag_settings["method"], 
                                                  rag_settings["embedding_model"](), 
                                                  rag_settings["collection_name"])
            retriever = vectorstore.as_retriever()
            # Save to manifest.json
            rag_chain = get_rag_chain(retriever, rag_settings["rag_llm"]())
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
        messages.append(HumanMessage(content=prompt))
        count += 1
        print(f'Fetching response #{count}!')
        if rag_mode:
            # RAG mode
            assert rag_chain is not None, "Set rag_chain after ingest command"
            response = rag_chain.invoke(prompt)
        else:
            try:
                response = chat_model.invoke(messages)
            except KeyboardInterrupt:
                print('Keyboard interrupt, aborting generation.')
                continue
        messages.append(response)
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
    # print all the attributes of the config object
    # print(config.__dict__)
    chat_is_persistent = PERSISTENCE_ENABLED
    if args.not_persistent:
        chat_is_persistent = False
        config.chat_config["persistence_enabled"] = chat_is_persistent

    if args.prompt is None and chat_is_persistent is False:
        if DEFAULT_TO_SAMPLE:
            excerpt_as_prompt = read_sample()
            if EXPLAIN_EXCERPT:
                excerpt_as_prompt = FORMATTED_PROMPT.format(excerpt=excerpt_as_prompt)
            args.prompt = excerpt_as_prompt
    
    if args.rag_mode:
        config.chat_config["rag_mode"] = True
    try:
        # main(args.prompt, persistent=chat_is_persistent)
        main(args.prompt, config=config)
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting.')
        raise SystemExit

# New architecture:
    # Rag commands: ingest, rag, reset
    # Regular chat
    # Type "ingest" as command
    # If no vectorstore, create one
    # Print "Now using vectorstore and solo responses"
    # Only command accepted is "rag or reset"
    # Rag command reformats the clipboard and responds to it
    # Reset returns to regular chat and clears vectorstore
    # Add restore as failsafe to retrieve last saved vectorstore