from langchain.schema import HumanMessage, SystemMessage
from helpers import save_response_to_markdown_file, save_history_to_markdown_file, read_sample
from constants import DEFAULT_QUERY, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES
from config import SAVE_ONESHOT_RESPONSE, DEFAULT_TO_SAMPLE, LOCAL_MODEL_ONLY
from classes import Config
from models import MODEL_DICT
from rag import vectorstore_from_inputs, get_rag_chain

def get_chat_settings(config: Config):
    if LOCAL_MODEL_ONLY:
        assert config.chat_config["primary_model"] == "get_local_model", "LOCAL_MODEL_ONLY is set to True"
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

def get_retriever_from_settings(rag_settings):
    # TODO:
    # Add error handling
    vectorstore = vectorstore_from_inputs(rag_settings["inputs"], 
                                          rag_settings["method"], 
                                          rag_settings["embedding_model"](), 
                                          rag_settings["collection_name"])
    retriever = vectorstore.as_retriever()
    return retriever


FORMATTED_PROMPT = '''Explain the following text using comprehensive bulletpoints:
"""
{excerpt}
"""
'''
EXPLAIN_EXCERPT = False # If set to true, -np on sample.txt will format prompt like above

def main(prompt=None, config=Config):
    # Note that by default with -np flag, main function reads prompt from sample.txt
    # TODO:
    # Add comments to explain the flow of the main function
    settings = get_chat_settings(config)
    rag_settings = get_rag_settings(config)
    chat_model = settings["primary_model"]
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
        rag_chain = get_rag_chain(retriever, rag_settings["rag_llm"]())

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
            prompt = paste().strip()
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
            messages = []
            if settings["enable_system_message"]:
                messages.append(SystemMessage(content=settings["system_message"]))
            rag_settings = get_rag_settings(new_config)
            continue
        # TODO:
        elif prompt == "saveall":
            # Save full chat history
            # Iterate through messages array and add to history.md
            save_history_to_markdown_file([msg.type.upper() + ": " + msg.content for msg in messages])
            continue
        # TODO:
        elif prompt == "info":
            # Print info about the model, # of exchanges, system message, etc.
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
    prompt = args.prompt
    chat_is_persistent = config.chat_config["persistence_enabled"]
    if args.not_persistent:
        chat_is_persistent = False
        config.chat_config["persistence_enabled"] = chat_is_persistent

    if prompt is None and chat_is_persistent is False:
        if DEFAULT_TO_SAMPLE:
            excerpt_as_prompt = read_sample()
            if EXPLAIN_EXCERPT:
                excerpt_as_prompt = FORMATTED_PROMPT.format(excerpt=excerpt_as_prompt)
            prompt = excerpt_as_prompt
    
    if args.rag_mode:
        config.chat_config["rag_mode"] = True
    try:
        # main(args.prompt, persistent=chat_is_persistent)
        main(prompt, config=config)
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting.')
        raise SystemExit
