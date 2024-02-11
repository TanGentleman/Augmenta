from langchain.schema import HumanMessage, SystemMessage
from models import get_together_mix, get_together_quen, get_mistral
from helpers import save_response_to_markdown_file, read_sample
from constants import DEFAULT_QUERY, DEFAULT_SYSTEM_MESSAGE, MAX_CHARS_IN_PROMPT, MAX_CHAT_EXCHANGES
from config import PERSISTENCE_ENABLED, ENABLE_SYSTEM_MESSAGE, ACTIVE_MODEL_TYPE, TOGETHER_API_ENABLED, SAVE_ONESHOT_RESPONSE


FORMATTED_PROMPT = '''Explain the following text using comprehensive bulletpoints:
"""
{excerpt}
"""
'''
EXPLAIN_EXCERPT = False

SYSTEM_MESSAGE = DEFAULT_SYSTEM_MESSAGE

if TOGETHER_API_ENABLED:
    assert ACTIVE_MODEL_TYPE == "together", "Set ACTIVE_MODEL_TYPE to 'together' in config.py"
    TOGETHER_MODEL = get_together_mix()
    LOCAL_MODEL = None
else:
    assert ACTIVE_MODEL_TYPE == "local", "Set ACTIVE_MODEL_TYPE to 'local' in config.py"
    TOGETHER_MODEL = None
    LOCAL_MODEL = get_mistral()

assert TOGETHER_MODEL is None or LOCAL_MODEL is None, "TOGETHER_MODEL and LOCAL_MODEL cannot both be enabled"

def main(prompt=DEFAULT_QUERY, persistent=False):
    # Note that by default, main function reads prompt from sample.txt
    model = ACTIVE_MODEL_TYPE
    assert model in ["local", "together"]
    if model == "together":
        assert TOGETHER_API_ENABLED, "Set TOGETHER_API_ENABLED in config.py"
        assert TOGETHER_MODEL is not None, "TOGETHER_MODEL is None"
    elif model == "local":
        assert LOCAL_MODEL is not None, "LOCAL_MODEL is None"
    
    save_response = False
    count = 0
    max_exchanges = MAX_CHAT_EXCHANGES
    force_prompt = False
    forced_prompt = ""
    if not persistent:
        max_exchanges = 1
        persistent = True
        force_prompt = True
        forced_prompt = prompt
        if SAVE_ONESHOT_RESPONSE:
            save_response = True

    messages = []
    if ENABLE_SYSTEM_MESSAGE:
        messages.append(SystemMessage(content=SYSTEM_MESSAGE))
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
            from pyperclip import paste
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
        elif prompt == "exit":
            print('Exiting.')
            return
        elif prompt == "save":
            save_response_to_markdown_file(messages[-1].content)
            print('Saved response to response.md')
            continue

        # add input to messages
        messages.append(HumanMessage(content=prompt))
        count += 1
        print(f'Fetching response #{count}!')
        if model == "local":
            response = LOCAL_MODEL.invoke(messages)
            messages.append(response)
            print()

        elif model == "together":
            response = TOGETHER_MODEL.invoke(messages)
            messages.append(response)
            print()

    if save_response:
        save_response_to_markdown_file(messages[-1].content)
        print('Saved response to response.md')
    print('Reached max exchanges, exiting.')
    return

# if __name__ == "__main__":
#     from sys import argv
#     if len(argv) > 1:
#         prompt = argv[1]
#     else:
#         if FORMAT_PROMPT:
#             prompt = FORMATTED_PROMPT.format(excerpt=read_sample())
#         else:
#             # prompt = read_sample()
#             prompt = "read"
#     main(prompt, persistent=PERSISTENCE_ENABLED)

# Argparse implementation
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Chat with an AI')
    parser.add_argument('prompt', type=str, nargs='?', help='Prompt for the AI')
    parser.add_argument('--persistent', action='store_true', help='Persistent chat mode')
    args = parser.parse_args()
    if args.prompt is None:
        if EXPLAIN_EXCERPT:
            args.prompt = FORMATTED_PROMPT.format(excerpt=read_sample())
        else:
            args.prompt = "read"
    main(args.prompt, persistent=args.persistent)
