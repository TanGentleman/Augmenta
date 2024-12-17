from langchain_core.messages import SystemMessage, HumanMessage

def insert_system_message(messages: list, system_content: str) -> None:
    """Insert or update system message at start of messages list."""
    if not messages:
        messages.append(SystemMessage(content=system_content))
    elif isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_content)
    else:
        messages.insert(0, SystemMessage(content=system_content))

def remove_last_message(messages: list) -> None:
    """Remove last non-system message from messages list."""
    if not messages:
        return
        
    if len(messages) == 1 and isinstance(messages[0], SystemMessage):
        return
        
    for i in range(len(messages)-1, -1, -1):
        if not isinstance(messages[i], SystemMessage):
            messages.pop(i)
            break

def clear_messages(messages: list, preserve_system: bool = True) -> None:
    """Clear messages list, optionally preserving system message."""
    if not messages:
        return
        
    if preserve_system and isinstance(messages[0], SystemMessage):
        system_msg = messages[0]
        messages.clear()
        messages.append(system_msg)
    else:
        messages.clear()

def get_last_user_message(messages: list) -> str:
    """Get the last user message from the messages list."""
    message = messages[-1]
    if isinstance(message, HumanMessage):
        return str(message.content)
    raise ValueError("No user message found!")
