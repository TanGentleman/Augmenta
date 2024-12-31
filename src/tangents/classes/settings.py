from pydantic import BaseModel

class ChatSettings(BaseModel):
    """Configuration settings for chat interactions.
    
    Attributes:
        primary_model: The default language model to use (default: "llama")
        stream: Whether to stream responses (default: True)
        system_message: Custom system message to use (default: "")
        disable_system_message: Override to disable system messages (default: False)
    """
    primary_model: str = "llama"
    stream: bool = True
    system_message: str = ""
    disable_system_message: bool = False

class RAGSettings(BaseModel):
    """Settings for Retrieval-Augmented Generation functionality.
    
    Attributes:
        enabled: Whether RAG features are enabled (default: False)
    """
    enabled: bool = False
    # Add RAG-specific settings here

class Config(BaseModel):
    """Global configuration container.
    
    Attributes:
        chat_settings: Settings for chat interactions
        rag_settings: Settings for RAG functionality
    """
    chat_settings: ChatSettings = ChatSettings()
    rag_settings: RAGSettings = RAGSettings()