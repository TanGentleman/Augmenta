from pydantic import BaseModel


class ChatSettings(BaseModel):
    """Configuration settings for chat interactions.

    Attributes:
        primary_model: The default language model to use (default: "llama")
        stream: Whether to stream responses (default: True)
        system_message: Custom system message to use (default: "")
        disable_system_message: Override to disable system messages (default: False)
    """

    primary_model: str = 'llama'
    stream: bool = True
    system_message: str = ''
    disable_system_message: bool = False


class RAGSettings(BaseModel):
    """Settings for Retrieval-Augmented Generation functionality.

    Attributes:
        enabled: Whether RAG features are enabled (default: False)
    """

    enabled: bool = False
    # Add RAG-specific settings here


class WorkflowSettings(BaseModel):
    """Settings for workflow execution limits and constraints.

    Attributes:
        max_mutations: Maximum state mutations before failing (default: 50)
        max_actions: Maximum actions per task before failing (default: 5)
    """

    max_mutations: int = 50
    max_actions: int = 5


class Config(BaseModel):
    """Global configuration container.

    Attributes:
        chat_settings: Settings for chat interactions
        rag_settings: Settings for RAG functionality
        workflow_settings: Settings for workflow execution
    """

    chat_settings: ChatSettings = ChatSettings()
    rag_settings: RAGSettings = RAGSettings()
    workflow_settings: WorkflowSettings = WorkflowSettings()
