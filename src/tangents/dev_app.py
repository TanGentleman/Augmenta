"""
Augmenta Chat Interface - Multi-Agent Version

A Gradio frontend supporting multiple graph workflows with different agent types.

## Adding New Agents

To add a new agent type, follow these steps:

1. **Create Agent Configuration**: Add a new entry to `GRAPH_CONFIGS` with:
   - `graph_type`: Unique identifier for your agent
   - `state_dict_func`: Function that returns initial state for your agent
   - `title`, `description`, `placeholder`, `emoji`: UI display properties
   - `validate_chat_history`: Custom validation function for chat history
   - `get_action_config`: Method to configure agent-specific actions

2. **Implement State Function**: Create a function that returns the initial state dict
   for your agent (similar to `get_default_state_dict` or `get_planning_state_dict`)

3. **Add Validation Logic**: Implement chat history validation specific to your agent's
   requirements in the `validate_chat_history` method

4. **Configure Actions**: Customize the action configuration in `get_action_config`
   method to include agent-specific parameters

5. **Test**: The agent will automatically appear in the dropdown and be available for use

Example:
```python
'my_agent': GraphConfig(
    graph_type='my_agent',
    state_dict_func=get_my_agent_state_dict,
    title="🔥 My Custom Agent",
    description="Description of what this agent does",
    placeholder="Type your message for the custom agent...",
    emoji="🔥",
)
```

Features:
- Real-time streaming responses
- Persistent chat sessions with checkpointing
- Human-In-The-Loop capabilities
- Dynamic agent switching
- Agent-specific validation and configuration
"""

import asyncio
import logging
from uuid import uuid4
from typing import Optional, Dict, Any, Generator, Literal, Callable, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import gradio as gr
from langgraph.types import Command as ResumeCommand

from tangents.classes.actions import Status
from tangents.tan_graph import create_workflow, get_planning_state_dict, get_default_state_dict
from tangents.experimental.utils import get_gradio_state_dict
from tangents.utils.chains import fast_get_llm
from tangents.utils.task_utils import get_task

# Application Configuration
MODEL_NAME = 'nebius/meta-llama/Llama-3.3-70B-Instruct'
SYSTEM_MESSAGE = 'You are a helpful assistant who responds playfully.'
POLLING_INTERVAL = 0.05
RECURSION_LIMIT = 50

# Planning Agent Configuration
PROPOSAL_SYSTEM_PROMPT = 'List the HTTP Functions, what they do, and a basic curl command for each one.'
REVISION_SYSTEM_PROMPT = 'Use the user input to revise the plan.'

# Only set true
DEFINE_CHAIN_IN_STATE = True

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class GraphConfig:
    """Configuration for different agent/graph types."""
    graph_type: str
    state_dict_func: callable
    title: str
    description: str
    placeholder: str
    emoji: str
    
    def validate_chat_history(self, chat_history: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Validate chat history for this agent type.
        
        Args:
            chat_history: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.graph_type == 'default':
            return self._validate_alternating_chat(chat_history)
        elif self.graph_type == 'planning':
            return self._validate_planning_chat(chat_history)
        else:
            raise ValueError("Invalid graph type")
            # Default validation for new agent types
            # return self._validate_basic_chat(chat_history)
    
    def _validate_alternating_chat(self, chat_history: List[Dict[str, str]]) -> Tuple[bool, str]:
        """Validate that chat alternates between user and assistant, ending with user."""
        if not chat_history:
            return False, "Chat history is empty"
        
        if chat_history[-1]["role"] != "user":
            return False, "Last message must be from user in chat mode"
        
        # Check alternating pattern
        for i in range(len(chat_history) - 1):
            current_role = chat_history[i]["role"]
            next_role = chat_history[i + 1]["role"]
            if current_role == next_role:
                return False, f"Chat messages must alternate between user and assistant"
        
        return True, ""
    
    def _validate_planning_chat(self, chat_history: List[Dict[str, str]]) -> Tuple[bool, str]:
        """Validate chat history for planning agents (more flexible rules)."""
        if not chat_history:
            return False, "Chat history is empty"
        
        # Planning agents can handle various message patterns
        # but still need the last message to be from user for processing
        if chat_history[-1]["role"] != "user":
            return False, "Last message must be from user to continue planning"
        
        return True, ""
    
    def _validate_basic_chat(self, chat_history: List[Dict[str, str]]) -> Tuple[bool, str]:
        """Basic validation for new agent types."""
        if not chat_history:
            return False, "Chat history is empty"
        
        if chat_history[-1]["role"] != "user":
            return False, "Last message must be from user"
        
        return True, ""
    
    def get_action_config(self, active_chain, response_streamer) -> Dict[str, Any]:
        """Get action config for this graph type."""
        base_config = {
            'stream_callback': response_streamer.add_content,
        }
        
        if self.graph_type == 'planning':
            base_config.update({
                'proposal_system_prompt': PROPOSAL_SYSTEM_PROMPT,
                'revision_system_prompt': REVISION_SYSTEM_PROMPT,
                'proposal_chain': active_chain,
                'revision_chain': active_chain,
            })
        else:
            base_config['chain'] = active_chain
            
        return base_config


# Graph configurations - easily extensible for new agent types
GRAPH_CONFIGS = {
    'default': GraphConfig(
        graph_type='default',
        state_dict_func=get_default_state_dict,
        title="🤖 Default Chat Agent",
        description="A standard chat interface with strict alternating user/assistant messages, "
                   "supporting streaming responses and resumable sessions.",
        placeholder="Start a conversation with the default agent...",
        emoji="🤖"
    ),
    'planning': GraphConfig(
        graph_type='planning',
        state_dict_func=get_planning_state_dict,
        title="📋 Planning Agent",
        description="A planning-focused agent with advanced planning capabilities and flexible message patterns.",
        placeholder="Start a planning conversation...",
        emoji="📋",
    )
}

# Default agent selection
DEFAULT_AGENT_TYPE = 'planning'


class ResponseStreamer:
    """
    Handles real-time streaming of assistant responses.
    """

    def __init__(self, chat_history: list):
        self.chat_history = chat_history
        self.current_response = ""
        self.is_active = False

    def start_new_response(self) -> None:
        """Start streaming a new response."""
        self.current_response = ""
        self.is_active = True
        self.chat_history.append({"role": "assistant", "content": ""})

    async def add_content(self, new_text: str) -> None:
        """Add text to the current streaming response."""
        if not self.is_active:
            logger.warning("Tried adding content without active streaming.")
            return

        self.current_response += new_text
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] = self.current_response

    def complete_response(self, final_content: Optional[str] = None) -> None:
        """Finish the current response stream."""
        if final_content:
            self.current_response = final_content
            if self.chat_history and self.chat_history[-1]["role"] == "assistant":
                self.chat_history[-1]["content"] = self.current_response

        self.is_active = False


class UnifiedWorkflowEngine:
    """
    Processes chat conversations using configurable workflow graphs.
    """

    def __init__(self, graph_config: GraphConfig):
        self.graph_config = graph_config
        self.workflow = create_workflow(checkpointer=True)
        self.is_workflow_completed = False
        # Create the active chain using the MODEL_NAME
        self.active_chain = fast_get_llm(MODEL_NAME)

    async def process_conversation(
        self,
        user_message: str,
        chat_history: list,
        response_streamer: ResponseStreamer,
        session_id: str,
        is_new_session: bool = False
    ) -> tuple[str, bool]:
        """
        Process a single conversation turn using the configured graph.
        """
        try:
            # Agent-specific chat history validation
            is_valid, error_msg = self.graph_config.validate_chat_history(chat_history)
            if not is_valid:
                logger.warning(f"{self.graph_config.graph_type} validation failed: {error_msg}")
                raise ValueError(f"Invalid chat history for {self.graph_config.graph_type} agent: {error_msg}")
            
            # Reset completion status at start of processing
            self.is_workflow_completed = False
            
            # Initialize active chain based on configuration
            if DEFINE_CHAIN_IN_STATE:
                # Chain will be created and stored in state
                logger.info('Active chain will only be defined in the state!')
                active_chain = None
            else:
                # Create chain now for immediate use
                active_chain = self.active_chain
                if active_chain is None:
                    raise ValueError('Failed to create active chain at runtime!')
                
            # Build configuration based on graph type
            config = {
                'recursion_limit': RECURSION_LIMIT,
                'configurable': {
                    'thread_id': session_id,
                    'action_config': self.graph_config.get_action_config(active_chain, response_streamer)
                }
            }

            if is_new_session:
                logger.info(f"Starting new {self.graph_config.graph_type} session: {session_id}")
                await self._start_new_workflow(user_message, chat_history, response_streamer, config)
            else:
                logger.info(f"Continuing {self.graph_config.graph_type} session: {session_id}")
                await self._continue_workflow(user_message, response_streamer, config)

            return response_streamer.current_response, self.is_workflow_completed

        except Exception as e:
            error_message = f"{self.graph_config.graph_type.title()} workflow error: {str(e)}"
            logger.error(error_message)
            response_streamer.complete_response(error_message)
            return error_message, self.is_workflow_completed

    async def _start_new_workflow(
        self,
        user_message: str,
        chat_history: list,
        response_streamer: ResponseStreamer,
        config: dict
    ) -> None:
        """Start a new workflow session."""
        # Use the configured state dict function
        if self.graph_config.graph_type == 'default':
            # For default graph, use the gradio state dict
            state_dict = get_gradio_state_dict(
                user_message=user_message,
                history=chat_history,
                model_name=MODEL_NAME,
                system_message=SYSTEM_MESSAGE,
                active_chain=self.active_chain if DEFINE_CHAIN_IN_STATE else None
            )
        else:
            # For planning graphs, use planning state dict
            state_dict = self.graph_config.state_dict_func()
            if self.graph_config.graph_type == 'planning' and DEFINE_CHAIN_IN_STATE:
                task_state = {
                    'plan_context': None,
                    'proposed_plan': None,
                    'plan': None,
                    'revision_count': 0,
                    'proposal_chain': self.active_chain,
                    'revision_chain': self.active_chain,
                }
                task = get_task(state_dict['task_dict'], status=Status.NOT_STARTED)
                task['status'] = Status.IN_PROGRESS
                task['state'] = task_state
            # TODO: Add user message and chat history to the state dict if needed
            # This is where you'd customize state initialization per graph type

        initial_state = {
            'keys': state_dict,
            'mutation_count': 0,
            'is_done': False,
        }

        response_streamer.start_new_response()
        await self._execute_workflow(initial_state, config)
        response_streamer.complete_response()

    async def _continue_workflow(
        self,
        user_input: str,
        response_streamer: ResponseStreamer,
        config: dict
    ) -> None:
        """Continue an existing workflow session."""
        response_streamer.start_new_response()
        await self._resume_from_checkpoint(user_input, config)
        response_streamer.complete_response()

    async def _execute_workflow(self, workflow_state: dict, config: dict) -> None:
        """Execute workflow from initial state."""
        async for output in self.workflow.astream(workflow_state, config, stream_mode='updates'):
            await self._process_workflow_output(output)

    async def _resume_from_checkpoint(self, resume_input: str, config: dict) -> None:
        """Resume workflow from a checkpoint."""
        async for output in self.workflow.astream(
            ResumeCommand(resume=resume_input),
            config,
            stream_mode='updates'
        ):
            await self._process_workflow_output(output)

    async def _process_workflow_output(self, output: Dict[str, Any]) -> None:
        """Handle workflow outputs and state transitions."""
        for node_name, node_updates in output.items():
            if node_name == '__interrupt__':
                logger.info(f"{self.graph_config.graph_type.title()} workflow paused, awaiting user input.")
                return

            try:
                if isinstance(node_updates, dict):
                    if node_updates.get('is_done') is True:
                        logger.info(f"{self.graph_config.graph_type.title()} workflow completed.")
                        self.is_workflow_completed = True
                        return
                else:
                    logger.warning(f"Node updates are not a dict: {type(node_updates)}")
                logger.debug(f"Processed {self.graph_config.graph_type} node '{node_name}': {type(node_updates)}")
            except Exception as e:
                logger.error(f"Error processing {self.graph_config.graph_type} node '{node_name}': {str(e)}")


class UnifiedChatInterface:
    """
    User interface for Multi-Agent Augmenta Chat.
    """

    def __init__(self):
        self.current_agent_type = DEFAULT_AGENT_TYPE
        self.workflow_engines = {
            agent_type: UnifiedWorkflowEngine(config) 
            for agent_type, config in GRAPH_CONFIGS.items()
        }

    def change_agent(self, agent_type: str) -> tuple[list, dict, gr.update]:
        """
        Change the active agent and clear session.
        
        Returns:
            Updated chat history, session data, and chatbot label
        """
        if agent_type not in GRAPH_CONFIGS:
            logger.warning(f"Unknown agent type: {agent_type}")
            return [], {"session_id": None, "message_count": 0}, gr.update()
        
        self.current_agent_type = agent_type
        config = GRAPH_CONFIGS[agent_type]
        logger.info(f"Switched to {config.title}")
        
        return (
            [], 
            {"session_id": None, "message_count": 0},
            gr.update(label=f"Chat with {config.title}")
        )

    def add_user_message(self, user_input: str, chat_history: list) -> tuple[str, list]:
        """Add user's message to chat history."""
        if self.current_agent_type == 'planning' and len(chat_history) == 0:
            chat_history.append({"role": "user", "content": "Start Graph."})
            return "", chat_history
        
        if not user_input.strip():
            logger.info("Empty user input received - not adding to chat history")
            return "", chat_history  # Return unchanged chat history
        
        chat_history.append({"role": "user", "content": user_input})
        logger.debug(f"Added user message to chat history: {user_input[:50]}...")
        return "", chat_history

    def generate_assistant_response(
        self,
        chat_history: list,
        session_data: dict
    ) -> Generator[tuple[list, dict, gr.update, gr.update], None, None]:
        """Generate assistant's response with agent-specific validation."""
        current_config = GRAPH_CONFIGS[self.current_agent_type]
        workflow_engine = self.workflow_engines[self.current_agent_type]
        
        # Early validation: check if we have any chat history
        if not chat_history:
            logger.info("No chat history available - nothing to process")
            yield (
                chat_history,
                session_data,
                gr.update(interactive=True),
                gr.update(interactive=True)
            )
            return
        
        # Agent-specific validation
        is_valid, error_msg = current_config.validate_chat_history(chat_history)
        if not is_valid:
            logger.warning(f"{current_config.graph_type} validation failed: {error_msg}")
            # Only add error message if this wasn't just an empty input case
            # if chat_history:  # We have history but it's invalid
            #     chat_history.append({"role": "assistant", "content": f"Error: {error_msg}"})
            yield (
                chat_history,
                session_data,
                gr.update(interactive=True),
                gr.update(interactive=True)
            )
            return
        
        if not session_data.get("session_id"):
            session_data["session_id"] = str(uuid4())
            session_data["message_count"] = 0

        session_data["message_count"] += 1
        session_id = session_data["session_id"]
        is_new_session = (session_data["message_count"] == 1)

        response_streamer = ResponseStreamer(chat_history)
        loop = None

        try:
            user_message = chat_history[-1]["content"]
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process_message():
                return await workflow_engine.process_conversation(
                    user_message, chat_history, response_streamer,
                    session_id=session_id,
                    is_new_session=is_new_session
                )

            task = loop.create_task(process_message())

            while not task.done():
                loop.run_until_complete(asyncio.sleep(POLLING_INTERVAL))
                yield (
                    chat_history,
                    session_data,
                    gr.update(interactive=True),
                    gr.update(interactive=False)
                )

            final_response, workflow_completed = loop.run_until_complete(task)

            if not final_response.strip():
                logger.warning(f"Empty response received from {current_config.graph_type} workflow.")
                if chat_history[-1]["role"] == "assistant" and not chat_history[-1]["content"].strip():
                    chat_history = chat_history[:-2]
                    logger.info("Removed incomplete exchange.")
                    
            # If graph is run to completion, remove the session from the gradio state
            if workflow_completed:
                logger.info(f"{current_config.graph_type.title()} workflow completed. Clearing session: {session_id}")
                session_data["session_id"] = None
                session_data["message_count"] = 0
                
            yield (
                chat_history,
                session_data,
                gr.update(interactive=True),
                gr.update(interactive=True)
            )

        except Exception as e:
            error_message = f"{current_config.graph_type.title()} chat error: {str(e)}"
            logger.error(error_message)
            chat_history.append({"role": "assistant", "content": error_message})
            yield (
                chat_history,
                session_data,
                gr.update(interactive=True),
                gr.update(interactive=True)
            )
        finally:
            if loop is not None:
                loop.close()
    
    def clear_session(self) -> tuple[list, dict]:
        """Clear chat history and session data."""
        current_config = GRAPH_CONFIGS[self.current_agent_type]
        logger.info(f"Clearing {current_config.graph_type} chat history and session data.")
        return [], {"session_id": None, "message_count": 0}
    
    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio chat interface."""
        with gr.Blocks(
            title="🚀 Multi-Agent Augmenta Chat",
            analytics_enabled=False
        ) as demo:
            gr.Markdown("# 🚀 Multi-Agent Augmenta Chat")
            gr.Markdown(
                "**Multi-Agent Chat Interface**\n\n"
                "Chat with different AI agents, each with specialized capabilities and interaction patterns. "
                "Switch between agents using the dropdown menu below."
            )
            
            # Session state management
            session_data = gr.State({})
            
            # Agent selection
            agent_choices = [(f"{config.emoji} {config.title}", agent_type) 
                           for agent_type, config in GRAPH_CONFIGS.items()]
            
            agent_selector = gr.Dropdown(
                choices=agent_choices,
                value=DEFAULT_AGENT_TYPE,
                label="Select Agent",
                info="Choose which agent to chat with"
            )
            
            # Display current agent info
            current_config = GRAPH_CONFIGS[DEFAULT_AGENT_TYPE]
            agent_info = gr.Markdown(f"**Current Agent:** {current_config.description}")
            
            with gr.Tabs():
                with gr.Tab("Chat"):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label=f"Chat with {current_config.title}",
                        height=500,
                        type="messages",
                        show_copy_button=True
                    )
                    
                    # Input controls
                    with gr.Row():
                        message_input = gr.Textbox(
                            placeholder=current_config.placeholder,
                            label="Your message",
                            lines=2,
                            scale=4,
                            show_label=False,
                        )
                        send_button = gr.Button("Send", variant="primary", scale=1)
                        clear_button = gr.Button("Clear", variant="secondary", scale=1)
                
                with gr.Tab("Session State"):
                    session_id_display = gr.Textbox(
                        label="Session ID",
                        value="No active session",
                        interactive=False
                    )
                    
                    graph_state_display = gr.JSON(
                        label="Graph State",
                        value={},
                        show_label=True
                    )
                    
                    refresh_state_button = gr.Button("Refresh State", variant="secondary")

            def update_session_state(session_data):
                """Update the session state display."""
                session_id = session_data.get("session_id", "No active session")
                
                # Get current graph state if session exists
                graph_state = {}
                if session_id != "No active session":
                    try:
                        current_engine = self.workflow_engines[self.current_agent_type]
                        # Get the current state from the workflow checkpoint
                        config = {'configurable': {'thread_id': session_id}}
                        state_snapshot = current_engine.workflow.get_state(config)
                        if state_snapshot and hasattr(state_snapshot, 'values'):
                            graph_state = state_snapshot.values
                    except Exception as e:
                        graph_state = {"error": f"Could not retrieve state: {str(e)}"}
                
                return session_id, graph_state

            refresh_state_button.click(
                fn=update_session_state,
                inputs=[session_data],
                outputs=[session_id_display, graph_state_display],
                queue=False
            )

            # Agent selection handling
            def update_agent_info_and_ui(agent_type):
                config = GRAPH_CONFIGS[agent_type]
                return (
                    gr.update(value=f"**Current Agent:** {config.description}"),
                    gr.update(placeholder=config.placeholder),
                    gr.update(label=f"Chat with {config.title}")
                )

            agent_selector.change(
                fn=update_agent_info_and_ui,
                inputs=[agent_selector],
                outputs=[agent_info, message_input, chatbot],
                queue=False
            ).then(
                fn=self.change_agent,
                inputs=[agent_selector],
                outputs=[chatbot, session_data, chatbot],
                queue=False
            )

            # Event handling with proper input management
            send_button.click(
                fn=self.add_user_message,
                inputs=[message_input, chatbot],
                outputs=[message_input, chatbot],
                queue=False
            ).then(
                fn=self.generate_assistant_response,
                inputs=[chatbot, session_data],
                outputs=[chatbot, session_data, message_input, send_button]
            )
            
            message_input.submit(
                fn=self.add_user_message,
                inputs=[message_input, chatbot],
                outputs=[message_input, chatbot],
                queue=False
            ).then(
                fn=self.generate_assistant_response,
                inputs=[chatbot, session_data],
                outputs=[chatbot, session_data, message_input, send_button]
            )

            clear_button.click(
                fn=self.clear_session,
                inputs=[],
                outputs=[chatbot, session_data],
                queue=False
            )

        return demo


def main():
    """Launch the Multi-Agent Augmenta Chat application."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info("Starting Multi-Agent Augmenta Chat application")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize and launch chat interface
        chat_app = UnifiedChatInterface()
        demo = chat_app.create_interface()
        
        logger.info("Launching Multi-Agent Chat interface")
        demo.launch(share=False, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start Multi-Agent Chat: {str(e)}")
        raise


if __name__ == "__main__":
    main() 