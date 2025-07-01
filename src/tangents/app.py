"""
Augmenta Chat Interface

A Gradio frontend supporting:
- Real-time streaming
- Persistent chat sessions
- Human-In-The-Loop within complex workflows
"""

import asyncio
import logging
from uuid import uuid4
from typing import Optional, Dict, Any, Generator

import gradio as gr
from langgraph.types import Command as ResumeCommand

from tangents.experimental.utils import get_gradio_state_dict
from tangents.tan_graph import create_workflow

# Application Configuration
MODEL_NAME = 'nebius/meta-llama/Llama-3.3-70B-Instruct'
SYSTEM_MESSAGE = 'You are a helpful assistant who responds playfully.'

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class WorkflowEngine:
    """
    Processes chat conversations using workflow graphs.
    """

    def __init__(self):
        self.workflow = create_workflow(checkpointer=True)

    async def process_conversation(
        self,
        user_message: str,
        chat_history: list,
        response_streamer: ResponseStreamer,
        session_id: str,
        is_new_session: bool = False
    ) -> str:
        """
        Process a single conversation turn.

        Args:
            user_message: User's input message.
            chat_history: Conversation history.
            response_streamer: Handles streaming responses.
            session_id: Unique session identifier.
            is_new_session: True if starting a new session.

        Returns:
            Final assistant response.
        """
        try:
            config = {
                'recursion_limit': 20,
                'configurable': {
                    'thread_id': session_id,
                    'stream_callback': response_streamer.add_content
                }
            }

            if is_new_session:
                logger.info(f"Starting new session: {session_id}")
                await self._start_new_workflow(user_message, chat_history, response_streamer, config)
            else:
                logger.info(f"Continuing session: {session_id}")
                await self._continue_workflow(user_message, response_streamer, config)

            return response_streamer.current_response

        except Exception as e:
            error_message = f"Workflow error: {str(e)}"
            logger.error(error_message)
            response_streamer.complete_response(error_message)
            return error_message

    async def _start_new_workflow(
        self,
        user_message: str,
        chat_history: list,
        response_streamer: ResponseStreamer,
        config: dict
    ) -> None:
        """Start a new workflow session."""
        state_dict = get_gradio_state_dict(
            user_message=user_message,
            history=chat_history.copy(),
            model_name=MODEL_NAME,
            system_message=SYSTEM_MESSAGE,
        )

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
                logger.info("Workflow paused, awaiting user input.")
                return

            try:
                if isinstance(node_updates, dict) and node_updates.get('is_done') is True:
                    logger.info("Workflow completed.")
                    return
                logger.debug(f"Processed node '{node_name}': {type(node_updates)}")
            except Exception as e:
                logger.error(f"Error processing node '{node_name}': {str(e)}")


class AugmentaChatInterface:
    """
    User interface for Augmenta Chat.
    """

    def __init__(self):
        self.workflow_engine = WorkflowEngine()

    def add_user_message(self, user_input: str, chat_history: list) -> tuple[str, list]:
        """Add user's message to chat history."""
        if not user_input.strip():
            return "", chat_history
        return "", chat_history + [{"role": "user", "content": user_input}]

    def generate_assistant_response(
        self,
        chat_history: list,
        session_data: dict
    ) -> Generator[tuple[list, dict, gr.update, gr.update], None, None]:
        """
        Generate assistant's response with streaming updates.

        Args:
            chat_history: Current chat history.
            session_data: Session metadata.

        Yields:
            Updates for chat history, session data, and UI elements.
        """
        if not chat_history or chat_history[-1]["role"] != "user":
            logger.error("Expected user message at end of history.")
            return

        user_message = chat_history[-1]["content"]

        if not session_data.get("session_id"):
            session_data["session_id"] = str(uuid4())
            session_data["message_count"] = 0

        session_data["message_count"] += 1
        session_id = session_data["session_id"]
        is_new_session = (session_data["message_count"] == 1)

        response_streamer = ResponseStreamer(chat_history)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def process_message():
                return await self.workflow_engine.process_conversation(
                    user_message, chat_history, response_streamer,
                    session_id=session_id,
                    is_new_session=is_new_session
                )

            task = loop.create_task(process_message())

            while not task.done():
                loop.run_until_complete(asyncio.sleep(0.1))
                yield (
                    chat_history,
                    session_data,
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )

            final_response = loop.run_until_complete(task)

            if not final_response.strip():
                logger.warning("Empty response received.")
                if chat_history[-1]["role"] == "assistant" and not chat_history[-1]["content"].strip():
                    chat_history = chat_history[:-2]
                    logger.info("Removed incomplete exchange.")

            yield (
                chat_history,
                session_data,
                gr.update(interactive=True),
                gr.update(interactive=True)
            )

        except Exception as e:
            error_message = f"Chat error: {str(e)}"
            logger.error(error_message)
            chat_history.append({"role": "assistant", "content": error_message})
            yield (
                chat_history,
                session_data,
                gr.update(interactive=True),
                gr.update(interactive=True)
            )
        finally:
            loop.close()
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and configure the Gradio chat interface.
        
        Returns:
            Configured Gradio Blocks interface ready for launch
        """
        with gr.Blocks(
            title="Augmenta Chat",
            analytics_enabled=False
        ) as demo:
            gr.Markdown("# 🤖 Augmenta Chat")
            gr.Markdown(
                "**Augmenta Chat Demo**\n\n"
                "A simple chat interface powered by LangGraph, supporting streaming responses, "
                "resumable chat sessions, and interactive workflows with Human-In-The-Loop capabilities."
            )
            
            # Session state management
            session_data = gr.State({})
            
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Chat with Augmenta",
                height=500,
                type="messages",
                show_copy_button=True
            )
            
            # Input controls
            with gr.Row():
                message_input = gr.Textbox(
                    placeholder="Start a conversation with Augmenta...",
                    label="Your message",
                    lines=2,
                    scale=4,
                    show_label=False,
                )
                send_button = gr.Button("Send", variant="primary", scale=1)
            
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
        
        return demo


def main():
    """Launch the Augmenta Chat application."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info("Starting Augmenta Chat application")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize and launch chat interface
        chat_app = AugmentaChatInterface()
        demo = chat_app.create_interface()
        
        logger.info("Launching Augmenta Chat interface")
        demo.launch(share=False, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start Augmenta Chat: {str(e)}")
        raise


if __name__ == "__main__":
    main() 

