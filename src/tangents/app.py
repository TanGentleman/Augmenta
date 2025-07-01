"""
LangGraph Chat Interface

A simple chat app that showcases LangGraph's powerful workflow capabilities:
- Real-time streaming responses
- Session continuity across conversations  
- Workflow interrupts and resumption
- Robust state management
"""

import asyncio
import logging
from uuid import uuid4
from typing import Optional, Dict, Any, Generator

import gradio as gr
from langgraph.types import Command as ResumeCommand

from tangents.experimental.utils import get_gradio_state_dict
from tangents.tan_graph import create_workflow

# Configuration
MODEL_NAME = 'nebius/meta-llama/Llama-3.3-70B-Instruct'
SYSTEM_MESSAGE = 'You are a helpful assistant who responds playfully.'

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StreamingHandler:
    """Manages real-time chat updates for a single response."""
    
    def __init__(self, chat_history: list):
        self.chat_history = chat_history
        self.current_response = ""
        self.is_active = False
        
    def start_new_response(self):
        """Begin streaming a new assistant response."""
        self.current_response = ""
        self.is_active = True
        self.chat_history.append({"role": "assistant", "content": ""})
    
    async def add_content(self, new_text: str):
        """Add text to the current streaming response."""
        if not self.is_active:
            logger.error("Not streaming, skipping content update")
            return
            
        self.current_response += new_text
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] = self.current_response
    
    def complete_response(self, final_content: Optional[str] = None):
        """Finish the current response."""
        if final_content:
            self.current_response = final_content
            if self.chat_history and self.chat_history[-1]["role"] == "assistant":
                self.chat_history[-1]["content"] = self.current_response
        
        self.is_active = False

class LangGraphExecutor:
    """Handles LangGraph workflow execution with streaming."""
    
    def __init__(self):
        self.workflow = create_workflow(checkpointer=True)
        self.active_sessions = {}
    
    async def run_with_streaming(
        self,
        user_message: str,
        chat_history: list,
        stream_handler: StreamingHandler,
        session_id: str,
        is_first_message: bool = False
    ) -> str:
        """
        Execute LangGraph workflow with real-time streaming.
        
        This showcases LangGraph's key features:
        - Persistent state across conversations
        - Workflow interrupts and resumption
        - Real-time streaming updates
        """
        try:
            config = {
                'recursion_limit': 20,
                'configurable': {
                    'thread_id': session_id,
                    'stream_callback': stream_handler.add_content
                }
            }
            
            if is_first_message:
                logger.info(f"Starting new LangGraph session {session_id}")
                # Create initial state for new conversation
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
                stream_handler.start_new_response()
                await self._run_workflow(initial_state, config)
                stream_handler.complete_response()
            else:
                logger.info(f"Resuming LangGraph session {session_id}")
                # Resume existing workflow with new input
                stream_handler.start_new_response()
                await self._resume_workflow(user_message, config)
                stream_handler.complete_response()
                
            return stream_handler.current_response
            
        except Exception as e:
            error_message = f"LangGraph execution error: {str(e)}"
            logger.error(error_message)
            stream_handler.complete_response(error_message)
            return error_message
    
    async def _run_workflow(self, workflow_state: dict, config: dict):
        """Execute the LangGraph workflow from initial state."""
        async for output in self.workflow.astream(workflow_state, config, stream_mode='updates'):
            await self._handle_workflow_output(output)
    
    async def _resume_workflow(self, resume_input: str, config: dict):
        """Resume LangGraph workflow from interrupt point."""
        async for output in self.workflow.astream(
            ResumeCommand(resume=resume_input), 
            config, 
            stream_mode='updates'
        ):
            await self._handle_workflow_output(output)
    
    async def _handle_workflow_output(self, output: Dict[str, Any]):
        """Process LangGraph workflow outputs and handle interrupts."""
        for node_name, node_updates in output.items():
            if node_name == '__interrupt__':
                logger.info("LangGraph workflow interrupted - waiting for user input")
                return  # Pause execution for user interaction
            try:
                if isinstance(node_updates, dict) and node_updates.get('is_done') is True:
                    logger.info("LangGraph workflow completed")
                    return
                logger.debug(f"Node {node_name} processed: {type(node_updates)}")
            except Exception as e:
                logger.error(f"Error processing {node_name} updates: {str(e)}")

class ChatInterface:
    """Simple chat interface showcasing LangGraph capabilities."""
    
    def __init__(self):
        self.graph_executor = LangGraphExecutor()
    
    def add_user_message(self, user_input: str, chat_history: list) -> tuple[str, list]:
        """Add user message to chat history."""
        if not user_input.strip():
            return "", chat_history
        return "", chat_history + [{"role": "user", "content": user_input}]
    
    def generate_bot_response(self, chat_history: list, session_data: dict) -> Generator[tuple[list, dict, gr.update, gr.update], None, None]:
        """
        Generate streaming bot response using LangGraph.
        
        Returns: (chat_history, session_data, textbox_update, send_button_update)
        """
        if not chat_history or chat_history[-1]["role"] != "user":
            logger.error("Invalid chat state: Expected user message")
            return
        
        user_message = chat_history[-1]["content"]
        
        # Initialize session if needed
        if not session_data.get("session_id"):
            session_data["session_id"] = str(uuid4())
            session_data["message_count"] = 0
        
        session_data["message_count"] += 1
        session_id = session_data["session_id"]
        is_first_message = (session_data["message_count"] == 1)
        
        stream_handler = StreamingHandler(chat_history)
        
        try:
            # Set up async execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def execute_langgraph():
                return await self.graph_executor.run_with_streaming(
                    user_message, chat_history, stream_handler,
                    session_id=session_id,
                    is_first_message=is_first_message
                )
            
            task = loop.create_task(execute_langgraph())
            
            # Stream updates while processing (disable user input)
            while not task.done():
                loop.run_until_complete(asyncio.sleep(0.1))
                yield (
                    chat_history, 
                    session_data, 
                    gr.update(interactive=False),  # Disable textbox
                    gr.update(interactive=False)   # Disable send button
                )
            
            # Get final result and re-enable input
            final_response = loop.run_until_complete(task)
            
            if final_response == "":
                logger.warning("Empty response from LangGraph")
                # Clean up failed exchanges
                if chat_history and chat_history[-1]["role"] == "assistant" and chat_history[-1]["content"] == "":
                    logger.warning("Removing empty response from history")
                    chat_history = chat_history[:-2]
                    
            yield (
                chat_history, 
                session_data, 
                gr.update(interactive=True),   # Re-enable textbox
                gr.update(interactive=True)    # Re-enable send button
            )
            
        except Exception as e:
            error_message = f"Chat error: {str(e)}"
            logger.error(error_message)
            
            # Add error message to chat
            if not chat_history or chat_history[-1]["role"] != "assistant":
                chat_history.append({"role": "assistant", "content": error_message})
            else:
                chat_history[-1]["content"] = error_message
                
            # Re-enable input after error
            yield (
                chat_history, 
                session_data, 
                gr.update(interactive=True),   # Re-enable textbox
                gr.update(interactive=True)    # Re-enable send button
            )
        finally:
            loop.close()
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio chat interface."""
        with gr.Blocks(
            title="LangGraph Chat Demo",
            analytics_enabled=False
        ) as demo:
            gr.Markdown("# 🕸️ LangGraph Chat Demo")
            gr.Markdown(
                "See LangGraph in action! This chat interface demonstrates:\n\n"
                "✨ **Real-time streaming** - Watch responses appear as they're generated\n"
                "🧵 **Session continuity** - Conversations persist with state management\n"
                "⚡ **Workflow interrupts** - Interactive workflows that can pause and resume\n"
                "🛡️ **Robust error handling** - Graceful handling of edge cases"
            )
            
            # Session state
            session_data = gr.State({})
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="LangGraph Assistant",
                height=500,
                type="messages",
                show_copy_button=True
            )
            
            with gr.Row():
                message_input = gr.Textbox(
                    placeholder="Type your message and see LangGraph in action...",
                    label="Message",
                    lines=2,
                    scale=4,
                    show_label=False,
                )
                send_button = gr.Button("Send", variant="primary", scale=1)
            
            # Event handlers with input locking
            send_button.click(
                fn=self.add_user_message,
                inputs=[message_input, chatbot],
                outputs=[message_input, chatbot],
                queue=False
            ).then(
                fn=self.generate_bot_response,
                inputs=[chatbot, session_data],
                outputs=[chatbot, session_data, message_input, send_button]
            )
            
            message_input.submit(
                fn=self.add_user_message,
                inputs=[message_input, chatbot],
                outputs=[message_input, chatbot],
                queue=False
            ).then(
                fn=self.generate_bot_response,
                inputs=[chatbot, session_data],
                outputs=[chatbot, session_data, message_input, send_button]
            )
        
        return demo


def main():
    """Launch the LangGraph chat demo."""
    try:
        # Setup logging
        logger.setLevel(logging.INFO)
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Create and launch the chat interface
        interface = ChatInterface()
        demo = interface.create_interface()
        demo.launch(share=False, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start LangGraph demo: {str(e)}")
        raise


if __name__ == "__main__":
    main() 

