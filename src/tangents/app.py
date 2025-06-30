"""
Tangents Experimental Graph Chat Interface

A Gradio-based chat interface that integrates with the experimental graph workflow system,
providing real-time streaming responses with robust error handling and interrupt management.

Key Components:
- StreamingHandler: Manages real-time chat updates with proper content isolation
- GraphExecutor: Handles graph execution and state management
- InterruptManager: Processes workflow interrupts through chat interface
- GradioInterface: UI setup and event coordination

Production Features:
- Clean content isolation between responses
- Simplified streaming with proper state management
- Robust error handling and logging
- Async/sync boundary management for Gradio
"""

import asyncio
import logging
from uuid import uuid4
from typing import Optional, Dict, Any, Generator

import gradio as gr
from langgraph.types import Command as ResumeCommand

from tangents.experimental.utils import get_gradio_state_dict
from tangents.tan_graph import create_workflow
from tangents.output_handlers import OutputProcessor

# Configuration
MODEL_NAME = 'nebius/meta-llama/Llama-3.3-70B-Instruct'
SYSTEM_MESSAGE = 'You are a helpful assistant who responds playfully.'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingHandler:
    """
    Handles real-time streaming updates for Gradio chat interface.
    
    Each instance manages content for a single response, preventing
    content leakage between different assistant responses.
    """
    
    def __init__(self, history: list):
        self.history = history
        self.current_content = ""
        self.is_streaming = False
        
    def start_response(self):
        """Start a new assistant response."""
        self.current_content = ""
        self.is_streaming = True
        
        # Add new assistant message to history
        self.history.append({"role": "assistant", "content": ""})
    
    async def update_content(self, text: str):
        """Update the current response content."""
        if not self.is_streaming:
            return
            
        # Filter out technical state data that shouldn't be shown
        if self._should_filter_content(text):
            return
            
        self.current_content += text
        
        # Update the last assistant message
        if self.history and self.history[-1]["role"] == "assistant":
            self.history[-1]["content"] = self.current_content
    
    def finish_response(self, final_message: Optional[str] = None):
        """Complete the current response."""
        if final_message:
            self.current_content = final_message
            if self.history and self.history[-1]["role"] == "assistant":
                self.history[-1]["content"] = self.current_content
        
        self.is_streaming = False
    
    def add_system_message(self, message: str):
        """Add a system message (like interrupts) to chat."""
        self.history.append({
            "role": "assistant",
            "content": f"🔄 **System**: {message}"
        })
    
    def _should_filter_content(self, text: str) -> bool:
        """Filter out technical state data that shouldn't be displayed."""
        if not text:
            return True
        
        # Don't filter whitespace-only content (legitimate spaces)
        if text.isspace():
            return False
            
        # Filter technical state indicators
        technical_terms = [
            'keysmutation_count', 'mutation_count', 'is_done',
            '{"keys"', '"mutation_count"', '"is_done"',
            'action_count', 'task_dict'
        ]
        
        text_lower = text.lower().strip()
        return any(term in text_lower for term in technical_terms)


class InterruptManager:
    """Manages workflow interrupts in the Gradio context."""
    
    @staticmethod
    async def handle_interrupt(interrupt_data: dict, streaming_handler: StreamingHandler) -> str:
        """
        Process workflow interrupt.
        
        Args:
            interrupt_data: Interrupt information from the graph
            streaming_handler: Handler for UI updates
            
        Returns:
            Placeholder indicating interrupt is pending
        """
        prompt = interrupt_data.get('prompt', 'Please provide input to continue:')
        
        # Skip generic/false interrupts
        if prompt in ['Enter your message (or /help for commands):', 'User input:']:
            logger.debug(f"Skipping generic interrupt: {prompt}")
            return "SKIP_INTERRUPT"
        
        logger.info(f"Processing interrupt: {prompt}")
        streaming_handler.add_system_message(f"Interrupt: {prompt}")
        
        return "INTERRUPT_PENDING"


class GraphExecutor:
    """Handles graph execution and streaming coordination."""
    
    def __init__(self):
        self.graph = create_workflow(checkpointer=True)
        self.active_threads = {}  # Track active sessions
    
    async def execute_with_streaming(
        self,
        user_message: str,
        history: list,
        streaming_handler: StreamingHandler,
        session_id: str,
        is_first_message: bool = False
    ) -> str:
        """
        Execute graph workflow with streaming support.
        
        Args:
            user_message: User input message
            history: Chat history (used only for context, not for state creation)
            streaming_handler: Streaming handler for UI updates
            session_id: Unique ID for session continuity
            is_first_message: Whether this is the first message in the session
        
        Returns:
            Final response content
        """
        try:
            # Configure execution
            config = {
                'recursion_limit': 20,
                'configurable': {
                    'thread_id': session_id,
                    'stream_callback': streaming_handler.update_content
                }
            }
            
            # Start streaming response
            streaming_handler.start_response()
            
            if is_first_message:
                # First message: create initial state with mock_inputs
                logger.info(f"Starting new session {session_id} with initial state")
                
                # Create clean history without empty last assistant message
                clean_history = history.copy()
                if clean_history and clean_history[-1].get("role") == "assistant" and clean_history[-1].get("content", "") == "":
                    clean_history.pop()
                
                state_dict = get_gradio_state_dict(
                    user_message=user_message,
                    history=clean_history,
                    model_name=MODEL_NAME,
                    system_message=SYSTEM_MESSAGE,
                )
                
                # Ensure required state keys
                state_dict.setdefault('task_dict', {})
                state_dict.setdefault('config', {})
                
                graph_state = {
                    'keys': state_dict,
                    'mutation_count': 0,
                    'is_done': False,
                }
                
                await self._execute_normal(graph_state, config, streaming_handler)
            else:
                # Continuation: use ResumeCommand
                logger.info(f"Continuing existing session {session_id} with ResumeCommand")
                await self._execute_resume(user_message, config, streaming_handler)
            
            # Check if we need to continue after interrupt
            if streaming_handler.is_streaming:
                return "INTERRUPT_PENDING"
                
            return streaming_handler.current_content
            
        except Exception as e:
            error_msg = f"Graph execution error: {str(e)}"
            logger.error(error_msg)
            streaming_handler.finish_response(error_msg)
            return error_msg
    
    async def _execute_normal(self, graph_state: dict, config: dict, streaming_handler: StreamingHandler):
        """Execute normal graph workflow."""
        async for output in self.graph.astream(graph_state, config, stream_mode='updates'):
            interrupt_status = await self._process_output(output, config, streaming_handler)
            if interrupt_status == "INTERRUPT_PENDING":
                return  # Pause execution on interrupt
    
    async def _execute_resume(self, resume_command: str, config: dict, streaming_handler: StreamingHandler):
        """Execute graph resume from interrupt."""
        async for output in self.graph.astream(
            ResumeCommand(resume=resume_command), 
            config, 
            stream_mode='updates'
        ):
            interrupt_status = await self._process_output(output, config, streaming_handler)
            if interrupt_status == "INTERRUPT_PENDING":
                return  # Pause execution on interrupt
    
    async def _process_output(self, output: Dict[str, Any], config: dict, streaming_handler: StreamingHandler):
        """Process graph output and handle interrupts."""
        for node, updates in output.items():
            if node == '__interrupt__':
                # Return the interrupt to pause execution
                return "INTERRUPT_PENDING"
            else:
                # Process regular node updates
                await self._extract_content_from_updates(node, updates, streaming_handler)
        
        return "CONTINUE"
    
    async def _extract_content_from_updates(self, node: str, updates: Any, streaming_handler: StreamingHandler):
        """Extract meaningful content from node updates."""
        try:
            # Check for completion
            if isinstance(updates, dict):
                # Handle completion signal
                keys = updates.get('keys', {})
                if keys.get('is_done', False) or updates.get('is_done', False):
                    streaming_handler.finish_response("✅ Task completed successfully.")
                    return
            
            # Log debug info without cluttering UI
            logger.debug(f"Node {node} processed: {type(updates)}")
            
        except Exception as e:
            logger.error(f"Error processing {node} updates: {str(e)}")
    



class GradioInterface:
    """Manages the Gradio chat interface."""
    
    def __init__(self):
        self.executor = GraphExecutor()
    
    def add_user_message(self, user_message: str, history: list) -> tuple[str, list]:
        """Add user message to history."""
        if not user_message.strip():
            return "", history
        return "", history + [{"role": "user", "content": user_message}]
    
    def generate_response(self, history: list, session_state: dict) -> Generator[tuple[list, dict], None, None]:
        """Generate bot response with streaming using proper session state."""
        if not history or history[-1]["role"] != "user":
            return
        
        user_message = history[-1]["content"]
        
        # Initialize session state if needed
        if not session_state.get("session_id"):
            session_state["session_id"] = str(uuid4())
            session_state["message_count"] = 0
        
        session_state["message_count"] += 1
        session_id = session_state["session_id"]
        
        # Create streaming handler
        streaming_handler = StreamingHandler(history)
        
        # Execute with proper async handling
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_graph():
                return await self.executor.execute_with_streaming(
                    user_message, history, streaming_handler,
                    session_id=session_id,
                    is_first_message=(session_state["message_count"] == 1)
                )
            
            # Execute with periodic yielding for UI updates
            task = loop.create_task(run_graph())
            
            # Yield updates while task is running
            while not task.done():
                loop.run_until_complete(asyncio.sleep(0.1))
                yield history, session_state
            
            # Get final result
            loop.run_until_complete(task)
            yield history, session_state
            
        except Exception as e:
            error_msg = f"Response generation error: {str(e)}"
            logger.error(error_msg)
            
            # Ensure we have an assistant message for the error
            if not history or history[-1]["role"] != "assistant":
                history.append({"role": "assistant", "content": error_msg})
            else:
                history[-1]["content"] = error_msg
            yield history, session_state
        finally:
            loop.close()
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Tangents Experimental Chat",
            analytics_enabled=False
        ) as demo:
            gr.Markdown("# 🔬 Tangents Experimental Graph Chat")
            gr.Markdown(
                "Production-ready chat interface with experimental graph workflow.\n\n"
                "**Features:**\n"
                "- Clean streaming responses\n"
                "- Proper session state management\n"
                "- Workflow interrupt handling\n"
                "- Robust error management"
            )
            
            # Session state management
            session_state = gr.State({})
            
            chatbot = gr.Chatbot(
                label="Assistant",
                height=500,
                type="messages",
                show_copy_button=True
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your message...",
                    label="Message",
                    lines=2,
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Wire up interactions
            send_btn.click(
                fn=self.add_user_message,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
                queue=False
            ).then(
                fn=self.generate_response,
                inputs=[chatbot, session_state],
                outputs=[chatbot, session_state]
            )
            
            user_input.submit(
                fn=self.add_user_message,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
                queue=False
            ).then(
                fn=self.generate_response,
                inputs=[chatbot, session_state],
                outputs=[chatbot, session_state]
            )
        
        return demo


def main():
    """Main entry point for the application."""
    try:
        # Setup
        logging.basicConfig(level=logging.INFO)
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Create and launch interface
        interface = GradioInterface()
        demo = interface.create_interface()
        demo.launch(share=False, debug=False)
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 

