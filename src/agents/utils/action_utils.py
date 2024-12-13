import logging
from typing import Dict, Any
from ..graph_classes import ActionType, ActionResult

def execute_action(action: str, state_dict: Dict[str, Any]) -> ActionResult:
    """Execute a specific action and return the result."""
    try:
        action_parts = action.split(":", 1)
        action_type = ActionType(action_parts[0])
        action_params = action_parts[1] if len(action_parts) > 1 else ""

        if action_type == ActionType.GENERATE:
            if state_dict["active_chain"] is None:
                raise ValueError("No active chain found")
            
            chain = state_dict["active_chain"]
            messages = state_dict["messages"]
            config = state_dict["config"]
            stream = config.chat_settings.stream

            try:
                if stream:
                    response_string = ""
                    for chunk in chain.stream(messages):
                        print(chunk.content, end="", flush=True)
                        response_string += chunk.content
                    print()
                    if not response_string:
                        raise ValueError('No response generated')
                else:
                    response = chain.invoke(messages)
                    print(response.content)
                    response_string = response.content
                return {
                    "success": True,
                    "data": response_string,
                    "error": None
                }

            except KeyboardInterrupt:
                print('Keyboard interrupt, aborting generation.')
                messages.pop()
                return {
                    "success": False,
                    "data": None, 
                    "error": "Generation interrupted"
                }
            except Exception as e:
                return {
                    "success": False,
                    "data": None,
                    "error": str(e)
                }
            
        elif action_type == ActionType.WEB_SEARCH:
            return {
                "success": True,
                "data": "Search results",
                "error": None
            }
            
        elif action_type == ActionType.SAVE_DATA:
            return {
                "success": True,
                "data": "Data saved",
                "error": None
            }
            
        elif action_type == ActionType.TOOL_CALL:
            return {
                "success": True,
                "data": "Tool called",
                "error": None
            }
            
        else:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown action type: {action_type}"
            }
            
    except Exception as e:
        logging.error(f"Action execution failed: {str(e)}")
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }