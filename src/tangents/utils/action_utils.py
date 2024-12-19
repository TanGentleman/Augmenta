import logging
from typing import Dict, Any

from ..graph_classes import Action, ActionType, ActionResult, PlanActionType

def execute_action(action: Action, state_dict: Dict[str, Any]) -> ActionResult:
    """Execute a specific action and return the result."""
    # TODO: Eventually reduce dependency on state_dict
    # Everything should fit as args to the action
    action_type = action["type"]
    action_args = action["args"]

    try:
        if action_type == ActionType.GENERATE:
            # Load state variables into action args
            if not action_args.get("active_chain"):
                action_args["active_chain"] = state_dict["active_chain"]
            if not action_args.get("messages"):
                action_args["messages"] = state_dict["messages"]
            
            if "stream" not in action_args:
                action_args["stream"] = state_dict["config"].chat_settings.stream
            
            stream = action_args["stream"]
            chain = action_args["active_chain"]
            messages = action_args["messages"]
            
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
        
        elif action_type == PlanActionType.FETCH:
            # Load state variables into action args
            
            source = action_args["source"]
            method = action_args["method"]
            result_string = "Fetched data."
            if method == "get_email_content":
                result_string = "This is an example email. Assign a task to Himanshu to review the updated docs."
            return {
                "success": True,
                "data": result_string,
                "error": None
            }
        
        elif action_type == PlanActionType.CREATE_PLAN:
            # Load state variables into action args
            if not action_args.get("plan_context"):
                action_args["plan_context"] = state_dict["plan_dict"]["context"]
            
            return {
                "success": True,
                "data": "CREATED PLAN.",
                "error": None
            }
        
        elif action_type == PlanActionType.REVISE_PLAN:
            # Load state variables into action args
            if not action_args.get("proposed_plan"):
                action_args["proposed_plan"] = state_dict["plan_dict"]["proposed_plan"]
            
            if "revision_count" not in action_args:
                action_args["revision_count"] = 0
            
            if "max_revisions" not in action_args:
                raise ValueError('Set "max_revisions" in REVISE_PLAN action args!')

            # If is_done is True, submit the plan
            revision_count = action_args["revision_count"]
            max_revisions = action_args["max_revisions"]
            if action_args.get("is_done", False):
                # Submit the plan
                return {
                    "success": True,
                    "data": f"Final draft ({revision_count} revisions).",
                    "error": None
                }
            
            # If max_revisions is reached, submit the plan
            if revision_count >= max_revisions:
                # Submit the plan
                logging.warning("Hit max revisions!")
                return {
                    "success": True,
                    "data": f"Final draft ({revision_count} revisions [max]).",
                    "error": None
                }
            
            # If max_revisions is not reached, revise the plan
            return {
                "success": False,
                "data": f"Plan revision {action_args['revision_count']}.",
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
    
def save_action_data(action: Action) -> bool:
    logging.info(f"Saving action data: {action}")
    return True

