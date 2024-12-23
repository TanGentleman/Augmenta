import logging
from tangents.classes.actions import Action, ActionType, ActionResult, PlanActionType
from tangents.classes.tasks import Task, TaskType

def execute_action(action: Action, current_task: Task) -> ActionResult:
    """Execute a specific action and return the result."""
    # TODO: Eventually reduce dependency on state_dict
    # Everything should fit as args to the action

    # TODO: Replace state_dict with task_state
    action_type = action["type"]
    action_args = action["args"]

    task_state = current_task["state"]

    try:
        if action_type == ActionType.GENERATE:
            if current_task["type"] == TaskType.CHAT:
                # Load state variables into action args
                if action_args.get("active_chain") is None:
                    action_args["active_chain"] = task_state["active_chain"]
            
                if action_args.get("messages") is None:
                    action_args["messages"] = task_state["messages"]
                
                if action_args.get("stream") is None:
                    action_args["stream"] = task_state["stream"]
                
                stream = action_args["stream"]
                chain = action_args["active_chain"]
                messages = action_args["messages"]
            else:
                raise ValueError("Task type not supported for generate action")
            
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
            if action_args.get("plan_context") is None:
                action_args["plan_context"] = task_state["context"]
            
            return {
                "success": True,
                "data": "CREATED PLAN.",
                "error": None
            }
        
        elif action_type == PlanActionType.REVISE_PLAN:
            # Load state variables into action args
            if action_args.get("proposed_plan") is None:
                action_args["proposed_plan"] = task_state["proposed_plan"]
            
            if action_args.get("revision_count") is None:
                action_args["revision_count"] = 0
            
            if action_args.get("max_revisions") is None:
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