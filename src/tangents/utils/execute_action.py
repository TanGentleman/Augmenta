import logging
from tangents.classes.actions import Action, ActionType, ActionResult, PlanActionType
from tangents.utils.experimental import run_healthcheck

async def execute_action(action: Action) -> ActionResult:
    """
    Execute a specific action and return the result.
    
    This function handles different action types and executes them appropriately:
    - GENERATE: Generates text using an LLM chain, with optional streaming
    - WEB_SEARCH: Performs web searches (placeholder)
    - SAVE_DATA: Saves data to storage (placeholder) 
    - TOOL_CALL: Makes external tool calls (placeholder)
    - HEALTHCHECK: Checks the health of an endpoint
    - FETCH: Retrieves data from specified sources
    - PROPOSE_PLAN: Creates initial task plans
    - REVISE_PLAN: Handles plan revisions and submissions

    Args:
        action (Action): The action to execute, containing type and arguments

    Returns:
        ActionResult: Result object with:
            - success (bool): Whether action completed successfully
            - data (Union[str, None]): Output data if successful
            - error (Union[str, None]): Error message if failed
    """
    action_type = action["type"]
    action_args = action["args"]

    try:
        if action_type == ActionType.GENERATE:
            stream = action_args["stream"]
            chain = action_args["active_chain"]
            messages = action_args["messages"]
            
            try:
                if stream:
                    response_string = ""
                    async for chunk in chain.astream(messages):
                        print(chunk.content, end="", flush=True)
                        response_string += chunk.content
                    print()
                    if not response_string:
                        raise ValueError('No response generated')
                else:
                    response = await chain.ainvoke(messages)
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
        
        elif action_type == ActionType.HEALTHCHECK:
            endpoint = action_args["endpoint"]
            if not endpoint:
                return {
                    "success": False,
                    "data": None,
                    "error": "No endpoint provided"
                }
            print(f"Running healthcheck on {endpoint}")
            result = await run_healthcheck(endpoint)
            return result
        
        elif action_type == PlanActionType.FETCH:
            source = action_args["source"]
            method = action_args["method"]
            print(f"Fetched data from {source}.")
            if method == "get_email_content":
                result_string = "This is an example email. Assign a task to Himanshu to review the updated docs."
            else:
                result_string = f"TODO: Implement method: {method}."
            return {
                "success": True,
                "data": result_string,
                "error": None
            }
        
        elif action_type == PlanActionType.PROPOSE_PLAN:
            context = action_args["plan_context"]
            def create_plan_fn(context: str) -> str:
                if not context:
                    return "This is a failed plan."
                else:
                    return "Aha! I've proposed a plan."
            plan = create_plan_fn(context)
            return {
                "success": True,
                "data": f"Proposed plan: {plan}",
                "error": None
            }
        
        elif action_type == PlanActionType.REVISE_PLAN:
            proposed_plan = action_args["proposed_plan"]
            assert proposed_plan, "No proposed plan found"
            revision_context = action_args["revision_context"]
            if action_args.get("is_done", False):
                return {
                    "success": True,
                    "data": f"Final draft submitted. ({revision_context})",
                    "error": None
                }            
            return {
                "success": False,
                "data": f"Revised plan. ({revision_context})",
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