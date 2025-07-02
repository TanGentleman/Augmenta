import logging
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from tangents.classes.actions import (
    Action,
    ActionType,
    ActionResult,
    PlanActionType,
    Status,
)
from tangents.classes.tasks import Task, TaskType
from tangents.utils.action_utils import add_human_action
from tangents.utils.experimental import run_healthcheck, HEALTHCHECK_ENDPOINT

# Experimental
from tangents.experimental.utils import parse_convex_result
from tangents.experimental.youtwo_mcp import create_entities


async def default_stream_callback(text: str):
    print(text, end='', flush=True)

def start_action(action: Action, task: Task, runtime_config: Optional[dict] = None) -> Action:
    """Initialize an action with required state before execution."""
    action['status'] = Status.IN_PROGRESS
    task_state = task['state']
    action_args = action['args']
    
    def get_arg_value(arg_name, default=None, required=False):
        """Get argument value with priority: action_args > runtime_config > task_state."""
        # 1. Check if argument exists in action_args
        if arg_name in action_args and action_args[arg_name] is not None:
            return action_args[arg_name]
        
        # 2. Check if argument exists directly in runtime_config
        if runtime_config is not None and arg_name in runtime_config and runtime_config[arg_name] is not None:
            return runtime_config[arg_name]
        
        # 3. Check if argument exists in task_state
        if arg_name in task_state and task_state[arg_name] is not None:
            return task_state[arg_name]
        
        if default is None and required is True:
            raise ValueError(f'Missing required argument: {arg_name}')
        
        return default

    match action['type']:
        case ActionType.GENERATE:
            match task['type']:
                case TaskType.CHAT:
                    # Apply consistent priority pattern for all arguments
                    action_args['active_chain'] = get_arg_value('active_chain', required=True)
                    action_args['messages'] = get_arg_value('messages', required=True)
                    action_args['stream'] = get_arg_value('stream', required=True)
                    
                    # Special handling for stream_callback with default
                    action_args['stream_callback'] = get_arg_value(
                        'stream_callback', 
                        default=default_stream_callback,
                        required=True
                    )
                    
                    # Validate required arguments
                    if action_args['active_chain'] is None:
                        raise ValueError('No active chain found!')
                case _:
                    raise ValueError('Missing support in start_action for ActionType.GENERATE!')

        case ActionType.HEALTHCHECK:
            action_args['endpoint'] = get_arg_value('endpoint',
                                                    default=HEALTHCHECK_ENDPOINT,
                                                    required=True)

        case PlanActionType.PROPOSE_PLAN:
            action_args['plan_context'] = get_arg_value('plan_context', required=True)

        case PlanActionType.REVISE_PLAN:
            action_args['proposed_plan'] = get_arg_value('proposed_plan', required=True)
        
        case ActionType.TOOL_CALL:
            action_args['tool_name'] = get_arg_value('tool_name', required=True)
            action_args['tool_args'] = get_arg_value('tool_args', required=True)

    return action


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
    action_type = action['type']
    action_args = action['args']

    try:
        match action_type:
            case ActionType.GENERATE:
                stream = action_args['stream']
                chain = action_args['active_chain']
                messages = action_args['messages']
                stream_callback = action_args['stream_callback']
                # NOTE: We may have to clean up the stream callback after the action is done
                print(f"Stream callback: {bool(stream_callback)}")
                try:
                    if stream and stream_callback:
                        response_string = ''
                        async for chunk in chain.astream(messages):
                            chunk_content = chunk.content
                            if chunk_content:
                                response_string += chunk_content
                                # Yield chunk through callback instead of printing
                                await stream_callback(chunk_content)
                        
                        if not response_string:
                            raise ValueError('No response generated')
                    # This should not be reached
                    elif stream:
                        # Fallback to printing if no callback provided
                        response_string = ''
                        async for chunk in chain.astream(messages):
                            print(chunk.content, end='', flush=True)
                            response_string += chunk.content
                        print()
                        if not response_string:
                            raise ValueError('No response generated')
                    else:
                        response = await chain.ainvoke(messages)
                        print(response.content)
                        response_string = response.content
                    return {'success': True, 'data': response_string, 'error': None}

                except KeyboardInterrupt:
                    print('Keyboard interrupt, aborting generation.')
                    return {
                        'success': False,
                        'data': None,
                        'error': 'Generation interrupted',
                    }
                except Exception as e:
                    return {'success': False, 'data': None, 'error': str(e)}

            case ActionType.WEB_SEARCH:
                return {'success': True, 'data': 'Search results', 'error': None}

            case ActionType.SAVE_DATA:
                return {'success': True, 'data': 'Data saved', 'error': None}

            case ActionType.TOOL_CALL:
                print("FYI: Tool output will be stringified.")
                tool_name = action_args['tool_name']
                tool_args = action_args['tool_args']
                response_string = f"Tool called: {tool_name}"
                if tool_name == 'create_entities':
                    res = await create_entities(tool_args['entities'])
                    res = parse_convex_result(res)
                    response_string = str(res)
                return {'success': True, 'data': response_string, 'error': None}

            case ActionType.HEALTHCHECK:
                endpoint = action_args['endpoint']
                if not endpoint:
                    return {'success': False, 'data': None, 'error': 'No endpoint provided'}
                logging.info(f'Running healthcheck on {endpoint}')
                result = await run_healthcheck(endpoint)
                return result

            case PlanActionType.FETCH:
                source = action_args['source']
                method = action_args['method']
                logging.info(f'Fetched data from {source}.')
                match method:
                    case 'get_email_content':
                        result_string = 'This is an example email. Assign a task to Himanshu to review the updated docs.'
                    case _:
                        result_string = f'TODO: Implement method: {method}.'
                return {'success': True, 'data': result_string, 'error': None}

            case PlanActionType.PROPOSE_PLAN:
                context = action_args['plan_context']

                def create_plan_fn(context: str) -> str:
                    return 'This is a failed plan.' if not context else "Aha! I've proposed a plan."

                plan = create_plan_fn(context)
                return {'success': True, 'data': f'Proposed plan: {plan}', 'error': None}

            case PlanActionType.REVISE_PLAN:
                proposed_plan = action_args['proposed_plan']
                assert proposed_plan, 'No proposed plan found'
                revision_context = action_args['revision_context']
                if action_args.get('is_done', False):
                    return {
                        'success': True,
                        'data': f'Final draft submitted. ({revision_context})',
                        'error': None,
                    }
                return {
                    'success': False,
                    'data': f'Revised plan. ({revision_context})',
                    'error': None,
                }

            case _:
                return {
                    'success': False,
                    'data': None,
                    'error': f'Unknown action type: {action_type}',
                }

    except Exception as e:
        logging.error(f'Action execution failed: {str(e)}')
        return {'success': False, 'data': None, 'error': str(e)}


def handle_action_result(task: Task, action_result: ActionResult) -> Task:
    """Handle the result of an action based on task and action types."""
    action = task['actions'][0]
    task_state = task['state']

    if action_result['success']:
        action['status'] = Status.DONE

        match (task['type'], action['type']):
            case (_, ActionType.HEALTHCHECK):
                print(f"{action_result['data']}")

            case (TaskType.CHAT, ActionType.GENERATE):
                task_state['messages'].append(AIMessage(content=action_result['data']))

            case (TaskType.PLANNING, PlanActionType.FETCH):
                task_state['context'] = action_result['data']

            case (TaskType.PLANNING, PlanActionType.PROPOSE_PLAN):
                task_state['proposed_plan'] = action_result['data']

            case (TaskType.PLANNING, PlanActionType.REVISE_PLAN):
                task_state['plan'] = action_result['data']
                task['status'] = Status.DONE

            case (TaskType.EXPERIMENTAL, ActionType.TOOL_CALL):
                task_state['tool_call_result'] = action_result['data']
                task_state['tool_call'] = None
                task_state['tool_call_args'] = None
                print(f"Updated state! Experimental MCP tool call was a success.")

            case _:
                raise ValueError(
                    f"Unsupported task/action combination: {task['type']}/{action['type']}"
                )

    elif action_result['error']:
        action['status'] = Status.FAILED
        # Handle error cleanup
        match (task['type'], action['type']):
            case (TaskType.CHAT, ActionType.GENERATE):
                if isinstance(task_state['messages'][-1], HumanMessage):
                    task_state['messages'].pop()
            case _:
                pass  # Other types may not need cleanup

    else:
        # Retry cases
        match (task['type'], action['type']):
            case (TaskType.PLANNING, PlanActionType.REVISE_PLAN):
                if 'revision_count' not in task_state:
                    task_state['revision_count'] = 0
                task_state['revision_count'] += 1
                action['args']['revision_context'] = f"Revision {task_state['revision_count']}"

                DEFAULT_MAX_REVISIONS = 3
                max_revisions = action['args'].get('max_revisions', DEFAULT_MAX_REVISIONS)
                if task_state['revision_count'] >= max_revisions:
                    action['args']['is_done'] = True
                else:
                    add_human_action(
                        task['actions'],
                        prompt=f"Review revision #{task_state['revision_count']}. Enter 'y' to finalize or any other input to revise again.",
                    )
            case _:
                pass

    return task
