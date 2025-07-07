import logging
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from tangents.classes.actions import (
    Action,
    ActionType,
    ActionResult,
    PlanActionType,
    Status,
)
from tangents.classes.tasks import Task, TaskType
from tangents.utils.action_utils import add_human_action, create_action
from tangents.utils.chains import fast_get_llm
from tangents.utils.experimental import run_healthcheck, HEALTHCHECK_ENDPOINT

# Experimental
from tangents.experimental.utils import get_proposed_plan, parse_convex_result
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
                    action_args['chain'] = get_arg_value('chain', required=True)
                    action_args['messages'] = get_arg_value('messages', required=True)
                    action_args['stream'] = get_arg_value('stream', required=True)
                    
                    # Special handling for stream_callback with default
                    action_args['stream_callback'] = get_arg_value(
                        'stream_callback', 
                        default=default_stream_callback,
                        required=True
                    )
                    
                    # Validate required arguments
                    if action_args['chain'] is None:
                        raise ValueError('No active chain found!')
                
                case TaskType.PLANNING:
                    # If we are on proposal, check for proposal chain
                    if task_state['proposed_plan'] is None:
                        action_args['chain'] = get_arg_value('proposal_chain', required=True)
                        proposal_system_prompt = get_arg_value('proposal_system_prompt', required=True)
                        action_args['messages'] = [
                            SystemMessage(content=proposal_system_prompt),
                            HumanMessage(content=task_state['plan_context']),
                        ]
                    else:
                        assert task_state['proposed_plan'], 'No plan context found!'
                        action_args['chain'] = get_arg_value('revision_chain', required=True)
                        revision_system_prompt = get_arg_value('revision_system_prompt', required=True)
                        action_args['messages'] = [
                            SystemMessage(content=revision_system_prompt),
                            AIMessage(content=task_state['proposed_plan']),
                            HumanMessage(content=task_state['human_feedback']),
                        ]
                    action_args['stream'] = get_arg_value('stream', default=True, required=True)
                    # Special handling for stream_callback with default
                    action_args['stream_callback'] = get_arg_value(
                        'stream_callback', 
                        default=default_stream_callback,
                        required=True
                    )

                case _:
                    raise ValueError('Missing support in start_action for ActionType.GENERATE!')

        case ActionType.HEALTHCHECK:
            action_args['endpoint'] = get_arg_value('endpoint',
                                                    default=HEALTHCHECK_ENDPOINT,
                                                    required=True)

        case PlanActionType.PROPOSE_PLAN:
            action_args['plan_context'] = get_arg_value('plan_context', required=True)
            action_args['proposed_plan'] = get_arg_value('proposed_plan', required=True)

        case PlanActionType.REVISE_PLAN:
            action_args['proposed_plan'] = get_arg_value('proposed_plan', required=True)
            action_args['revision_context'] = get_arg_value('revision_context', required=True, default='No revision context found')
        
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
                chain = action_args['chain']
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
                        result_string = 'This is an example email. Assign a task to review the updated docs.'
                    case 'convex_mcp_spec':
                        filepath = Path(source).expanduser()
                        if not filepath.exists():
                            return {
                                'success': False,
                                'data': None,
                                'error': f'File not found: {source}',
                            }
                        with open(filepath, 'r') as file:
                            result_string = file.read()
                            # Validate that the result_string is valid MCP spec
                        if not result_string:
                            return {
                                'success': False,
                                'data': None,
                                'error': 'Failed to read MCP spec file',
                            }
                    case _:
                        result_string = f'TODO: Implement method: {method}.'
                return {'success': True, 'data': result_string, 'error': None}

            case PlanActionType.PROPOSE_PLAN:
                context = action_args['plan_context']
                proposed_plan = action_args['proposed_plan']
                if context is None and proposed_plan is None:
                    return {
                        'success': False,
                        'data': None,
                        'error': 'Missing plan context or proposed plan',
                    }
                # Reformat the proposed plan using reformatter
                logging.info(f"This action ensures context->proposal is complete and can reformat!")
                return {'success': True, 'data': f'Proposed plan: {proposed_plan}', 'error': None}

            case PlanActionType.REVISE_PLAN:
                proposed_plan = action_args['proposed_plan']
                revision_context = action_args['revision_context']
                assert proposed_plan, 'No proposed plan found'
                assert revision_context, 'No revision context found'
                # NOTE: Pass proposed plan as the data for the action result
                if action_args.get('is_done') is True:
                    return {
                        'success': True,
                        'data': f'Final plan: {proposed_plan}\n\n({revision_context})',
                        'error': None,
                    }
                print("Simulating modification of plan...")
                return {
                    'success': False,
                    'data': proposed_plan,
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

    # Clean up non-serializable objects from action args to prevent serialization issues
    non_serializable_keys = ['stream_callback', 'chain', 'proposal_chain', 'revision_chain']
    for key in non_serializable_keys:
        if key in action['args']:
            del action['args'][key]

    if action_result['success']:
        action['status'] = Status.DONE

        match (task['type'], action['type']):
            case (_, ActionType.HEALTHCHECK):
                print(f"{action_result['data']}")

            case (TaskType.CHAT, ActionType.GENERATE):
                task_state['messages'].append(AIMessage(content=action_result['data']))

            case (TaskType.PLANNING, ActionType.GENERATE):
                # Check if next action is to propose a plan
                logging.info(f"Using Generate action response as proposed plan!")
                task_state['proposed_plan'] = action_result['data']

            case (TaskType.PLANNING, PlanActionType.FETCH):
                task_state['plan_context'] = action_result['data']
                # Add a generate action to the task
                generate_action = create_action(ActionType.GENERATE)
                task['actions'].insert(1, generate_action)
            case (TaskType.PLANNING, PlanActionType.PROPOSE_PLAN):
                task_state['proposed_plan'] = action_result['data']
                # Generation for revision
                task['actions'].insert(1, create_action(ActionType.HUMAN_INPUT, {
                    'prompt': f"Review the proposed plan. Enter 'y' to finalize or any other input to revise again.",
                }))
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
                # NOTE: We can update the shared state before action is fully executed
                task_state['proposed_plan'] = action_result['data']
                # Clear human feedback?
                # action['args']['revision_context'] = f"Revision {task_state['revision_count']}"

                if task_state['revision_count'] >= action['args']['max_revisions']:
                    action['args']['is_done'] = True
                else:
                    add_human_action(
                        task['actions'],
                        prompt=f"Review revision #{task_state['revision_count']}. Enter 'y' to finalize or any other input to revise again.",
                    )
            case _:
                pass

    return task
