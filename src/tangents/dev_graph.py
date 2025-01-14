import asyncio
import logging
from uuid import uuid4

from tangents.classes.actions import ActionType, Status
from tangents.tan_graph import (
    create_workflow,
    get_default_state_dict,
    get_initial_graph_state,
    get_planning_state_dict,
    process_graph_updates,
)
from tangents.utils.action_utils import create_action
from tangents.utils.task_utils import get_task

RECURSION_LIMIT = 50
DEV_MODE = False


async def main_async(stream_mode: str = 'updates') -> None:
    """
    Run the async workflow with streaming support.

    Args:
        stream_mode: Output processing mode ("updates" or "values")
    """
    app = create_workflow()

    # Initialize workflow state
    graph_state = get_initial_graph_state()
    graph_state['keys'] = get_default_state_dict() if not DEV_MODE else get_planning_state_dict()

    state_dict = graph_state['keys']
    current_task = get_task(state_dict['task_dict'], status=Status.NOT_STARTED)
    if not current_task:
        raise ValueError('Chat task not found!')

    # Add a healthcheck action
    healthcheck_action = create_action(ActionType.HEALTHCHECK)
    current_task['actions'].append(healthcheck_action)

    # Add a human action
    human_action = create_action(
        ActionType.HUMAN_INPUT,
        args={'prompt': 'What is the answer to life, the universe, and everything?'},
    )
    current_task['actions'].append(human_action)

    # # Add a mock input
    state_dict['mock_inputs'] = ['42']

    # Add a stash action to skip generation
    SKIP_GENERATION = True
    if SKIP_GENERATION:
        stash_action = create_action(ActionType.STASH)
        current_task['actions'].append(stash_action)

    # Now generation is queued for the mock input!

    # Configure runtime settings
    app_config = {
        'recursion_limit': RECURSION_LIMIT,
        'configurable': {'thread_id': uuid4()},
    }

    # Process workflow outputs
    async for output in app.astream(graph_state, app_config, stream_mode=stream_mode):
        await process_graph_updates(output, app, app_config)

    # get state
    final_state = app.get_state(app_config).values
    current_task = get_task(final_state['keys']['task_dict'])
    logging.info(f'Current task: {current_task}')

    # TODO: Allow a better "resume" workflow (Must fetch the task from the database)
    # remove the stash action
    # assert current_task['actions'][0]['type'] == ActionType.STASH
    # current_task['actions'].pop(0)
    # # resume the workflow
    # async for output in app.astream(final_state, app_config, stream_mode=stream_mode):
    #     await process_graph_updates(output, app, app_config)

    print('Workflow completed.')


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main_async())
