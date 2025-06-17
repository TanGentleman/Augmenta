import asyncio
import logging
from uuid import uuid4

from tangents.classes.actions import Status
from tangents.tan_graph import (
    create_workflow,
    process_graph_updates,
)
from tangents.utils.task_utils import get_task
from tangents.experimental.utils import get_experimental_state_dict
from tangents.template import get_initial_graph_state

RECURSION_LIMIT = 20
DEV_MODE = True

MOCK_INPUTS: list[str] = ['entity', '/quit']


async def main_async(stream_mode: str = 'updates') -> None:
    """
    Run the async workflow with streaming support.

    Args:
        stream_mode: Output processing mode ("updates" or "values")
    """
    app = create_workflow()

    # Initialize workflow state
    graph_state = get_initial_graph_state()
    graph_state['keys'] = get_experimental_state_dict()

    state_dict = graph_state['keys']
    current_task = get_task(state_dict['task_dict'], status=Status.NOT_STARTED)
    if not current_task:
        raise ValueError('Experimental task not found!')

    state_dict['mock_inputs'] = MOCK_INPUTS

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

    print('Workflow completed.')


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main_async())
