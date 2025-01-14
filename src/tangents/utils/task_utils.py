import logging
from typing import Optional

from tangents.classes.settings import Config
from tangents.classes.tasks import Status, Task, TaskType
from tangents.utils.action_utils import is_stash_action_next
from tangents.utils.chains import fast_get_llm
from tangents.utils.message_utils import insert_system_message


def get_task(
    task_dict: dict[str, Task],
    task_name: Optional[str] = None,
    status: Optional[Status] = Status.IN_PROGRESS,
) -> Task | None:
    """Get a task from the task dictionary by name or status."""
    if task_name:
        task = task_dict.get(task_name)
    elif status:
        task = next((task for task in task_dict.values() if task['status'] == status), None)
    else:
        raise ValueError('Either task_name or status must be provided')

    if task is None:
        logging.debug(f'Task not found! {task_name or str(status)}')
        return None

    return task


def get_current_task(task_dict: dict[str, Task]) -> Task | None:
    """Get the currently in-progress task."""
    return get_task(task_dict, status=Status.IN_PROGRESS)


def save_task_to_convex(task_name: str, task: Task) -> bool:
    """Save task to Convex database."""
    # TODO: Implement this
    logging.info(f'Saving task {task_name} to convex')
    return True


def save_tasks_by_status(task_dict: dict[str, Task], status: Status) -> list[str]:
    """Save tasks with given status and return list of saved task names."""
    saved_tasks = []
    for task_name, task in task_dict.items():
        if task['status'] == status:
            success = save_task_to_convex(task_name, task)
            if success:
                logging.info(f'Saved {status.value} task')
                saved_tasks.append(task_name)
            else:
                logging.error(f'Failed to save {status.value} task {task_name}')
    return saved_tasks


def save_completed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save completed tasks and return list of saved task names."""
    return save_tasks_by_status(task_dict, Status.DONE)


def save_failed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save failed tasks and return list of saved task names."""
    return save_tasks_by_status(task_dict, Status.FAILED)


def stash_task(task: Task) -> bool:
    """Stash a task for later processing."""
    logging.info('Stashed task')
    return True


def save_stashed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save stashable tasks and return list of stashed task names."""
    stashed_tasks = []
    for task_name, task in task_dict.items():
        if task['status'] == Status.IN_PROGRESS and is_stash_action_next(task['actions']):
            success = stash_task(task)
            if success:
                stashed_tasks.append(task_name)
            else:
                logging.error(f'Failed to stash task {task_name}')
    return stashed_tasks


def initialize_task_state(task: Task, config: Config) -> dict:
    """Initialize task state based on task type."""
    match task['type']:
        case TaskType.CHAT:
            task_state = {
                'messages': [],
                'active_chain': None,
                'stream': config.chat_settings.stream,
            }
            if not config.chat_settings.disable_system_message:
                insert_system_message(task_state['messages'], config.chat_settings.system_message)

            model_name = config.chat_settings.primary_model
            logging.info(f'Initializing chain with model {model_name}!')
            llm = fast_get_llm(model_name)
            if llm is None:
                raise ValueError('Chain not initialized!')
            task_state['active_chain'] = llm

        case TaskType.RAG:
            if not config.rag_settings.enabled:
                raise ValueError('RAG task is disabled!')
            task_state = {}

        case TaskType.PLANNING:
            task_state = {
                'context': None,
                'proposed_plan': None,
                'plan': None,
                'revision_count': 0,
            }

        case _:
            raise ValueError('Invalid task type!')

    return task_state


def start_task(task: Task, config: Config) -> Task:
    """Initialize and start a new task."""
    if task['status'] != Status.NOT_STARTED:
        raise ValueError('Task must have NOT_STARTED status to start!')

    if task['state'] is None:
        task['state'] = initialize_task_state(task, config)

    task['status'] = Status.IN_PROGRESS
    print(f"Started task: {task['type']}")
    return task
