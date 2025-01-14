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
    """Retrieve a task by name or status from the task dictionary."""
    if task_name is not None:
        task = task_dict.get(task_name)
    elif status is not None:
        task = next((task for task in task_dict.values() if task['status'] == status), None)
    else:
        raise ValueError('Either task_name or status must be provided')

    if task is None:
        logging.debug(f'Task not found! {task_name or str(status)}')
        return None

    return task


def get_current_task(task_dict: dict[str, Task]) -> Task | None:
    """Get the task currently in progress."""
    return get_task(task_dict, status=Status.IN_PROGRESS)


def save_task_to_convex(task_name: str, task: Task) -> bool:
    """Save a task to the Convex database (placeholder)."""
    # TODO: Implement this
    logging.info(f'Saving task {task_name} to convex')
    return True


def save_tasks_by_status(task_dict: dict[str, Task], status: Status) -> list[str]:
    """Save tasks with a specific status and return their names."""
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
    """Save tasks marked as DONE and return their names."""
    return save_tasks_by_status(task_dict, Status.DONE)


def save_failed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save tasks marked as FAILED and return their names."""
    return save_tasks_by_status(task_dict, Status.FAILED)


def stash_task(task: Task) -> bool:
    """Stash a task for later processing. (placeholder)"""
    logging.info('Stashed task')
    return True


def save_stashed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save tasks ready for stashing and return their names."""
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
    """Initialize the state of a task based on its type."""
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
    """Set task status to IN_PROGRESS and initialize its state."""
    if task['status'] != Status.NOT_STARTED:
        raise ValueError('Task must have NOT_STARTED status to start!')

    if task['state'] is None:
        task['state'] = initialize_task_state(task, config)

    task['status'] = Status.IN_PROGRESS
    print(f"Started task: {task['type']}")
    return task
