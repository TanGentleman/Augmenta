
import logging
from typing import Optional
from tangents.classes.tasks import Task, Status
from tangents.utils.action_utils import is_stash_action_next

def get_task(task_dict: dict[str, Task], task_name: Optional[str] = None, status: Optional[Status] = Status.IN_PROGRESS) -> Task | None:
    """Get a task from the task dictionary"""
    if task_name:
        task = task_dict.get(task_name)
    elif status:
        task = next((task for task in task_dict.values() if task["status"] == status), None)
    else:
        raise ValueError("Either task_name or status must be provided")
    
    if not task:
        logging.debug(f"Task not found! {task_name or str(status)}")
        return None
    
    return task

def get_current_task(task_dict: dict[str, Task]) -> Task | None:
    """Get the next in progress task from the task dictionary"""
    return get_task(task_dict, status=Status.IN_PROGRESS)

def save_task_to_convex(task_name: str, task: Task, to_fail_table: bool = False) -> bool:
    """Save the task to the convex database"""
    # TODO: Implement this
    logging.info(f"Saving task {task_name} to convex")
    return True

def save_completed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save completed tasks to the task dictionary"""
    completed_tasks = []
    for task_name, task in task_dict.items():
        if task["status"] == Status.DONE:
            success = save_task_to_convex(task_name, task)
            if success:
                logging.info("Saved completed task")
                completed_tasks.append(task_name)
            else:
                logging.error(f"Failed to save completed task {task_name}")

    return completed_tasks

def save_failed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save failed tasks to the task dictionary"""
    failed_tasks = []
    for task_name, task in task_dict.items():
        if task["status"] == Status.FAILED:
            success = save_task_to_convex(task_name, task, to_fail_table=True)
            if success:
                logging.info("Saved failed task")
                failed_tasks.append(task_name)
            else:
                logging.error(f"Failed to save failed task {task_name}")

    return failed_tasks

def stash_task(task: Task) -> bool:
    """Stash a task."""
    logging.info("Stashed task")
    return True

def save_stashed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save stashed tasks to the task dictionary"""
    stashed_tasks = []
    for task_name, task in task_dict.items():
        if task["status"] == Status.IN_PROGRESS:
            if is_stash_action_next(task["actions"]):
                success = stash_task(task)
                if success:
                    logging.info("Stashed task")
                    stashed_tasks.append(task_name)
                else:
                    logging.error(f"Failed to stash task {task_name}")

    return stashed_tasks

