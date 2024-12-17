
import logging
from typing import Optional
from tangents.graph_classes import Task, TaskStatus

def get_task(task_dict: dict[str, Task], task_name: Optional[str] = None, status: Optional[TaskStatus] = TaskStatus.IN_PROGRESS) -> Task | None:
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
    return get_task(task_dict, status=TaskStatus.IN_PROGRESS)

def save_task_to_convex(task_name: str, task: Task, to_fail_table: bool = False) -> bool:
    """Save the task to the convex database"""
    # TODO: Implement this
    return True

def save_completed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save completed tasks to the task dictionary"""
    completed_tasks = []
    for task_name, task in task_dict.items():
        if task["status"] == TaskStatus.DONE:
            success = save_task_to_convex(task_name, task)
            if success:
                completed_tasks.append(task_name)
            else:
                logging.error(f"Failed to save task {task_name} to convex")

    return completed_tasks

def save_failed_tasks(task_dict: dict[str, Task]) -> list[str]:
    """Save failed tasks to the task dictionary"""
    failed_tasks = []
    for task_name, task in task_dict.items():
        if task["status"] == TaskStatus.FAILED:
            success = save_task_to_convex(task_name, task, to_fail_table=True)
            if success:
                failed_tasks.append(task_name)
            else:
                logging.error(f"Failed to save task {task_name} to convex")

    return failed_tasks

def start_next_task(task_dict: dict[str, Task]) -> bool:
    """Start the next task in the task dictionary. Returns True if a task is in progress."""
    current_task = get_task(task_dict, status=TaskStatus.IN_PROGRESS)
    if current_task:
        print("Task is already in progress!")
        return True
    
    next_task = get_task(task_dict, status=TaskStatus.NOT_STARTED)
    if next_task:
        next_task["status"] = TaskStatus.IN_PROGRESS
        return True
    return False
