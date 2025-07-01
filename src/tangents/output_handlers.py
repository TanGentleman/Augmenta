"""Module for handling workflow output processing and display."""

import logging

from .classes.states import GraphState

logger = logging.getLogger(__name__)


class OutputProcessor:
    """Processes and displays workflow output."""

    @staticmethod
    def format_task_status(task_dict) -> str:
        """Format task status information for display."""
        if not task_dict:
            return 'Task: None'
        
        status_lines = []
        for task_name, task in task_dict.items():
            if hasattr(task, 'get') and task.get('status'):
                status_lines.append(f"{task_name}: {task['status'].value}")
            elif hasattr(task, '__getitem__') and 'status' in task:
                status_lines.append(f"{task_name}: {task['status'].value}")
            else:
                status_lines.append(f"{task_name}: Unknown status")
        
        return '\n'.join(status_lines) if status_lines else 'Task: None'

    @staticmethod
    def print_values(graph_state: GraphState) -> None:
        """Process and display full graph state output."""
        task_dict = graph_state['keys'].get('task_dict')
        status = OutputProcessor.format_task_status(task_dict)
        if status:
            print(status or 'Task: None')
            print('\n---\n')

    @staticmethod
    def print_updates(node: str, updates: dict) -> None:
        """Process and display node updates."""
        if not isinstance(updates, dict):
            raise ValueError('Updates must be a dictionary.')
        hops = updates.get('mutation_count', None)
        print(f'\nNode: {node}\nHops: {hops}')

        if 'task_dict' in updates.get('keys', {}):
            task_dict = updates['keys']['task_dict']
            for task_name, task in task_dict.items():
                print(f"{task_name}: {task['status'].value}")
        else:
            print('Task: None')
        print('\n---\n')
