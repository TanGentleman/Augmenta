"""Module for handling workflow output processing and display."""

import logging
from .classes.states import GraphState

logger = logging.getLogger(__name__)


class TaskOutputFormatter:
    """Handles formatting and display of task-related output."""

    @staticmethod
    def format_task_status(task_dict: dict) -> str:
        """Format the status of the first task in task_dict."""
        if not task_dict:
            return ''

        for task_name, task in task_dict.items():
            return f"{task_name}: {task['status'].name}"
        raise ValueError('Unhandled task status.')

    @staticmethod
    def format_node_info(node: str, updates: dict) -> str:
        """Format node information including mutation count."""
        hops = updates.get('mutation_count', None) if isinstance(updates, dict) else updates
        return f'\nNode: {node}\nHops: {hops}'


class OutputProcessor:
    """Processes and displays workflow output."""

    def __init__(self, formatter: TaskOutputFormatter = None):
        self.formatter = formatter or TaskOutputFormatter()

    def process_values(self, graph_state: GraphState) -> None:
        """Process and display full graph state output."""
        task_dict = graph_state['keys'].get('task_dict')
        status = self.formatter.format_task_status(task_dict)
        if status:
            print(status or 'Task: None')
            print('\n---\n')

    def process_updates(self, node: str, updates: dict) -> None:
        """Process and display node updates."""
        print(self.formatter.format_node_info(node, updates))

        if 'task_dict' in updates.get('keys', {}):
            status = self.formatter.format_task_status(updates['keys']['task_dict'])
            if status:
                print(status or 'Task: None')
        print('\n---\n')
