from typing import Optional
from datetime import datetime
from uuid import UUID
from typing_extensions import TypedDict
from pydantic import Field

from tangents.classes.actions import Action
from tangents.classes.tasks import Status, TaskState, TaskType
from tangents.db_schemas.base import BaseDocument


# Removing abstraction for now
class TaskDocument(TypedDict):
    id: UUID
    name: str
    type: str
    status: str
    created_at: datetime
    updated_at: datetime
    actions: list[dict]
    state: Optional[dict]


class TaskModel(BaseDocument):
    name: str = Field(...)
    type: TaskType
    status: Status = Field(default=Status.NOT_STARTED)
    actions: list[Action] = Field(default_factory=list)
    state: Optional[TaskState] = None

    # class Config:
    #     schema_extra = {
    #         'example': {
    #             '_id': '066de609-b04a-4b30-b46c-32537c7f1f6e',
    #             'name': 'Chat Task',
    #             'type': 'CHAT',
    #             'status': 'NOT_STARTED',
    #             'actions': [],
    #             'state': None,
    #         }
    #     }

    @classmethod
    def from_task(cls, task, name: str) -> "TaskModel":
        """Create a TaskModel instance from a Task object.
        
        Args:
            task: The Task object to convert
            name: Name for the task record
            
        Returns:
            A new TaskModel instance
        """
        return cls(
            name=name,
            type=task["type"],
            status=task["status"],
            actions=task["actions"],
            state=task["state"]
        )
    
    def to_task_document(self) -> TaskDocument:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'actions': self.actions,
            'state': self.state,
        }

    def to_dict(self) -> TaskDocument:
        # TODO: Ensure that actions and state only have serializable values
        return {
            'id': str(self.id),
            'name': self.name,
            'type': self.type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'actions': self.actions,
            'state': self.state,
        }
