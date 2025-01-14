from typing import Optional
from datetime import datetime
from typing_extensions import TypedDict
from uuid import uuid4
from pydantic import BaseModel, Field

from tangents.classes.actions import Action
from tangents.classes.tasks import Status, TaskState, TaskType
from tangents.db_schemas.base import BaseDocument


# Removing abstraction for now
class TaskDocument(TypedDict):
    id: str
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

    def to_dict(self) -> TaskDocument:
        # TODO: Ensure that actions and state only have serializable values
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
