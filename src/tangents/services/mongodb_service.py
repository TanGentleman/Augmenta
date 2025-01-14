from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from tangents.classes.tasks import Status
from tangents.db_schemas.tasks import TaskModel


class MongoDBConnection:
    """Base MongoDB connection handler."""
    
    def __init__(self, connection_string: str, db_name: str):
        """Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection URI
            db_name: Name of database to connect to
        """
        self.client = MongoClient(connection_string)
        self.db: Database = self.client[db_name]

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()


class TaskService(MongoDBConnection):
    """Service for task-related database operations."""

    def __init__(self, connection_string: str, task_db_name: str):
        super().__init__(connection_string, db_name=task_db_name)
        self.collection: Collection = self.db.tasks

    def get_task(
        self, task_name: Optional[str] = None, status: Optional[Status] = None
    ) -> Optional[TaskModel]:
        """Retrieve a task by name or status.
        
        Args:
            task_name: Name of task to retrieve
            status: Status to filter by
            
        Returns:
            TaskModel if found, None otherwise
        """
        if task_name:
            task = self.collection.find_one({'name': task_name})
        elif status:
            task = self.collection.find_one({'status': status.value})
        else:
            return None

        return TaskModel(**task) if task else None

    def save_task(self, task: TaskModel) -> bool:
        """Save or update a task.
        
        Args:
            task: Task model to save
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            task_document = task.to_dict()
            result = self.collection.update_one(
                {'name': task.name}, {'$set': task_document}, upsert=True
            )
            return result.acknowledged
        except Exception as e:
            print(f'Error saving task: {e}')
            return False
