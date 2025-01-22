"""MongoDB collection setup and initialization."""
import logging
from pymongo import IndexModel, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid
from tangents.services.db_init import init_mongodb_connection

logger = logging.getLogger(__name__)

# Schema validation for tasks collection
TASK_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["id", "name", "type", "status", "created_at", "updated_at", "actions", "state"],
        "properties": {
            "id": {
                "bsonType": "uuid",
                "description": "Unique task identifier"
            },
            "name": {
                "bsonType": "string", 
                "description": "Task name - required and must be unique"
            },
            "type": {
                "bsonType": "string",
                "description": "Task type"
            },
            "status": {
                "bsonType": "string",
                "description": "Task status"
            },
            "created_at": {
                "bsonType": "date",
                "description": "Creation timestamp"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "Last update timestamp"
            },
            "actions": {
                "bsonType": "array",
                "description": "List of task actions",
                "items": {
                    "bsonType": "object"
                }
            },
            "state": {
                "bsonType": ["object", "null"],
                "description": "Task state information"
            }
        }
    }
}

def setup_tasks_collection():
    """Initialize tasks collection with indexes and schema validation."""
    try:
        service = init_mongodb_connection()
        if not service:
            logger.error("Failed to initialize MongoDB connection")
            return False

        db = service.db
        
        # Create collection with validation
        try:
            db.create_collection(
                "tasks",
                validator=TASK_SCHEMA,
                validationAction="error"
            )
            logger.info("Tasks collection created successfully")
        except CollectionInvalid:
            # Collection already exists
            db.command({
                "collMod": "tasks",
                "validator": TASK_SCHEMA,
                "validationAction": "error"
            })
            logger.info("Updated existing tasks collection schema")

        # Create indexes
        collection = db.tasks
        indexes = [
            IndexModel([("name", ASCENDING)], unique=True),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("updated_at", DESCENDING)])
        ]
        
        collection.create_indexes(indexes)
        logger.info("Created indexes for tasks collection")

        # Verify indexes
        index_info = collection.index_information()
        logger.info(f"Current indexes: {list(index_info.keys())}")

        return True

    except Exception as e:
        logger.error(f"Error setting up tasks collection: {e}")
        return False
    finally:
        if service:
            service.close()

if __name__ == "__main__":
    if setup_tasks_collection():
        logger.info("Successfully initialized MongoDB collections")
    else:
        logger.error("Failed to initialize MongoDB collections")
        exit(1) 