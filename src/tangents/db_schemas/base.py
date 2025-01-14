from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

class BaseDocument(BaseModel):
    """Base document model with common MongoDB fields"""
    id: Optional[UUID] = Field(default_factory=uuid4, alias="_id")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        # Allow population by field name or alias
        populate_by_name = True
        # Configure for MongoDB compatibility
        json_encoders = {
            UUID: str,
            datetime: lambda dt: dt.isoformat()
        }
