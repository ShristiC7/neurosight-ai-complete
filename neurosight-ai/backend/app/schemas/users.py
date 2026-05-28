"""User request/response schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, ConfigDict


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: EmailStr
    avatar_url: str | None = None
    is_active: bool
    timezone: str
    preferences: dict
    created_at: datetime
