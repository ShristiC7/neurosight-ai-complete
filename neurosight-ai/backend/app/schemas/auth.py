"""Auth request/response schemas."""

from pydantic import BaseModel, EmailStr, Field
from app.schemas.users import UserResponse


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128,
                          description="Minimum 8 characters")


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: int  # Unix timestamp milliseconds
    user: UserResponse
