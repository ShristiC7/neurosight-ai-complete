"""
NeuroSight AI — FastAPI Dependencies
Re-exports commonly used dependencies for clean imports.
"""

from app.core.security import get_current_user, get_current_user_optional

__all__ = ["get_current_user", "get_current_user_optional"]
