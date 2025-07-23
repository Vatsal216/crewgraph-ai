"""
Authentication and Authorization for CrewGraph AI + Langflow Integration

This module provides JWT-based authentication and RBAC authorization
leveraging the existing CrewGraph security infrastructure.

Created by: Vatsal216
Date: 2025-07-23
"""

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Import existing CrewGraph security components
try:
    from ....security.security_manager import SecurityManager, SessionStatus
    from ....security.rbac import RoleManager, Permission, User
    from ....security.audit_logger import AuditLogger, AuditEvent
    from ....utils.logging import get_logger
except ImportError:
    # Fallback for development/testing
    SecurityManager = None
    RoleManager = None
    Permission = None
    User = None
    AuditLogger = None
    AuditEvent = None
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

from ..config import get_config

logger = get_logger(__name__)
security = HTTPBearer()


class TokenData(BaseModel):
    """JWT token data model"""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = []
    permissions: List[str] = []
    session_id: Optional[str] = None
    expires_at: datetime


class AuthManager:
    """
    Authentication and Authorization Manager for Langflow Integration
    
    Integrates with existing CrewGraph security infrastructure to provide:
    - JWT token generation and validation
    - Role-based access control (RBAC)
    - Session management
    - Audit logging
    """
    
    def __init__(self):
        self.config = get_config()
        self.security_manager = SecurityManager() if SecurityManager else None
        self.role_manager = RoleManager() if RoleManager else None
        self.audit_logger = AuditLogger() if AuditLogger else None
        
        # Required permissions for different operations
        self.permissions_map = {
            "workflow:export": ["langflow.workflow.export"],
            "workflow:import": ["langflow.workflow.import"],
            "workflow:execute": ["langflow.workflow.execute"],
            "workflow:status": ["langflow.workflow.read"],
            "component:register": ["langflow.component.create"],
            "component:list": ["langflow.component.read"],
            "admin:all": ["langflow.admin.*"],
        }
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """
        Create JWT access token for user
        
        Args:
            user_data: User information including id, username, roles, etc.
            
        Returns:
            JWT token string
        """
        if not self.config.jwt_secret_key:
            raise ValueError("JWT secret key not configured")
        
        # Calculate expiration time
        expires_at = datetime.now(timezone.utc) + timedelta(hours=self.config.jwt_expire_hours)
        
        # Prepare token payload
        payload = {
            "user_id": user_data.get("user_id", user_data.get("id")),
            "username": user_data.get("username"),
            "email": user_data.get("email"),
            "roles": user_data.get("roles", []),
            "permissions": user_data.get("permissions", []),
            "session_id": user_data.get("session_id"),
            "exp": expires_at.timestamp(),
            "iat": datetime.now(timezone.utc).timestamp(),
            "iss": "crewgraph-langflow-api",
            "aud": "langflow-integration"
        }
        
        # Create token
        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        # Log authentication event
        if self.audit_logger:
            self.audit_logger.log_event(
                AuditEvent(
                    event_type="authentication.token_created",
                    user_id=payload["user_id"],
                    resource="langflow_api",
                    details={"username": payload["username"], "expires_at": expires_at.isoformat()}
                )
            )
        
        return token
    
    def validate_token(self, token: str) -> TokenData:
        """
        Validate JWT token and return token data
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData object with user information
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                audience="langflow-integration"
            )
            
            # Extract token data
            token_data = TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload.get("email"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                session_id=payload.get("session_id"),
                expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            )
            
            # Check if token is expired
            if token_data.expires_at < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            # Validate session if session management is enabled
            if self.security_manager and token_data.session_id:
                session = self.security_manager.get_session(token_data.session_id)
                if not session or session.status != SessionStatus.ACTIVE:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Session invalid or expired"
                    )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed"
            )
    
    def check_permissions(self, token_data: TokenData, required_permissions: List[str]) -> bool:
        """
        Check if user has required permissions
        
        Args:
            token_data: User token data
            required_permissions: List of required permissions
            
        Returns:
            True if user has all required permissions
        """
        if not self.config.enable_rbac:
            return True  # RBAC disabled, allow all
        
        user_permissions = set(token_data.permissions)
        
        # Check for admin permissions
        if "langflow.admin.*" in user_permissions:
            return True
        
        # Check specific permissions
        for perm in required_permissions:
            if perm not in user_permissions:
                # Check for wildcard permissions
                perm_parts = perm.split(".")
                for i in range(len(perm_parts)):
                    wildcard_perm = ".".join(perm_parts[:i+1]) + ".*"
                    if wildcard_perm in user_permissions:
                        break
                else:
                    return False
        
        return True
    
    def get_user_from_crewgraph(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from CrewGraph security system
        
        Args:
            username: Username to lookup
            
        Returns:
            User data dictionary or None if not found
        """
        if not self.role_manager:
            return None
        
        try:
            # This would integrate with the actual CrewGraph user system
            # For now, return a mock user for development
            return {
                "user_id": hashlib.md5(username.encode()).hexdigest()[:12],
                "username": username,
                "email": f"{username}@example.com",
                "roles": ["user"],
                "permissions": [
                    "langflow.workflow.read",
                    "langflow.workflow.execute",
                    "langflow.component.read"
                ]
            }
        except Exception as e:
            logger.error(f"Error getting user from CrewGraph: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and return access token
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token if authentication successful, None otherwise
        """
        try:
            # In a real implementation, this would validate against the CrewGraph user system
            # For development, we'll accept any non-empty username/password
            if not username or not password:
                return None
            
            # Get user data
            user_data = self.get_user_from_crewgraph(username)
            if not user_data:
                return None
            
            # Create session if security manager available
            session_id = None
            if self.security_manager:
                session = self.security_manager.create_session(
                    user_id=user_data["user_id"],
                    metadata={"source": "langflow_api", "username": username}
                )
                session_id = session.session_id
            
            user_data["session_id"] = session_id
            
            # Create access token
            token = self.create_access_token(user_data)
            
            # Log authentication success
            if self.audit_logger:
                self.audit_logger.log_event(
                    AuditEvent(
                        event_type="authentication.login_success",
                        user_id=user_data["user_id"],
                        resource="langflow_api",
                        details={"username": username, "method": "password"}
                    )
                )
            
            return token
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            
            # Log authentication failure
            if self.audit_logger:
                self.audit_logger.log_event(
                    AuditEvent(
                        event_type="authentication.login_failed",
                        user_id=username,
                        resource="langflow_api",
                        details={"username": username, "error": str(e)}
                    )
                )
            
            return None


# Global auth manager instance
auth_manager = AuthManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """
    FastAPI dependency to get current authenticated user
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        TokenData for the authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        return auth_manager.validate_token(token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication dependency error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


def require_permissions(*permissions: str):
    """
    Decorator factory to require specific permissions
    
    Args:
        *permissions: Required permission strings
        
    Returns:
        FastAPI dependency function
    """
    def permission_dependency(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        """Check if user has required permissions"""
        if not auth_manager.check_permissions(current_user, list(permissions)):
            # Log authorization failure
            if auth_manager.audit_logger:
                auth_manager.audit_logger.log_event(
                    AuditEvent(
                        event_type="authorization.access_denied",
                        user_id=current_user.user_id,
                        resource="langflow_api",
                        details={
                            "required_permissions": list(permissions),
                            "user_permissions": current_user.permissions
                        }
                    )
                )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {', '.join(permissions)}"
            )
        
        return current_user
    
    return permission_dependency


# Convenience permission dependencies
require_workflow_export = require_permissions("langflow.workflow.export")
require_workflow_import = require_permissions("langflow.workflow.import")
require_workflow_execute = require_permissions("langflow.workflow.execute")
require_workflow_read = require_permissions("langflow.workflow.read")
require_component_create = require_permissions("langflow.component.create")
require_component_read = require_permissions("langflow.component.read")
require_admin = require_permissions("langflow.admin.*")