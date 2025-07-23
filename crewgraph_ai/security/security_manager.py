"""
Security Manager - Comprehensive security management for CrewGraph AI

This module provides centralized security management including:
- Authentication and authorization
- Session management
- Security policy enforcement
- Integration with RBAC, audit logging, and encryption
- Security validation and compliance

Features:
- Configurable security policies
- Session-based access control
- Security event monitoring
- Compliance reporting
- Integration with existing workflows

Created by: Vatsal216
Date: 2025-07-23
"""

import time
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import threading

from .rbac import RoleManager, Role, Permission, User
from .audit_logger import AuditLogger, AuditEvent
from .encryption import EncryptionManager, CryptoConfig
from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class SessionStatus(Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    description: str = ""
    enabled: bool = True
    min_security_level: SecurityLevel = SecurityLevel.INTERNAL
    max_session_duration: int = 3600  # seconds
    require_encryption: bool = True
    audit_all_operations: bool = True
    allow_delegation: bool = True
    ip_whitelist: List[str] = field(default_factory=list)
    forbidden_operations: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecuritySession:
    """Security session for user interactions"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    user_permissions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1))
    status: SessionStatus = SessionStatus.ACTIVE
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if session is valid and not expired"""
        now = datetime.now(timezone.utc)
        return (self.status == SessionStatus.ACTIVE and 
                now < self.expires_at)
    
    def refresh(self, duration_hours: int = 1):
        """Refresh session expiration"""
        self.last_activity = datetime.now(timezone.utc)
        self.expires_at = self.last_activity + timedelta(hours=duration_hours)


@dataclass
class SecurityContext:
    """Security context for operations"""
    session: Optional[SecuritySession] = None
    user: Optional[User] = None
    operation: str = ""
    resource: str = ""
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    additional_data: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """
    Comprehensive security manager for CrewGraph AI.
    
    Provides centralized security management including authentication,
    authorization, session management, and security policy enforcement.
    """
    
    def __init__(self, 
                 enable_encryption: bool = True,
                 enable_audit_logging: bool = True,
                 default_session_duration: int = 3600):
        """
        Initialize the security manager.
        
        Args:
            enable_encryption: Enable data encryption
            enable_audit_logging: Enable audit logging
            default_session_duration: Default session duration in seconds
        """
        self.default_session_duration = default_session_duration
        
        # Initialize components
        self.role_manager = RoleManager()
        self.audit_logger = AuditLogger() if enable_audit_logging else None
        self.encryption_manager = EncryptionManager() if enable_encryption else None
        
        # Session management
        self._sessions: Dict[str, SecuritySession] = {}
        self._user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        
        # Security policies
        self._policies: Dict[str, SecurityPolicy] = {}
        self._default_policy = self._create_default_policy()
        
        # Security validators and handlers
        self._security_validators: List[Callable[[SecurityContext], bool]] = []
        self._security_handlers: Dict[str, Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default roles and permissions
        self._initialize_default_roles()
        
        logger.info("SecurityManager initialized")
        logger.info(f"Encryption: {'enabled' if enable_encryption else 'disabled'}")
        logger.info(f"Audit logging: {'enabled' if enable_audit_logging else 'disabled'}")
    
    def _create_default_policy(self) -> SecurityPolicy:
        """Create default security policy"""
        return SecurityPolicy(
            name="default",
            description="Default security policy for CrewGraph AI",
            min_security_level=SecurityLevel.INTERNAL,
            max_session_duration=self.default_session_duration,
            require_encryption=self.encryption_manager is not None,
            audit_all_operations=self.audit_logger is not None
        )
    
    def _initialize_default_roles(self):
        """Initialize default roles and permissions"""
        # Create default permissions
        permissions = [
            Permission("workflow.create", "Create workflows"),
            Permission("workflow.read", "Read workflow information"),
            Permission("workflow.update", "Update workflows"),
            Permission("workflow.delete", "Delete workflows"),
            Permission("workflow.execute", "Execute workflows"),
            Permission("agent.create", "Create agents"),
            Permission("agent.manage", "Manage agents"),
            Permission("communication.send", "Send messages"),
            Permission("communication.broadcast", "Broadcast messages"),
            Permission("template.use", "Use workflow templates"),
            Permission("template.create", "Create templates"),
            Permission("security.manage", "Manage security settings"),
            Permission("audit.view", "View audit logs"),
            Permission("admin.full", "Full administrative access")
        ]
        
        for perm in permissions:
            self.role_manager.create_permission(perm)
        
        # Create default roles
        roles = [
            Role("viewer", "Read-only access", ["workflow.read"]),
            Role("user", "Standard user access", [
                "workflow.create", "workflow.read", "workflow.update", "workflow.execute",
                "agent.create", "communication.send", "template.use"
            ]),
            Role("developer", "Developer access", [
                "workflow.create", "workflow.read", "workflow.update", "workflow.delete", "workflow.execute",
                "agent.create", "agent.manage", "communication.send", "communication.broadcast",
                "template.use", "template.create"
            ]),
            Role("admin", "Administrative access", [
                "workflow.create", "workflow.read", "workflow.update", "workflow.delete", "workflow.execute",
                "agent.create", "agent.manage", "communication.send", "communication.broadcast",
                "template.use", "template.create", "security.manage", "audit.view", "admin.full"
            ])
        ]
        
        for role in roles:
            self.role_manager.create_role(role)
        
        logger.info("Default roles and permissions initialized")
    
    def create_user(self, 
                   user_id: str,
                   username: str,
                   email: str,
                   roles: List[str],
                   metadata: Optional[Dict[str, Any]] = None) -> User:
        """
        Create a new user with specified roles.
        
        Args:
            user_id: Unique user identifier
            username: Username
            email: User email
            roles: List of role names to assign
            metadata: Additional user metadata
            
        Returns:
            Created User object
        """
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            metadata=metadata or {}
        )
        
        # Validate roles exist
        for role_name in roles:
            if not self.role_manager.get_role(role_name):
                raise ValidationError(f"Role '{role_name}' does not exist")
        
        self.role_manager.create_user(user)
        
        if self.audit_logger:
            self.audit_logger.log_event(AuditEvent(
                event_type="user.created",
                user_id=user_id,
                resource="user",
                action="create",
                details={"username": username, "roles": roles}
            ))
        
        logger.info(f"User '{username}' created with roles: {roles}")
        return user
    
    def authenticate_user(self, 
                         user_id: str,
                         ip_address: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> SecuritySession:
        """
        Authenticate user and create security session.
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            metadata: Additional session metadata
            
        Returns:
            Security session
            
        Raises:
            ValidationError: If user not found or authentication fails
        """
        user = self.role_manager.get_user(user_id)
        if not user:
            raise ValidationError(f"User '{user_id}' not found")
        
        # Check IP whitelist if configured
        if (self._default_policy.ip_whitelist and 
            ip_address and 
            ip_address not in self._default_policy.ip_whitelist):
            raise ValidationError(f"IP address '{ip_address}' not authorized")
        
        # Get user permissions
        permissions = self.role_manager.get_user_permissions(user_id)
        
        # Create session
        session = SecuritySession(
            user_id=user_id,
            user_permissions=set(permissions),
            ip_address=ip_address,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.default_session_duration),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._sessions[session.session_id] = session
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session.session_id)
        
        if self.audit_logger:
            self.audit_logger.log_event(AuditEvent(
                event_type="auth.login",
                user_id=user_id,
                session_id=session.session_id,
                resource="session",
                action="create",
                details={"ip_address": ip_address}
            ))
        
        logger.info(f"User '{user_id}' authenticated, session: {session.session_id}")
        return session
    
    def validate_session(self, session_id: str) -> Optional[SecuritySession]:
        """
        Validate security session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Valid session or None if invalid
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if not session:
                return None
            
            if not session.is_valid():
                # Clean up expired session
                self._cleanup_session(session_id)
                return None
            
            # Refresh last activity
            session.last_activity = datetime.now(timezone.utc)
            
        return session
    
    def authorize_operation(self, 
                           session_id: str,
                           operation: str,
                           resource: str = "",
                           security_level: SecurityLevel = SecurityLevel.INTERNAL,
                           additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Authorize operation for session.
        
        Args:
            session_id: Session identifier
            operation: Operation being performed
            resource: Resource being accessed
            security_level: Required security level
            additional_data: Additional context data
            
        Returns:
            True if authorized, False otherwise
        """
        session = self.validate_session(session_id)
        if not session:
            logger.warning(f"Authorization failed: Invalid session {session_id}")
            return False
        
        user = self.role_manager.get_user(session.user_id)
        if not user:
            logger.warning(f"Authorization failed: User not found {session.user_id}")
            return False
        
        # Create security context
        context = SecurityContext(
            session=session,
            user=user,
            operation=operation,
            resource=resource,
            security_level=security_level,
            additional_data=additional_data or {}
        )
        
        # Check security policy
        if not self._check_security_policy(context):
            logger.warning(f"Authorization failed: Policy violation for {operation}")
            return False
        
        # Check permissions
        required_permission = self._get_required_permission(operation)
        if required_permission and required_permission not in session.user_permissions:
            logger.warning(f"Authorization failed: Missing permission {required_permission}")
            return False
        
        # Run custom validators
        for validator in self._security_validators:
            try:
                if not validator(context):
                    logger.warning(f"Authorization failed: Custom validator rejected {operation}")
                    return False
            except Exception as e:
                logger.error(f"Security validator error: {e}")
                return False
        
        # Log authorization event
        if self.audit_logger:
            self.audit_logger.log_event(AuditEvent(
                event_type="auth.authorize",
                user_id=session.user_id,
                session_id=session_id,
                resource=resource,
                action=operation,
                success=True,
                details={"security_level": security_level.value}
            ))
        
        return True
    
    def _check_security_policy(self, context: SecurityContext) -> bool:
        """Check if operation complies with security policy"""
        policy = self._get_applicable_policy(context)
        
        # Check minimum security level
        if context.security_level.value < policy.min_security_level.value:
            return False
        
        # Check forbidden operations
        if context.operation in policy.forbidden_operations:
            return False
        
        # Check required permissions
        if policy.required_permissions:
            for req_perm in policy.required_permissions:
                if req_perm not in context.session.user_permissions:
                    return False
        
        return True
    
    def _get_applicable_policy(self, context: SecurityContext) -> SecurityPolicy:
        """Get applicable security policy for context"""
        # For now, return default policy
        # Could be extended to support resource-specific policies
        return self._default_policy
    
    def _get_required_permission(self, operation: str) -> Optional[str]:
        """Get required permission for operation"""
        # Map operations to permissions
        operation_permissions = {
            "workflow.create": "workflow.create",
            "workflow.read": "workflow.read",
            "workflow.update": "workflow.update",
            "workflow.delete": "workflow.delete",
            "workflow.execute": "workflow.execute",
            "agent.create": "agent.create",
            "agent.manage": "agent.manage",
            "communication.send": "communication.send",
            "communication.broadcast": "communication.broadcast",
            "template.use": "template.use",
            "template.create": "template.create",
            "security.manage": "security.manage",
            "audit.view": "audit.view"
        }
        
        return operation_permissions.get(operation)
    
    def logout_user(self, session_id: str) -> bool:
        """
        Logout user and terminate session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if logout successful
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            session.status = SessionStatus.TERMINATED
            
            if self.audit_logger:
                self.audit_logger.log_event(AuditEvent(
                    event_type="auth.logout",
                    user_id=session.user_id,
                    session_id=session_id,
                    resource="session",
                    action="terminate"
                ))
            
            self._cleanup_session(session_id)
            
        logger.info(f"User session {session_id} terminated")
        return True
    
    def _cleanup_session(self, session_id: str):
        """Clean up session from internal storage"""
        session = self._sessions.get(session_id)
        if session:
            user_id = session.user_id
            if user_id in self._user_sessions:
                self._user_sessions[user_id].discard(session_id)
                if not self._user_sessions[user_id]:
                    del self._user_sessions[user_id]
            del self._sessions[session_id]
    
    def get_user_sessions(self, user_id: str) -> List[SecuritySession]:
        """Get all active sessions for user"""
        session_ids = self._user_sessions.get(user_id, set())
        sessions = []
        
        for session_id in list(session_ids):  # Create copy to avoid modification during iteration
            session = self.validate_session(session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        expired_count = 0
        
        with self._lock:
            expired_sessions = []
            for session_id, session in self._sessions.items():
                if not session.is_valid():
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self._cleanup_session(session_id)
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
        
        return expired_count
    
    def add_security_validator(self, validator: Callable[[SecurityContext], bool]):
        """Add custom security validator"""
        self._security_validators.append(validator)
        logger.info("Custom security validator added")
    
    def create_security_policy(self, policy: SecurityPolicy) -> bool:
        """Create or update security policy"""
        self._policies[policy.name] = policy
        logger.info(f"Security policy '{policy.name}' created/updated")
        return True
    
    def encrypt_data(self, data: Any, context: Optional[SecurityContext] = None) -> Optional[str]:
        """Encrypt sensitive data"""
        if not self.encryption_manager:
            return None
        
        try:
            return self.encryption_manager.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str, context: Optional[SecurityContext] = None) -> Optional[Any]:
        """Decrypt sensitive data"""
        if not self.encryption_manager:
            return None
        
        try:
            return self.encryption_manager.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics"""
        with self._lock:
            active_sessions = sum(1 for s in self._sessions.values() if s.is_valid())
            total_users = len(self.role_manager.list_users())
            
        metrics = {
            "active_sessions": active_sessions,
            "total_sessions": len(self._sessions),
            "total_users": total_users,
            "total_roles": len(self.role_manager.list_roles()),
            "encryption_enabled": self.encryption_manager is not None,
            "audit_logging_enabled": self.audit_logger is not None,
            "policies_count": len(self._policies),
            "validators_count": len(self._security_validators)
        }
        
        if self.audit_logger:
            audit_stats = self.audit_logger.get_statistics()
            metrics.update(audit_stats)
        
        return metrics
    
    def export_audit_log(self, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export audit log for compliance reporting"""
        if not self.audit_logger:
            return []
        
        return self.audit_logger.export_events(start_time, end_time, user_id)
    
    def validate_compliance(self) -> Dict[str, Any]:
        """Validate security compliance"""
        compliance = {
            "encryption_enabled": self.encryption_manager is not None,
            "audit_logging_enabled": self.audit_logger is not None,
            "rbac_configured": len(self.role_manager.list_roles()) > 0,
            "session_management": True,
            "policies_defined": len(self._policies) > 0,
            "security_validators": len(self._security_validators) > 0
        }
        
        compliance["overall_score"] = sum(compliance.values()) / len(compliance) * 100
        
        return compliance