"""
Security utilities for CrewGraph AI
"""

import os
import hashlib
import secrets
import jwt
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .logging import get_logger
from .exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


class PermissionLevel(Enum):
    """Security permission levels"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    enable_encryption: bool = True
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_audit_logging: bool = True
    password_min_length: int = 8
    session_timeout: int = 3600  # 1 hour
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # 5 minutes


class EncryptionUtils:
    """Encryption utilities for secure data handling"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption utilities.
        
        Args:
            encryption_key: Base encryption key (will be derived)
        """
        self.encryption_key = encryption_key or self._generate_key()
        self._cipher_suite = self._create_cipher_suite()
        
        logger.info("EncryptionUtils initialized")
    
    def _generate_key(self) -> str:
        """Generate a secure encryption key"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode()
    
    def _create_cipher_suite(self) -> Fernet:
        """Create cipher suite from key"""
        # Derive key using PBKDF2
        password = self.encryption_key.encode()
        salt = b'crewgraph_salt_2025'  # In production, use random salt per user
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt string data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data (base64 encoded)
        """
        try:
            encrypted_data = self._cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise CrewGraphError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt string data.
        
        Args:
            encrypted_data: Encrypted data (base64 encoded)
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise CrewGraphError(f"Decryption failed: {e}")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data"""
        import json
        json_data = json.dumps(data, default=str)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data"""
        import json
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash password with salt.
        
        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for password hashing
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            computed_hash, _ = self.hash_password(password, salt)
            return secrets.compare_digest(computed_hash, hashed_password)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False


class AuthenticationManager:
    """Handle user authentication and session management"""
    
    def __init__(self, 
                 jwt_secret: str,
                 session_timeout: int = 3600,
                 max_failed_attempts: int = 5):
        """
        Initialize authentication manager.
        
        Args:
            jwt_secret: Secret key for JWT tokens
            session_timeout: Session timeout in seconds
            max_failed_attempts: Maximum failed login attempts
        """
        self.jwt_secret = jwt_secret
        self.session_timeout = session_timeout
        self.max_failed_attempts = max_failed_attempts
        
        # Track failed attempts
        self._failed_attempts: Dict[str, int] = {}
        self._lockout_times: Dict[str, float] = {}
        
        # Active sessions
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("AuthenticationManager initialized")
    
    def create_token(self, 
                    user_id: str,
                    permissions: List[str],
                    expires_in: Optional[int] = None) -> str:
        """
        Create JWT authentication token.
        
        Args:
            user_id: User identifier
            permissions: List of user permissions
            expires_in: Token expiration time (seconds)
            
        Returns:
            JWT token string
        """
        expires_in = expires_in or self.session_timeout
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': time.time(),
            'exp': time.time() + expires_in,
            'iss': 'crewgraph-ai',
            'created_by': 'Vatsal216',
            'created_at': '2025-07-22 10:27:20'
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        # Store session info
        self._active_sessions[token] = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'permissions': permissions
        }
        
        logger.info(f"JWT token created for user: {user_id}")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token and return payload.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Update session activity
            if token in self._active_sessions:
                self._active_sessions[token]['last_activity'] = time.time()
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            self._cleanup_session(token)
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh an existing token"""
        payload = self.verify_token(token)
        
        if payload:
            # Create new token with same permissions
            new_token = self.create_token(
                payload['user_id'],
                payload['permissions']
            )
            
            # Remove old session
            self._cleanup_session(token)
            
            return new_token
        
        return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        if token in self._active_sessions:
            self._cleanup_session(token)
            logger.info("Token revoked successfully")
            return True
        
        return False
    
    def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user is rate limited due to failed attempts.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user can attempt login, False if locked out
        """
        current_time = time.time()
        
        # Check if user is in lockout
        if user_id in self._lockout_times:
            lockout_time = self._lockout_times[user_id]
            if current_time - lockout_time < 300:  # 5 minute lockout
                return False
            else:
                # Lockout expired
                del self._lockout_times[user_id]
                self._failed_attempts[user_id] = 0
        
        return True
    
    def record_failed_attempt(self, user_id: str) -> None:
        """Record failed login attempt"""
        self._failed_attempts[user_id] = self._failed_attempts.get(user_id, 0) + 1
        
        if self._failed_attempts[user_id] >= self.max_failed_attempts:
            self._lockout_times[user_id] = time.time()
            logger.warning(f"User {user_id} locked out due to failed attempts")
    
    def record_successful_login(self, user_id: str) -> None:
        """Record successful login (reset failed attempts)"""
        self._failed_attempts[user_id] = 0
        if user_id in self._lockout_times:
            del self._lockout_times[user_id]
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        # Clean up expired sessions first
        current_time = time.time()
        expired_tokens = []
        
        for token, session in self._active_sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            self._cleanup_session(token)
        
        return self._active_sessions.copy()
    
    def _cleanup_session(self, token: str) -> None:
        """Clean up session data"""
        if token in self._active_sessions:
            del self._active_sessions[token]


class SecurityManager:
    """Comprehensive security management for CrewGraph AI"""
    
    def __init__(self, 
                 policy: Optional[SecurityPolicy] = None,
                 encryption_key: Optional[str] = None,
                 jwt_secret: Optional[str] = None):
        """
        Initialize security manager.
        
        Args:
            policy: Security policy configuration
            encryption_key: Encryption key for data protection
            jwt_secret: JWT secret for authentication
        """
        self.policy = policy or SecurityPolicy()
        
        # Initialize security components
        self.encryption = EncryptionUtils(encryption_key) if self.policy.enable_encryption else None
        
        if self.policy.enable_authentication:
            jwt_secret = jwt_secret or self._generate_jwt_secret()
            self.auth = AuthenticationManager(
                jwt_secret=jwt_secret,
                session_timeout=self.policy.session_timeout,
                max_failed_attempts=self.policy.max_failed_attempts
            )
        else:
            self.auth = None
        
        # User permissions and roles
        self._user_permissions: Dict[str, List[str]] = {}
        self._role_permissions: Dict[str, List[str]] = {}
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
        
        logger.info("SecurityManager initialized with comprehensive security features")
    
    def _generate_jwt_secret(self) -> str:
        """Generate secure JWT secret"""
        return base64.urlsafe_b64encode(os.urandom(64)).decode()
    
    def authenticate_user(self, 
                         username: str, 
                         password: str,
                         permissions: Optional[List[str]] = None) -> Optional[str]:
        """
        Authenticate user and return token.
        
        Args:
            username: Username
            password: Password
            permissions: User permissions (if known)
            
        Returns:
            JWT token if authentication successful
        """
        if not self.auth:
            raise CrewGraphError("Authentication not enabled")
        
        # Check rate limiting
        if not self.auth.check_rate_limit(username):
            self.audit_log("authentication_failed", {
                "username": username,
                "reason": "rate_limited"
            })
            return None
        
        # In production, verify against user database
        # For demo, we'll use a simple check
        if self._verify_user_credentials(username, password):
            user_permissions = permissions or self._user_permissions.get(username, [])
            token = self.auth.create_token(username, user_permissions)
            
            self.auth.record_successful_login(username)
            self.audit_log("authentication_success", {"username": username})
            
            return token
        else:
            self.auth.record_failed_attempt(username)
            self.audit_log("authentication_failed", {
                "username": username,
                "reason": "invalid_credentials"
            })
            return None
    
    def _verify_user_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (placeholder implementation)"""
        # In production, this would check against a secure user database
        # For demo purposes, accept specific test credentials
        test_users = {
            "Vatsal216": "secure_password_2025",
            "admin": "admin_password",
            "demo_user": "demo_password"
        }
        
        return test_users.get(username) == password
    
    def authorize_operation(self, 
                           token: str, 
                           operation: str,
                           resource: str,
                           required_permission: PermissionLevel = PermissionLevel.READ) -> bool:
        """
        Authorize user operation.
        
        Args:
            token: JWT authentication token
            operation: Operation being performed
            resource: Resource being accessed
            required_permission: Required permission level
            
        Returns:
            True if authorized, False otherwise
        """
        if not self.auth:
            return True  # No authorization enabled
        
        # Verify token
        payload = self.auth.verify_token(token)
        if not payload:
            self.audit_log("authorization_failed", {
                "operation": operation,
                "resource": resource,
                "reason": "invalid_token"
            })
            return False
        
        user_id = payload['user_id']
        user_permissions = payload.get('permissions', [])
        
        # Check permissions
        required_perm = required_permission.value
        has_permission = (
            required_perm in user_permissions or
            "admin" in user_permissions
        )
        
        if has_permission:
            self.audit_log("authorization_success", {
                "user_id": user_id,
                "operation": operation,
                "resource": resource
            })
        else:
            self.audit_log("authorization_failed", {
                "user_id": user_id,
                "operation": operation,
                "resource": resource,
                "reason": "insufficient_permissions"
            })
        
        return has_permission
    
    def encrypt_sensitive_data(self, data: Any) -> str:
        """Encrypt sensitive data"""
        if not self.encryption:
            return str(data)
        
        if isinstance(data, dict):
            return self.encryption.encrypt_dict(data)
        else:
            return self.encryption.encrypt(str(data))
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Any:
        """Decrypt sensitive data"""
        if not self.encryption:
            return encrypted_data
        
        try:
            # Try to decrypt as dict first
            return self.encryption.decrypt_dict(encrypted_data)
        except:
            # Fall back to string decryption
            return self.encryption.decrypt(encrypted_data)
    
    def add_user_permission(self, user_id: str, permission: str) -> None:
        """Add permission to user"""
        if user_id not in self._user_permissions:
            self._user_permissions[user_id] = []
        
        if permission not in self._user_permissions[user_id]:
            self._user_permissions[user_id].append(permission)
            
        self.audit_log("permission_granted", {
            "user_id": user_id,
            "permission": permission
        })
    
    def remove_user_permission(self, user_id: str, permission: str) -> None:
        """Remove permission from user"""
        if user_id in self._user_permissions:
            if permission in self._user_permissions[user_id]:
                self._user_permissions[user_id].remove(permission)
                
        self.audit_log("permission_revoked", {
            "user_id": user_id,
            "permission": permission
        })
    
    def create_role(self, role_name: str, permissions: List[str]) -> None:
        """Create role with permissions"""
        self._role_permissions[role_name] = permissions
        
        self.audit_log("role_created", {
            "role_name": role_name,
            "permissions": permissions
        })
    
    def assign_role_to_user(self, user_id: str, role_name: str) -> None:
        """Assign role to user"""
        if role_name in self._role_permissions:
            role_permissions = self._role_permissions[role_name]
            
            if user_id not in self._user_permissions:
                self._user_permissions[user_id] = []
            
            for permission in role_permissions:
                if permission not in self._user_permissions[user_id]:
                    self._user_permissions[user_id].append(permission)
            
            self.audit_log("role_assigned", {
                "user_id": user_id,
                "role_name": role_name
            })
    
    def audit_log(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security audit event"""
        if self.policy.enable_audit_logging:
            audit_entry = {
                "timestamp": time.time(),
                "event_type": event_type,
                "details": details,
                "created_by": "Vatsal216",
                "system_time": "2025-07-22 10:27:20"
            }
            
            self._audit_log.append(audit_entry)
            
            # Log to system logger
            logger.info(f"Security audit: {event_type}", **details)
    
    def get_audit_log(self, 
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered audit log"""
        filtered_log = self._audit_log.copy()
        
        if start_time:
            filtered_log = [entry for entry in filtered_log if entry['timestamp'] >= start_time]
        
        if end_time:
            filtered_log = [entry for entry in filtered_log if entry['timestamp'] <= end_time]
        
        if event_type:
            filtered_log = [entry for entry in filtered_log if entry['event_type'] == event_type]
        
        return filtered_log
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        status = {
            "policy": {
                "encryption_enabled": self.policy.enable_encryption,
                "authentication_enabled": self.policy.enable_authentication,
                "authorization_enabled": self.policy.enable_authorization,
                "audit_logging_enabled": self.policy.enable_audit_logging
            },
            "active_sessions": len(self.auth.get_active_sessions()) if self.auth else 0,
            "total_users": len(self._user_permissions),
            "total_roles": len(self._role_permissions),
            "audit_events": len(self._audit_log)
        }
        
        if self.auth:
            status["session_info"] = {
                "timeout": self.policy.session_timeout,
                "max_failed_attempts": self.policy.max_failed_attempts
            }
        
        return status


# Global security manager instance
_global_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> Optional[SecurityManager]:
    """Get global security manager instance"""
    return _global_security_manager


def initialize_security(policy: Optional[SecurityPolicy] = None,
                       encryption_key: Optional[str] = None,
                       jwt_secret: Optional[str] = None) -> SecurityManager:
    """Initialize global security manager"""
    global _global_security_manager
    
    _global_security_manager = SecurityManager(
        policy=policy,
        encryption_key=encryption_key,
        jwt_secret=jwt_secret
    )
    
    return _global_security_manager