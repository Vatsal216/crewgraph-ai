"""
Security Module for CrewGraph AI

This module provides comprehensive security features for enterprise deployments including:
- Role-based access control (RBAC)
- Audit logging system
- Data encryption for sensitive workflows
- Session management
- Security validation

Created by: Vatsal216
Date: 2025-07-23
"""

from .audit_logger import AuditEvent, AuditLogger
from .encryption import CryptoConfig, EncryptionManager
from .rbac import Permission, Role, RoleManager, User
from .security_manager import SecurityManager

__all__ = [
    "SecurityManager",
    "RoleManager",
    "Role",
    "Permission",
    "User",
    "AuditLogger",
    "AuditEvent",
    "EncryptionManager",
    "CryptoConfig",
]

# Version info
__version__ = "1.0.0"
__author__ = "Vatsal216"
