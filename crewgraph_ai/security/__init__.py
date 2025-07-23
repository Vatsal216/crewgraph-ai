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

from .security_manager import SecurityManager
from .rbac import RoleManager, Role, Permission, User
from .audit_logger import AuditLogger, AuditEvent
from .encryption import EncryptionManager, CryptoConfig

__all__ = [
    "SecurityManager",
    "RoleManager",
    "Role", 
    "Permission",
    "User",
    "AuditLogger",
    "AuditEvent",
    "EncryptionManager",
    "CryptoConfig"
]

# Version info
__version__ = "1.0.0"
__author__ = "Vatsal216"