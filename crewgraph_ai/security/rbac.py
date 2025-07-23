"""
Role-Based Access Control (RBAC) - Comprehensive authorization system

This module provides role-based access control functionality including:
- User management with roles and permissions
- Role hierarchy and inheritance
- Permission validation and checking
- Dynamic role assignment
- Integration with security manager

Features:
- Flexible role and permission system
- Role inheritance and composition
- User group management
- Permission caching for performance
- Audit integration

Created by: Vatsal216
Date: 2025-07-23
"""

import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


@dataclass
class Permission:
    """Permission definition for RBAC system"""
    name: str
    description: str = ""
    resource_type: str = ""
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.name:
            raise ValidationError("Permission name cannot be empty")
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Permission):
            return self.name == other.name
        return False


@dataclass
class Role:
    """Role definition with permissions and hierarchy"""
    name: str
    description: str = ""
    permissions: List[str] = field(default_factory=list)
    parent_roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.name:
            raise ValidationError("Role name cannot be empty")
    
    def add_permission(self, permission_name: str):
        """Add permission to role"""
        if permission_name not in self.permissions:
            self.permissions.append(permission_name)
    
    def remove_permission(self, permission_name: str):
        """Remove permission from role"""
        if permission_name in self.permissions:
            self.permissions.remove(permission_name)
    
    def has_permission(self, permission_name: str) -> bool:
        """Check if role has specific permission"""
        return permission_name in self.permissions


@dataclass
class User:
    """User definition with roles and profile"""
    user_id: str
    username: str
    email: str = ""
    roles: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    is_active: bool = True
    is_system_user: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.user_id:
            raise ValidationError("User ID cannot be empty")
        if not self.username:
            raise ValidationError("Username cannot be empty")
    
    def add_role(self, role_name: str):
        """Add role to user"""
        if role_name not in self.roles:
            self.roles.append(role_name)
    
    def remove_role(self, role_name: str):
        """Remove role from user"""
        if role_name in self.roles:
            self.roles.remove(role_name)
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has specific role"""
        return role_name in self.roles


@dataclass
class UserGroup:
    """User group for organizing users"""
    name: str
    description: str = ""
    members: List[str] = field(default_factory=list)  # user_ids
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_member(self, user_id: str):
        """Add user to group"""
        if user_id not in self.members:
            self.members.append(user_id)
    
    def remove_member(self, user_id: str):
        """Remove user from group"""
        if user_id in self.members:
            self.members.remove(user_id)
    
    def has_member(self, user_id: str) -> bool:
        """Check if user is in group"""
        return user_id in self.members


class RoleManager:
    """
    Comprehensive role-based access control manager.
    
    Manages users, roles, permissions, and groups with support for
    role inheritance, permission validation, and efficient access checks.
    """
    
    def __init__(self):
        # Core storage
        self._permissions: Dict[str, Permission] = {}
        self._roles: Dict[str, Role] = {}
        self._users: Dict[str, User] = {}
        self._groups: Dict[str, UserGroup] = {}
        
        # Performance caches
        self._user_permissions_cache: Dict[str, Set[str]] = {}
        self._role_inheritance_cache: Dict[str, Set[str]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("RoleManager initialized")
    
    def create_permission(self, permission: Permission) -> bool:
        """
        Create a new permission.
        
        Args:
            permission: Permission object to create
            
        Returns:
            True if creation successful
            
        Raises:
            ValidationError: If permission already exists
        """
        with self._lock:
            if permission.name in self._permissions:
                raise ValidationError(f"Permission '{permission.name}' already exists")
            
            self._permissions[permission.name] = permission
            
        logger.info(f"Permission '{permission.name}' created")
        return True
    
    def get_permission(self, name: str) -> Optional[Permission]:
        """Get permission by name"""
        return self._permissions.get(name)
    
    def list_permissions(self) -> List[str]:
        """List all permission names"""
        return list(self._permissions.keys())
    
    def delete_permission(self, name: str) -> bool:
        """
        Delete permission and remove from all roles.
        
        Args:
            name: Permission name to delete
            
        Returns:
            True if deletion successful
        """
        with self._lock:
            if name not in self._permissions:
                return False
            
            # Remove from all roles
            for role in self._roles.values():
                role.remove_permission(name)
            
            del self._permissions[name]
            
            # Clear caches
            self._clear_caches()
            
        logger.info(f"Permission '{name}' deleted")
        return True
    
    def create_role(self, role: Role) -> bool:
        """
        Create a new role.
        
        Args:
            role: Role object to create
            
        Returns:
            True if creation successful
            
        Raises:
            ValidationError: If role already exists or has invalid permissions
        """
        with self._lock:
            if role.name in self._roles:
                raise ValidationError(f"Role '{role.name}' already exists")
            
            # Validate permissions exist
            for perm_name in role.permissions:
                if perm_name not in self._permissions:
                    raise ValidationError(f"Permission '{perm_name}' does not exist")
            
            # Validate parent roles exist
            for parent_name in role.parent_roles:
                if parent_name not in self._roles:
                    raise ValidationError(f"Parent role '{parent_name}' does not exist")
            
            self._roles[role.name] = role
            
            # Clear inheritance cache
            self._role_inheritance_cache.clear()
            
        logger.info(f"Role '{role.name}' created with {len(role.permissions)} permissions")
        return True
    
    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name"""
        return self._roles.get(name)
    
    def list_roles(self) -> List[str]:
        """List all role names"""
        return list(self._roles.keys())
    
    def delete_role(self, name: str) -> bool:
        """
        Delete role and remove from all users.
        
        Args:
            name: Role name to delete
            
        Returns:
            True if deletion successful
        """
        with self._lock:
            if name not in self._roles:
                return False
            
            # Check if role is system role
            role = self._roles[name]
            if role.is_system_role:
                raise ValidationError(f"Cannot delete system role '{name}'")
            
            # Remove from all users
            for user in self._users.values():
                user.remove_role(name)
            
            # Remove from all groups
            for group in self._groups.values():
                if name in group.roles:
                    group.roles.remove(name)
            
            # Remove as parent role from other roles
            for other_role in self._roles.values():
                if name in other_role.parent_roles:
                    other_role.parent_roles.remove(name)
            
            del self._roles[name]
            
            # Clear caches
            self._clear_caches()
            
        logger.info(f"Role '{name}' deleted")
        return True
    
    def get_role_permissions(self, role_name: str, include_inherited: bool = True) -> Set[str]:
        """
        Get all permissions for a role.
        
        Args:
            role_name: Role name
            include_inherited: Include permissions from parent roles
            
        Returns:
            Set of permission names
        """
        role = self.get_role(role_name)
        if not role:
            return set()
        
        permissions = set(role.permissions)
        
        if include_inherited:
            # Check cache first
            if role_name in self._role_inheritance_cache:
                inherited_roles = self._role_inheritance_cache[role_name]
            else:
                inherited_roles = self._get_inherited_roles(role_name)
                self._role_inheritance_cache[role_name] = inherited_roles
            
            # Add permissions from inherited roles
            for inherited_role_name in inherited_roles:
                inherited_role = self.get_role(inherited_role_name)
                if inherited_role:
                    permissions.update(inherited_role.permissions)
        
        return permissions
    
    def _get_inherited_roles(self, role_name: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """Get all inherited roles (recursive)"""
        if visited is None:
            visited = set()
        
        if role_name in visited:
            # Circular dependency detected
            logger.warning(f"Circular role dependency detected for role '{role_name}'")
            return set()
        
        visited.add(role_name)
        inherited = set()
        
        role = self.get_role(role_name)
        if role:
            for parent_name in role.parent_roles:
                inherited.add(parent_name)
                inherited.update(self._get_inherited_roles(parent_name, visited.copy()))
        
        return inherited
    
    def create_user(self, user: User) -> bool:
        """
        Create a new user.
        
        Args:
            user: User object to create
            
        Returns:
            True if creation successful
            
        Raises:
            ValidationError: If user already exists or has invalid roles
        """
        with self._lock:
            if user.user_id in self._users:
                raise ValidationError(f"User '{user.user_id}' already exists")
            
            # Validate roles exist
            for role_name in user.roles:
                if role_name not in self._roles:
                    raise ValidationError(f"Role '{role_name}' does not exist")
            
            self._users[user.user_id] = user
            
            # Clear user permissions cache
            self._user_permissions_cache.pop(user.user_id, None)
            
        logger.info(f"User '{user.username}' created with roles: {user.roles}")
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def list_users(self) -> List[str]:
        """List all user IDs"""
        return list(self._users.keys())
    
    def update_user_roles(self, user_id: str, roles: List[str]) -> bool:
        """
        Update user roles.
        
        Args:
            user_id: User ID
            roles: New list of role names
            
        Returns:
            True if update successful
        """
        with self._lock:
            user = self.get_user(user_id)
            if not user:
                return False
            
            # Validate roles exist
            for role_name in roles:
                if role_name not in self._roles:
                    raise ValidationError(f"Role '{role_name}' does not exist")
            
            old_roles = user.roles.copy()
            user.roles = roles.copy()
            
            # Clear user permissions cache
            self._user_permissions_cache.pop(user_id, None)
            
        logger.info(f"User '{user_id}' roles updated from {old_roles} to {roles}")
        return True
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """
        Get all permissions for a user (including from roles and groups).
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permission names
        """
        # Check cache first
        if user_id in self._user_permissions_cache:
            return self._user_permissions_cache[user_id]
        
        user = self.get_user(user_id)
        if not user:
            return set()
        
        permissions = set()
        
        # Get permissions from user's direct roles
        for role_name in user.roles:
            role_permissions = self.get_role_permissions(role_name, include_inherited=True)
            permissions.update(role_permissions)
        
        # Get permissions from user's groups
        for group_name in user.groups:
            group = self.get_group(group_name)
            if group:
                for role_name in group.roles:
                    role_permissions = self.get_role_permissions(role_name, include_inherited=True)
                    permissions.update(role_permissions)
        
        # Cache the result
        self._user_permissions_cache[user_id] = permissions
        
        return permissions
    
    def has_permission(self, user_id: str, permission_name: str) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            user_id: User ID
            permission_name: Permission name to check
            
        Returns:
            True if user has permission
        """
        user_permissions = self.get_user_permissions(user_id)
        return permission_name in user_permissions
    
    def has_role(self, user_id: str, role_name: str, include_inherited: bool = True) -> bool:
        """
        Check if user has specific role.
        
        Args:
            user_id: User ID
            role_name: Role name to check
            include_inherited: Check inherited roles from groups
            
        Returns:
            True if user has role
        """
        user = self.get_user(user_id)
        if not user:
            return False
        
        # Check direct roles
        if role_name in user.roles:
            return True
        
        if include_inherited:
            # Check roles from groups
            for group_name in user.groups:
                group = self.get_group(group_name)
                if group and role_name in group.roles:
                    return True
        
        return False
    
    def create_group(self, group: UserGroup) -> bool:
        """
        Create a new user group.
        
        Args:
            group: UserGroup object to create
            
        Returns:
            True if creation successful
        """
        with self._lock:
            if group.name in self._groups:
                raise ValidationError(f"Group '{group.name}' already exists")
            
            # Validate roles exist
            for role_name in group.roles:
                if role_name not in self._roles:
                    raise ValidationError(f"Role '{role_name}' does not exist")
            
            # Validate members exist
            for user_id in group.members:
                if user_id not in self._users:
                    raise ValidationError(f"User '{user_id}' does not exist")
            
            self._groups[group.name] = group
            
            # Update user group memberships
            for user_id in group.members:
                user = self._users[user_id]
                if group.name not in user.groups:
                    user.groups.append(group.name)
                # Clear user permissions cache
                self._user_permissions_cache.pop(user_id, None)
            
        logger.info(f"Group '{group.name}' created with {len(group.members)} members")
        return True
    
    def get_group(self, name: str) -> Optional[UserGroup]:
        """Get group by name"""
        return self._groups.get(name)
    
    def list_groups(self) -> List[str]:
        """List all group names"""
        return list(self._groups.keys())
    
    def add_user_to_group(self, user_id: str, group_name: str) -> bool:
        """
        Add user to group.
        
        Args:
            user_id: User ID
            group_name: Group name
            
        Returns:
            True if addition successful
        """
        with self._lock:
            user = self.get_user(user_id)
            group = self.get_group(group_name)
            
            if not user or not group:
                return False
            
            if not group.has_member(user_id):
                group.add_member(user_id)
            
            if group_name not in user.groups:
                user.groups.append(group_name)
            
            # Clear user permissions cache
            self._user_permissions_cache.pop(user_id, None)
            
        logger.info(f"User '{user_id}' added to group '{group_name}'")
        return True
    
    def remove_user_from_group(self, user_id: str, group_name: str) -> bool:
        """
        Remove user from group.
        
        Args:
            user_id: User ID
            group_name: Group name
            
        Returns:
            True if removal successful
        """
        with self._lock:
            user = self.get_user(user_id)
            group = self.get_group(group_name)
            
            if not user or not group:
                return False
            
            group.remove_member(user_id)
            
            if group_name in user.groups:
                user.groups.remove(group_name)
            
            # Clear user permissions cache
            self._user_permissions_cache.pop(user_id, None)
            
        logger.info(f"User '{user_id}' removed from group '{group_name}'")
        return True
    
    def get_user_effective_roles(self, user_id: str) -> Set[str]:
        """Get all effective roles for user (direct + from groups)"""
        user = self.get_user(user_id)
        if not user:
            return set()
        
        roles = set(user.roles)
        
        # Add roles from groups
        for group_name in user.groups:
            group = self.get_group(group_name)
            if group:
                roles.update(group.roles)
        
        return roles
    
    def _clear_caches(self):
        """Clear all performance caches"""
        self._user_permissions_cache.clear()
        self._role_inheritance_cache.clear()
    
    def get_rbac_statistics(self) -> Dict[str, Any]:
        """Get RBAC system statistics"""
        with self._lock:
            stats = {
                "total_users": len(self._users),
                "active_users": sum(1 for u in self._users.values() if u.is_active),
                "total_roles": len(self._roles),
                "system_roles": sum(1 for r in self._roles.values() if r.is_system_role),
                "total_permissions": len(self._permissions),
                "total_groups": len(self._groups),
                "cached_user_permissions": len(self._user_permissions_cache),
                "cached_role_inheritance": len(self._role_inheritance_cache)
            }
        
        return stats
    
    def validate_rbac_integrity(self) -> Dict[str, Any]:
        """Validate RBAC system integrity"""
        issues = []
        
        # Check for orphaned permissions in roles
        for role_name, role in self._roles.items():
            for perm_name in role.permissions:
                if perm_name not in self._permissions:
                    issues.append(f"Role '{role_name}' has non-existent permission '{perm_name}'")
        
        # Check for orphaned roles in users
        for user_id, user in self._users.items():
            for role_name in user.roles:
                if role_name not in self._roles:
                    issues.append(f"User '{user_id}' has non-existent role '{role_name}'")
        
        # Check for circular role dependencies
        for role_name in self._roles:
            visited = set()
            if self._has_circular_dependency(role_name, visited):
                issues.append(f"Circular dependency detected in role hierarchy starting from '{role_name}'")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "issues_count": len(issues)
        }
    
    def _has_circular_dependency(self, role_name: str, visited: Set[str], path: Optional[Set[str]] = None) -> bool:
        """Check for circular dependencies in role hierarchy"""
        if path is None:
            path = set()
        
        if role_name in path:
            return True
        
        if role_name in visited:
            return False
        
        visited.add(role_name)
        path.add(role_name)
        
        role = self.get_role(role_name)
        if role:
            for parent_role in role.parent_roles:
                if self._has_circular_dependency(parent_role, visited, path.copy()):
                    return True
        
        path.remove(role_name)
        return False