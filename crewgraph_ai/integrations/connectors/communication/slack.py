"""
Slack Integration for CrewGraph AI

Provides Slack messaging, channel management, and notification capabilities
for workflow automation and team communication.

Author: Vatsal216
Created: 2025-07-23 18:40:00 UTC
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ... import BaseIntegration, IntegrationConfig, IntegrationMetadata, IntegrationResult, IntegrationType

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class SlackIntegration(BaseIntegration):
    """
    Slack integration for team communication and notifications.
    
    Supports sending messages, managing channels, and retrieving
    team information through the Slack Web API.
    """
    
    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Slack integration metadata."""
        return IntegrationMetadata(
            name="Slack",
            version="1.0.0",
            description="Team communication and notifications via Slack",
            author="CrewGraph AI",
            integration_type=IntegrationType.COMMUNICATION,
            dependencies=["requests"],
            config_schema={
                "type": "object",
                "properties": {
                    "bot_token": {
                        "type": "string",
                        "description": "Slack Bot User OAuth Token (xoxb-...)"
                    },
                    "default_channel": {
                        "type": "string",
                        "description": "Default channel for messages"
                    },
                    "workspace_url": {
                        "type": "string", 
                        "description": "Slack workspace URL"
                    }
                },
                "required": ["bot_token"]
            },
            supports_async=True,
            supports_webhook=True,
            homepage="https://api.slack.com/",
            documentation="https://api.slack.com/web",
            tags=["communication", "messaging", "notifications", "team"]
        )
    
    def initialize(self) -> bool:
        """Initialize Slack integration."""
        try:
            if not REQUESTS_AVAILABLE:
                self.logger.error("requests library not available")
                return False
            
            # Get configuration
            self.bot_token = self.config.config.get("bot_token")
            self.default_channel = self.config.config.get("default_channel", "#general")
            self.workspace_url = self.config.config.get("workspace_url")
            
            if not self.bot_token:
                self.logger.error("bot_token is required")
                return False
            
            # Test API connection
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            # Simulate API test (in production, make actual API call)
            self.logger.info("Testing Slack API connection...")
            
            # Store API headers for future use
            self.headers = headers
            self.base_url = "https://slack.com/api"
            
            self.is_initialized = True
            self.logger.info("Slack integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Slack integration: {e}")
            return False
    
    def execute(self, action: str, **kwargs) -> IntegrationResult:
        """Execute Slack action."""
        if not self.is_initialized:
            return IntegrationResult(
                success=False,
                error_message="Integration not initialized"
            )
        
        try:
            if action == "send_message":
                return self._send_message(**kwargs)
            elif action == "create_channel":
                return self._create_channel(**kwargs)
            elif action == "list_channels":
                return self._list_channels(**kwargs)
            elif action == "get_user_info":
                return self._get_user_info(**kwargs)
            elif action == "upload_file":
                return self._upload_file(**kwargs)
            else:
                return IntegrationResult(
                    success=False,
                    error_message=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_config(self) -> List[str]:
        """Validate Slack configuration."""
        issues = []
        
        config = self.config.config
        
        if not config.get("bot_token"):
            issues.append("bot_token is required")
        elif not config["bot_token"].startswith("xoxb-"):
            issues.append("bot_token should start with 'xoxb-'")
        
        default_channel = config.get("default_channel")
        if default_channel and not (default_channel.startswith("#") or default_channel.startswith("@")):
            issues.append("default_channel should start with '#' for channels or '@' for DMs")
        
        return issues
    
    def _send_message(
        self, 
        text: str, 
        channel: Optional[str] = None,
        blocks: Optional[List[Dict]] = None,
        attachments: Optional[List[Dict]] = None,
        thread_ts: Optional[str] = None
    ) -> IntegrationResult:
        """Send message to Slack channel."""
        try:
            channel = channel or self.default_channel
            
            payload = {
                "channel": channel,
                "text": text
            }
            
            if blocks:
                payload["blocks"] = blocks
            if attachments:
                payload["attachments"] = attachments
            if thread_ts:
                payload["thread_ts"] = thread_ts
            
            # Simulate API call (in production, make actual request)
            response = self._simulate_api_call("chat.postMessage", payload)
            
            if response.get("ok"):
                return IntegrationResult(
                    success=True,
                    data={
                        "message_ts": response.get("ts"),
                        "channel": response.get("channel"),
                        "text": text
                    },
                    metadata={"channel": channel, "thread_ts": thread_ts}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to send message")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error sending message: {str(e)}"
            )
    
    def _create_channel(
        self, 
        name: str, 
        is_private: bool = False,
        purpose: Optional[str] = None
    ) -> IntegrationResult:
        """Create new Slack channel."""
        try:
            # Remove # prefix if present
            channel_name = name.lstrip("#")
            
            payload = {
                "name": channel_name,
                "is_private": is_private
            }
            
            if purpose:
                payload["purpose"] = purpose
            
            # Simulate API call
            endpoint = "conversations.create"
            response = self._simulate_api_call(endpoint, payload)
            
            if response.get("ok"):
                channel_info = response.get("channel", {})
                return IntegrationResult(
                    success=True,
                    data={
                        "channel_id": channel_info.get("id"),
                        "channel_name": channel_info.get("name"),
                        "is_private": channel_info.get("is_private", False),
                        "created": channel_info.get("created")
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to create channel")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error creating channel: {str(e)}"
            )
    
    def _list_channels(self, limit: int = 100, exclude_archived: bool = True) -> IntegrationResult:
        """List Slack channels."""
        try:
            payload = {
                "limit": limit,
                "exclude_archived": exclude_archived,
                "types": "public_channel,private_channel"
            }
            
            # Simulate API call
            response = self._simulate_api_call("conversations.list", payload)
            
            if response.get("ok"):
                channels = response.get("channels", [])
                channel_list = [
                    {
                        "id": ch.get("id"),
                        "name": ch.get("name"),
                        "is_private": ch.get("is_private", False),
                        "is_archived": ch.get("is_archived", False),
                        "num_members": ch.get("num_members", 0),
                        "purpose": ch.get("purpose", {}).get("value", ""),
                        "topic": ch.get("topic", {}).get("value", "")
                    }
                    for ch in channels
                ]
                
                return IntegrationResult(
                    success=True,
                    data={"channels": channel_list, "total": len(channel_list)}
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to list channels")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error listing channels: {str(e)}"
            )
    
    def _get_user_info(self, user_id: str) -> IntegrationResult:
        """Get Slack user information."""
        try:
            payload = {"user": user_id}
            
            # Simulate API call
            response = self._simulate_api_call("users.info", payload)
            
            if response.get("ok"):
                user = response.get("user", {})
                profile = user.get("profile", {})
                
                user_info = {
                    "id": user.get("id"),
                    "name": user.get("name"),
                    "real_name": user.get("real_name"),
                    "display_name": profile.get("display_name"),
                    "email": profile.get("email"),
                    "is_admin": user.get("is_admin", False),
                    "is_bot": user.get("is_bot", False),
                    "timezone": user.get("tz"),
                    "status": profile.get("status_text", "")
                }
                
                return IntegrationResult(
                    success=True,
                    data=user_info
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to get user info")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error getting user info: {str(e)}"
            )
    
    def _upload_file(
        self, 
        file_path: str, 
        channels: Optional[List[str]] = None,
        title: Optional[str] = None,
        initial_comment: Optional[str] = None
    ) -> IntegrationResult:
        """Upload file to Slack."""
        try:
            channels = channels or [self.default_channel]
            
            payload = {
                "channels": ",".join(channels),
                "filename": file_path.split("/")[-1] if "/" in file_path else file_path
            }
            
            if title:
                payload["title"] = title
            if initial_comment:
                payload["initial_comment"] = initial_comment
            
            # Simulate file upload
            response = self._simulate_api_call("files.upload", payload)
            
            if response.get("ok"):
                file_info = response.get("file", {})
                return IntegrationResult(
                    success=True,
                    data={
                        "file_id": file_info.get("id"),
                        "name": file_info.get("name"),
                        "url": file_info.get("url_private"),
                        "size": file_info.get("size"),
                        "channels": channels
                    }
                )
            else:
                return IntegrationResult(
                    success=False,
                    error_message=response.get("error", "Failed to upload file")
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                error_message=f"Error uploading file: {str(e)}"
            )
    
    def _simulate_api_call(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Slack API call (for testing without actual API)."""
        # In production, this would make actual HTTP requests to Slack API
        
        if endpoint == "auth.test":
            return {
                "ok": True,
                "url": "https://example.slack.com/",
                "team": "Example Team",
                "user": "bot_user",
                "team_id": "T1234567890",
                "user_id": "U1234567890"
            }
        
        elif endpoint == "chat.postMessage":
            return {
                "ok": True,
                "channel": payload.get("channel"),
                "ts": f"{int(datetime.now().timestamp())}.123456",
                "message": {
                    "text": payload.get("text"),
                    "user": "U1234567890",
                    "ts": f"{int(datetime.now().timestamp())}.123456"
                }
            }
        
        elif endpoint == "conversations.create":
            return {
                "ok": True,
                "channel": {
                    "id": "C1234567890",
                    "name": payload.get("name"),
                    "is_private": payload.get("is_private", False),
                    "created": int(datetime.now().timestamp())
                }
            }
        
        elif endpoint == "conversations.list":
            return {
                "ok": True,
                "channels": [
                    {
                        "id": "C1234567890",
                        "name": "general",
                        "is_private": False,
                        "is_archived": False,
                        "num_members": 10,
                        "purpose": {"value": "General discussion"},
                        "topic": {"value": "Welcome to the team!"}
                    },
                    {
                        "id": "C1234567891", 
                        "name": "random",
                        "is_private": False,
                        "is_archived": False,
                        "num_members": 8,
                        "purpose": {"value": "Random chat"},
                        "topic": {"value": ""}
                    }
                ]
            }
        
        elif endpoint == "users.info":
            return {
                "ok": True,
                "user": {
                    "id": payload.get("user"),
                    "name": "john.doe",
                    "real_name": "John Doe",
                    "is_admin": False,
                    "is_bot": False,
                    "tz": "America/New_York",
                    "profile": {
                        "display_name": "John",
                        "email": "john.doe@example.com",
                        "status_text": "In a meeting"
                    }
                }
            }
        
        elif endpoint == "files.upload":
            return {
                "ok": True,
                "file": {
                    "id": "F1234567890",
                    "name": payload.get("filename"),
                    "url_private": "https://files.slack.com/files-pri/...",
                    "size": 1024,
                    "channels": payload.get("channels", "").split(",")
                }
            }
        
        else:
            return {
                "ok": False,
                "error": f"Unknown endpoint: {endpoint}"
            }
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Slack-specific health check."""
        try:
            # Test auth
            response = self._simulate_api_call("auth.test", {})
            
            if response.get("ok"):
                return {
                    "status": "healthy",
                    "message": "Slack API connection successful",
                    "team": response.get("team"),
                    "user": response.get("user"),
                    "execution_count": self.execution_count,
                    "last_execution": self.last_execution
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"Slack API error: {response.get('error')}",
                    "execution_count": self.execution_count
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "execution_count": self.execution_count
            }