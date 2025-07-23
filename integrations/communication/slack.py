"""
Slack Integration for CrewGraph AI

Provides comprehensive Slack messaging and notifications integration with:
- Message sending and receiving
- Channel management
- Workflow notifications
- Interactive components
- File sharing capabilities

Author: CrewGraph AI Team
Version: 1.1.0
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    WebClient = SlackApiError = None
    SLACK_AVAILABLE = False

from ....marketplace.plugins import BasePlugin, PluginContext
from ....utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SlackConfig:
    """Slack integration configuration."""
    
    token: str
    default_channel: str
    
    # Optional settings
    username: str = "CrewGraph AI"
    icon_emoji: str = ":robot_face:"
    timeout: int = 30
    
    # Webhook settings (optional)
    webhook_url: Optional[str] = None


class SlackIntegration(BasePlugin):
    """
    Slack messaging integration plugin.
    
    Provides messaging, notifications, and collaboration capabilities
    for CrewGraph AI workflows.
    """
    
    def __init__(self, context: PluginContext):
        """Initialize Slack integration."""
        super().__init__(context)
        
        self.config = self._parse_config()
        self.client = None
        self.bot_info = None
        self.message_stats = {
            "messages_sent": 0,
            "messages_failed": 0,
            "files_uploaded": 0,
            "channels_accessed": set()
        }
    
    def _parse_config(self) -> SlackConfig:
        """Parse plugin configuration."""
        config_data = self.context.config
        
        return SlackConfig(
            token=config_data.get("token"),
            default_channel=config_data.get("channel"),
            username=config_data.get("username", "CrewGraph AI"),
            icon_emoji=config_data.get("icon_emoji", ":robot_face:"),
            timeout=config_data.get("timeout", 30),
            webhook_url=config_data.get("webhook_url")
        )
    
    async def initialize(self) -> bool:
        """Initialize Slack client."""
        if not SLACK_AVAILABLE:
            self.context.log_error("slack-sdk library not available")
            return False
        
        if not self.config.token:
            self.context.log_error("Slack token is required")
            return False
        
        try:
            # Initialize Slack client
            self.client = WebClient(
                token=self.config.token,
                timeout=self.config.timeout
            )
            
            # Test connection and get bot info
            response = self.client.auth_test()
            self.bot_info = {
                "user_id": response["user_id"],
                "user": response["user"],
                "team": response["team"],
                "team_id": response["team_id"]
            }
            
            self.context.log_info(f"Connected to Slack team: {self.bot_info['team']}")
            return True
            
        except SlackApiError as e:
            self.context.log_error(f"Slack API error: {e.response['error']}")
            return False
        except Exception as e:
            self.context.log_error(f"Failed to initialize Slack: {e}")
            return False
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Slack operations.
        
        Supported operations:
        - send_message: Send text messages
        - send_rich_message: Send formatted messages with blocks
        - upload_file: Upload files to channels
        - create_channel: Create new channels
        - get_channel_info: Get channel information
        - send_notification: Send workflow notifications
        """
        operation = task.get("operation")
        
        if operation == "send_message":
            return await self._send_message(task)
        elif operation == "send_rich_message":
            return await self._send_rich_message(task)
        elif operation == "upload_file":
            return await self._upload_file(task)
        elif operation == "create_channel":
            return await self._create_channel(task)
        elif operation == "get_channel_info":
            return await self._get_channel_info(task)
        elif operation == "send_notification":
            return await self._send_notification(task)
        elif operation == "get_messages":
            return await self._get_messages(task)
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }
    
    async def _send_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Send a text message to Slack."""
        text = task.get("text")
        channel = task.get("channel", self.config.default_channel)
        thread_ts = task.get("thread_ts")
        
        if not text:
            return {"success": False, "error": "Message text is required"}
        
        if not channel:
            return {"success": False, "error": "Channel is required"}
        
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=text,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji,
                thread_ts=thread_ts
            )
            
            self.message_stats["messages_sent"] += 1
            self.message_stats["channels_accessed"].add(channel)
            
            return {
                "success": True,
                "message_ts": response["ts"],
                "channel": response["channel"],
                "text": text
            }
            
        except SlackApiError as e:
            self.message_stats["messages_failed"] += 1
            self.context.log_error(f"Failed to send message: {e.response['error']}")
            
            return {
                "success": False,
                "error": e.response["error"],
                "text": text,
                "channel": channel
            }
    
    async def _send_rich_message(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Send a rich message with blocks and attachments."""
        blocks = task.get("blocks")
        attachments = task.get("attachments")
        text = task.get("text", "")
        channel = task.get("channel", self.config.default_channel)
        
        if not blocks and not attachments:
            return {"success": False, "error": "Blocks or attachments are required"}
        
        if not channel:
            return {"success": False, "error": "Channel is required"}
        
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=text,
                blocks=blocks,
                attachments=attachments,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji
            )
            
            self.message_stats["messages_sent"] += 1
            self.message_stats["channels_accessed"].add(channel)
            
            return {
                "success": True,
                "message_ts": response["ts"],
                "channel": response["channel"],
                "blocks_count": len(blocks) if blocks else 0,
                "attachments_count": len(attachments) if attachments else 0
            }
            
        except SlackApiError as e:
            self.message_stats["messages_failed"] += 1
            self.context.log_error(f"Failed to send rich message: {e.response['error']}")
            
            return {
                "success": False,
                "error": e.response["error"],
                "channel": channel
            }
    
    async def _upload_file(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Upload a file to Slack."""
        file_path = task.get("file_path")
        file_content = task.get("file_content")
        filename = task.get("filename")
        title = task.get("title")
        initial_comment = task.get("initial_comment")
        channels = task.get("channels", [self.config.default_channel])
        
        if not file_path and not file_content:
            return {"success": False, "error": "File path or content is required"}
        
        try:
            if file_path:
                # Upload from file path
                response = self.client.files_upload(
                    channels=channels,
                    file=file_path,
                    title=title,
                    initial_comment=initial_comment
                )
            else:
                # Upload from content
                response = self.client.files_upload(
                    channels=channels,
                    content=file_content,
                    filename=filename,
                    title=title,
                    initial_comment=initial_comment
                )
            
            self.message_stats["files_uploaded"] += 1
            for channel in channels:
                self.message_stats["channels_accessed"].add(channel)
            
            return {
                "success": True,
                "file_id": response["file"]["id"],
                "file_url": response["file"]["url_private"],
                "filename": response["file"]["name"],
                "channels": channels
            }
            
        except SlackApiError as e:
            self.context.log_error(f"Failed to upload file: {e.response['error']}")
            
            return {
                "success": False,
                "error": e.response["error"],
                "filename": filename or file_path
            }
    
    async def _create_channel(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Slack channel."""
        name = task.get("name")
        is_private = task.get("is_private", False)
        
        if not name:
            return {"success": False, "error": "Channel name is required"}
        
        try:
            if is_private:
                response = self.client.conversations_create(
                    name=name,
                    is_private=True
                )
            else:
                response = self.client.conversations_create(name=name)
            
            channel_info = response["channel"]
            
            return {
                "success": True,
                "channel_id": channel_info["id"],
                "channel_name": channel_info["name"],
                "is_private": channel_info.get("is_private", False),
                "created": channel_info["created"]
            }
            
        except SlackApiError as e:
            self.context.log_error(f"Failed to create channel: {e.response['error']}")
            
            return {
                "success": False,
                "error": e.response["error"],
                "channel_name": name
            }
    
    async def _get_channel_info(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about a Slack channel."""
        channel = task.get("channel")
        
        if not channel:
            return {"success": False, "error": "Channel is required"}
        
        try:
            response = self.client.conversations_info(channel=channel)
            channel_info = response["channel"]
            
            return {
                "success": True,
                "channel_id": channel_info["id"],
                "channel_name": channel_info["name"],
                "is_private": channel_info.get("is_private", False),
                "is_archived": channel_info.get("is_archived", False),
                "member_count": channel_info.get("num_members", 0),
                "topic": channel_info.get("topic", {}).get("value", ""),
                "purpose": channel_info.get("purpose", {}).get("value", "")
            }
            
        except SlackApiError as e:
            self.context.log_error(f"Failed to get channel info: {e.response['error']}")
            
            return {
                "success": False,
                "error": e.response["error"],
                "channel": channel
            }
    
    async def _send_notification(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Send workflow notification with status and details."""
        workflow_id = task.get("workflow_id")
        status = task.get("status")
        message = task.get("message")
        details = task.get("details", {})
        channel = task.get("channel", self.config.default_channel)
        
        if not workflow_id or not status:
            return {"success": False, "error": "Workflow ID and status are required"}
        
        # Create rich notification blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Workflow {status.title()}: {workflow_id}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message or f"Workflow {workflow_id} has {status}"
                }
            }
        ]
        
        # Add status indicator
        status_emoji = {
            "started": ":arrow_forward:",
            "completed": ":white_check_mark:",
            "failed": ":x:",
            "warning": ":warning:",
            "paused": ":pause_button:"
        }
        
        emoji = status_emoji.get(status.lower(), ":information_source:")
        
        # Add details if provided
        if details:
            detail_text = "\n".join([f"*{k}:* {v}" for k, v in details.items()])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": detail_text
                }
            })
        
        # Add timestamp
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{emoji} {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
                }
            ]
        })
        
        return await self._send_rich_message({
            "blocks": blocks,
            "channel": channel,
            "text": f"Workflow {status}: {workflow_id}"
        })
    
    async def _get_messages(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get messages from a channel."""
        channel = task.get("channel", self.config.default_channel)
        limit = task.get("limit", 50)
        oldest = task.get("oldest")
        latest = task.get("latest")
        
        if not channel:
            return {"success": False, "error": "Channel is required"}
        
        try:
            response = self.client.conversations_history(
                channel=channel,
                limit=limit,
                oldest=oldest,
                latest=latest
            )
            
            messages = []
            for msg in response["messages"]:
                messages.append({
                    "ts": msg["ts"],
                    "user": msg.get("user"),
                    "text": msg.get("text", ""),
                    "type": msg.get("type"),
                    "subtype": msg.get("subtype"),
                    "thread_ts": msg.get("thread_ts")
                })
            
            return {
                "success": True,
                "messages": messages,
                "channel": channel,
                "message_count": len(messages)
            }
            
        except SlackApiError as e:
            self.context.log_error(f"Failed to get messages: {e.response['error']}")
            
            return {
                "success": False,
                "error": e.response["error"],
                "channel": channel
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Slack client not initialized"
                }
            
            # Test API connection
            response = self.client.auth_test()
            
            return {
                "status": "healthy",
                "bot_info": self.bot_info,
                "team": response.get("team"),
                "statistics": {
                    **self.message_stats,
                    "channels_accessed": len(self.message_stats["channels_accessed"])
                }
            }
            
        except SlackApiError as e:
            return {
                "status": "unhealthy",
                "error": e.response["error"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup Slack resources."""
        self.client = None
        self.context.log_info("Slack integration cleaned up")


# Plugin entry point
Plugin = SlackIntegration

# Plugin manifest
PLUGIN_MANIFEST = {
    "id": "slack",
    "name": "Slack Integration",
    "version": "1.1.0",
    "description": "Comprehensive Slack messaging and notifications integration with rich message support",
    "author": "CrewGraph AI Team",
    "api_version": "1.0.0",
    "min_crewgraph_version": "1.0.0",
    "python_version": ">=3.8",
    "dependencies": ["slack-sdk>=3.19.0"],
    "entry_point": "slack_integration.py",
    "plugin_class": "SlackIntegration",
    "category": "communication",
    "tags": ["communication", "slack", "notifications", "messaging"],
    "permissions": ["network_access"],
    "sandbox_enabled": True,
    "network_access": True,
    "file_access": ["data"],
    "config_schema": {
        "token": {"type": "string", "required": True, "description": "Slack Bot User OAuth Token"},
        "channel": {"type": "string", "required": True, "description": "Default channel for messages"},
        "username": {"type": "string", "default": "CrewGraph AI", "description": "Bot username"},
        "icon_emoji": {"type": "string", "default": ":robot_face:", "description": "Bot icon emoji"},
        "timeout": {"type": "integer", "default": 30, "description": "API timeout in seconds"},
        "webhook_url": {"type": "string", "required": False, "description": "Optional webhook URL"}
    }
}