"""
Agent Communication Hub - Inter-agent communication system for CrewGraph AI

This module provides sophisticated communication capabilities allowing agents to:
- Send direct messages to other agents
- Broadcast messages to multiple agents
- Subscribe to communication channels
- Maintain conversation history
- Route messages intelligently

Features:
- Thread-safe message handling
- Message persistence and history
- Channel-based subscriptions
- Intelligent message routing
- Event-driven architecture
- Integration with existing AgentWrapper

Created by: Vatsal216
Date: 2025-07-23
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timezone

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError
from ..memory.base import BaseMemory
from ..memory.dict_memory import DictMemory

logger = get_logger(__name__)


class MessageType(Enum):
    """Types of messages in the communication system"""
    DIRECT = "direct"           # Direct message to specific agent
    BROADCAST = "broadcast"     # Message to all subscribed agents
    CHANNEL = "channel"         # Message to channel subscribers
    SYSTEM = "system"           # System-generated message
    REQUEST = "request"         # Request requiring response
    RESPONSE = "response"       # Response to a request


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Message:
    """Communication message structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None
    channel: Optional[str] = None
    content: Any = None
    message_type: MessageType = MessageType.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    response_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "channel": self.channel,
            "content": self.content,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "requires_response": self.requires_response,
            "response_to": self.response_to,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        msg = cls()
        msg.id = data.get("id", str(uuid.uuid4()))
        msg.sender_id = data.get("sender_id", "")
        msg.recipient_id = data.get("recipient_id")
        msg.channel = data.get("channel")
        msg.content = data.get("content")
        msg.message_type = MessageType(data.get("message_type", "direct"))
        msg.priority = MessagePriority(data.get("priority", 2))
        msg.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat()))
        msg.metadata = data.get("metadata", {})
        msg.requires_response = data.get("requires_response", False)
        msg.response_to = data.get("response_to")
        if data.get("expires_at"):
            msg.expires_at = datetime.fromisoformat(data["expires_at"])
        return msg


@dataclass
class Channel:
    """Communication channel for agent subscriptions"""
    name: str
    description: str = ""
    subscribers: Set[str] = field(default_factory=set)
    message_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_subscribers: Optional[int] = None
    is_private: bool = False
    
    def add_subscriber(self, agent_id: str) -> bool:
        """Add agent to channel subscribers"""
        if self.max_subscribers and len(self.subscribers) >= self.max_subscribers:
            return False
        self.subscribers.add(agent_id)
        return True
    
    def remove_subscriber(self, agent_id: str) -> bool:
        """Remove agent from channel subscribers"""
        if agent_id in self.subscribers:
            self.subscribers.remove(agent_id)
            return True
        return False
    
    def add_message(self, message: Message):
        """Add message to channel history"""
        self.message_history.append(message)


class CommunicationProtocol(Protocol):
    """Protocol for agent communication integration"""
    
    def receive_message(self, message: Message) -> None:
        """Handle incoming message"""
        ...
    
    def get_agent_id(self) -> str:
        """Get agent identifier"""
        ...


class AgentCommunicationHub:
    """
    Sophisticated inter-agent communication system.
    
    Provides comprehensive messaging capabilities including direct messaging,
    broadcasting, channel subscriptions, message routing, and conversation history.
    """
    
    def __init__(self, 
                 memory: Optional[BaseMemory] = None,
                 max_message_history: int = 10000,
                 cleanup_interval: int = 3600):
        """
        Initialize the communication hub.
        
        Args:
            memory: Memory backend for persistence
            max_message_history: Maximum messages to keep in history
            cleanup_interval: Interval in seconds for cleanup tasks
        """
        self.memory = memory or DictMemory()
        self.max_message_history = max_message_history
        self.cleanup_interval = cleanup_interval
        
        # Agent registry and message handling
        self._agents: Dict[str, CommunicationProtocol] = {}
        self._message_queue: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._message_history: deque = deque(maxlen=max_message_history)
        self._channels: Dict[str, Channel] = {}
        
        # Message routing and callbacks
        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._routing_rules: List[Callable[[Message], Optional[str]]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("AgentCommunicationHub initialized")
        logger.info(f"Max message history: {max_message_history}, Cleanup interval: {cleanup_interval}s")
    
    def start(self):
        """Start the communication hub background tasks"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            
            # Start cleanup task if in async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._cleanup_task = loop.create_task(self._cleanup_loop())
            except RuntimeError:
                # No async context available
                pass
            
            logger.info("AgentCommunicationHub started")
    
    def stop(self):
        """Stop the communication hub"""
        with self._lock:
            self._running = False
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                self._cleanup_task = None
            
            logger.info("AgentCommunicationHub stopped")
    
    def register_agent(self, agent: CommunicationProtocol) -> bool:
        """
        Register an agent with the communication hub.
        
        Args:
            agent: Agent implementing CommunicationProtocol
            
        Returns:
            True if registration successful
        """
        agent_id = agent.get_agent_id()
        
        with self._lock:
            if agent_id in self._agents:
                logger.warning(f"Agent {agent_id} already registered")
                return False
            
            self._agents[agent_id] = agent
            self._message_queue[agent_id] = deque(maxlen=1000)
            
            logger.info(f"Agent {agent_id} registered with communication hub")
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the communication hub.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            True if unregistration successful
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            # Remove from all channels
            for channel in self._channels.values():
                channel.remove_subscriber(agent_id)
            
            # Clean up queues
            if agent_id in self._message_queue:
                del self._message_queue[agent_id]
            
            del self._agents[agent_id]
            
            logger.info(f"Agent {agent_id} unregistered from communication hub")
            return True
    
    def send_message(self, 
                    sender_id: str,
                    recipient_id: Optional[str] = None,
                    content: Any = None,
                    message_type: MessageType = MessageType.DIRECT,
                    channel: Optional[str] = None,
                    priority: MessagePriority = MessagePriority.NORMAL,
                    requires_response: bool = False,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message through the communication hub.
        
        Args:
            sender_id: ID of sending agent
            recipient_id: ID of recipient agent (for direct messages)
            content: Message content
            message_type: Type of message
            channel: Channel name (for channel messages)
            priority: Message priority
            requires_response: Whether message requires a response
            metadata: Additional message metadata
            
        Returns:
            Message ID
            
        Raises:
            ValidationError: If message parameters are invalid
        """
        if not content:
            raise ValidationError("Message content cannot be empty")
        
        if message_type == MessageType.DIRECT and not recipient_id:
            raise ValidationError("Direct messages require recipient_id")
        
        if message_type == MessageType.CHANNEL and not channel:
            raise ValidationError("Channel messages require channel name")
        
        message = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            channel=channel,
            content=content,
            message_type=message_type,
            priority=priority,
            requires_response=requires_response,
            metadata=metadata or {}
        )
        
        return self._route_message(message)
    
    def send_direct_message(self, 
                           sender_id: str,
                           recipient_id: str,
                           content: Any,
                           priority: MessagePriority = MessagePriority.NORMAL,
                           requires_response: bool = False,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send a direct message to specific agent"""
        return self.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            message_type=MessageType.DIRECT,
            priority=priority,
            requires_response=requires_response,
            metadata=metadata
        )
    
    def broadcast_message(self,
                         sender_id: str,
                         content: Any,
                         priority: MessagePriority = MessagePriority.NORMAL,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Broadcast message to all registered agents"""
        return self.send_message(
            sender_id=sender_id,
            content=content,
            message_type=MessageType.BROADCAST,
            priority=priority,
            metadata=metadata
        )
    
    def send_channel_message(self,
                            sender_id: str,
                            channel: str,
                            content: Any,
                            priority: MessagePriority = MessagePriority.NORMAL,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Send message to channel subscribers"""
        return self.send_message(
            sender_id=sender_id,
            channel=channel,
            content=content,
            message_type=MessageType.CHANNEL,
            priority=priority,
            metadata=metadata
        )
    
    def create_channel(self,
                      name: str,
                      description: str = "",
                      max_subscribers: Optional[int] = None,
                      is_private: bool = False,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new communication channel.
        
        Args:
            name: Channel name
            description: Channel description
            max_subscribers: Maximum number of subscribers
            is_private: Whether channel is private
            metadata: Additional channel metadata
            
        Returns:
            True if channel created successfully
        """
        with self._lock:
            if name in self._channels:
                logger.warning(f"Channel {name} already exists")
                return False
            
            channel = Channel(
                name=name,
                description=description,
                max_subscribers=max_subscribers,
                is_private=is_private,
                metadata=metadata or {}
            )
            
            self._channels[name] = channel
            logger.info(f"Channel {name} created")
            return True
    
    def subscribe_to_channel(self, agent_id: str, channel_name: str) -> bool:
        """
        Subscribe agent to a channel.
        
        Args:
            agent_id: ID of agent to subscribe
            channel_name: Name of channel to subscribe to
            
        Returns:
            True if subscription successful
        """
        with self._lock:
            if channel_name not in self._channels:
                logger.warning(f"Channel {channel_name} does not exist")
                return False
            
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return False
            
            channel = self._channels[channel_name]
            success = channel.add_subscriber(agent_id)
            
            if success:
                logger.info(f"Agent {agent_id} subscribed to channel {channel_name}")
            else:
                logger.warning(f"Failed to subscribe agent {agent_id} to channel {channel_name}")
            
            return success
    
    def unsubscribe_from_channel(self, agent_id: str, channel_name: str) -> bool:
        """Unsubscribe agent from a channel"""
        with self._lock:
            if channel_name not in self._channels:
                return False
            
            channel = self._channels[channel_name]
            success = channel.remove_subscriber(agent_id)
            
            if success:
                logger.info(f"Agent {agent_id} unsubscribed from channel {channel_name}")
            
            return success
    
    def get_messages(self, agent_id: str, limit: int = 100) -> List[Message]:
        """Get pending messages for an agent"""
        with self._lock:
            if agent_id not in self._message_queue:
                return []
            
            messages = []
            queue = self._message_queue[agent_id]
            
            for _ in range(min(limit, len(queue))):
                if queue:
                    message = queue.popleft()
                    if not message.is_expired():
                        messages.append(message)
            
            return sorted(messages, key=lambda m: m.priority.value, reverse=True)
    
    def get_conversation_history(self,
                               agent_id: str,
                               other_agent_id: Optional[str] = None,
                               channel: Optional[str] = None,
                               limit: int = 100) -> List[Message]:
        """
        Get conversation history for an agent.
        
        Args:
            agent_id: ID of agent
            other_agent_id: ID of other agent (for direct conversations)
            channel: Channel name (for channel conversations)
            limit: Maximum number of messages
            
        Returns:
            List of messages in conversation
        """
        messages = []
        
        for message in self._message_history:
            if other_agent_id:
                # Direct conversation
                if ((message.sender_id == agent_id and message.recipient_id == other_agent_id) or
                    (message.sender_id == other_agent_id and message.recipient_id == agent_id)):
                    messages.append(message)
            elif channel:
                # Channel conversation
                if message.channel == channel:
                    messages.append(message)
            else:
                # All messages involving this agent
                if (message.sender_id == agent_id or 
                    message.recipient_id == agent_id or
                    (message.channel and channel in self._channels and 
                     agent_id in self._channels[message.channel].subscribers)):
                    messages.append(message)
            
            if len(messages) >= limit:
                break
        
        return sorted(messages, key=lambda m: m.timestamp)
    
    def add_routing_rule(self, rule: Callable[[Message], Optional[str]]):
        """Add a custom message routing rule"""
        self._routing_rules.append(rule)
        logger.info("Custom routing rule added")
    
    def add_message_handler(self, message_type: MessageType, handler: Callable[[Message], None]):
        """Add a message handler for specific message type"""
        self._message_handlers[message_type].append(handler)
        logger.info(f"Message handler added for type: {message_type.value}")
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get communication statistics for an agent"""
        stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "channels_subscribed": [],
            "pending_messages": len(self._message_queue.get(agent_id, [])),
            "is_registered": agent_id in self._agents
        }
        
        # Count messages in history
        for message in self._message_history:
            if message.sender_id == agent_id:
                stats["messages_sent"] += 1
            if message.recipient_id == agent_id:
                stats["messages_received"] += 1
        
        # Get channel subscriptions
        for channel_name, channel in self._channels.items():
            if agent_id in channel.subscribers:
                stats["channels_subscribed"].append(channel_name)
        
        return stats
    
    def get_hub_statistics(self) -> Dict[str, Any]:
        """Get overall hub statistics"""
        return {
            "total_agents": len(self._agents),
            "total_channels": len(self._channels),
            "total_messages": len(self._message_history),
            "pending_messages": sum(len(queue) for queue in self._message_queue.values()),
            "active_channels": [name for name, channel in self._channels.items() if channel.subscribers],
            "message_types": {msg_type.value: 0 for msg_type in MessageType}
        }
    
    def _route_message(self, message: Message) -> str:
        """Route message to appropriate recipients"""
        with self._lock:
            # Apply custom routing rules
            for rule in self._routing_rules:
                try:
                    custom_recipient = rule(message)
                    if custom_recipient:
                        message.recipient_id = custom_recipient
                        break
                except Exception as e:
                    logger.error(f"Error in routing rule: {e}")
            
            # Route based on message type
            if message.message_type == MessageType.DIRECT:
                self._route_direct_message(message)
            elif message.message_type == MessageType.BROADCAST:
                self._route_broadcast_message(message)
            elif message.message_type == MessageType.CHANNEL:
                self._route_channel_message(message)
            
            # Add to message history
            self._message_history.append(message)
            
            # Trigger message handlers
            for handler in self._message_handlers[message.message_type]:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
            
            logger.debug(f"Message {message.id} routed successfully")
            return message.id
    
    def _route_direct_message(self, message: Message):
        """Route direct message to specific recipient"""
        if not message.recipient_id or message.recipient_id not in self._agents:
            logger.warning(f"Recipient {message.recipient_id} not found for direct message")
            return
        
        self._message_queue[message.recipient_id].append(message)
        
        # Deliver immediately if possible
        try:
            agent = self._agents[message.recipient_id]
            agent.receive_message(message)
        except Exception as e:
            logger.error(f"Error delivering message to {message.recipient_id}: {e}")
    
    def _route_broadcast_message(self, message: Message):
        """Route broadcast message to all registered agents"""
        for agent_id in self._agents:
            if agent_id != message.sender_id:  # Don't send to sender
                message_copy = Message(
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    content=message.content,
                    message_type=MessageType.BROADCAST,
                    priority=message.priority,
                    metadata=message.metadata.copy()
                )
                self._message_queue[agent_id].append(message_copy)
                
                # Deliver immediately if possible
                try:
                    agent = self._agents[agent_id]
                    agent.receive_message(message_copy)
                except Exception as e:
                    logger.error(f"Error delivering broadcast to {agent_id}: {e}")
    
    def _route_channel_message(self, message: Message):
        """Route channel message to subscribers"""
        if not message.channel or message.channel not in self._channels:
            logger.warning(f"Channel {message.channel} not found for channel message")
            return
        
        channel = self._channels[message.channel]
        channel.add_message(message)
        
        for agent_id in channel.subscribers:
            if agent_id != message.sender_id:  # Don't send to sender
                message_copy = Message(
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    channel=message.channel,
                    content=message.content,
                    message_type=MessageType.CHANNEL,
                    priority=message.priority,
                    metadata=message.metadata.copy()
                )
                self._message_queue[agent_id].append(message_copy)
                
                # Deliver immediately if possible
                try:
                    agent = self._agents[agent_id]
                    agent.receive_message(message_copy)
                except Exception as e:
                    logger.error(f"Error delivering channel message to {agent_id}: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup_expired_messages()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_expired_messages(self):
        """Clean up expired messages"""
        with self._lock:
            # Clean message history
            expired_count = 0
            for message in list(self._message_history):
                if message.is_expired():
                    self._message_history.remove(message)
                    expired_count += 1
            
            # Clean agent queues
            for agent_id, queue in self._message_queue.items():
                expired_messages = []
                for message in list(queue):
                    if message.is_expired():
                        expired_messages.append(message)
                
                for message in expired_messages:
                    queue.remove(message)
                    expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired messages")