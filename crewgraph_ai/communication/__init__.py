"""
Communication Module for CrewGraph AI

This module provides sophisticated inter-agent communication capabilities including:
- Direct messaging between agents
- Broadcasting to multiple agents
- Channel-based communication
- Message routing and history
- Subscription management

Created by: Vatsal216
Date: 2025-07-23
"""

from .agent_hub import (
    AgentCommunicationHub,
    Channel,
    CommunicationProtocol,
    Message,
    MessagePriority,
    MessageType,
)

__all__ = [
    "AgentCommunicationHub",
    "Message",
    "MessageType",
    "MessagePriority",
    "Channel",
    "CommunicationProtocol",
]

# Version info
__version__ = "1.0.0"
__author__ = "Vatsal216"
