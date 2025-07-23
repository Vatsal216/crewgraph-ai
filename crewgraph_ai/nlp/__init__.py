"""
<<<<<<< HEAD
Natural Language Processing for CrewGraph AI

AI-powered natural language workflow building and documentation.

Author: Vatsal216
Created: 2025-07-23 06:25:00 UTC
"""

from .parser import RequirementsParser, WorkflowParser
from .converter import NLToWorkflowConverter, WorkflowToNLConverter
from .builder import ConversationalWorkflowBuilder
from .generator import DocumentationGenerator, CodeGenerator

__all__ = [
    "RequirementsParser",
    "WorkflowParser", 
    "NLToWorkflowConverter",
    "WorkflowToNLConverter",
    "ConversationalWorkflowBuilder",
    "DocumentationGenerator",
    "CodeGenerator"
]

__version__ = "1.0.0"
__author__ = "Vatsal216"
__created__ = "2025-07-23 06:25:00"

print(f"🗣️ CrewGraph AI NLP Module v{__version__} loaded")
print(f"📝 Natural language workflow building enabled")
print(f"👤 Created by: {__author__}")
print(f"⏰ Timestamp: {__created__}")
=======
Natural Language Processing Module for CrewGraph AI

This module provides NLP capabilities for converting natural language descriptions
into executable workflows, enabling conversational workflow building.

Key Components:
    - Workflow Parser: Convert natural language to workflow definitions
    - Intent Classifier: Understand user intentions and requirements
    - Task Extractor: Extract individual tasks from descriptions
    - Dependency Analyzer: Identify task dependencies from context

Features:
    - Natural language to workflow conversion
    - Conversational workflow building
    - Auto-documentation generation
    - Context-aware task extraction

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from .parser import WorkflowParser, ParsedWorkflow
from .classifier import IntentClassifier, WorkflowIntent
from .extractor import TaskExtractor, ExtractedTask
from .generator import DocumentationGenerator, WorkflowDocumentation

__all__ = [
    "WorkflowParser",
    "ParsedWorkflow",
    "IntentClassifier", 
    "WorkflowIntent",
    "TaskExtractor",
    "ExtractedTask",
    "DocumentationGenerator",
    "WorkflowDocumentation"
]

__version__ = "1.0.0"
__author__ = "Vatsal216"
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
