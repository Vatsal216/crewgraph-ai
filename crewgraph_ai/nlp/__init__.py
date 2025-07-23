"""
Natural Language Processing for CrewGraph AI

AI-powered natural language workflow building and documentation system.
Provides comprehensive NLP capabilities for converting natural language descriptions
into executable workflows, enabling conversational workflow building.

Key Components:
    - Workflow Parser: Convert natural language to workflow definitions
    - Requirements Parser: Parse natural language requirements into structured data
    - Intent Classifier: Understand user intentions and requirements
    - Task Extractor: Extract individual tasks from descriptions
    - Dependency Analyzer: Identify task dependencies from context
    - NL Converters: Bidirectional natural language conversion
    - Conversational Builder: Interactive workflow building
    - Documentation Generator: Auto-generate comprehensive documentation
    - Code Generator: Generate executable code from workflows

Features:
    - Natural language to workflow conversion
    - Conversational workflow building
    - Auto-documentation generation
    - Code generation for multiple languages
    - Context-aware task extraction
    - Bidirectional natural language conversion

Author: Vatsal216
Created: 2025-07-23 10:33:54 UTC
"""

# Core parsing and workflow components
from .parser import WorkflowParser, RequirementsParser, ParsedWorkflow
from .classifier import IntentClassifier, WorkflowIntent
from .extractor import TaskExtractor, ExtractedTask

# Conversion and building components
from .converter import NLToWorkflowConverter, WorkflowToNLConverter
from .builder import ConversationalWorkflowBuilder

# Generation components
from .generator import DocumentationGenerator, CodeGenerator, WorkflowDocumentation

__all__ = [
    # Parser components
    "WorkflowParser",
    "RequirementsParser",
    "ParsedWorkflow",
    
    # Classification components
    "IntentClassifier", 
    "WorkflowIntent",
    
    # Extraction components
    "TaskExtractor",
    "ExtractedTask",
    
    # Conversion components
    "NLToWorkflowConverter",
    "WorkflowToNLConverter",
    
    # Building components
    "ConversationalWorkflowBuilder",
    
    # Generation components
    "DocumentationGenerator",
    "CodeGenerator",
    "WorkflowDocumentation"
]

__version__ = "1.0.0"
__author__ = "Vatsal216"
__created__ = "2025-07-23 10:33:54 UTC"

print(f"🗣️ CrewGraph AI NLP Module v{__version__} loaded")
print(f"📝 Natural language workflow building enabled")
print(f"🤖 AI-powered workflow conversion and generation")
print(f"👤 Created by: {__author__}")
print(f"⏰ Updated: {__created__}")
