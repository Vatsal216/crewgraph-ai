"""
Workflow Templates Module for CrewGraph AI

This module provides pre-built workflow templates for common use cases including:
- Data processing pipeline template
- Research automation workflow
- Content generation with review cycles
- Customer service automation
- Multi-step analysis workflows

Created by: Vatsal216
Date: 2025-07-23
"""

from .content_generation import ContentGenerationTemplate
from .data_pipeline import DataPipelineTemplate
from .research_workflow import ResearchWorkflowTemplate
from .workflow_templates import (
    TemplateBuilder,
    TemplateCategory,
    TemplateRegistry,
    WorkflowTemplate,
)

__all__ = [
    "WorkflowTemplate",
    "TemplateRegistry",
    "TemplateCategory",
    "TemplateBuilder",
    "DataPipelineTemplate",
    "ResearchWorkflowTemplate",
    "ContentGenerationTemplate",
]

# Version info
__version__ = "1.0.0"
__author__ = "Vatsal216"
