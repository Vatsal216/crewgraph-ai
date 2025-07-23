"""
Workflow Templates Module for CrewGraph AI

This module provides pre-built workflow templates for common use cases including:
- Data processing pipeline template
- Research automation workflow
- Content generation with review cycles
- Customer service automation
- Multi-step analysis workflows
- Marketing campaign management
- Quality assurance testing
- Financial analysis
- Document processing
- Template marketplace functionality
- YAML/JSON format support
- Template inheritance and composition

Created by: Vatsal216
Date: 2025-07-23
"""

from .additional_templates import (
    CustomerSupportTemplate,
    DocumentProcessingTemplate,
    FinancialAnalysisTemplate,
    MarketingCampaignTemplate,
    QualityAssuranceTemplate,
)
from .content_generation import ContentGenerationTemplate
from .data_pipeline import DataPipelineTemplate
from .formats import (
    DynamicWorkflowTemplate,
    ParameterInjector,
    TemplateDeserializer,
    TemplateFormatError,
    TemplateInheritance,
    TemplateSchema,
    TemplateSerializer,
    create_template_from_json,
    create_template_from_yaml,
    load_template_from_file,
    save_template_as_json,
    save_template_as_yaml,
)
from .marketplace import (
    MarketplaceTemplate,
    TemplateMarketplace,
    TemplateRating,
    TemplateSource,
    TemplateStats,
    TemplateStatus,
    get_featured_templates,
    get_template_marketplace,
    search_marketplace_templates,
)
from .research_workflow import ResearchWorkflowTemplate
from .workflow_templates import (
    TemplateBuilder,
    TemplateCategory,
    TemplateMetadata,
    TemplateParameter,
    TemplateRegistry,
    TemplateStep,
    WorkflowTemplate,
    create_workflow_from_template,
    get_template_registry,
    register_template,
)

__all__ = [
    # Core template classes
    "WorkflowTemplate",
    "TemplateRegistry",
    "TemplateCategory",
    "TemplateBuilder",
    "TemplateMetadata",
    "TemplateParameter",
    "TemplateStep",
    
    # Built-in templates
    "DataPipelineTemplate",
    "ResearchWorkflowTemplate", 
    "ContentGenerationTemplate",
    "CustomerSupportTemplate",
    "MarketingCampaignTemplate",
    "QualityAssuranceTemplate",
    "FinancialAnalysisTemplate",
    "DocumentProcessingTemplate",
    
    # Marketplace functionality
    "TemplateMarketplace",
    "MarketplaceTemplate",
    "TemplateRating",
    "TemplateStats",
    "TemplateSource",
    "TemplateStatus",
    "get_template_marketplace",
    "search_marketplace_templates",
    "get_featured_templates",
    
    # Format support
    "TemplateSerializer",
    "TemplateDeserializer", 
    "DynamicWorkflowTemplate",
    "TemplateSchema",
    "TemplateFormatError",
    "TemplateInheritance",
    "ParameterInjector",
    "save_template_as_yaml",
    "save_template_as_json",
    "load_template_from_file",
    "create_template_from_yaml",
    "create_template_from_json",
    
    # Convenience functions
    "register_template",
    "get_template_registry",
    "create_workflow_from_template",
]

# Version info
__version__ = "1.0.0"
__author__ = "Vatsal216"

# Initialize global registries with built-in templates
def _initialize_builtin_templates():
    """Initialize the template registry with built-in templates"""
    registry = get_template_registry()
    marketplace = get_template_marketplace()
    
    # Register all built-in templates
    builtin_templates = [
        DataPipelineTemplate(),
        ResearchWorkflowTemplate(),
        ContentGenerationTemplate(),
        CustomerSupportTemplate(),
        MarketingCampaignTemplate(),
        QualityAssuranceTemplate(),
        FinancialAnalysisTemplate(),
        DocumentProcessingTemplate(),
    ]
    
    for template in builtin_templates:
        registry.register_template(template)
        marketplace.add_template(template, TemplateSource.OFFICIAL)
    
    # Mark some templates as featured
    featured_template_names = [
        "Data Processing Pipeline",
        "Research Automation Workflow", 
        "Content Generation Pipeline",
        "Customer Support Automation"
    ]
    
    for template_id, marketplace_template in marketplace.templates.items():
        if marketplace_template.template.metadata.name in featured_template_names:
            marketplace_template.featured = True
            marketplace_template.verified = True

# Initialize built-in templates when module is imported
_initialize_builtin_templates()
