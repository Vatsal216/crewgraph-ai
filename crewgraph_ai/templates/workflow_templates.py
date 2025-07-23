"""
Workflow Templates - Pre-built workflow templates for common use cases

This module provides a template system for creating standardized workflows including:
- Base template class for creating custom templates
- Template registry for organizing and discovering templates
- Template builder for constructing workflows from templates
- Pre-built templates for common scenarios

Features:
- Standardized template structure
- Parameterizable workflows
- Template validation and verification
- Easy customization and extension
- Integration with existing CrewGraph components

Created by: Vatsal216
Date: 2025-07-23
"""

import uuid
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy
from datetime import datetime, timezone

from ..core.agents import AgentWrapper
from ..core.tasks import TaskWrapper
from ..core.orchestrator import GraphOrchestrator
from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError, ValidationError

logger = get_logger(__name__)


class TemplateCategory(Enum):
    """Categories for workflow templates"""
    DATA_PROCESSING = "data_processing"
    RESEARCH = "research"
    CONTENT_GENERATION = "content_generation"
    CUSTOMER_SERVICE = "customer_service"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    CUSTOM = "custom"


@dataclass
class TemplateMetadata:
    """Metadata for workflow templates"""
    name: str
    description: str
    version: str = "1.0.0"
    category: TemplateCategory = TemplateCategory.CUSTOM
    author: str = "CrewGraph AI"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    complexity: str = "medium"  # simple, medium, complex
    estimated_time: str = "5-10 minutes"
    requirements: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TemplateParameter:
    """Parameter definition for workflow templates"""
    name: str
    description: str
    param_type: str = "str"  # str, int, float, bool, list, dict
    required: bool = True
    default_value: Any = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)


@dataclass
class TemplateStep:
    """Individual step in a workflow template"""
    step_id: str
    name: str
    description: str
    agent_role: str
    task_description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False


class WorkflowTemplate(ABC):
    """
    Base class for workflow templates.
    
    Provides the foundation for creating standardized, reusable workflows
    with configurable parameters and validation.
    """
    
    def __init__(self):
        self.template_id = str(uuid.uuid4())
        self.metadata = self._define_metadata()
        self.parameters = self._define_parameters()
        self.steps = self._define_steps()
        self._validate_template()
        
        logger.info(f"Template '{self.metadata.name}' initialized with ID: {self.template_id}")
    
    @abstractmethod
    def _define_metadata(self) -> TemplateMetadata:
        """Define template metadata."""
        pass
    
    @abstractmethod
    def _define_parameters(self) -> List[TemplateParameter]:
        """Define template parameters."""
        pass
    
    @abstractmethod
    def _define_steps(self) -> List[TemplateStep]:
        """Define workflow steps."""
        pass
    
    def _validate_template(self):
        """Validate template configuration."""
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValidationError("Duplicate step IDs found in template")
        
        # Validate dependencies
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValidationError(f"Step '{step.step_id}' has invalid dependency: {dep}")
        
        logger.debug(f"Template '{self.metadata.name}' validation completed")
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize template parameters.
        
        Args:
            params: Parameter values to validate
            
        Returns:
            Validated and normalized parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        validated = {}
        
        for param in self.parameters:
            value = params.get(param.name, param.default_value)
            
            # Check required parameters
            if param.required and value is None:
                raise ValidationError(f"Required parameter '{param.name}' is missing")
            
            # Type validation
            if value is not None:
                try:
                    validated[param.name] = self._validate_parameter_type(value, param)
                except (TypeError, ValueError) as e:
                    raise ValidationError(f"Invalid type for parameter '{param.name}': {e}")
            
            # Apply validation rules
            if value is not None and param.validation_rules:
                self._apply_validation_rules(param.name, value, param.validation_rules)
        
        return validated
    
    def _validate_parameter_type(self, value: Any, param: TemplateParameter) -> Any:
        """Validate and convert parameter type."""
        if param.param_type == "str":
            return str(value)
        elif param.param_type == "int":
            return int(value)
        elif param.param_type == "float":
            return float(value)
        elif param.param_type == "bool":
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif param.param_type == "list":
            if isinstance(value, str):
                # Try to parse as comma-separated values
                return [item.strip() for item in value.split(",")]
            return list(value)
        elif param.param_type == "dict":
            if isinstance(value, str):
                import json
                return json.loads(value)
            return dict(value)
        else:
            return value
    
    def _apply_validation_rules(self, name: str, value: Any, rules: Dict[str, Any]):
        """Apply validation rules to parameter value."""
        if "min_length" in rules and len(str(value)) < rules["min_length"]:
            raise ValidationError(f"Parameter '{name}' is too short")
        
        if "max_length" in rules and len(str(value)) > rules["max_length"]:
            raise ValidationError(f"Parameter '{name}' is too long")
        
        if "min_value" in rules and value < rules["min_value"]:
            raise ValidationError(f"Parameter '{name}' is below minimum value")
        
        if "max_value" in rules and value > rules["max_value"]:
            raise ValidationError(f"Parameter '{name}' exceeds maximum value")
        
        if "allowed_values" in rules and value not in rules["allowed_values"]:
            raise ValidationError(f"Parameter '{name}' has invalid value")
    
    def create_workflow(self, 
                       params: Dict[str, Any],
                       workflow_name: Optional[str] = None) -> GraphOrchestrator:
        """
        Create a workflow instance from the template.
        
        Args:
            params: Template parameters
            workflow_name: Name for the workflow instance
            
        Returns:
            Configured GraphOrchestrator instance
        """
        # Validate parameters
        validated_params = self.validate_parameters(params)
        
        # Create workflow name
        if not workflow_name:
            workflow_name = f"{self.metadata.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create orchestrator
        orchestrator = GraphOrchestrator(workflow_name)
        
        # Create and configure agents and tasks
        agents = self._create_agents(validated_params)
        tasks = self._create_tasks(validated_params, agents)
        
        # Add agents and tasks to orchestrator
        for agent in agents.values():
            orchestrator.add_agent(agent)
        
        for task in tasks.values():
            orchestrator.add_task(task)
        
        # Configure workflow structure
        self._configure_workflow_structure(orchestrator, validated_params, agents, tasks)
        
        logger.info(f"Workflow created from template '{self.metadata.name}': {workflow_name}")
        return orchestrator
    
    @abstractmethod
    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, AgentWrapper]:
        """Create agents for the workflow."""
        pass
    
    @abstractmethod
    def _create_tasks(self, 
                     params: Dict[str, Any], 
                     agents: Dict[str, AgentWrapper]) -> Dict[str, TaskWrapper]:
        """Create tasks for the workflow."""
        pass
    
    @abstractmethod
    def _configure_workflow_structure(self,
                                    orchestrator: GraphOrchestrator,
                                    params: Dict[str, Any],
                                    agents: Dict[str, AgentWrapper],
                                    tasks: Dict[str, TaskWrapper]):
        """Configure the workflow structure."""
        pass
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for template parameters."""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in self.parameters:
            param_schema = {
                "type": param.param_type,
                "description": param.description
            }
            
            if param.default_value is not None:
                param_schema["default"] = param.default_value
            
            if param.examples:
                param_schema["examples"] = param.examples
            
            # Add validation rules to schema
            param_schema.update(param.validation_rules)
            
            schema["properties"][param.name] = param_schema
            
            if param.required:
                schema["required"].append(param.name)
        
        return schema
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive template information."""
        return {
            "template_id": self.template_id,
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "version": self.metadata.version,
                "category": self.metadata.category.value,
                "author": self.metadata.author,
                "created_at": self.metadata.created_at.isoformat(),
                "tags": self.metadata.tags,
                "complexity": self.metadata.complexity,
                "estimated_time": self.metadata.estimated_time,
                "requirements": self.metadata.requirements,
                "examples": self.metadata.examples
            },
            "parameters": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.param_type,
                    "required": p.required,
                    "default": p.default_value,
                    "examples": p.examples
                }
                for p in self.parameters
            ],
            "steps": [
                {
                    "id": s.step_id,
                    "name": s.name,
                    "description": s.description,
                    "agent_role": s.agent_role,
                    "dependencies": s.dependencies,
                    "optional": s.optional
                }
                for s in self.steps
            ]
        }


class TemplateRegistry:
    """
    Registry for managing workflow templates.
    
    Provides template discovery, registration, and management capabilities.
    """
    
    def __init__(self):
        self._templates: Dict[str, WorkflowTemplate] = {}
        self._categories: Dict[TemplateCategory, List[str]] = {}
        
        # Register built-in templates
        self._register_builtin_templates()
        
        logger.info("TemplateRegistry initialized")
    
    def register_template(self, template: WorkflowTemplate) -> bool:
        """
        Register a workflow template.
        
        Args:
            template: Template to register
            
        Returns:
            True if registration successful
        """
        template_name = template.metadata.name
        
        if template_name in self._templates:
            logger.warning(f"Template '{template_name}' already registered")
            return False
        
        self._templates[template_name] = template
        
        # Add to category index
        category = template.metadata.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(template_name)
        
        logger.info(f"Template '{template_name}' registered successfully")
        return True
    
    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get template by name."""
        return self._templates.get(name)
    
    def list_templates(self, 
                      category: Optional[TemplateCategory] = None,
                      tags: Optional[List[str]] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            List of template names
        """
        templates = []
        
        for name, template in self._templates.items():
            # Filter by category
            if category and template.metadata.category != category:
                continue
            
            # Filter by tags
            if tags and not any(tag in template.metadata.tags for tag in tags):
                continue
            
            templates.append(name)
        
        return sorted(templates)
    
    def search_templates(self, query: str) -> List[str]:
        """
        Search templates by name or description.
        
        Args:
            query: Search query
            
        Returns:
            List of matching template names
        """
        query = query.lower()
        results = []
        
        for name, template in self._templates.items():
            if (query in name.lower() or 
                query in template.metadata.description.lower() or
                any(query in tag.lower() for tag in template.metadata.tags)):
                results.append(name)
        
        return sorted(results)
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a template."""
        template = self.get_template(name)
        return template.get_info() if template else None
    
    def get_categories(self) -> Dict[str, List[str]]:
        """Get templates organized by category."""
        return {cat.value: templates for cat, templates in self._categories.items()}
    
    def _register_builtin_templates(self):
        """Register built-in templates."""
        # This will be populated as we create the specific template classes
        pass


class TemplateBuilder:
    """
    Builder for creating workflows from templates.
    
    Provides a fluent interface for configuring and building workflows
    from registered templates.
    """
    
    def __init__(self, registry: Optional[TemplateRegistry] = None):
        self.registry = registry or TemplateRegistry()
        self._template: Optional[WorkflowTemplate] = None
        self._params: Dict[str, Any] = {}
        self._workflow_name: Optional[str] = None
        
        logger.info("TemplateBuilder initialized")
    
    def use_template(self, name: str) -> "TemplateBuilder":
        """
        Select a template to use.
        
        Args:
            name: Template name
            
        Returns:
            Self for method chaining
            
        Raises:
            ValidationError: If template not found
        """
        template = self.registry.get_template(name)
        if not template:
            raise ValidationError(f"Template '{name}' not found")
        
        self._template = template
        self._params = {}  # Reset parameters
        
        logger.info(f"Template '{name}' selected for building")
        return self
    
    def set_parameter(self, name: str, value: Any) -> "TemplateBuilder":
        """
        Set a template parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Self for method chaining
        """
        self._params[name] = value
        return self
    
    def set_parameters(self, params: Dict[str, Any]) -> "TemplateBuilder":
        """
        Set multiple template parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Self for method chaining
        """
        self._params.update(params)
        return self
    
    def set_workflow_name(self, name: str) -> "TemplateBuilder":
        """
        Set the workflow name.
        
        Args:
            name: Workflow name
            
        Returns:
            Self for method chaining
        """
        self._workflow_name = name
        return self
    
    def build(self) -> GraphOrchestrator:
        """
        Build the workflow.
        
        Returns:
            Configured GraphOrchestrator instance
            
        Raises:
            ValidationError: If template not selected or parameters invalid
        """
        if not self._template:
            raise ValidationError("No template selected")
        
        return self._template.create_workflow(self._params, self._workflow_name)
    
    def get_parameter_info(self) -> Optional[List[Dict[str, Any]]]:
        """Get information about template parameters."""
        if not self._template:
            return None
        
        return [
            {
                "name": p.name,
                "description": p.description,
                "type": p.param_type,
                "required": p.required,
                "default": p.default_value,
                "examples": p.examples,
                "current_value": self._params.get(p.name)
            }
            for p in self._template.parameters
        ]
    
    def validate_current_parameters(self) -> Dict[str, Any]:
        """
        Validate current parameters.
        
        Returns:
            Validation results
        """
        if not self._template:
            return {"valid": False, "error": "No template selected"}
        
        try:
            validated = self._template.validate_parameters(self._params)
            return {"valid": True, "parameters": validated}
        except ValidationError as e:
            return {"valid": False, "error": str(e)}


# Global template registry instance
_global_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Get the global template registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = TemplateRegistry()
    return _global_registry


def register_template(template: WorkflowTemplate) -> bool:
    """Register a template with the global registry."""
    return get_template_registry().register_template(template)


def create_workflow_from_template(template_name: str, 
                                params: Dict[str, Any],
                                workflow_name: Optional[str] = None) -> GraphOrchestrator:
    """
    Convenience function to create workflow from template.
    
    Args:
        template_name: Name of template to use
        params: Template parameters
        workflow_name: Optional workflow name
        
    Returns:
        Configured GraphOrchestrator instance
    """
    registry = get_template_registry()
    template = registry.get_template(template_name)
    
    if not template:
        raise ValidationError(f"Template '{template_name}' not found")
    
    return template.create_workflow(params, workflow_name)