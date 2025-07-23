"""
Template Format Support for CrewGraph AI

Provides support for YAML/JSON template formats including:
- Template serialization to YAML/JSON
- Template deserialization from YAML/JSON  
- Template validation and schema checking
- Template inheritance and composition
- Template parameter injection

Created by: Vatsal216
Date: 2025-07-23
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Type, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..utils.exceptions import CrewGraphError, ValidationError
from ..utils.logging import get_logger
from .workflow_templates import (
    TemplateMetadata,
    TemplateParameter,
    TemplateStep,
    WorkflowTemplate,
    TemplateCategory
)

logger = get_logger(__name__)


class TemplateFormatError(CrewGraphError):
    """Template format related errors"""
    pass


class TemplateSchema:
    """Schema definition for template formats"""
    
    REQUIRED_FIELDS = ["metadata", "parameters", "steps"]
    
    METADATA_SCHEMA = {
        "name": {"type": str, "required": True},
        "description": {"type": str, "required": True},
        "version": {"type": str, "required": False, "default": "1.0.0"},
        "category": {"type": str, "required": False, "default": "custom"},
        "author": {"type": str, "required": False, "default": "Unknown"},
        "tags": {"type": list, "required": False, "default": []},
        "complexity": {"type": str, "required": False, "default": "medium"},
        "estimated_time": {"type": str, "required": False, "default": "5-10 minutes"},
        "requirements": {"type": list, "required": False, "default": []},
        "examples": {"type": list, "required": False, "default": []}
    }
    
    PARAMETER_SCHEMA = {
        "name": {"type": str, "required": True},
        "description": {"type": str, "required": True},
        "param_type": {"type": str, "required": False, "default": "str"},
        "required": {"type": bool, "required": False, "default": True},
        "default_value": {"type": "any", "required": False, "default": None},
        "validation_rules": {"type": dict, "required": False, "default": {}},
        "examples": {"type": list, "required": False, "default": []}
    }
    
    STEP_SCHEMA = {
        "step_id": {"type": str, "required": True},
        "name": {"type": str, "required": True},
        "description": {"type": str, "required": True},
        "agent_role": {"type": str, "required": True},
        "task_description": {"type": str, "required": True},
        "inputs": {"type": list, "required": False, "default": []},
        "outputs": {"type": list, "required": False, "default": []},
        "dependencies": {"type": list, "required": False, "default": []},
        "tools": {"type": list, "required": False, "default": []},
        "configuration": {"type": dict, "required": False, "default": {}},
        "optional": {"type": bool, "required": False, "default": False}
    }


class TemplateSerializer:
    """Serializes templates to various formats"""
    
    @staticmethod
    def to_dict(template: WorkflowTemplate) -> Dict[str, Any]:
        """Convert template to dictionary representation"""
        return {
            "metadata": {
                "name": template.metadata.name,
                "description": template.metadata.description,
                "version": template.metadata.version,
                "category": template.metadata.category.value,
                "author": template.metadata.author,
                "created_at": template.metadata.created_at.isoformat(),
                "tags": template.metadata.tags,
                "complexity": template.metadata.complexity,
                "estimated_time": template.metadata.estimated_time,
                "requirements": template.metadata.requirements,
                "examples": template.metadata.examples
            },
            "parameters": [
                {
                    "name": param.name,
                    "description": param.description,
                    "param_type": param.param_type,
                    "required": param.required,
                    "default_value": param.default_value,
                    "validation_rules": param.validation_rules,
                    "examples": param.examples
                }
                for param in template.parameters
            ],
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "description": step.description,
                    "agent_role": step.agent_role,
                    "task_description": step.task_description,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                    "dependencies": step.dependencies,
                    "tools": step.tools,
                    "configuration": step.configuration,
                    "optional": step.optional
                }
                for step in template.steps
            ]
        }
    
    @staticmethod
    def to_json(template: WorkflowTemplate, indent: int = 2) -> str:
        """Convert template to JSON string"""
        template_dict = TemplateSerializer.to_dict(template)
        return json.dumps(template_dict, indent=indent, default=str)
    
    @staticmethod
    def to_yaml(template: WorkflowTemplate) -> str:
        """Convert template to YAML string"""
        if not YAML_AVAILABLE:
            raise TemplateFormatError("PyYAML is not installed. Install with: pip install PyYAML")
        
        template_dict = TemplateSerializer.to_dict(template)
        return yaml.dump(template_dict, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def save_template(template: WorkflowTemplate, filepath: str, format: str = "auto") -> None:
        """
        Save template to file.
        
        Args:
            template: Template to save
            filepath: File path to save to
            format: Format to save in (json, yaml, auto)
        """
        if format == "auto":
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                format = "yaml"
            elif filepath.endswith(".json"):
                format = "json"
            else:
                format = "json"  # Default to JSON
        
        if format == "json":
            content = TemplateSerializer.to_json(template)
        elif format == "yaml":
            content = TemplateSerializer.to_yaml(template)
        else:
            raise TemplateFormatError(f"Unsupported format: {format}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Template saved to {filepath} in {format} format")


class TemplateDeserializer:
    """Deserializes templates from various formats"""
    
    @staticmethod
    def _validate_dict(data: Dict[str, Any]) -> None:
        """Validate template dictionary structure"""
        for field in TemplateSchema.REQUIRED_FIELDS:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate metadata
        metadata = data["metadata"]
        for field, schema in TemplateSchema.METADATA_SCHEMA.items():
            if schema["required"] and field not in metadata:
                raise ValidationError(f"Missing required metadata field: {field}")
        
        # Validate parameters
        if not isinstance(data["parameters"], list):
            raise ValidationError("Parameters must be a list")
        
        for i, param in enumerate(data["parameters"]):
            for field, schema in TemplateSchema.PARAMETER_SCHEMA.items():
                if schema["required"] and field not in param:
                    raise ValidationError(f"Missing required parameter field '{field}' in parameter {i}")
        
        # Validate steps
        if not isinstance(data["steps"], list):
            raise ValidationError("Steps must be a list")
        
        for i, step in enumerate(data["steps"]):
            for field, schema in TemplateSchema.STEP_SCHEMA.items():
                if schema["required"] and field not in step:
                    raise ValidationError(f"Missing required step field '{field}' in step {i}")
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DynamicWorkflowTemplate':
        """Create template from dictionary"""
        TemplateDeserializer._validate_dict(data)
        return DynamicWorkflowTemplate(data)
    
    @staticmethod
    def from_json(json_str: str) -> 'DynamicWorkflowTemplate':
        """Create template from JSON string"""
        try:
            data = json.loads(json_str)
            return TemplateDeserializer.from_dict(data)
        except json.JSONDecodeError as e:
            raise TemplateFormatError(f"Invalid JSON format: {e}")
    
    @staticmethod
    def from_yaml(yaml_str: str) -> 'DynamicWorkflowTemplate':
        """Create template from YAML string"""
        if not YAML_AVAILABLE:
            raise TemplateFormatError("PyYAML is not installed. Install with: pip install PyYAML")
        
        try:
            data = yaml.safe_load(yaml_str)
            return TemplateDeserializer.from_dict(data)
        except yaml.YAMLError as e:
            raise TemplateFormatError(f"Invalid YAML format: {e}")
    
    @staticmethod
    def load_template(filepath: str, format: str = "auto") -> 'DynamicWorkflowTemplate':
        """
        Load template from file.
        
        Args:
            filepath: File path to load from
            format: Format to load (json, yaml, auto)
            
        Returns:
            Loaded template
        """
        if format == "auto":
            if filepath.endswith(".yaml") or filepath.endswith(".yml"):
                format = "yaml"
            elif filepath.endswith(".json"):
                format = "json"
            else:
                # Try to detect from content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content.startswith('{'):
                        format = "json"
                    else:
                        format = "yaml"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if format == "json":
            template = TemplateDeserializer.from_json(content)
        elif format == "yaml":
            template = TemplateDeserializer.from_yaml(content)
        else:
            raise TemplateFormatError(f"Unsupported format: {format}")
        
        logger.info(f"Template loaded from {filepath}")
        return template


class DynamicWorkflowTemplate(WorkflowTemplate):
    """
    Dynamic template created from dictionary/file data.
    
    This allows creating templates from YAML/JSON without defining
    a new class for each template.
    """
    
    def __init__(self, template_data: Dict[str, Any]):
        """
        Initialize dynamic template from data.
        
        Args:
            template_data: Template data dictionary
        """
        self._template_data = template_data
        super().__init__()
    
    def _define_metadata(self) -> TemplateMetadata:
        """Define metadata from template data"""
        metadata = self._template_data["metadata"]
        
        # Convert category string to enum
        category_value = metadata.get("category", "custom")
        try:
            category = TemplateCategory(category_value)
        except ValueError:
            category = TemplateCategory.CUSTOM
        
        return TemplateMetadata(
            name=metadata["name"],
            description=metadata["description"],
            version=metadata.get("version", "1.0.0"),
            category=category,
            author=metadata.get("author", "Unknown"),
            tags=metadata.get("tags", []),
            complexity=metadata.get("complexity", "medium"),
            estimated_time=metadata.get("estimated_time", "5-10 minutes"),
            requirements=metadata.get("requirements", []),
            examples=metadata.get("examples", [])
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        """Define parameters from template data"""
        parameters = []
        
        for param_data in self._template_data["parameters"]:
            param = TemplateParameter(
                name=param_data["name"],
                description=param_data["description"],
                param_type=param_data.get("param_type", "str"),
                required=param_data.get("required", True),
                default_value=param_data.get("default_value"),
                validation_rules=param_data.get("validation_rules", {}),
                examples=param_data.get("examples", [])
            )
            parameters.append(param)
        
        return parameters
    
    def _define_steps(self) -> List[TemplateStep]:
        """Define steps from template data"""
        steps = []
        
        for step_data in self._template_data["steps"]:
            step = TemplateStep(
                step_id=step_data["step_id"],
                name=step_data["name"],
                description=step_data["description"],
                agent_role=step_data["agent_role"],
                task_description=step_data["task_description"],
                inputs=step_data.get("inputs", []),
                outputs=step_data.get("outputs", []),
                dependencies=step_data.get("dependencies", []),
                tools=step_data.get("tools", []),
                configuration=step_data.get("configuration", {}),
                optional=step_data.get("optional", False)
            )
            steps.append(step)
        
        return steps


class TemplateInheritance:
    """Handles template inheritance and composition"""
    
    @staticmethod
    def extend_template(base_template: WorkflowTemplate, extensions: Dict[str, Any]) -> DynamicWorkflowTemplate:
        """
        Extend a base template with additional configuration.
        
        Args:
            base_template: Base template to extend
            extensions: Extensions to apply
            
        Returns:
            Extended template
        """
        # Convert base template to dict
        base_data = TemplateSerializer.to_dict(base_template)
        
        # Apply extensions
        if "metadata" in extensions:
            base_data["metadata"].update(extensions["metadata"])
        
        if "parameters" in extensions:
            # Add new parameters
            existing_param_names = {p["name"] for p in base_data["parameters"]}
            for param in extensions["parameters"]:
                if param["name"] not in existing_param_names:
                    base_data["parameters"].append(param)
        
        if "steps" in extensions:
            # Add new steps
            existing_step_ids = {s["step_id"] for s in base_data["steps"]}
            for step in extensions["steps"]:
                if step["step_id"] not in existing_step_ids:
                    base_data["steps"].append(step)
        
        return DynamicWorkflowTemplate(base_data)
    
    @staticmethod
    def compose_templates(templates: List[WorkflowTemplate], composition_rules: Dict[str, Any] = None) -> DynamicWorkflowTemplate:
        """
        Compose multiple templates into a single template.
        
        Args:
            templates: List of templates to compose
            composition_rules: Rules for composition
            
        Returns:
            Composed template
        """
        if not templates:
            raise ValidationError("At least one template required for composition")
        
        composition_rules = composition_rules or {}
        
        # Start with first template as base
        base_data = TemplateSerializer.to_dict(templates[0])
        
        # Update metadata for composition
        base_data["metadata"]["name"] = composition_rules.get("name", "Composed Template")
        base_data["metadata"]["description"] = composition_rules.get("description", "Template composed from multiple sources")
        
        # Combine parameters and steps from all templates
        all_param_names = set()
        all_step_ids = set()
        
        for template in templates[1:]:
            template_data = TemplateSerializer.to_dict(template)
            
            # Add parameters
            for param in template_data["parameters"]:
                if param["name"] not in all_param_names:
                    base_data["parameters"].append(param)
                    all_param_names.add(param["name"])
            
            # Add steps
            for step in template_data["steps"]:
                if step["step_id"] not in all_step_ids:
                    base_data["steps"].append(step)
                    all_step_ids.add(step["step_id"])
        
        return DynamicWorkflowTemplate(base_data)


class ParameterInjector:
    """Handles parameter injection into templates"""
    
    @staticmethod
    def inject_parameters(template_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject parameters into template data.
        
        Args:
            template_data: Template data with parameter placeholders
            parameters: Parameter values to inject
            
        Returns:
            Template data with injected parameters
        """
        injected_data = json.loads(json.dumps(template_data))  # Deep copy
        
        # Replace parameter placeholders
        def replace_placeholders(obj):
            if isinstance(obj, str):
                # Replace ${param_name} placeholders
                for param_name, param_value in parameters.items():
                    placeholder = f"${{{param_name}}}"
                    if placeholder in obj:
                        obj = obj.replace(placeholder, str(param_value))
                return obj
            elif isinstance(obj, dict):
                return {k: replace_placeholders(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item) for item in obj]
            else:
                return obj
        
        return replace_placeholders(injected_data)


# Convenience functions
def save_template_as_yaml(template: WorkflowTemplate, filepath: str) -> None:
    """Save template as YAML file"""
    TemplateSerializer.save_template(template, filepath, "yaml")


def save_template_as_json(template: WorkflowTemplate, filepath: str) -> None:
    """Save template as JSON file"""
    TemplateSerializer.save_template(template, filepath, "json")


def load_template_from_file(filepath: str) -> DynamicWorkflowTemplate:
    """Load template from YAML or JSON file"""
    return TemplateDeserializer.load_template(filepath)


def create_template_from_yaml(yaml_content: str) -> DynamicWorkflowTemplate:
    """Create template from YAML string"""
    return TemplateDeserializer.from_yaml(yaml_content)


def create_template_from_json(json_content: str) -> DynamicWorkflowTemplate:
    """Create template from JSON string"""
    return TemplateDeserializer.from_json(json_content)