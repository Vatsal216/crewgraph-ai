"""
Base Langflow Component for CrewGraph AI Integration

This module provides the base class for all CrewGraph Langflow components.

Created by: Vatsal216
Date: 2025-07-23
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# Import CrewGraph utilities
try:
    from ....utils.logging import get_logger
    from ....utils.exceptions import CrewGraphError
except ImportError:
    def get_logger(name):
        import logging
        return logging.getLogger(name)
    
    class CrewGraphError(Exception):
        pass

logger = get_logger(__name__)


class ComponentInput(BaseModel):
    """Input field definition for Langflow component"""
    name: str = Field(..., description="Input field name")
    display_name: str = Field(..., description="Display name in UI")
    input_type: str = Field(..., description="Input type (str, int, float, bool, dict, list)")
    required: bool = Field(True, description="Whether input is required")
    default_value: Any = Field(None, description="Default value")
    description: str = Field("", description="Field description")
    options: Optional[List[str]] = Field(None, description="Available options for select inputs")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value for numeric inputs")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value for numeric inputs")
    multiline: bool = Field(False, description="Whether text input should be multiline")
    file_types: Optional[List[str]] = Field(None, description="Allowed file types for file inputs")


class ComponentOutput(BaseModel):
    """Output field definition for Langflow component"""
    name: str = Field(..., description="Output field name")
    display_name: str = Field(..., description="Display name in UI")
    output_type: str = Field(..., description="Output type (str, int, float, bool, dict, list)")
    description: str = Field("", description="Field description")


class ComponentMetadata(BaseModel):
    """Metadata for Langflow component"""
    component_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique component ID")
    name: str = Field(..., description="Component name")
    display_name: str = Field(..., description="Display name in UI")
    description: str = Field("", description="Component description")
    category: str = Field("CrewGraph", description="Component category")
    tags: List[str] = Field(default_factory=list, description="Component tags")
    version: str = Field("1.0.0", description="Component version")
    author: str = Field("Vatsal216", description="Component author")
    icon: Optional[str] = Field(None, description="Component icon")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    beta: bool = Field(False, description="Whether component is in beta")


class LangflowComponent(ABC):
    """
    Base class for all CrewGraph Langflow components
    
    This class provides the foundation for creating custom Langflow components
    that integrate with CrewGraph AI functionality.
    """
    
    def __init__(self):
        self.metadata = self._get_metadata()
        self.inputs = self._get_inputs()
        self.outputs = self._get_outputs()
        self.logger = get_logger(self.__class__.__name__)
        
        # Validate component definition
        self._validate_component()
    
    @abstractmethod
    def _get_metadata(self) -> ComponentMetadata:
        """
        Get component metadata
        
        Returns:
            ComponentMetadata object defining the component
        """
        pass
    
    @abstractmethod
    def _get_inputs(self) -> List[ComponentInput]:
        """
        Get component input definitions
        
        Returns:
            List of ComponentInput objects defining inputs
        """
        pass
    
    @abstractmethod
    def _get_outputs(self) -> List[ComponentOutput]:
        """
        Get component output definitions
        
        Returns:
            List of ComponentOutput objects defining outputs
        """
        pass
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the component with given inputs
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values
            
        Raises:
            CrewGraphError: If execution fails
        """
        pass
    
    def _validate_component(self) -> None:
        """Validate component definition"""
        try:
            # Validate metadata
            if not self.metadata.name:
                raise ValueError("Component name is required")
            
            # Validate inputs
            input_names = set()
            for input_def in self.inputs:
                if input_def.name in input_names:
                    raise ValueError(f"Duplicate input name: {input_def.name}")
                input_names.add(input_def.name)
            
            # Validate outputs
            output_names = set()
            for output_def in self.outputs:
                if output_def.name in output_names:
                    raise ValueError(f"Duplicate output name: {output_def.name}")
                output_names.add(output_def.name)
            
            self.logger.info(f"Component '{self.metadata.name}' validated successfully")
            
        except Exception as e:
            self.logger.error(f"Component validation failed: {e}")
            raise CrewGraphError(f"Component validation failed: {e}")
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and process input values
        
        Args:
            inputs: Raw input values
            
        Returns:
            Validated and processed input values
            
        Raises:
            CrewGraphError: If validation fails
        """
        validated_inputs = {}
        
        for input_def in self.inputs:
            value = inputs.get(input_def.name)
            
            # Check required inputs
            if input_def.required and value is None:
                if input_def.default_value is not None:
                    value = input_def.default_value
                else:
                    raise CrewGraphError(f"Required input '{input_def.name}' is missing")
            
            # Type validation and conversion
            if value is not None:
                try:
                    value = self._convert_input_type(value, input_def)
                except Exception as e:
                    raise CrewGraphError(f"Invalid type for input '{input_def.name}': {e}")
                
                # Range validation for numeric types
                if input_def.input_type in ["int", "float"] and value is not None:
                    if input_def.min_value is not None and value < input_def.min_value:
                        raise CrewGraphError(f"Input '{input_def.name}' is below minimum value {input_def.min_value}")
                    if input_def.max_value is not None and value > input_def.max_value:
                        raise CrewGraphError(f"Input '{input_def.name}' is above maximum value {input_def.max_value}")
                
                # Options validation
                if input_def.options and value not in input_def.options:
                    raise CrewGraphError(f"Input '{input_def.name}' must be one of: {input_def.options}")
            
            validated_inputs[input_def.name] = value
        
        return validated_inputs
    
    def _convert_input_type(self, value: Any, input_def: ComponentInput) -> Any:
        """Convert input value to the expected type"""
        if value is None:
            return None
        
        input_type = input_def.input_type.lower()
        
        if input_type == "str":
            return str(value)
        elif input_type == "int":
            return int(value)
        elif input_type == "float":
            return float(value)
        elif input_type == "bool":
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ["true", "1", "yes", "on"]
            else:
                return bool(value)
        elif input_type in ["dict", "object"]:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, dict):
                return value
            else:
                raise ValueError(f"Cannot convert {type(value)} to dict")
        elif input_type in ["list", "array"]:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, list):
                return value
            else:
                raise ValueError(f"Cannot convert {type(value)} to list")
        else:
            return value
    
    def format_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format output values according to output definitions
        
        Args:
            outputs: Raw output values
            
        Returns:
            Formatted output values
        """
        formatted_outputs = {}
        
        for output_def in self.outputs:
            value = outputs.get(output_def.name)
            
            if value is not None:
                try:
                    value = self._convert_output_type(value, output_def)
                except Exception as e:
                    self.logger.warning(f"Failed to format output '{output_def.name}': {e}")
                    # Keep original value if conversion fails
            
            formatted_outputs[output_def.name] = value
        
        return formatted_outputs
    
    def _convert_output_type(self, value: Any, output_def: ComponentOutput) -> Any:
        """Convert output value to the expected type"""
        if value is None:
            return None
        
        output_type = output_def.output_type.lower()
        
        if output_type == "str":
            return str(value)
        elif output_type == "int":
            return int(value)
        elif output_type == "float":
            return float(value)
        elif output_type == "bool":
            return bool(value)
        elif output_type in ["dict", "object"]:
            if isinstance(value, dict):
                return value
            else:
                return {"value": value}
        elif output_type in ["list", "array"]:
            if isinstance(value, list):
                return value
            else:
                return [value]
        else:
            return value
    
    def get_langflow_definition(self) -> Dict[str, Any]:
        """
        Get Langflow component definition
        
        Returns:
            Dictionary containing the Langflow component definition
        """
        return {
            "name": self.metadata.name,
            "display_name": self.metadata.display_name,
            "description": self.metadata.description,
            "category": self.metadata.category,
            "tags": self.metadata.tags,
            "version": self.metadata.version,
            "author": self.metadata.author,
            "icon": self.metadata.icon,
            "beta": self.metadata.beta,
            "inputs": [
                {
                    "name": inp.name,
                    "display_name": inp.display_name,
                    "type": inp.input_type,
                    "required": inp.required,
                    "default": inp.default_value,
                    "description": inp.description,
                    "options": inp.options,
                    "min": inp.min_value,
                    "max": inp.max_value,
                    "multiline": inp.multiline,
                    "file_types": inp.file_types,
                }
                for inp in self.inputs
            ],
            "outputs": [
                {
                    "name": out.name,
                    "display_name": out.display_name,
                    "type": out.output_type,
                    "description": out.description,
                }
                for out in self.outputs
            ],
            "documentation_url": self.metadata.documentation_url,
        }
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Main entry point for component execution
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            Dictionary of output values
        """
        try:
            # Validate inputs
            validated_inputs = self.validate_inputs(kwargs)
            
            self.logger.info(f"Executing component '{self.metadata.name}' with inputs: {list(validated_inputs.keys())}")
            
            # Execute component
            raw_outputs = await self.execute(validated_inputs)
            
            # Format outputs
            formatted_outputs = self.format_outputs(raw_outputs)
            
            self.logger.info(f"Component '{self.metadata.name}' executed successfully")
            
            return formatted_outputs
            
        except Exception as e:
            self.logger.error(f"Component execution failed: {e}")
            raise CrewGraphError(f"Component execution failed: {e}")
    
    def __str__(self) -> str:
        return f"LangflowComponent(name='{self.metadata.name}', version='{self.metadata.version}')"
    
    def __repr__(self) -> str:
        return self.__str__()