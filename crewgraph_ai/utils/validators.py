"""
Comprehensive validation utilities for CrewGraph AI
"""

import inspect
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .exceptions import ValidationError
from .logging import get_logger

logger = get_logger(__name__)


class ValidationType(Enum):
    """Validation type enumeration"""

    TYPE = "type"
    RANGE = "range"
    FORMAT = "format"
    CUSTOM = "custom"
    REQUIRED = "required"


@dataclass
class ValidationRule:
    """Validation rule definition"""

    name: str
    validation_type: ValidationType
    validator: Callable[[Any], bool]
    message: str
    severity: str = "error"  # error, warning

    def validate(self, value: Any) -> bool:
        """Execute validation"""
        try:
            return self.validator(value)
        except Exception as e:
            logger.error(f"Validation rule '{self.name}' failed: {e}")
            return False


class ParameterValidator:
    """Comprehensive parameter validation system"""

    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default validation rules"""

        # String validations
        self.add_rule(
            "string",
            ValidationRule(
                name="string_type",
                validation_type=ValidationType.TYPE,
                validator=lambda x: isinstance(x, str),
                message="Value must be a string",
            ),
        )

        self.add_rule(
            "non_empty_string",
            ValidationRule(
                name="non_empty",
                validation_type=ValidationType.REQUIRED,
                validator=lambda x: isinstance(x, str) and len(x.strip()) > 0,
                message="String cannot be empty",
            ),
        )

        # Numeric validations
        self.add_rule(
            "integer",
            ValidationRule(
                name="integer_type",
                validation_type=ValidationType.TYPE,
                validator=lambda x: isinstance(x, int),
                message="Value must be an integer",
            ),
        )

        self.add_rule(
            "positive_integer",
            ValidationRule(
                name="positive",
                validation_type=ValidationType.RANGE,
                validator=lambda x: isinstance(x, int) and x > 0,
                message="Value must be a positive integer",
            ),
        )

        self.add_rule(
            "float",
            ValidationRule(
                name="float_type",
                validation_type=ValidationType.TYPE,
                validator=lambda x: isinstance(x, (int, float)),
                message="Value must be a number",
            ),
        )

        # Collection validations
        self.add_rule(
            "list",
            ValidationRule(
                name="list_type",
                validation_type=ValidationType.TYPE,
                validator=lambda x: isinstance(x, list),
                message="Value must be a list",
            ),
        )

        self.add_rule(
            "dict",
            ValidationRule(
                name="dict_type",
                validation_type=ValidationType.TYPE,
                validator=lambda x: isinstance(x, dict),
                message="Value must be a dictionary",
            ),
        )

        # Format validations
        self.add_rule(
            "email",
            ValidationRule(
                name="email_format",
                validation_type=ValidationType.FORMAT,
                validator=self._validate_email,
                message="Value must be a valid email address",
            ),
        )

        self.add_rule(
            "url",
            ValidationRule(
                name="url_format",
                validation_type=ValidationType.FORMAT,
                validator=self._validate_url,
                message="Value must be a valid URL",
            ),
        )

        self.add_rule(
            "json",
            ValidationRule(
                name="json_format",
                validation_type=ValidationType.FORMAT,
                validator=self._validate_json,
                message="Value must be valid JSON",
            ),
        )

    def add_rule(self, parameter_type: str, rule: ValidationRule):
        """Add validation rule for parameter type"""
        if parameter_type not in self.rules:
            self.rules[parameter_type] = []
        self.rules[parameter_type].append(rule)

    def validate_parameter(
        self,
        value: Any,
        parameter_type: str,
        parameter_name: str = "parameter",
        custom_rules: Optional[List[ValidationRule]] = None,
    ) -> bool:
        """
        Validate parameter value.

        Args:
            value: Value to validate
            parameter_type: Parameter type to validate against
            parameter_name: Parameter name for error messages
            custom_rules: Additional custom validation rules

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        errors = []
        warnings = []

        # Get rules for parameter type
        type_rules = self.rules.get(parameter_type, [])
        all_rules = type_rules + (custom_rules or [])

        if not all_rules:
            logger.warning(f"No validation rules found for type '{parameter_type}'")
            return True

        # Execute validation rules
        for rule in all_rules:
            try:
                if not rule.validate(value):
                    message = f"Parameter '{parameter_name}': {rule.message}"

                    if rule.severity == "error":
                        errors.append(message)
                    else:
                        warnings.append(message)

            except Exception as e:
                error_msg = (
                    f"Parameter '{parameter_name}': Validation rule '{rule.name}' failed - {str(e)}"
                )
                errors.append(error_msg)

        # Log warnings
        for warning in warnings:
            logger.warning(warning)

        # Raise errors
        if errors:
            raise ValidationError(
                f"Parameter validation failed for '{parameter_name}'",
                field=parameter_name,
                value=value,
                expected_type=parameter_type,
                details={"errors": errors, "warnings": warnings},
            )

        return True

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        schema: Dict[str, str],
        custom_rules: Optional[Dict[str, List[ValidationRule]]] = None,
    ) -> bool:
        """
        Validate multiple parameters against schema.

        Args:
            parameters: Parameters to validate
            schema: Parameter type schema
            custom_rules: Custom validation rules per parameter

        Returns:
            True if all valid

        Raises:
            ValidationError: If any validation fails
        """
        all_errors = []

        for param_name, param_type in schema.items():
            if param_name in parameters:
                try:
                    param_custom_rules = custom_rules.get(param_name, []) if custom_rules else []
                    self.validate_parameter(
                        parameters[param_name], param_type, param_name, param_custom_rules
                    )
                except ValidationError as e:
                    all_errors.append(str(e))

        if all_errors:
            raise ValidationError(
                "Multiple parameter validation errors", details={"errors": all_errors}
            )

        return True

    def create_range_rule(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        inclusive: bool = True,
    ) -> ValidationRule:
        """Create range validation rule"""

        def range_validator(value):
            if not isinstance(value, (int, float)):
                return False

            if min_value is not None:
                if inclusive and value < min_value:
                    return False
                elif not inclusive and value <= min_value:
                    return False

            if max_value is not None:
                if inclusive and value > max_value:
                    return False
                elif not inclusive and value >= max_value:
                    return False

            return True

        message = f"Value must be "
        if min_value is not None and max_value is not None:
            message += f"between {min_value} and {max_value}"
        elif min_value is not None:
            message += f"greater than {'or equal to ' if inclusive else ''}{min_value}"
        elif max_value is not None:
            message += f"less than {'or equal to ' if inclusive else ''}{max_value}"

        return ValidationRule(
            name="range_validation",
            validation_type=ValidationType.RANGE,
            validator=range_validator,
            message=message,
        )

    def create_regex_rule(self, pattern: str, message: str) -> ValidationRule:
        """Create regex validation rule"""
        compiled_pattern = re.compile(pattern)

        def regex_validator(value):
            if not isinstance(value, str):
                return False
            return bool(compiled_pattern.match(value))

        return ValidationRule(
            name="regex_validation",
            validation_type=ValidationType.FORMAT,
            validator=regex_validator,
            message=message,
        )

    def _validate_email(self, value: Any) -> bool:
        """Validate email format"""
        if not isinstance(value, str):
            return False

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(email_pattern, value))

    def _validate_url(self, value: Any) -> bool:
        """Validate URL format"""
        if not isinstance(value, str):
            return False

        url_pattern = r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$"
        return bool(re.match(url_pattern, value))

    def _validate_json(self, value: Any) -> bool:
        """Validate JSON format"""
        if isinstance(value, str):
            try:
                json.loads(value)
                return True
            except json.JSONDecodeError:
                return False
        elif isinstance(value, (dict, list)):
            try:
                json.dumps(value)
                return True
            except (TypeError, ValueError):
                return False

        return False


class StateValidator:
    """Validate workflow state objects"""

    def __init__(self):
        self.required_fields = {"workflow_id": str, "current_step": str, "status": str}

        self.optional_fields = {
            "metadata": dict,
            "results": dict,
            "errors": list,
            "start_time": (int, float),
            "end_time": (int, float),
        }

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate workflow state structure.

        Args:
            state: State dictionary to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        errors = []

        # Check required fields
        for field, expected_type in self.required_fields.items():
            if field not in state:
                errors.append(f"Required field '{field}' missing")
            elif not isinstance(state[field], expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")

        # Check optional fields (if present)
        for field, expected_type in self.optional_fields.items():
            if field in state:
                if isinstance(expected_type, tuple):
                    if not isinstance(state[field], expected_type):
                        type_names = [t.__name__ for t in expected_type]
                        errors.append(f"Field '{field}' must be of type {' or '.join(type_names)}")
                else:
                    if not isinstance(state[field], expected_type):
                        errors.append(f"Field '{field}' must be of type {expected_type.__name__}")

        if errors:
            raise ValidationError("State validation failed", details={"errors": errors})

        return True


class WorkflowValidator:
    """Validate workflow definitions"""

    def __init__(self):
        self.param_validator = ParameterValidator()

    def validate_workflow(self, workflow_config: Dict[str, Any]) -> bool:
        """
        Validate workflow configuration.

        Args:
            workflow_config: Workflow configuration to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        errors = []

        # Check required workflow fields
        required_fields = ["name", "agents", "tasks"]
        for field in required_fields:
            if field not in workflow_config:
                errors.append(f"Required field '{field}' missing")

        # Validate agents
        if "agents" in workflow_config:
            try:
                self._validate_agents(workflow_config["agents"])
            except ValidationError as e:
                errors.extend(e.details.get("errors", [str(e)]))

        # Validate tasks
        if "tasks" in workflow_config:
            try:
                self._validate_tasks(workflow_config["tasks"])
            except ValidationError as e:
                errors.extend(e.details.get("errors", [str(e)]))

        # Validate dependencies
        if "agents" in workflow_config and "tasks" in workflow_config:
            try:
                self._validate_dependencies(workflow_config["agents"], workflow_config["tasks"])
            except ValidationError as e:
                errors.extend(e.details.get("errors", [str(e)]))

        if errors:
            raise ValidationError("Workflow validation failed", details={"errors": errors})

        return True

    def _validate_agents(self, agents: List[Dict[str, Any]]):
        """Validate agent configurations"""
        errors = []
        agent_names = set()

        for i, agent in enumerate(agents):
            # Check required agent fields
            if "name" not in agent:
                errors.append(f"Agent {i}: 'name' field required")
            else:
                name = agent["name"]
                if name in agent_names:
                    errors.append(f"Agent {i}: Duplicate agent name '{name}'")
                agent_names.add(name)

            if "role" not in agent:
                errors.append(f"Agent {i}: 'role' field required")

        if errors:
            raise ValidationError("Agent validation failed", details={"errors": errors})

    def _validate_tasks(self, tasks: List[Dict[str, Any]]):
        """Validate task configurations"""
        errors = []
        task_names = set()

        for i, task in enumerate(tasks):
            # Check required task fields
            if "name" not in task:
                errors.append(f"Task {i}: 'name' field required")
            else:
                name = task["name"]
                if name in task_names:
                    errors.append(f"Task {i}: Duplicate task name '{name}'")
                task_names.add(name)

            if "description" not in task:
                errors.append(f"Task {i}: 'description' field required")

            # Validate dependencies if present
            if "dependencies" in task:
                if not isinstance(task["dependencies"], list):
                    errors.append(f"Task {i}: 'dependencies' must be a list")

        if errors:
            raise ValidationError("Task validation failed", details={"errors": errors})

    def _validate_dependencies(self, agents: List[Dict[str, Any]], tasks: List[Dict[str, Any]]):
        """Validate task dependencies and agent assignments"""
        errors = []

        agent_names = {agent["name"] for agent in agents if "name" in agent}
        task_names = {task["name"] for task in tasks if "name" in task}

        for task in tasks:
            if "name" not in task:
                continue

            task_name = task["name"]

            # Validate agent assignment
            if "agent" in task:
                agent_name = task["agent"]
                if agent_name not in agent_names:
                    errors.append(f"Task '{task_name}': Unknown agent '{agent_name}'")

            # Validate dependencies
            if "dependencies" in task:
                for dep in task["dependencies"]:
                    if dep not in task_names:
                        errors.append(f"Task '{task_name}': Unknown dependency '{dep}'")

        # Check for circular dependencies
        try:
            self._check_circular_dependencies(tasks)
        except ValidationError as e:
            errors.extend(e.details.get("errors", [str(e)]))

        if errors:
            raise ValidationError("Dependency validation failed", details={"errors": errors})

    def _check_circular_dependencies(self, tasks: List[Dict[str, Any]]):
        """Check for circular dependencies in tasks"""
        # Build dependency graph
        graph = {}
        for task in tasks:
            if "name" in task:
                task_name = task["name"]
                dependencies = task.get("dependencies", [])
                graph[task_name] = dependencies

        # DFS to detect cycles
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for task_name in graph:
            if task_name not in visited:
                if has_cycle(task_name, visited, set()):
                    raise ValidationError(
                        "Circular dependency detected",
                        details={"errors": [f"Circular dependency involving task '{task_name}'"]},
                    )


class ConfigValidator:
    """Validate configuration objects"""

    def __init__(self):
        self.param_validator = ParameterValidator()

    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.

        Args:
            config: Configuration to validate
            schema: Validation schema

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        errors = []

        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in config:
                errors.append(f"Required configuration field '{field}' missing")

        # Validate field types and values
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in config:
                try:
                    self._validate_config_field(config[field], field_schema, field)
                except ValidationError as e:
                    errors.extend(e.details.get("errors", [str(e)]))

        if errors:
            raise ValidationError("Configuration validation failed", details={"errors": errors})

        return True

    def _validate_config_field(self, value: Any, field_schema: Dict[str, Any], field_name: str):
        """Validate individual configuration field"""
        field_type = field_schema.get("type")

        if field_type:
            # Type validation
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }

            expected_type = type_map.get(field_type)
            if expected_type and not isinstance(value, expected_type):
                raise ValidationError(
                    f"Configuration field '{field_name}' must be of type {field_type}",
                    field=field_name,
                    value=value,
                    expected_type=field_type,
                )

        # Additional validations
        if "minimum" in field_schema and isinstance(value, (int, float)):
            if value < field_schema["minimum"]:
                raise ValidationError(
                    f"Configuration field '{field_name}' must be >= {field_schema['minimum']}",
                    field=field_name,
                    value=value,
                )

        if "maximum" in field_schema and isinstance(value, (int, float)):
            if value > field_schema["maximum"]:
                raise ValidationError(
                    f"Configuration field '{field_name}' must be <= {field_schema['maximum']}",
                    field=field_name,
                    value=value,
                )

        if "enum" in field_schema:
            if value not in field_schema["enum"]:
                raise ValidationError(
                    f"Configuration field '{field_name}' must be one of {field_schema['enum']}",
                    field=field_name,
                    value=value,
                )


# Global validator instances
parameter_validator = ParameterValidator()
state_validator = StateValidator()
workflow_validator = WorkflowValidator()
config_validator = ConfigValidator()
