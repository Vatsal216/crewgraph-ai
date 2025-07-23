"""
CrewGraph AI Tool Validator
Comprehensive validation system for tools, ensuring quality and security

Author: Vatsal216
Created: 2025-07-22 12:29:42 UTC
"""

import ast
import inspect
import json
import re
import threading
import time
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from crewai.tools import BaseTool as CrewAIBaseTool

from ..utils.exceptions import ValidationError
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ValidationLevel(Enum):
    """Validation strictness levels"""

    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    NONE = "none"


class ValidationSeverity(Enum):
    """Validation issue severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""

    severity: ValidationSeverity
    category: str
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None
    line_number: Optional[int] = None
    code: Optional[str] = None
    created_at: str = "2025-07-22 12:29:42"
    validator: str = "Vatsal216"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "line_number": self.line_number,
            "code": self.code,
            "created_at": self.created_at,
            "validator": self.validator,
        }


@dataclass
class ValidationResult:
    """Result of tool validation"""

    is_valid: bool
    tool_name: str
    validation_level: ValidationLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_at: str = "2025-07-22 12:29:42"
    validated_by: str = "Vatsal216"

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity level"""
        return [issue for issue in self.issues if issue.severity == severity]

    def has_critical_issues(self) -> bool:
        """Check if there are critical issues"""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = len(self.get_issues_by_severity(severity))

        return {
            "is_valid": self.is_valid,
            "tool_name": self.tool_name,
            "score": self.score,
            "total_issues": len(self.issues),
            "severity_breakdown": severity_counts,
            "validation_level": self.validation_level.value,
            "execution_time": self.execution_time,
            "validated_by": self.validated_by,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "tool_name": self.tool_name,
            "validation_level": self.validation_level.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "score": self.score,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "validated_at": self.validated_at,
            "validated_by": self.validated_by,
            "summary": self.get_summary(),
        }


@dataclass
class ValidationConfig:
    """Configuration for tool validation"""

    validation_level: ValidationLevel = ValidationLevel.MODERATE
    enable_security_checks: bool = True
    enable_performance_checks: bool = True
    enable_code_quality_checks: bool = True
    enable_documentation_checks: bool = True
    enable_signature_validation: bool = True
    enable_runtime_testing: bool = False
    max_execution_time: float = 5.0
    allowed_imports: Set[str] = field(
        default_factory=lambda: {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "math",
            "random",
            "requests",
            "urllib",
            "re",
            "typing",
            "dataclasses",
            "enum",
            "pathlib",
            "collections",
            "itertools",
            "functools",
        }
    )
    forbidden_imports: Set[str] = field(
        default_factory=lambda: {"subprocess", "exec", "eval", "compile", "__import__"}
    )
    forbidden_functions: Set[str] = field(
        default_factory=lambda: {
            "exec",
            "eval",
            "compile",
            "__import__",
            "globals",
            "locals",
            "vars",
            "dir",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
        }
    )
    max_complexity: int = 10
    min_documentation_coverage: float = 0.8
    require_type_hints: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.max_execution_time <= 0:
            self.max_execution_time = 5.0

        if not 0 <= self.min_documentation_coverage <= 1:
            self.min_documentation_coverage = 0.8

        if self.max_complexity < 1:
            self.max_complexity = 10


class ToolValidator:
    """
    Comprehensive tool validation system for CrewGraph AI.

    Validates tools across multiple dimensions:
    - Security: Prevents dangerous operations and code injection
    - Performance: Checks for efficiency and resource usage
    - Code Quality: Analyzes code structure and best practices
    - Documentation: Ensures proper documentation and type hints
    - Functionality: Tests tool execution and error handling
    - Compatibility: Verifies CrewAI integration compatibility

    Features:
    - Multi-level validation (strict, moderate, lenient)
    - Detailed issue reporting with suggestions
    - Performance scoring and metrics
    - Security vulnerability detection
    - Code complexity analysis
    - Runtime testing capabilities

    Created by: Vatsal216
    Date: 2025-07-22 12:29:42 UTC
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize tool validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "critical_issues_found": 0,
            "total_execution_time": 0.0,
            "last_validation_time": None,
        }

        # Thread safety
        self._lock = threading.RLock()

        # Validation rule sets
        self._security_rules = self._initialize_security_rules()
        self._quality_rules = self._initialize_quality_rules()
        self._performance_rules = self._initialize_performance_rules()

        logger.info("ToolValidator initialized")
        logger.info(f"Validation level: {self.config.validation_level.value}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:29:42")

        # Record initialization metrics
        metrics.increment_counter(
            "crewgraph_tool_validators_created_total",
            labels={"validation_level": self.config.validation_level.value, "user": "Vatsal216"},
        )

    def validate_tool(
        self, tool: Union[CrewAIBaseTool, Callable], tool_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a tool comprehensively.

        Args:
            tool: CrewAI BaseTool to validate (CrewAI Tool or callable)
            tool_name: Optional tool name override

        Returns:
            Validation result with issues and score
        """
        start_time = time.time()

        # Determine tool name and function
        if isinstance(tool, CrewAIBaseTool):
            name = tool_name or tool.name
            func = tool.func
        else:
            name = tool_name or getattr(tool, "__name__", "unknown_tool")
            func = tool

        logger.info(f"Validating tool: {name}")

        # Initialize result
        result = ValidationResult(
            is_valid=True,
            tool_name=name,
            validation_level=self.config.validation_level,
            validated_by="Vatsal216",
        )

        try:
            with self._lock:
                # Run validation checks based on configuration
                if self.config.enable_security_checks:
                    self._validate_security(func, result)

                if self.config.enable_signature_validation:
                    self._validate_signature(func, result)

                if self.config.enable_code_quality_checks:
                    self._validate_code_quality(func, result)

                if self.config.enable_documentation_checks:
                    self._validate_documentation(func, result)

                if self.config.enable_performance_checks:
                    self._validate_performance(func, result)

                if self.config.enable_runtime_testing:
                    self._validate_runtime(func, result)

                # Calculate overall score
                result.score = self._calculate_score(result)

                # Determine if valid based on issues and level
                result.is_valid = self._determine_validity(result)

                # Update statistics
                self.validation_stats["total_validations"] += 1
                if result.is_valid:
                    self.validation_stats["passed_validations"] += 1
                else:
                    self.validation_stats["failed_validations"] += 1

                critical_issues = len(result.get_issues_by_severity(ValidationSeverity.CRITICAL))
                self.validation_stats["critical_issues_found"] += critical_issues

                execution_time = time.time() - start_time
                result.execution_time = execution_time
                self.validation_stats["total_execution_time"] += execution_time
                self.validation_stats["last_validation_time"] = time.time()

        except Exception as e:
            logger.error(f"Error during validation of tool {name}: {e}")
            result.is_valid = False
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="validation_error",
                    message=f"Validation failed with error: {str(e)}",
                    validator="Vatsal216",
                )
            )

        # Record metrics
        metrics.record_duration(
            "crewgraph_tool_validation_duration_seconds",
            result.execution_time,
            labels={
                "tool_name": name,
                "validation_level": self.config.validation_level.value,
                "is_valid": str(result.is_valid),
                "user": "Vatsal216",
            },
        )

        metrics.increment_counter(
            "crewgraph_tools_validated_total",
            labels={
                "validation_level": self.config.validation_level.value,
                "is_valid": str(result.is_valid),
                "user": "Vatsal216",
            },
        )

        logger.info(
            f"Validation completed for {name}: {'PASSED' if result.is_valid else 'FAILED'} "
            f"(Score: {result.score:.2f}, Issues: {len(result.issues)})"
        )

        return result

    def _validate_security(self, func: Callable, result: ValidationResult) -> None:
        """Validate security aspects of the tool"""
        try:
            # Get function source code
            source = inspect.getsource(func)

            # Parse AST for security analysis
            tree = ast.parse(source)

            # Check for dangerous imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.config.forbidden_imports:
                            result.issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.CRITICAL,
                                    category="security",
                                    message=f"Forbidden import detected: {alias.name}",
                                    suggestion="Remove dangerous import or use safer alternative",
                                    validator="Vatsal216",
                                )
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.config.forbidden_imports:
                        result.issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.CRITICAL,
                                category="security",
                                message=f"Forbidden module import: {node.module}",
                                suggestion="Use safer alternative module",
                                validator="Vatsal216",
                            )
                        )

                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.config.forbidden_functions:
                            result.issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.HIGH,
                                    category="security",
                                    message=f"Dangerous function call: {node.func.id}",
                                    suggestion="Avoid using potentially unsafe functions",
                                    line_number=getattr(node, "lineno", None),
                                    validator="Vatsal216",
                                )
                            )

            # Check for string concatenation in exec-like contexts
            self._check_code_injection_patterns(source, result)

            # Check for file system operations
            self._check_filesystem_operations(tree, result)

            # Check for network operations
            self._check_network_operations(tree, result)

        except Exception as e:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="security",
                    message=f"Could not perform complete security analysis: {str(e)}",
                    validator="Vatsal216",
                )
            )

    def _validate_signature(self, func: Callable, result: ValidationResult) -> None:
        """Validate function signature"""
        try:
            sig = inspect.signature(func)

            # Check for type hints if required
            if self.config.require_type_hints:
                missing_hints = []

                # Check parameters
                for param_name, param in sig.parameters.items():
                    if param.annotation == inspect.Parameter.empty:
                        missing_hints.append(f"parameter '{param_name}'")

                # Check return type
                if sig.return_annotation == inspect.Signature.empty:
                    missing_hints.append("return type")

                if missing_hints:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.MEDIUM,
                            category="signature",
                            message=f"Missing type hints for: {', '.join(missing_hints)}",
                            suggestion="Add type hints to improve code clarity and tooling support",
                            validator="Vatsal216",
                        )
                    )

            # Check parameter defaults
            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    # Check for mutable defaults
                    if isinstance(param.default, (list, dict, set)):
                        result.issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.HIGH,
                                category="signature",
                                message=f"Mutable default argument in parameter '{param_name}'",
                                suggestion="Use None as default and create mutable object inside function",
                                validator="Vatsal216",
                            )
                        )

            # Check for too many parameters
            param_count = len(sig.parameters)
            if param_count > 10:
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="signature",
                        message=f"Function has too many parameters ({param_count})",
                        suggestion="Consider grouping parameters into a configuration object",
                        validator="Vatsal216",
                    )
                )

        except Exception as e:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category="signature",
                    message=f"Could not analyze function signature: {str(e)}",
                    validator="Vatsal216",
                )
            )

    def _validate_code_quality(self, func: Callable, result: ValidationResult) -> None:
        """Validate code quality aspects"""
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)

            # Calculate cyclomatic complexity
            complexity = self._calculate_complexity(tree)
            if complexity > self.config.max_complexity:
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.HIGH,
                        category="code_quality",
                        message=f"High cyclomatic complexity: {complexity} (max: {self.config.max_complexity})",
                        suggestion="Break down function into smaller, more focused functions",
                        validator="Vatsal216",
                    )
                )

            # Check for code smells
            self._check_code_smells(tree, source, result)

            # Check naming conventions
            self._check_naming_conventions(tree, result)

            # Check for magic numbers
            self._check_magic_numbers(tree, result)

        except Exception as e:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category="code_quality",
                    message=f"Could not perform code quality analysis: {str(e)}",
                    validator="Vatsal216",
                )
            )

    def _validate_documentation(self, func: Callable, result: ValidationResult) -> None:
        """Validate documentation quality"""
        try:
            # Check for docstring
            docstring = inspect.getdoc(func)
            if not docstring:
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="documentation",
                        message="Function lacks docstring",
                        suggestion="Add comprehensive docstring describing purpose, parameters, and return value",
                        validator="Vatsal216",
                    )
                )
                return

            # Check docstring length and quality
            if len(docstring.strip()) < 20:
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.LOW,
                        category="documentation",
                        message="Docstring is too short",
                        suggestion="Provide more detailed description of function purpose and usage",
                        validator="Vatsal216",
                    )
                )

            # Check for parameter documentation
            sig = inspect.signature(func)
            if len(sig.parameters) > 0:
                # Simple check for parameter mentions in docstring
                documented_params = 0
                for param_name in sig.parameters.keys():
                    if param_name in docstring.lower():
                        documented_params += 1

                coverage = documented_params / len(sig.parameters)
                if coverage < self.config.min_documentation_coverage:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.MEDIUM,
                            category="documentation",
                            message=f"Poor parameter documentation coverage: {coverage:.1%}",
                            suggestion="Document all parameters in the docstring",
                            validator="Vatsal216",
                        )
                    )

            # Check for return value documentation
            if "return" not in docstring.lower() and "returns" not in docstring.lower():
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.LOW,
                        category="documentation",
                        message="Return value not documented",
                        suggestion="Document what the function returns",
                        validator="Vatsal216",
                    )
                )

        except Exception as e:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category="documentation",
                    message=f"Could not analyze documentation: {str(e)}",
                    validator="Vatsal216",
                )
            )

    def _validate_performance(self, func: Callable, result: ValidationResult) -> None:
        """Validate performance aspects"""
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)

            # Check for performance anti-patterns
            for node in ast.walk(tree):
                # Check for loops that could be optimized
                if isinstance(node, (ast.For, ast.While)):
                    # Check for nested loops
                    nested_loops = [
                        n
                        for n in ast.walk(node)
                        if isinstance(n, (ast.For, ast.While)) and n != node
                    ]
                    if len(nested_loops) >= 2:
                        result.issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.MEDIUM,
                                category="performance",
                                message="Multiple nested loops detected",
                                suggestion="Consider optimizing nested loops or using more efficient algorithms",
                                line_number=getattr(node, "lineno", None),
                                validator="Vatsal216",
                            )
                        )

                # Check for inefficient string operations
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    if self._is_string_operation(node):
                        result.issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.LOW,
                                category="performance",
                                message="String concatenation in loop (potential performance issue)",
                                suggestion="Use join() or f-strings for better performance",
                                line_number=getattr(node, "lineno", None),
                                validator="Vatsal216",
                            )
                        )

        except Exception as e:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category="performance",
                    message=f"Could not perform performance analysis: {str(e)}",
                    validator="Vatsal216",
                )
            )

    def _validate_runtime(self, func: Callable, result: ValidationResult) -> None:
        """Validate runtime behavior with test execution"""
        try:
            # Only perform runtime testing if explicitly enabled
            if not self.config.enable_runtime_testing:
                return

            logger.debug(f"Performing runtime validation for {func.__name__}")

            # Get function signature for test input generation
            sig = inspect.signature(func)

            # Generate test inputs (simplified)
            test_inputs = self._generate_test_inputs(sig)

            for test_input in test_inputs:
                try:
                    # Execute with timeout
                    start_time = time.time()

                    # Call function with test input
                    if test_input:
                        func(**test_input)
                    else:
                        func()

                    execution_time = time.time() - start_time

                    # Check execution time
                    # Continuing from _validate_runtime method...

                    # Check execution time
                    if execution_time > self.config.max_execution_time:
                        result.issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.HIGH,
                                category="performance",
                                message=f"Function execution too slow: {execution_time:.2f}s (max: {self.config.max_execution_time}s)",
                                suggestion="Optimize function performance or increase timeout",
                                validator="Vatsal216",
                            )
                        )

                except TimeoutError:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="runtime",
                            message="Function execution timed out",
                            suggestion="Optimize function or handle long-running operations properly",
                            validator="Vatsal216",
                        )
                    )

                except Exception as e:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.MEDIUM,
                            category="runtime",
                            message=f"Runtime error during test: {str(e)}",
                            suggestion="Add proper error handling and input validation",
                            validator="Vatsal216",
                        )
                    )

        except Exception as e:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category="runtime",
                    message=f"Could not perform runtime testing: {str(e)}",
                    validator="Vatsal216",
                )
            )

    def _check_code_injection_patterns(self, source: str, result: ValidationResult) -> None:
        """Check for potential code injection patterns"""
        dangerous_patterns = [
            r"exec\s*\(",
            r"eval\s*\(",
            r"compile\s*\(",
            r"__import__\s*\(",
            r'getattr\s*\([^,]+,\s*["\'][^"\']*["\']',
            r'setattr\s*\([^,]+,\s*["\'][^"\']*["\']',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, source, re.IGNORECASE):
                result.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.HIGH,
                        category="security",
                        message=f"Potential code injection pattern detected",
                        details=f"Pattern: {pattern}",
                        suggestion="Use safer alternatives to dynamic code execution",
                        validator="Vatsal216",
                    )
                )

    def _check_filesystem_operations(self, tree: ast.AST, result: ValidationResult) -> None:
        """Check for filesystem operations"""
        filesystem_functions = {"open", "read", "write", "remove", "rmdir", "mkdir"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in filesystem_functions:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.MEDIUM,
                            category="security",
                            message=f"Filesystem operation detected: {node.func.id}",
                            suggestion="Ensure proper input validation and path sanitization",
                            line_number=getattr(node, "lineno", None),
                            validator="Vatsal216",
                        )
                    )

    def _check_network_operations(self, tree: ast.AST, result: ValidationResult) -> None:
        """Check for network operations"""
        network_modules = {"requests", "urllib", "http", "socket"}
        network_functions = {"get", "post", "put", "delete", "request", "urlopen"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for module.function calls
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if (
                            node.func.value.id in network_modules
                            or node.func.attr in network_functions
                        ):
                            result.issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.MEDIUM,
                                    category="security",
                                    message="Network operation detected",
                                    details=f"Operation: {node.func.value.id}.{node.func.attr}",
                                    suggestion="Validate URLs and implement proper error handling",
                                    line_number=getattr(node, "lineno", None),
                                    validator="Vatsal216",
                                )
                            )

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1

        return complexity

    def _check_code_smells(self, tree: ast.AST, source: str, result: ValidationResult) -> None:
        """Check for common code smells"""
        # Check for too many return statements
        return_count = len([node for node in ast.walk(tree) if isinstance(node, ast.Return)])
        if return_count > 5:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="code_quality",
                    message=f"Too many return statements: {return_count}",
                    suggestion="Consider refactoring to reduce return points",
                    validator="Vatsal216",
                )
            )

        # Check for long functions (by line count)
        line_count = len(source.split("\n"))
        if line_count > 50:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="code_quality",
                    message=f"Function is too long: {line_count} lines",
                    suggestion="Break function into smaller, more focused functions",
                    validator="Vatsal216",
                )
            )

        # Check for deeply nested code
        max_nesting = self._calculate_max_nesting(tree)
        if max_nesting > 4:
            result.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="code_quality",
                    message=f"Code is deeply nested: {max_nesting} levels",
                    suggestion="Reduce nesting by extracting functions or using early returns",
                    validator="Vatsal216",
                )
            )

    def _check_naming_conventions(self, tree: ast.AST, result: ValidationResult) -> None:
        """Check naming conventions"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check snake_case for functions
                if not re.match(r"^[a-z_][a-z0-9_]*$", node.name):
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.LOW,
                            category="code_quality",
                            message=f"Function name '{node.name}' doesn't follow snake_case convention",
                            suggestion="Use snake_case for function names",
                            line_number=getattr(node, "lineno", None),
                            validator="Vatsal216",
                        )
                    )

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Check variable naming
                if len(node.id) == 1 and node.id not in ["i", "j", "k", "x", "y", "z"]:
                    result.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.LOW,
                            category="code_quality",
                            message=f"Single-letter variable name: '{node.id}'",
                            suggestion="Use descriptive variable names",
                            line_number=getattr(node, "lineno", None),
                            validator="Vatsal216",
                        )
                    )

    def _check_magic_numbers(self, tree: ast.AST, result: ValidationResult) -> None:
        """Check for magic numbers"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    # Skip common acceptable numbers
                    if node.value not in [0, 1, -1, 2, 10, 100, 1000]:
                        result.issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.LOW,
                                category="code_quality",
                                message=f"Magic number detected: {node.value}",
                                suggestion="Define as named constant for better maintainability",
                                line_number=getattr(node, "lineno", None),
                                validator="Vatsal216",
                            )
                        )

    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting level"""

        def get_nesting(node, current_level=0):
            max_level = current_level

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.Try)):
                    child_max = get_nesting(child, current_level + 1)
                    max_level = max(max_level, child_max)
                else:
                    child_max = get_nesting(child, current_level)
                    max_level = max(max_level, child_max)

            return max_level

        return get_nesting(tree)

    def _is_string_operation(self, node: ast.BinOp) -> bool:
        """Check if binary operation is likely a string concatenation"""
        # This is a simplified check - in practice, you'd want more sophisticated analysis
        return True  # Placeholder - would need type inference for accurate detection

    def _generate_test_inputs(self, sig: inspect.Signature) -> List[Dict[str, Any]]:
        """Generate test inputs for runtime validation"""
        test_inputs = []

        try:
            # Generate basic test case
            basic_input = {}

            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    # Use default value
                    continue

                # Generate test value based on annotation
                if param.annotation != inspect.Parameter.empty:
                    test_value = self._generate_test_value(param.annotation)
                    if test_value is not None:
                        basic_input[param_name] = test_value
                else:
                    # Generate generic test value
                    basic_input[param_name] = "test_value"

            test_inputs.append(basic_input)

            # Limit test cases to avoid long execution times
            return test_inputs[:3]

        except Exception as e:
            logger.debug(f"Could not generate test inputs: {e}")
            return [{}]  # Return empty input for parameter-less functions

    def _generate_test_value(self, annotation: Any) -> Any:
        """Generate test value based on type annotation"""
        try:
            if annotation == str:
                return "test_string"
            elif annotation == int:
                return 42
            elif annotation == float:
                return 3.14
            elif annotation == bool:
                return True
            elif annotation == list:
                return ["test_item"]
            elif annotation == dict:
                return {"test_key": "test_value"}
            else:
                return "test_value"
        except (TypeError, AttributeError, ValueError) as e:
            # Log the error for debugging but return a safe default
            logger = get_logger(__name__)
            logger.debug(f"Failed to generate test value for {annotation}: {e}")
            return "test_value"

    def _calculate_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score"""
        if not result.issues:
            return 100.0

        # Weight penalties by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 25,
            ValidationSeverity.HIGH: 15,
            ValidationSeverity.MEDIUM: 10,
            ValidationSeverity.LOW: 5,
            ValidationSeverity.INFO: 1,
        }

        total_penalty = 0
        for issue in result.issues:
            total_penalty += severity_weights.get(issue.severity, 5)

        # Calculate score (max 100, min 0)
        score = max(0, min(100, 100 - total_penalty))
        return score

    def _determine_validity(self, result: ValidationResult) -> bool:
        """Determine if tool is valid based on issues and validation level"""
        critical_issues = len(result.get_issues_by_severity(ValidationSeverity.CRITICAL))
        high_issues = len(result.get_issues_by_severity(ValidationSeverity.HIGH))

        if self.config.validation_level == ValidationLevel.STRICT:
            return critical_issues == 0 and high_issues == 0
        elif self.config.validation_level == ValidationLevel.MODERATE:
            return critical_issues == 0
        elif self.config.validation_level == ValidationLevel.LENIENT:
            return critical_issues <= 1
        else:  # NONE
            return True

    def _initialize_security_rules(self) -> Dict[str, Any]:
        """Initialize security validation rules"""
        return {
            "forbidden_imports": self.config.forbidden_imports,
            "forbidden_functions": self.config.forbidden_functions,
            "injection_patterns": [r"exec\s*\(", r"eval\s*\(", r"compile\s*\(", r"__import__\s*\("],
        }

    def _initialize_quality_rules(self) -> Dict[str, Any]:
        """Initialize code quality validation rules"""
        return {
            "max_complexity": self.config.max_complexity,
            "max_function_length": 50,
            "max_nesting_level": 4,
            "max_return_statements": 5,
        }

    def _initialize_performance_rules(self) -> Dict[str, Any]:
        """Initialize performance validation rules"""
        return {
            "max_execution_time": self.config.max_execution_time,
            "max_nested_loops": 2,
            "inefficient_patterns": [
                "string_concatenation_in_loop",
                "repeated_expensive_operations",
            ],
        }

    def validate_batch(
        self, tools: List[Union[CrewAIBaseTool, Callable]], tool_names: Optional[List[str]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple tools in batch.

        Args:
            tools: List of tools to validate
            tool_names: Optional list of tool names

        Returns:
            Dictionary mapping tool names to validation results
        """
        logger.info(f"Starting batch validation of {len(tools)} tools")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:35:41")

        results = {}

        for i, tool in enumerate(tools):
            tool_name = None
            if tool_names and i < len(tool_names):
                tool_name = tool_names[i]

            try:
                result = self.validate_tool(tool, tool_name)
                name = result.tool_name
                results[name] = result

                logger.debug(f"Validated tool {i+1}/{len(tools)}: {name}")

            except Exception as e:
                name = tool_name or f"tool_{i}"
                logger.error(f"Failed to validate tool {name}: {e}")

                # Create failure result
                results[name] = ValidationResult(
                    is_valid=False,
                    tool_name=name,
                    validation_level=self.config.validation_level,
                    issues=[
                        ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="validation_error",
                            message=f"Validation failed: {str(e)}",
                            validator="Vatsal216",
                        )
                    ],
                )

        # Record batch metrics
        valid_count = sum(1 for r in results.values() if r.is_valid)
        invalid_count = len(results) - valid_count

        metrics.record_gauge(
            "crewgraph_tool_batch_validation_results",
            valid_count,
            labels={"result": "valid", "user": "Vatsal216"},
        )

        metrics.record_gauge(
            "crewgraph_tool_batch_validation_results",
            invalid_count,
            labels={"result": "invalid", "user": "Vatsal216"},
        )

        logger.info(f"Batch validation completed: {valid_count} valid, {invalid_count} invalid")

        return results

    def get_validation_report(
        self, results: Union[ValidationResult, List[ValidationResult], Dict[str, ValidationResult]]
    ) -> str:
        """
        Generate comprehensive validation report.

        Args:
            results: Validation results to report on

        Returns:
            Formatted validation report
        """
        if isinstance(results, ValidationResult):
            results = [results]
        elif isinstance(results, dict):
            results = list(results.values())

        report = []
        report.append("🔍 CrewGraph AI Tool Validation Report")
        report.append(f"👤 Generated by: Vatsal216")
        report.append(f"⏰ Generated at: 2025-07-22 12:35:41 UTC")
        report.append("=" * 60)

        # Summary statistics
        total_tools = len(results)
        valid_tools = sum(1 for r in results if r.is_valid)
        invalid_tools = total_tools - valid_tools

        total_issues = sum(len(r.issues) for r in results)
        avg_score = sum(r.score for r in results) / total_tools if total_tools > 0 else 0

        report.append(f"\n📊 SUMMARY STATISTICS")
        report.append(f"Total Tools Validated: {total_tools}")
        report.append(f"Valid Tools: {valid_tools} ({valid_tools/total_tools:.1%})")
        report.append(f"Invalid Tools: {invalid_tools} ({invalid_tools/total_tools:.1%})")
        report.append(f"Total Issues Found: {total_issues}")
        report.append(f"Average Score: {avg_score:.1f}/100")

        # Severity breakdown
        severity_counts = {}
        for severity in ValidationSeverity:
            count = sum(len(r.get_issues_by_severity(severity)) for r in results)
            severity_counts[severity.value] = count

        report.append(f"\n🚨 ISSUES BY SEVERITY")
        for severity, count in severity_counts.items():
            if count > 0:
                report.append(f"  {severity.upper()}: {count}")

        # Individual tool results
        report.append(f"\n🔧 INDIVIDUAL TOOL RESULTS")
        report.append("-" * 60)

        for result in sorted(results, key=lambda x: x.score, reverse=True):
            status = "✅ VALID" if result.is_valid else "❌ INVALID"
            report.append(f"\n{result.tool_name}: {status} (Score: {result.score:.1f}/100)")

            if result.issues:
                report.append(f"  Issues ({len(result.issues)}):")

                # Group issues by severity
                for severity in [
                    ValidationSeverity.CRITICAL,
                    ValidationSeverity.HIGH,
                    ValidationSeverity.MEDIUM,
                    ValidationSeverity.LOW,
                ]:
                    severity_issues = result.get_issues_by_severity(severity)
                    if severity_issues:
                        report.append(f"    {severity.value.upper()}:")
                        for issue in severity_issues:
                            report.append(f"      • {issue.message}")
                            if issue.suggestion:
                                report.append(f"        → {issue.suggestion}")

        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        report.append("-" * 30)

        if invalid_tools > 0:
            report.append("• Address critical and high-severity issues first")
            report.append("• Review security-related findings carefully")
            report.append("• Consider refactoring complex functions")

        if total_issues > 0:
            report.append("• Add comprehensive documentation and type hints")
            report.append("• Follow Python naming conventions")
            report.append("• Implement proper error handling")

        if avg_score < 80:
            report.append("• Focus on improving overall code quality")
            report.append("• Consider using automated code quality tools")

        report.append(f"\n📋 VALIDATION CONFIGURATION")
        report.append(f"Validation Level: {self.config.validation_level.value}")
        report.append(f"Security Checks: {self.config.enable_security_checks}")
        report.append(f"Performance Checks: {self.config.enable_performance_checks}")
        report.append(f"Code Quality Checks: {self.config.enable_code_quality_checks}")
        report.append(f"Documentation Checks: {self.config.enable_documentation_checks}")
        report.append(f"Runtime Testing: {self.config.enable_runtime_testing}")

        report.append(f"\n" + "=" * 60)
        report.append(f"🔗 CrewGraph AI Tool Validator v1.0.0")
        report.append(f"👤 Report generated by: Vatsal216")
        report.append(f"⏰ Completed at: 2025-07-22 12:35:41 UTC")

        return "\n".join(report)

    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics"""
        with self._lock:
            stats = self.validation_stats.copy()

            # Calculate additional metrics
            if stats["total_validations"] > 0:
                stats["success_rate"] = stats["passed_validations"] / stats["total_validations"]
                stats["failure_rate"] = stats["failed_validations"] / stats["total_validations"]
                stats["average_execution_time"] = (
                    stats["total_execution_time"] / stats["total_validations"]
                )
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
                stats["average_execution_time"] = 0.0

            stats.update(
                {
                    "validation_level": self.config.validation_level.value,
                    "security_checks_enabled": self.config.enable_security_checks,
                    "performance_checks_enabled": self.config.enable_performance_checks,
                    "runtime_testing_enabled": self.config.enable_runtime_testing,
                    "created_by": "Vatsal216",
                    "created_at": "2025-07-22 12:35:41",
                }
            )

        return stats

    def update_config(self, **kwargs) -> None:
        """Update validator configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} = {value}")

        logger.info("Validator configuration updated by Vatsal216")

    def export_results(
        self, results: Dict[str, ValidationResult], format_type: str = "json"
    ) -> str:
        """Export validation results"""
        if format_type.lower() == "json":
            export_data = {
                "validation_results": {name: result.to_dict() for name, result in results.items()},
                "export_metadata": {
                    "total_tools": len(results),
                    "valid_tools": sum(1 for r in results.values() if r.is_valid),
                    "total_issues": sum(len(r.issues) for r in results.values()),
                    "exported_by": "Vatsal216",
                    "exported_at": "2025-07-22 12:35:41",
                },
            }
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def __repr__(self) -> str:
        return (
            f"ToolValidator(level={self.config.validation_level.value}, "
            f"validations={self.validation_stats['total_validations']})"
        )


def create_tool_validator(validation_level: str = "moderate", **kwargs) -> ToolValidator:
    """
    Factory function to create tool validator.

    Args:
        validation_level: Validation strictness level
        **kwargs: Additional configuration options

    Returns:
        Configured ToolValidator instance
    """
    config = ValidationConfig(validation_level=ValidationLevel(validation_level.lower()), **kwargs)

    logger.info(f"Creating ToolValidator with level: {validation_level}")
    logger.info(f"User: Vatsal216, Time: 2025-07-22 12:35:41")

    return ToolValidator(config)
