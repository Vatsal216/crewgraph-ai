"""
<<<<<<< HEAD
Natural Language Parsers for CrewGraph AI

Parse natural language requirements and descriptions into structured data.

Author: Vatsal216
Created: 2025-07-23 06:25:00 UTC
=======
Workflow Parser for CrewGraph AI NLP Module

This module provides natural language parsing capabilities to convert
descriptions into structured workflow definitions.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
"""

import re
import json
<<<<<<< HEAD
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import threading

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class TaskType(Enum):
    """Types of tasks that can be identified"""
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    DECISION = "decision"
    NOTIFICATION = "notification"
    CALCULATION = "calculation"
    SEARCH = "search"


class AgentRole(Enum):
    """Types of agent roles that can be identified"""
    ANALYST = "analyst"
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    REPORTER = "reporter"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    MONITOR = "monitor"


@dataclass
class ParsedTask:
    """Parsed task information"""
    name: str
    description: str
    task_type: TaskType
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str]
    parameters: Dict[str, Any]
    estimated_complexity: float
    agent_role_required: Optional[AgentRole]
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:25:00"


@dataclass
class ParsedAgent:
    """Parsed agent information"""
    name: str
    role: AgentRole
    description: str
    capabilities: List[str]
    tools: List[str]
    specialization: str
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:25:00"
=======
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger
from ..types import WorkflowId

logger = get_logger(__name__)


class WorkflowType(Enum):
    """Types of workflows that can be parsed."""
    DATA_PROCESSING = "data_processing"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    RESEARCH = "research"
    CONTENT_GENERATION = "content_generation"
    CUSTOMER_SERVICE = "customer_service"
    GENERAL = "general"
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d


@dataclass
class ParsedWorkflow:
<<<<<<< HEAD
    """Parsed workflow information"""
    name: str
    description: str
    objectives: List[str]
    tasks: List[ParsedTask]
    agents: List[ParsedAgent]
    dependencies: List[Tuple[str, str]]
    constraints: List[str]
    success_criteria: List[str]
    estimated_duration: float
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:25:00"


class RequirementsParser:
    """
    Parse natural language requirements into structured workflow components.
    
    Uses pattern matching and keyword analysis to identify tasks, agents,
    and workflow structure from natural language descriptions.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:25:00 UTC
    """
    
    def __init__(self):
        """Initialize requirements parser"""
        self._task_patterns = self._build_task_patterns()
        self._agent_patterns = self._build_agent_patterns()
        self._dependency_patterns = self._build_dependency_patterns()
        self._constraint_patterns = self._build_constraint_patterns()
        
        self._lock = threading.RLock()
        
        logger.info("RequirementsParser initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:25:00")
    
    def parse_requirements(self, requirements_text: str) -> Dict[str, Any]:
        """
        Parse natural language requirements into structured data.
        
        Args:
            requirements_text: Natural language requirements description
            
        Returns:
            Structured requirements data
        """
        with self._lock:
            # Preprocess text
            text = self._preprocess_text(requirements_text)
            
            # Extract components
            objectives = self._extract_objectives(text)
            tasks = self._extract_tasks(text)
            agents = self._extract_agents(text, tasks)
            dependencies = self._extract_dependencies(text, tasks)
            constraints = self._extract_constraints(text)
            success_criteria = self._extract_success_criteria(text)
            
            # Estimate workflow metadata
            workflow_name = self._generate_workflow_name(objectives)
            description = self._generate_description(objectives, tasks)
            estimated_duration = self._estimate_duration(tasks)
            
            parsed_data = {
                "workflow_name": workflow_name,
                "description": description,
                "objectives": objectives,
                "tasks": [task.__dict__ for task in tasks],
                "agents": [agent.__dict__ for agent in agents],
                "dependencies": dependencies,
                "constraints": constraints,
                "success_criteria": success_criteria,
                "estimated_duration": estimated_duration,
                "metadata": {
                    "parser_version": "1.0.0",
                    "created_by": "Vatsal216",
                    "created_at": "2025-07-23 06:25:00"
                }
            }
            
            metrics.record_metric("requirements_parsed_total", 1.0)
            
            logger.info(f"Parsed requirements: {len(tasks)} tasks, {len(agents)} agents")
            
            return parsed_data
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better parsing"""
        # Convert to lowercase for pattern matching
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Join back with consistent formatting
        return '. '.join(sentence.strip() for sentence in sentences if sentence.strip())
    
    def _extract_objectives(self, text: str) -> List[str]:
        """Extract workflow objectives from text"""
        objectives = []
        
        # Objective patterns
        objective_patterns = [
            r'(?:goal|objective|aim|purpose|intent)(?:\s+is\s+to|\s*:)\s*([^.!?]+)',
            r'(?:need|want|require|should)\s+to\s+([^.!?]+)',
            r'(?:will|must|shall)\s+([^.!?]+)',
            r'in order to\s+([^.!?]+)',
            r'to\s+([^.!?]+?)(?:\s+by|\s+using|\s+with|$)'
        ]
        
        for pattern in objective_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                objective = match.strip()
                if len(objective) > 10 and objective not in objectives:
                    objectives.append(objective)
        
        # If no specific objectives found, infer from action words
        if not objectives:
            action_patterns = [
                r'(analyze\s+[^.!?]+)',
                r'(process\s+[^.!?]+)',
                r'(generate\s+[^.!?]+)',
                r'(create\s+[^.!?]+)',
                r'(validate\s+[^.!?]+)'
            ]
            
            for pattern in action_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) > 10:
                        objectives.append(match.strip())
        
        return objectives[:5]  # Limit to 5 main objectives
    
    def _extract_tasks(self, text: str) -> List[ParsedTask]:
        """Extract tasks from text"""
        tasks = []
        task_counter = 1
        
        # Split text into potential task sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check for task indicators
            task_info = self._identify_task_in_sentence(sentence)
            if task_info:
                task_name = f"task_{task_counter}" if not task_info.get('name') else task_info['name']
                
                parsed_task = ParsedTask(
                    name=task_name,
                    description=sentence,
                    task_type=task_info.get('type', TaskType.DATA_PROCESSING),
                    inputs=task_info.get('inputs', []),
                    outputs=task_info.get('outputs', []),
                    dependencies=task_info.get('dependencies', []),
                    parameters=task_info.get('parameters', {}),
                    estimated_complexity=task_info.get('complexity', 1.0),
                    agent_role_required=task_info.get('agent_role')
                )
                
                tasks.append(parsed_task)
                task_counter += 1
        
        return tasks
    
    def _identify_task_in_sentence(self, sentence: str) -> Optional[Dict[str, Any]]:
        """Identify if a sentence describes a task"""
        
        # Task action verbs
        action_verbs = [
            'analyze', 'process', 'generate', 'create', 'validate', 'transform',
            'calculate', 'extract', 'filter', 'sort', 'merge', 'split',
            'send', 'receive', 'notify', 'alert', 'check', 'verify',
            'load', 'save', 'read', 'write', 'update', 'delete'
        ]
        
        # Check if sentence contains action verbs
        has_action = any(verb in sentence.lower() for verb in action_verbs)
        
        if not has_action:
            return None
        
        task_info = {}
        
        # Identify task type
        task_info['type'] = self._classify_task_type(sentence)
        
        # Extract inputs
        task_info['inputs'] = self._extract_inputs(sentence)
        
        # Extract outputs
        task_info['outputs'] = self._extract_outputs(sentence)
        
        # Estimate complexity
        task_info['complexity'] = self._estimate_task_complexity(sentence)
        
        # Determine required agent role
        task_info['agent_role'] = self._determine_agent_role(sentence, task_info['type'])
        
        return task_info
    
    def _classify_task_type(self, sentence: str) -> TaskType:
        """Classify the type of task based on sentence content"""
        sentence_lower = sentence.lower()
        
        # Task type keywords
        type_keywords = {
            TaskType.DATA_PROCESSING: ['process', 'transform', 'clean', 'normalize', 'filter'],
            TaskType.FILE_OPERATION: ['file', 'save', 'load', 'read', 'write', 'upload', 'download'],
            TaskType.API_CALL: ['api', 'call', 'request', 'endpoint', 'service', 'http'],
            TaskType.ANALYSIS: ['analyze', 'examine', 'study', 'investigate', 'review'],
            TaskType.VALIDATION: ['validate', 'verify', 'check', 'confirm', 'ensure'],
            TaskType.TRANSFORMATION: ['convert', 'transform', 'format', 'restructure'],
            TaskType.DECISION: ['decide', 'choose', 'select', 'determine', 'if', 'condition'],
            TaskType.NOTIFICATION: ['notify', 'alert', 'send', 'email', 'message'],
            TaskType.CALCULATION: ['calculate', 'compute', 'sum', 'count', 'aggregate'],
            TaskType.SEARCH: ['search', 'find', 'locate', 'lookup', 'query']
        }
        
        # Score each task type
        type_scores = {}
        for task_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > 0:
                type_scores[task_type] = score
        
        # Return highest scoring type or default
        if type_scores:
            return max(type_scores.keys(), key=lambda k: type_scores[k])
        
        return TaskType.DATA_PROCESSING
    
    def _extract_inputs(self, sentence: str) -> List[str]:
        """Extract input parameters from sentence"""
        inputs = []
        
        # Input patterns
        input_patterns = [
            r'(?:from|using|with|input)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:data|file|information)',
            r'(?:the|this)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        for pattern in input_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if len(match) > 2 and match.lower() not in ['data', 'file', 'information']:
                    inputs.append(match.lower())
        
        return list(set(inputs))  # Remove duplicates
    
    def _extract_outputs(self, sentence: str) -> List[str]:
        """Extract output parameters from sentence"""
        outputs = []
        
        # Output patterns
        output_patterns = [
            r'(?:generate|create|produce|output)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'(?:result|outcome)\s+(?:is|will be)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'(?:to|into)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:format|file)'
        ]
        
        for pattern in output_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if len(match) > 2:
                    outputs.append(match.lower())
        
        return list(set(outputs))  # Remove duplicates
    
    def _estimate_task_complexity(self, sentence: str) -> float:
        """Estimate task complexity based on sentence content"""
        complexity = 1.0
        
        # Complexity indicators
        complexity_keywords = {
            'simple': 0.5, 'easy': 0.5, 'basic': 0.7,
            'complex': 2.0, 'complicated': 2.5, 'advanced': 2.0,
            'multiple': 1.5, 'several': 1.3, 'many': 1.5,
            'large': 1.5, 'big': 1.3, 'huge': 2.0,
            'optimize': 2.0, 'machine learning': 3.0, 'ai': 2.5
        }
        
        sentence_lower = sentence.lower()
        for keyword, multiplier in complexity_keywords.items():
            if keyword in sentence_lower:
                complexity *= multiplier
        
        # Limit complexity range
        return min(max(complexity, 0.5), 5.0)
    
    def _determine_agent_role(self, sentence: str, task_type: TaskType) -> Optional[AgentRole]:
        """Determine required agent role for task"""
        sentence_lower = sentence.lower()
        
        # Role keywords
        role_keywords = {
            AgentRole.ANALYST: ['analyze', 'examine', 'study', 'investigate'],
            AgentRole.PROCESSOR: ['process', 'transform', 'convert', 'clean'],
            AgentRole.VALIDATOR: ['validate', 'verify', 'check', 'confirm'],
            AgentRole.COORDINATOR: ['coordinate', 'manage', 'orchestrate', 'organize'],
            AgentRole.REPORTER: ['report', 'notify', 'inform', 'communicate'],
            AgentRole.RESEARCHER: ['research', 'find', 'search', 'discover'],
            AgentRole.EXECUTOR: ['execute', 'run', 'perform', 'carry out'],
            AgentRole.MONITOR: ['monitor', 'watch', 'track', 'observe']
        }
        
        # Score roles based on keywords
        role_scores = {}
        for role, keywords in role_keywords.items():
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > 0:
                role_scores[role] = score
        
        # Return highest scoring role
        if role_scores:
            return max(role_scores.keys(), key=lambda k: role_scores[k])
        
        # Default based on task type
        task_role_mapping = {
            TaskType.ANALYSIS: AgentRole.ANALYST,
            TaskType.DATA_PROCESSING: AgentRole.PROCESSOR,
            TaskType.VALIDATION: AgentRole.VALIDATOR,
            TaskType.NOTIFICATION: AgentRole.REPORTER,
            TaskType.SEARCH: AgentRole.RESEARCHER
        }
        
        return task_role_mapping.get(task_type, AgentRole.EXECUTOR)
    
    def _extract_agents(self, text: str, tasks: List[ParsedTask]) -> List[ParsedAgent]:
        """Extract required agents based on tasks"""
        agents = []
        
        # Collect required roles from tasks
        required_roles = set()
        for task in tasks:
            if task.agent_role_required:
                required_roles.add(task.agent_role_required)
        
        # Create agents for each required role
        agent_counter = 1
        for role in required_roles:
            agent_name = f"{role.value}_agent"
            
            # Determine capabilities based on role
            capabilities = self._get_role_capabilities(role)
            
            # Determine tools based on tasks
            tools = self._get_role_tools(role, tasks)
            
            # Generate description
            description = f"Agent specialized in {role.value} tasks"
            
            agent = ParsedAgent(
                name=agent_name,
                role=role,
                description=description,
                capabilities=capabilities,
                tools=tools,
                specialization=role.value
            )
            
            agents.append(agent)
            agent_counter += 1
        
        return agents
    
    def _get_role_capabilities(self, role: AgentRole) -> List[str]:
        """Get capabilities for an agent role"""
        capabilities_map = {
            AgentRole.ANALYST: ['data analysis', 'pattern recognition', 'statistical analysis', 'reporting'],
            AgentRole.PROCESSOR: ['data processing', 'transformation', 'cleaning', 'formatting'],
            AgentRole.VALIDATOR: ['data validation', 'quality checks', 'compliance verification'],
            AgentRole.COORDINATOR: ['task coordination', 'workflow management', 'resource allocation'],
            AgentRole.REPORTER: ['report generation', 'communication', 'notification', 'alerting'],
            AgentRole.RESEARCHER: ['information retrieval', 'search', 'data collection', 'investigation'],
            AgentRole.EXECUTOR: ['task execution', 'process automation', 'system integration'],
            AgentRole.MONITOR: ['system monitoring', 'performance tracking', 'alerting', 'logging']
        }
        
        return capabilities_map.get(role, ['general task execution'])
    
    def _get_role_tools(self, role: AgentRole, tasks: List[ParsedTask]) -> List[str]:
        """Get tools needed for an agent role based on tasks"""
        tools = []
        
        # Base tools for each role
        base_tools = {
            AgentRole.ANALYST: ['data_analyzer', 'statistics_calculator', 'report_generator'],
            AgentRole.PROCESSOR: ['data_processor', 'file_handler', 'transformer'],
            AgentRole.VALIDATOR: ['validator', 'quality_checker', 'compliance_checker'],
            AgentRole.COORDINATOR: ['task_manager', 'workflow_controller', 'resource_manager'],
            AgentRole.REPORTER: ['report_generator', 'notifier', 'communicator'],
            AgentRole.RESEARCHER: ['search_engine', 'data_collector', 'web_scraper'],
            AgentRole.EXECUTOR: ['process_executor', 'system_integrator', 'automation_tools'],
            AgentRole.MONITOR: ['system_monitor', 'performance_tracker', 'alerting_system']
        }
        
        tools.extend(base_tools.get(role, ['basic_tools']))
        
        # Add task-specific tools
        for task in tasks:
            if task.agent_role_required == role:
                if task.task_type == TaskType.FILE_OPERATION:
                    tools.append('file_handler')
                elif task.task_type == TaskType.API_CALL:
                    tools.append('api_client')
                elif task.task_type == TaskType.CALCULATION:
                    tools.append('calculator')
        
        return list(set(tools))  # Remove duplicates
    
    def _extract_dependencies(self, text: str, tasks: List[ParsedTask]) -> List[Tuple[str, str]]:
        """Extract task dependencies from text"""
        dependencies = []
        
        # Simple dependency patterns
        dependency_patterns = [
            r'(?:after|once|when)\s+([^,]+),?\s+(?:then\s+)?([^.!?]+)',
            r'([^.!?]+)\s+(?:before|prior to)\s+([^.!?]+)',
            r'([^.!?]+)\s+(?:requires|needs|depends on)\s+([^.!?]+)'
        ]
        
        task_names = [task.name for task in tasks]
        
        for pattern in dependency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for first, second in matches:
                # Try to map to actual task names
                first_task = self._find_matching_task(first.strip(), task_names)
                second_task = self._find_matching_task(second.strip(), task_names)
                
                if first_task and second_task and first_task != second_task:
                    dependencies.append((first_task, second_task))
        
        return dependencies
    
    def _find_matching_task(self, text: str, task_names: List[str]) -> Optional[str]:
        """Find the best matching task name for a text fragment"""
        text_lower = text.lower()
        
        # Direct name match
        for name in task_names:
            if name.lower() in text_lower or text_lower in name.lower():
                return name
        
        # Keyword match
        for name in task_names:
            name_words = name.lower().split('_')
            if any(word in text_lower for word in name_words if len(word) > 3):
                return name
        
        return None
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract workflow constraints from text"""
        constraints = []
        
        # Constraint patterns
        constraint_patterns = [
            r'(?:must|should|cannot|must not)\s+([^.!?]+)',
            r'(?:within|before|after)\s+(\d+\s+\w+)',
            r'(?:budget|cost|limit)\s+(?:is|of|under)\s+([^.!?]+)',
            r'(?:requires|needs)\s+([^.!?]+)'
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                constraint = match.strip()
                if len(constraint) > 5:
                    constraints.append(constraint)
        
        return constraints
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria from text"""
        criteria = []
        
        # Success criteria patterns
        criteria_patterns = [
            r'(?:success|successful|complete)\s+(?:when|if)\s+([^.!?]+)',
            r'(?:goal|objective)\s+(?:is|achieved|met)\s+([^.!?]+)',
            r'(?:should|must)\s+(?:result in|produce|generate)\s+([^.!?]+)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                criterion = match.strip()
                if len(criterion) > 5:
                    criteria.append(criterion)
        
        return criteria
    
    def _generate_workflow_name(self, objectives: List[str]) -> str:
        """Generate a workflow name based on objectives"""
        if not objectives:
            return "generated_workflow"
        
        # Extract key words from first objective
        first_objective = objectives[0].lower()
        key_words = re.findall(r'\b\w{4,}\b', first_objective)
        
        if key_words:
            # Take first 2-3 significant words
            name_words = key_words[:3]
            return '_'.join(name_words) + '_workflow'
        
        return "workflow_from_requirements"
    
    def _generate_description(self, objectives: List[str], tasks: List[ParsedTask]) -> str:
        """Generate workflow description"""
        if objectives:
            base_description = f"Workflow to {objectives[0]}"
        else:
            base_description = "Generated workflow"
        
        if len(tasks) > 1:
            base_description += f" involving {len(tasks)} tasks"
        
        return base_description
    
    def _estimate_duration(self, tasks: List[ParsedTask]) -> float:
        """Estimate workflow duration in minutes"""
        if not tasks:
            return 30.0
        
        # Base time per task type
        base_times = {
            TaskType.DATA_PROCESSING: 15.0,
            TaskType.FILE_OPERATION: 5.0,
            TaskType.API_CALL: 10.0,
            TaskType.ANALYSIS: 20.0,
            TaskType.VALIDATION: 8.0,
            TaskType.TRANSFORMATION: 12.0,
            TaskType.DECISION: 5.0,
            TaskType.NOTIFICATION: 3.0,
            TaskType.CALCULATION: 10.0,
            TaskType.SEARCH: 15.0
        }
        
        total_time = 0.0
        for task in tasks:
            base_time = base_times.get(task.task_type, 10.0)
            task_time = base_time * task.estimated_complexity
            total_time += task_time
        
        # Assume some parallelization reduces total time
        parallelization_factor = 0.7 if len(tasks) > 3 else 0.9
        
        return total_time * parallelization_factor
    
    def _build_task_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for task identification"""
        return {
            'action_verbs': [
                'analyze', 'process', 'generate', 'create', 'validate',
                'transform', 'calculate', 'extract', 'filter', 'sort'
            ],
            'data_objects': [
                'data', 'file', 'document', 'record', 'information',
                'report', 'dataset', 'table', 'list', 'collection'
            ]
        }
    
    def _build_agent_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for agent identification"""
        return {
            'role_indicators': [
                'agent', 'service', 'component', 'module', 'processor',
                'analyzer', 'validator', 'generator', 'handler'
            ]
        }
    
    def _build_dependency_patterns(self) -> List[str]:
        """Build patterns for dependency identification"""
        return [
            r'(?:after|once|when)\s+(.+?)\s+(?:then|,)\s+(.+)',
            r'(.+?)\s+(?:before|prior to)\s+(.+)',
            r'(.+?)\s+(?:requires|needs|depends on)\s+(.+)'
        ]
    
    def _build_constraint_patterns(self) -> List[str]:
        """Build patterns for constraint identification"""
        return [
            r'(?:must|should|cannot|must not)\s+(.+)',
            r'(?:within|before|after)\s+(\d+\s+\w+)',
            r'(?:budget|cost|limit)\s+(?:is|of|under)\s+(.+)'
        ]
=======
    """Represents a workflow parsed from natural language."""
    name: str
    description: str
    workflow_type: WorkflowType
    tasks: List[Dict[str, Any]]
    dependencies: List[Dict[str, str]]
    estimated_duration: int  # in minutes
    confidence_score: float
    metadata: Dict[str, Any]
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d


class WorkflowParser:
    """
<<<<<<< HEAD
    Parse workflow descriptions and convert them to structured formats.
    
    Specialized parser for workflow-specific language and patterns.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:25:00 UTC
    """
    
    def __init__(self):
        """Initialize workflow parser"""
        self.requirements_parser = RequirementsParser()
        
        logger.info("WorkflowParser initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:25:00")
    
    def parse_workflow_description(self, description: str) -> ParsedWorkflow:
        """
        Parse a workflow description into structured components.
        
        Args:
            description: Natural language workflow description
=======
    Parses natural language descriptions into structured workflow definitions.
    
    Uses pattern matching and keyword analysis to understand workflow requirements
    and convert them into executable CrewGraph AI workflows.
    """
    
    def __init__(self):
        """Initialize the workflow parser."""
        self.task_patterns = self._initialize_task_patterns()
        self.dependency_patterns = self._initialize_dependency_patterns()
        self.workflow_type_keywords = self._initialize_type_keywords()
        
        logger.info("WorkflowParser initialized with pattern matching")
    
    def parse_description(self, description: str, context: Optional[Dict[str, Any]] = None) -> ParsedWorkflow:
        """
        Parse a natural language workflow description.
        
        Args:
            description: Natural language description of the workflow
            context: Additional context for parsing (user preferences, domain, etc.)
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
            
        Returns:
            Parsed workflow structure
        """
<<<<<<< HEAD
        # Use requirements parser as base
        parsed_data = self.requirements_parser.parse_requirements(description)
        
        # Convert to ParsedWorkflow
        tasks = [ParsedTask(**task_dict) for task_dict in parsed_data['tasks']]
        agents = [ParsedAgent(**agent_dict) for agent_dict in parsed_data['agents']]
        
        workflow = ParsedWorkflow(
            name=parsed_data['workflow_name'],
            description=parsed_data['description'],
            objectives=parsed_data['objectives'],
            tasks=tasks,
            agents=agents,
            dependencies=parsed_data['dependencies'],
            constraints=parsed_data['constraints'],
            success_criteria=parsed_data['success_criteria'],
            estimated_duration=parsed_data['estimated_duration']
        )
        
        logger.info(f"Parsed workflow: {workflow.name}")
        
        return workflow
    
    def validate_workflow_structure(self, workflow: ParsedWorkflow) -> Dict[str, Any]:
        """
        Validate the parsed workflow structure.
        
        Args:
            workflow: Parsed workflow to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # Check for required components
        if not workflow.tasks:
            validation_results['errors'].append("Workflow must have at least one task")
            validation_results['is_valid'] = False
        
        if not workflow.agents:
            validation_results['warnings'].append("No agents specified - may need manual assignment")
        
        if not workflow.objectives:
            validation_results['warnings'].append("No clear objectives identified")
        
        # Check task dependencies
        task_names = [task.name for task in workflow.tasks]
        for dep_from, dep_to in workflow.dependencies:
            if dep_from not in task_names:
                validation_results['errors'].append(f"Dependency references unknown task: {dep_from}")
                validation_results['is_valid'] = False
            
            if dep_to not in task_names:
                validation_results['errors'].append(f"Dependency references unknown task: {dep_to}")
                validation_results['is_valid'] = False
        
        # Check for circular dependencies
        if self._has_circular_dependencies(workflow.dependencies):
            validation_results['errors'].append("Circular dependencies detected")
            validation_results['is_valid'] = False
        
        # Generate suggestions
        if len(workflow.tasks) > 10:
            validation_results['suggestions'].append("Consider breaking large workflow into sub-workflows")
        
        if workflow.estimated_duration > 480:  # 8 hours
            validation_results['suggestions'].append("Long-running workflow - consider adding checkpoints")
        
        return validation_results
    
    def _has_circular_dependencies(self, dependencies: List[Tuple[str, str]]) -> bool:
        """Check for circular dependencies in task list"""
        if not dependencies:
            return False
        
        # Build adjacency list
        graph = {}
        for from_task, to_task in dependencies:
            if from_task not in graph:
                graph[from_task] = []
            graph[from_task].append(to_task)
        
        # DFS to detect cycles
=======
        # Clean and preprocess the description
        cleaned_description = self._preprocess_description(description)
        
        # Extract workflow metadata
        workflow_name = self._extract_workflow_name(cleaned_description)
        workflow_type = self._classify_workflow_type(cleaned_description)
        
        # Extract tasks
        tasks = self._extract_tasks(cleaned_description)
        
        # Identify dependencies
        dependencies = self._extract_dependencies(cleaned_description, tasks)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(tasks)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(cleaned_description, tasks)
        
        # Extract additional metadata
        metadata = self._extract_metadata(cleaned_description, context)
        
        parsed_workflow = ParsedWorkflow(
            name=workflow_name,
            description=cleaned_description,
            workflow_type=workflow_type,
            tasks=tasks,
            dependencies=dependencies,
            estimated_duration=estimated_duration,
            confidence_score=confidence_score,
            metadata=metadata
        )
        
        logger.info(f"Parsed workflow '{workflow_name}' with {len(tasks)} tasks "
                   f"(confidence: {confidence_score:.2f})")
        
        return parsed_workflow
    
    def parse_conversational_input(self, 
                                 conversation_history: List[str],
                                 current_input: str) -> ParsedWorkflow:
        """
        Parse workflow from conversational input, considering context.
        
        Args:
            conversation_history: Previous conversation turns
            current_input: Current user input
            
        Returns:
            Parsed workflow incorporating conversational context
        """
        # Combine conversation history for context
        full_context = " ".join(conversation_history) + " " + current_input
        
        # Extract refinements from conversation
        refinements = self._extract_refinements(conversation_history)
        
        # Parse with enhanced context
        parsed = self.parse_description(current_input, {"refinements": refinements})
        
        # Apply conversational refinements
        if refinements:
            parsed = self._apply_refinements(parsed, refinements)
        
        logger.info(f"Parsed conversational workflow with {len(conversation_history)} context turns")
        return parsed
    
    def validate_parsed_workflow(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """
        Validate a parsed workflow for completeness and correctness.
        
        Args:
            parsed_workflow: Workflow to validate
            
        Returns:
            Validation result with issues and suggestions
        """
        issues = []
        suggestions = []
        
        # Check for minimum requirements
        if not parsed_workflow.tasks:
            issues.append("No tasks identified in workflow")
            suggestions.append("Please provide more specific task descriptions")
        
        if len(parsed_workflow.tasks) == 1:
            suggestions.append("Consider breaking down the task into smaller steps")
        
        # Check task completeness
        incomplete_tasks = [task for task in parsed_workflow.tasks 
                          if not task.get("description") or len(task.get("description", "")) < 10]
        
        if incomplete_tasks:
            issues.append(f"{len(incomplete_tasks)} tasks have insufficient descriptions")
            suggestions.append("Provide more detailed descriptions for better execution")
        
        # Check for dependency cycles
        if self._has_dependency_cycles(parsed_workflow.dependencies):
            issues.append("Circular dependencies detected")
            suggestions.append("Review task order to eliminate cycles")
        
        # Check confidence score
        if parsed_workflow.confidence_score < 0.6:
            issues.append("Low parsing confidence - workflow may be ambiguous")
            suggestions.append("Provide more specific details about tasks and their order")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "completeness_score": self._calculate_completeness(parsed_workflow),
            "validation_timestamp": "2025-07-23T06:03:54Z"
        }
    
    def enhance_workflow_with_suggestions(self, 
                                        parsed_workflow: ParsedWorkflow) -> ParsedWorkflow:
        """
        Enhance a parsed workflow with intelligent suggestions.
        
        Args:
            parsed_workflow: Original parsed workflow
            
        Returns:
            Enhanced workflow with additional tasks and optimizations
        """
        enhanced_tasks = parsed_workflow.tasks.copy()
        enhanced_dependencies = parsed_workflow.dependencies.copy()
        
        # Add missing common tasks based on workflow type
        suggested_tasks = self._suggest_missing_tasks(parsed_workflow)
        enhanced_tasks.extend(suggested_tasks)
        
        # Add error handling tasks
        if not any("error" in task.get("description", "").lower() for task in enhanced_tasks):
            enhanced_tasks.append({
                "id": f"error_handling_{len(enhanced_tasks)}",
                "description": "Handle errors and implement retry logic",
                "type": "error_handling",
                "suggested": True
            })
        
        # Add logging and monitoring
        if not any("log" in task.get("description", "").lower() for task in enhanced_tasks):
            enhanced_tasks.append({
                "id": f"logging_{len(enhanced_tasks)}",
                "description": "Log workflow execution and metrics",
                "type": "logging",
                "suggested": True
            })
        
        # Update dependencies for suggested tasks
        enhanced_dependencies.extend(self._generate_suggested_dependencies(enhanced_tasks))
        
        enhanced_workflow = ParsedWorkflow(
            name=parsed_workflow.name,
            description=parsed_workflow.description,
            workflow_type=parsed_workflow.workflow_type,
            tasks=enhanced_tasks,
            dependencies=enhanced_dependencies,
            estimated_duration=parsed_workflow.estimated_duration + 5,  # Add time for enhancements
            confidence_score=min(0.95, parsed_workflow.confidence_score + 0.1),
            metadata={**parsed_workflow.metadata, "enhanced": True}
        )
        
        logger.info(f"Enhanced workflow with {len(suggested_tasks)} suggested tasks")
        return enhanced_workflow
    
    def _preprocess_description(self, description: str) -> str:
        """Clean and preprocess the input description."""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', description.strip())
        
        # Convert to lowercase for pattern matching
        cleaned = cleaned.lower()
        
        # Remove common filler words that don't add value
        filler_words = ['um', 'uh', 'you know', 'like', 'basically', 'actually']
        for filler in filler_words:
            cleaned = cleaned.replace(filler, '')
        
        return cleaned
    
    def _extract_workflow_name(self, description: str) -> str:
        """Extract a suitable name for the workflow."""
        # Look for explicit name mentions
        name_patterns = [
            r'workflow (?:called|named) "([^"]+)"',
            r'process (?:called|named) "([^"]+)"',
            r'create a "([^"]+)" workflow',
            r'build a "([^"]+)" process'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, description)
            if match:
                return match.group(1).title()
        
        # Generate name from content
        if "data" in description and "process" in description:
            return "Data Processing Workflow"
        elif "analysis" in description:
            return "Analysis Workflow"
        elif "customer" in description and "service" in description:
            return "Customer Service Workflow"
        elif "content" in description and "generat" in description:
            return "Content Generation Workflow"
        else:
            # Extract first few meaningful words
            words = description.split()[:3]
            meaningful_words = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with']]
            if meaningful_words:
                return " ".join(meaningful_words).title() + " Workflow"
        
        return "Custom Workflow"
    
    def _classify_workflow_type(self, description: str) -> WorkflowType:
        """Classify the type of workflow based on keywords."""
        type_scores = {}
        
        for workflow_type, keywords in self.workflow_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description)
            if score > 0:
                type_scores[workflow_type] = score
        
        if type_scores:
            best_type = max(type_scores.keys(), key=lambda x: type_scores[x])
            return WorkflowType(best_type)
        
        return WorkflowType.GENERAL
    
    def _extract_tasks(self, description: str) -> List[Dict[str, Any]]:
        """Extract individual tasks from the description."""
        tasks = []
        
        # Split description into sentences
        sentences = re.split(r'[.!?]', description)
        
        # Look for task indicators
        task_indicators = [
            r'(?:first|then|next|after that|finally),?\s*(.+)',
            r'step \d+:?\s*(.+)',
            r'(?:need to|should|must|will)\s+(.+)',
            r'(?:please|can you)\s+(.+)',
            r'(?:i want to|i need to)\s+(.+)'
        ]
        
        task_id = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:  # Skip very short sentences
                continue
            
            # Check if sentence contains task indicators
            is_task = False
            task_description = sentence
            
            for pattern in task_indicators:
                match = re.search(pattern, sentence)
                if match:
                    task_description = match.group(1).strip()
                    is_task = True
                    break
            
            # Also consider sentences with action verbs as tasks
            action_verbs = ['analyze', 'process', 'create', 'generate', 'send', 'collect', 'transform', 'validate']
            if not is_task and any(verb in sentence for verb in action_verbs):
                is_task = True
            
            if is_task and len(task_description) > 5:
                task_type = self._classify_task_type(task_description)
                
                tasks.append({
                    "id": f"task_{task_id}",
                    "description": task_description,
                    "type": task_type,
                    "estimated_duration": self._estimate_task_duration(task_description),
                    "source_sentence": sentence
                })
                task_id += 1
        
        # If no tasks found through patterns, create a default task
        if not tasks:
            tasks.append({
                "id": "task_0",
                "description": description[:100] + "..." if len(description) > 100 else description,
                "type": "general",
                "estimated_duration": 10
            })
        
        return tasks
    
    def _extract_dependencies(self, description: str, tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract task dependencies from the description."""
        dependencies = []
        
        # Look for explicit dependency indicators
        dependency_patterns = [
            r'(?:after|once|when)\s+(.+?),?\s+(?:then|next)\s+(.+)',
            r'(?:first|initially)\s+(.+?),?\s+(?:then|followed by)\s+(.+)',
            r'(.+?)\s+(?:before|prior to)\s+(.+)'
        ]
        
        for pattern in dependency_patterns:
            matches = re.finditer(pattern, description)
            for match in matches:
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                # Find matching tasks
                source_task = self._find_matching_task(source_text, tasks)
                target_task = self._find_matching_task(target_text, tasks)
                
                if source_task and target_task and source_task != target_task:
                    dependencies.append({
                        "source": source_task["id"],
                        "target": target_task["id"]
                    })
        
        # Create sequential dependencies if no explicit dependencies found
        if not dependencies and len(tasks) > 1:
            for i in range(len(tasks) - 1):
                dependencies.append({
                    "source": tasks[i]["id"],
                    "target": tasks[i + 1]["id"]
                })
        
        return dependencies
    
    def _classify_task_type(self, task_description: str) -> str:
        """Classify the type of a task based on its description."""
        task_type_keywords = {
            "data_processing": ["process", "transform", "clean", "filter", "convert"],
            "analysis": ["analyze", "examine", "evaluate", "assess", "review"],
            "io": ["read", "write", "save", "load", "fetch", "download", "upload"],
            "communication": ["send", "notify", "email", "message", "alert"],
            "validation": ["validate", "verify", "check", "test", "confirm"],
            "ml": ["predict", "model", "train", "classify", "recommend"]
        }
        
        for task_type, keywords in task_type_keywords.items():
            if any(keyword in task_description.lower() for keyword in keywords):
                return task_type
        
        return "general"
    
    def _estimate_task_duration(self, task_description: str) -> int:
        """Estimate task duration in minutes based on description."""
        # Simple heuristic based on task complexity indicators
        if any(keyword in task_description.lower() for keyword in ["analyze", "process", "transform"]):
            return 15
        elif any(keyword in task_description.lower() for keyword in ["send", "notify", "save"]):
            return 2
        elif any(keyword in task_description.lower() for keyword in ["train", "model", "ml"]):
            return 60
        else:
            return 5
    
    def _estimate_duration(self, tasks: List[Dict[str, Any]]) -> int:
        """Estimate total workflow duration."""
        if not tasks:
            return 10
        
        total_duration = sum(task.get("estimated_duration", 5) for task in tasks)
        
        # Add overhead for coordination and setup
        overhead = len(tasks) * 2
        
        return total_duration + overhead
    
    def _calculate_confidence(self, description: str, tasks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the parsing."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for clear task indicators
        task_indicators = ["first", "then", "next", "finally", "step"]
        indicator_count = sum(1 for indicator in task_indicators if indicator in description)
        confidence += min(0.3, indicator_count * 0.1)
        
        # Increase confidence for action verbs
        action_verbs = ["analyze", "process", "create", "send", "validate"]
        action_count = sum(1 for verb in action_verbs if verb in description)
        confidence += min(0.2, action_count * 0.05)
        
        # Decrease confidence for very short or very long descriptions
        word_count = len(description.split())
        if word_count < 10:
            confidence -= 0.2
        elif word_count > 200:
            confidence -= 0.1
        
        # Increase confidence for more tasks (suggests clearer structure)
        if len(tasks) > 1:
            confidence += min(0.2, len(tasks) * 0.05)
        
        return max(0.1, min(0.95, confidence))
    
    def _extract_metadata(self, description: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract additional metadata from description and context."""
        metadata = {
            "original_description": description,
            "word_count": len(description.split()),
            "parsing_timestamp": "2025-07-23T06:03:54Z",
            "parser_version": "1.0.0"
        }
        
        # Extract priority indicators
        if any(keyword in description for keyword in ["urgent", "asap", "immediately", "critical"]):
            metadata["priority"] = "high"
        elif any(keyword in description for keyword in ["low priority", "when possible", "eventually"]):
            metadata["priority"] = "low"
        else:
            metadata["priority"] = "normal"
        
        # Extract resource hints
        if any(keyword in description for keyword in ["large", "big", "massive", "huge"]):
            metadata["resource_intensive"] = True
        
        # Add context metadata
        if context:
            metadata.update(context)
        
        return metadata
    
    def _initialize_task_patterns(self) -> List[str]:
        """Initialize patterns for task extraction."""
        return [
            r'(?:step \d+:?\s*)(.+)',
            r'(?:first|then|next|after|finally),?\s*(.+)',
            r'(?:need to|should|must|will)\s+(.+)',
            r'(?:please|can you)\s+(.+)'
        ]
    
    def _initialize_dependency_patterns(self) -> List[str]:
        """Initialize patterns for dependency extraction."""
        return [
            r'(?:after|once|when)\s+(.+?),?\s+(?:then|next)\s+(.+)',
            r'(?:before|prior to)\s+(.+?),?\s+(?:we need to|should)\s+(.+)',
            r'(.+?)\s+(?:depends on|requires)\s+(.+)'
        ]
    
    def _initialize_type_keywords(self) -> Dict[str, List[str]]:
        """Initialize keywords for workflow type classification."""
        return {
            "data_processing": ["data", "process", "transform", "etl", "pipeline"],
            "analysis": ["analyze", "analysis", "report", "insights", "metrics"],
            "automation": ["automate", "schedule", "trigger", "workflow", "process"],
            "research": ["research", "investigate", "study", "gather", "information"],
            "content_generation": ["generate", "create", "write", "content", "document"],
            "customer_service": ["customer", "support", "ticket", "service", "help"]
        }
    
    def _find_matching_task(self, text: str, tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find a task that best matches the given text."""
        for task in tasks:
            task_desc = task.get("description", "").lower()
            if any(word in task_desc for word in text.lower().split() if len(word) > 3):
                return task
        return None
    
    def _extract_refinements(self, conversation_history: List[str]) -> Dict[str, Any]:
        """Extract refinements from conversation history."""
        refinements = {}
        
        # Look for modifications in conversation
        for turn in conversation_history:
            if "change" in turn.lower() or "modify" in turn.lower():
                refinements["has_modifications"] = True
            if "add" in turn.lower():
                refinements["has_additions"] = True
            if "remove" in turn.lower() or "delete" in turn.lower():
                refinements["has_removals"] = True
        
        return refinements
    
    def _apply_refinements(self, parsed: ParsedWorkflow, refinements: Dict[str, Any]) -> ParsedWorkflow:
        """Apply conversational refinements to parsed workflow."""
        # For now, just mark that refinements were applied
        parsed.metadata["refinements_applied"] = refinements
        parsed.confidence_score = min(0.95, parsed.confidence_score + 0.05)
        
        return parsed
    
    def _has_dependency_cycles(self, dependencies: List[Dict[str, str]]) -> bool:
        """Check if dependencies contain cycles."""
        # Simple cycle detection using DFS
        from collections import defaultdict
        
        graph = defaultdict(list)
        for dep in dependencies:
            graph[dep["source"]].append(dep["target"])
        
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
<<<<<<< HEAD
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
=======
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
                    return True
            
            rec_stack.remove(node)
            return False
        
<<<<<<< HEAD
        # Check all nodes
=======
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
<<<<<<< HEAD
        return False
=======
        return False
    
    def _calculate_completeness(self, parsed_workflow: ParsedWorkflow) -> float:
        """Calculate completeness score for validation."""
        score = 0.0
        
        # Check workflow has name
        if parsed_workflow.name and parsed_workflow.name != "Custom Workflow":
            score += 0.2
        
        # Check has tasks
        if parsed_workflow.tasks:
            score += 0.3
        
        # Check task descriptions
        complete_tasks = [t for t in parsed_workflow.tasks 
                         if t.get("description") and len(t.get("description", "")) >= 10]
        if complete_tasks:
            score += 0.3 * (len(complete_tasks) / len(parsed_workflow.tasks))
        
        # Check has dependencies (if more than one task)
        if len(parsed_workflow.tasks) > 1 and parsed_workflow.dependencies:
            score += 0.2
        
        return min(1.0, score)
    
    def _suggest_missing_tasks(self, parsed_workflow: ParsedWorkflow) -> List[Dict[str, Any]]:
        """Suggest missing tasks based on workflow type."""
        suggested = []
        
        if parsed_workflow.workflow_type == WorkflowType.DATA_PROCESSING:
            # Check for common data processing tasks
            has_validation = any("valid" in task.get("description", "").lower() 
                               for task in parsed_workflow.tasks)
            if not has_validation:
                suggested.append({
                    "id": f"validation_{len(parsed_workflow.tasks)}",
                    "description": "Validate input data quality and format",
                    "type": "validation",
                    "suggested": True,
                    "estimated_duration": 5
                })
        
        elif parsed_workflow.workflow_type == WorkflowType.ANALYSIS:
            # Check for result saving
            has_output = any("save" in task.get("description", "").lower() or 
                           "output" in task.get("description", "").lower()
                           for task in parsed_workflow.tasks)
            if not has_output:
                suggested.append({
                    "id": f"save_results_{len(parsed_workflow.tasks)}",
                    "description": "Save analysis results to output format",
                    "type": "io",
                    "suggested": True,
                    "estimated_duration": 3
                })
        
        return suggested
    
    def _generate_suggested_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate dependencies for suggested tasks."""
        dependencies = []
        
        # Find suggested tasks
        suggested_tasks = [t for t in tasks if t.get("suggested", False)]
        original_tasks = [t for t in tasks if not t.get("suggested", False)]
        
        # Connect suggested tasks to appropriate original tasks
        for suggested in suggested_tasks:
            if suggested.get("type") == "validation" and original_tasks:
                # Validation should come first
                dependencies.append({
                    "source": suggested["id"],
                    "target": original_tasks[0]["id"]
                })
            elif suggested.get("type") in ["io", "logging"] and original_tasks:
                # Output tasks should come last
                dependencies.append({
                    "source": original_tasks[-1]["id"],
                    "target": suggested["id"]
                })
        
        return dependencies
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
