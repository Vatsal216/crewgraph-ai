"""
Natural Language Parsers for CrewGraph AI

Parse natural language requirements and descriptions into structured data.

Author: Vatsal216
Created: 2025-07-23 06:25:00 UTC
"""

import re
import json
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


@dataclass
class ParsedWorkflow:
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


class WorkflowParser:
    """
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
            
        Returns:
            Parsed workflow structure
        """
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
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False