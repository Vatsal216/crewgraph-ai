"""
Natural Language to Workflow Converters for CrewGraph AI

Convert between natural language descriptions and executable workflows.

Author: Vatsal216
Created: 2025-07-23 06:25:00 UTC
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import threading

from .parser import RequirementsParser, WorkflowParser, ParsedWorkflow, ParsedTask, ParsedAgent
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class NLToWorkflowConverter:
    """
    Convert natural language descriptions to executable workflow definitions.
    
    Transforms parsed natural language requirements into CrewAI/LangGraph
    compatible workflow configurations.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:25:00 UTC
    """
    
    def __init__(self):
        """Initialize NL to workflow converter"""
        self.requirements_parser = RequirementsParser()
        self.workflow_parser = WorkflowParser()
        
        self._conversion_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        logger.info("NLToWorkflowConverter initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:25:00")
    
    def convert_to_workflow(self, 
                          natural_language: str,
                          workflow_format: str = "crewai") -> Dict[str, Any]:
        """
        Convert natural language to executable workflow.
        
        Args:
            natural_language: Natural language description
            workflow_format: Target format ("crewai", "langgraph", "generic")
            
        Returns:
            Executable workflow definition
        """
        with self._lock:
            start_time = time.time()
            
            # Parse natural language
            parsed_workflow = self.workflow_parser.parse_workflow_description(natural_language)
            
            # Convert to target format
            if workflow_format.lower() == "crewai":
                workflow_def = self._convert_to_crewai(parsed_workflow)
            elif workflow_format.lower() == "langgraph":
                workflow_def = self._convert_to_langgraph(parsed_workflow)
            else:
                workflow_def = self._convert_to_generic(parsed_workflow)
            
            # Add metadata
            workflow_def['metadata'] = {
                'source': 'natural_language',
                'converter_version': '1.0.0',
                'created_by': 'Vatsal216',
                'created_at': time.time(),
                'conversion_time': time.time() - start_time,
                'original_text_length': len(natural_language),
                'parsed_tasks': len(parsed_workflow.tasks),
                'parsed_agents': len(parsed_workflow.agents)
            }
            
            # Record conversion
            self._record_conversion(natural_language, parsed_workflow, workflow_def, workflow_format)
            
            conversion_time = time.time() - start_time
            metrics.record_metric("nl_to_workflow_conversions_total", 1.0)
            metrics.record_metric("conversion_time_seconds", conversion_time)
            
            logger.info(f"Converted NL to {workflow_format} workflow in {conversion_time:.3f}s")
            
            return workflow_def
    
    def _convert_to_crewai(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """Convert parsed workflow to CrewAI format"""
        
        # Convert agents
        crewai_agents = []
        for agent in parsed_workflow.agents:
            agent_def = {
                'role': agent.role.value,
                'goal': f"Execute {agent.specialization} tasks efficiently",
                'backstory': f"I am a specialized {agent.role.value} agent with expertise in {agent.specialization}",
                'tools': agent.tools,
                'verbose': True,
                'allow_delegation': False,
                'max_iter': 3,
                'memory': True,
                'step_callback': None,
                'system_template': None,
                'prompt_template': None,
                'response_template': None,
                'llm': None,  # Will be set by the framework
                'function_calling_llm': None,
                'callbacks': [],
                'agent_ops': False,
                'created_by': 'Vatsal216'
            }
            crewai_agents.append(agent_def)
        
        # Convert tasks
        crewai_tasks = []
        for task in parsed_workflow.tasks:
            task_def = {
                'description': task.description,
                'expected_output': f"Completed {task.name} with results",
                'tools': self._get_task_tools(task),
                'agent': self._find_agent_for_task(task, parsed_workflow.agents),
                'async_execution': False,
                'context': [],
                'config': {
                    'max_execution_time': None,
                    'max_retries': 3,
                    'retry_delay': 1
                },
                'output_json': None,
                'output_pydantic': None,
                'output_file': None,
                'callback': None,
                'human_input': False,
                'converter_metadata': {
                    'original_name': task.name,
                    'task_type': task.task_type.value,
                    'complexity': task.estimated_complexity,
                    'inputs': task.inputs,
                    'outputs': task.outputs
                }
            }
            crewai_tasks.append(task_def)
        
        # Set up task dependencies
        self._setup_crewai_dependencies(crewai_tasks, parsed_workflow.dependencies)
        
        # Create crew configuration
        crew_config = {
            'agents': crewai_agents,
            'tasks': crewai_tasks,
            'process': 'sequential',  # Default to sequential, could be hierarchical
            'verbose': 2,
            'memory': True,
            'cache': True,
            'max_rpm': 10,
            'language': 'en',
            'full_output': True,
            'step_callback': None,
            'task_callback': None,
            'share_crew': False,
            'function_calling_llm': None,
            'config': {
                'workflow_name': parsed_workflow.name,
                'description': parsed_workflow.description,
                'objectives': parsed_workflow.objectives,
                'constraints': parsed_workflow.constraints,
                'success_criteria': parsed_workflow.success_criteria,
                'estimated_duration_minutes': parsed_workflow.estimated_duration,
                'created_by': 'Vatsal216'
            }
        }
        
        return {
            'format': 'crewai',
            'crew': crew_config,
            'execution_config': {
                'inputs': self._extract_workflow_inputs(parsed_workflow),
                'expected_outputs': self._extract_workflow_outputs(parsed_workflow)
            }
        }
    
    def _convert_to_langgraph(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """Convert parsed workflow to LangGraph format"""
        
        # Create nodes for each task
        nodes = {}
        for task in parsed_workflow.tasks:
            node_def = {
                'function': f"execute_{task.name}",
                'input_schema': {
                    'type': 'object',
                    'properties': {inp: {'type': 'string'} for inp in task.inputs},
                    'required': task.inputs
                },
                'output_schema': {
                    'type': 'object', 
                    'properties': {out: {'type': 'string'} for out in task.outputs},
                    'required': task.outputs
                },
                'metadata': {
                    'description': task.description,
                    'task_type': task.task_type.value,
                    'complexity': task.estimated_complexity,
                    'agent_role': task.agent_role_required.value if task.agent_role_required else None,
                    'created_by': 'Vatsal216'
                }
            }
            nodes[task.name] = node_def
        
        # Create edges for dependencies
        edges = []
        for from_task, to_task in parsed_workflow.dependencies:
            edge_def = {
                'from': from_task,
                'to': to_task,
                'condition': None,  # Unconditional edge
                'metadata': {
                    'dependency_type': 'sequential',
                    'created_by': 'Vatsal216'
                }
            }
            edges.append(edge_def)
        
        # Add conditional edges if needed
        conditional_edges = self._create_conditional_edges(parsed_workflow)
        
        # Create state schema
        state_schema = self._create_state_schema(parsed_workflow)
        
        # Create LangGraph configuration
        langgraph_config = {
            'nodes': nodes,
            'edges': edges,
            'conditional_edges': conditional_edges,
            'state_schema': state_schema,
            'entry_point': self._find_entry_point(parsed_workflow),
            'finish_points': self._find_finish_points(parsed_workflow),
            'checkpointer': {
                'type': 'memory',
                'config': {}
            },
            'interrupt_before': [],
            'interrupt_after': [],
            'debug': False,
            'stream_mode': 'values',
            'output_keys': None,
            'config': {
                'workflow_name': parsed_workflow.name,
                'description': parsed_workflow.description,
                'objectives': parsed_workflow.objectives,
                'created_by': 'Vatsal216'
            }
        }
        
        return {
            'format': 'langgraph',
            'graph': langgraph_config,
            'execution_config': {
                'initial_state': self._create_initial_state(parsed_workflow),
                'config': {
                    'recursion_limit': 100,
                    'max_concurrency': 4
                }
            }
        }
    
    def _convert_to_generic(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """Convert parsed workflow to generic format"""
        
        return {
            'format': 'generic',
            'workflow': {
                'name': parsed_workflow.name,
                'description': parsed_workflow.description,
                'version': '1.0.0',
                'objectives': parsed_workflow.objectives,
                'constraints': parsed_workflow.constraints,
                'success_criteria': parsed_workflow.success_criteria,
                'estimated_duration': parsed_workflow.estimated_duration,
                'tasks': [asdict(task) for task in parsed_workflow.tasks],
                'agents': [asdict(agent) for agent in parsed_workflow.agents],
                'dependencies': parsed_workflow.dependencies,
                'execution_plan': {
                    'entry_tasks': self._find_entry_tasks(parsed_workflow),
                    'exit_tasks': self._find_exit_tasks(parsed_workflow),
                    'parallel_groups': self._find_parallel_groups(parsed_workflow),
                    'critical_path': self._find_critical_path(parsed_workflow)
                },
                'resource_requirements': {
                    'estimated_memory_mb': sum(256 for _ in parsed_workflow.agents),
                    'estimated_cpu_cores': len(parsed_workflow.agents) * 0.5,
                    'estimated_storage_mb': 1024,
                    'network_required': any('api' in task.task_type.value for task in parsed_workflow.tasks)
                },
                'created_by': 'Vatsal216',
                'created_at': '2025-07-23 06:25:00'
            }
        }
    
    def _get_task_tools(self, task: ParsedTask) -> List[str]:
        """Get tools needed for a specific task"""
        tools = []
        
        # Task type specific tools
        if task.task_type.value == 'file_operation':
            tools.extend(['file_reader', 'file_writer'])
        elif task.task_type.value == 'api_call':
            tools.extend(['http_client', 'json_parser'])
        elif task.task_type.value == 'data_processing':
            tools.extend(['data_processor', 'csv_handler'])
        elif task.task_type.value == 'analysis':
            tools.extend(['data_analyzer', 'statistics_calculator'])
        elif task.task_type.value == 'validation':
            tools.extend(['validator', 'schema_checker'])
        elif task.task_type.value == 'calculation':
            tools.extend(['calculator', 'math_utils'])
        elif task.task_type.value == 'search':
            tools.extend(['search_engine', 'query_builder'])
        elif task.task_type.value == 'notification':
            tools.extend(['email_sender', 'notification_service'])
        
        # Add generic tools
        tools.extend(['logger', 'error_handler'])
        
        return list(set(tools))  # Remove duplicates
    
    def _find_agent_for_task(self, task: ParsedTask, agents: List[ParsedAgent]) -> str:
        """Find the best agent for a task"""
        if task.agent_role_required:
            for agent in agents:
                if agent.role == task.agent_role_required:
                    return agent.name
        
        # Fallback to first agent or create default
        if agents:
            return agents[0].name
        
        return 'default_agent'
    
    def _setup_crewai_dependencies(self, tasks: List[Dict], dependencies: List[Tuple[str, str]]):
        """Set up task dependencies in CrewAI format"""
        # Create task name to index mapping
        task_map = {}
        for i, task in enumerate(tasks):
            original_name = task.get('converter_metadata', {}).get('original_name')
            if original_name:
                task_map[original_name] = i
        
        # Set up context dependencies
        for from_task, to_task in dependencies:
            if from_task in task_map and to_task in task_map:
                from_idx = task_map[from_task]
                to_idx = task_map[to_task]
                
                # Add from_task to to_task's context
                if 'context' not in tasks[to_idx]:
                    tasks[to_idx]['context'] = []
                
                tasks[to_idx]['context'].append(from_idx)
    
    def _create_conditional_edges(self, parsed_workflow: ParsedWorkflow) -> List[Dict]:
        """Create conditional edges for LangGraph"""
        conditional_edges = []
        
        # Look for decision tasks
        for task in parsed_workflow.tasks:
            if task.task_type.value == 'decision':
                # Create conditional edge based on decision outcome
                conditional_edge = {
                    'source': task.name,
                    'path_map': {
                        'true': f"{task.name}_success_path",
                        'false': f"{task.name}_failure_path",
                        'default': f"{task.name}_default_path"
                    },
                    'condition_function': f"evaluate_{task.name}_condition",
                    'metadata': {
                        'type': 'decision',
                        'created_by': 'Vatsal216'
                    }
                }
                conditional_edges.append(conditional_edge)
        
        return conditional_edges
    
    def _create_state_schema(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """Create state schema for LangGraph"""
        state_properties = {}
        
        # Add workflow-level state
        state_properties['workflow_state'] = {
            'type': 'object',
            'properties': {
                'status': {'type': 'string', 'enum': ['running', 'completed', 'failed']},
                'progress': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'current_task': {'type': 'string'},
                'results': {'type': 'object'},
                'errors': {'type': 'array', 'items': {'type': 'string'}}
            }
        }
        
        # Add task-specific state
        for task in parsed_workflow.tasks:
            task_state = {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string', 'enum': ['pending', 'running', 'completed', 'failed']},
                    'input': {'type': 'object'},
                    'output': {'type': 'object'},
                    'error': {'type': 'string'},
                    'start_time': {'type': 'number'},
                    'end_time': {'type': 'number'}
                }
            }
            state_properties[f"{task.name}_state"] = task_state
        
        return {
            'type': 'object',
            'properties': state_properties,
            'required': ['workflow_state']
        }
    
    def _find_entry_point(self, parsed_workflow: ParsedWorkflow) -> str:
        """Find the entry point for LangGraph"""
        # Find tasks with no dependencies
        all_tasks = [task.name for task in parsed_workflow.tasks]
        dependent_tasks = set(dep[1] for dep in parsed_workflow.dependencies)
        
        entry_tasks = [task for task in all_tasks if task not in dependent_tasks]
        
        # Return first entry task or first task overall
        if entry_tasks:
            return entry_tasks[0]
        elif parsed_workflow.tasks:
            return parsed_workflow.tasks[0].name
        
        return 'start'
    
    def _find_finish_points(self, parsed_workflow: ParsedWorkflow) -> List[str]:
        """Find finish points for LangGraph"""
        # Find tasks that are not dependencies for other tasks
        all_tasks = [task.name for task in parsed_workflow.tasks]
        dependency_sources = set(dep[0] for dep in parsed_workflow.dependencies)
        
        finish_tasks = [task for task in all_tasks if task not in dependency_sources]
        
        # If no clear finish tasks, use last task
        if not finish_tasks and parsed_workflow.tasks:
            finish_tasks = [parsed_workflow.tasks[-1].name]
        
        return finish_tasks or ['end']
    
    def _create_initial_state(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """Create initial state for LangGraph execution"""
        initial_state = {
            'workflow_state': {
                'status': 'running',
                'progress': 0.0,
                'current_task': None,
                'results': {},
                'errors': []
            }
        }
        
        # Initialize task states
        for task in parsed_workflow.tasks:
            initial_state[f"{task.name}_state"] = {
                'status': 'pending',
                'input': {},
                'output': {},
                'error': None,
                'start_time': None,
                'end_time': None
            }
        
        return initial_state
    
    def _extract_workflow_inputs(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """Extract required workflow inputs"""
        inputs = {}
        
        # Collect all unique inputs from tasks
        all_inputs = set()
        for task in parsed_workflow.tasks:
            all_inputs.update(task.inputs)
        
        # Create input schema
        for input_name in all_inputs:
            inputs[input_name] = {
                'type': 'string',
                'description': f'Input parameter: {input_name}',
                'required': True
            }
        
        return inputs
    
    def _extract_workflow_outputs(self, parsed_workflow: ParsedWorkflow) -> Dict[str, Any]:
        """Extract expected workflow outputs"""
        outputs = {}
        
        # Collect all unique outputs from tasks
        all_outputs = set()
        for task in parsed_workflow.tasks:
            all_outputs.update(task.outputs)
        
        # Create output schema
        for output_name in all_outputs:
            outputs[output_name] = {
                'type': 'string',
                'description': f'Output result: {output_name}'
            }
        
        return outputs
    
    def _find_entry_tasks(self, parsed_workflow: ParsedWorkflow) -> List[str]:
        """Find entry tasks (no dependencies)"""
        all_tasks = [task.name for task in parsed_workflow.tasks]
        dependent_tasks = set(dep[1] for dep in parsed_workflow.dependencies)
        
        return [task for task in all_tasks if task not in dependent_tasks]
    
    def _find_exit_tasks(self, parsed_workflow: ParsedWorkflow) -> List[str]:
        """Find exit tasks (no dependents)"""
        all_tasks = [task.name for task in parsed_workflow.tasks]
        dependency_sources = set(dep[0] for dep in parsed_workflow.dependencies)
        
        return [task for task in all_tasks if task not in dependency_sources]
    
    def _find_parallel_groups(self, parsed_workflow: ParsedWorkflow) -> List[List[str]]:
        """Find groups of tasks that can run in parallel"""
        # Simple implementation - could be more sophisticated
        parallel_groups = []
        
        # Find tasks at same dependency level
        dependency_levels = {}
        self._calculate_dependency_levels(parsed_workflow, dependency_levels)
        
        # Group by dependency level
        level_groups = {}
        for task_name, level in dependency_levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(task_name)
        
        # Return groups with more than one task
        for level, tasks in level_groups.items():
            if len(tasks) > 1:
                parallel_groups.append(tasks)
        
        return parallel_groups
    
    def _calculate_dependency_levels(self, parsed_workflow: ParsedWorkflow, levels: Dict[str, int]):
        """Calculate dependency levels for tasks"""
        # Initialize all tasks at level 0
        for task in parsed_workflow.tasks:
            levels[task.name] = 0
        
        # Calculate levels based on dependencies
        changed = True
        while changed:
            changed = False
            for from_task, to_task in parsed_workflow.dependencies:
                new_level = levels[from_task] + 1
                if levels[to_task] < new_level:
                    levels[to_task] = new_level
                    changed = True
    
    def _find_critical_path(self, parsed_workflow: ParsedWorkflow) -> List[str]:
        """Find the critical path through the workflow"""
        # Simple implementation - return longest path
        if not parsed_workflow.tasks:
            return []
        
        # Build dependency graph
        graph = {}
        for from_task, to_task in parsed_workflow.dependencies:
            if from_task not in graph:
                graph[from_task] = []
            graph[from_task].append(to_task)
        
        # Find longest path using DFS
        longest_path = []
        
        def dfs(node, current_path):
            nonlocal longest_path
            current_path = current_path + [node]
            
            if len(current_path) > len(longest_path):
                longest_path = current_path[:]
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, current_path)
        
        # Start DFS from entry tasks
        entry_tasks = self._find_entry_tasks(parsed_workflow)
        for entry_task in entry_tasks:
            dfs(entry_task, [])
        
        return longest_path
    
    def _record_conversion(self, 
                         natural_language: str,
                         parsed_workflow: ParsedWorkflow,
                         workflow_def: Dict[str, Any],
                         workflow_format: str):
        """Record conversion for analysis and improvement"""
        record = {
            'timestamp': time.time(),
            'input_text': natural_language,
            'input_length': len(natural_language),
            'parsed_workflow': asdict(parsed_workflow),
            'output_format': workflow_format,
            'output_size': len(json.dumps(workflow_def)),
            'tasks_count': len(parsed_workflow.tasks),
            'agents_count': len(parsed_workflow.agents),
            'dependencies_count': len(parsed_workflow.dependencies),
            'created_by': 'Vatsal216'
        }
        
        self._conversion_history.append(record)
        
        # Keep only last 100 conversions
        if len(self._conversion_history) > 100:
            self._conversion_history = self._conversion_history[-100:]
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get statistics about conversion performance"""
        if not self._conversion_history:
            return {'total_conversions': 0}
        
        total_conversions = len(self._conversion_history)
        avg_input_length = sum(c['input_length'] for c in self._conversion_history) / total_conversions
        avg_tasks = sum(c['tasks_count'] for c in self._conversion_history) / total_conversions
        avg_agents = sum(c['agents_count'] for c in self._conversion_history) / total_conversions
        
        format_counts = {}
        for conversion in self._conversion_history:
            format_name = conversion['output_format']
            format_counts[format_name] = format_counts.get(format_name, 0) + 1
        
        return {
            'total_conversions': total_conversions,
            'avg_input_length': avg_input_length,
            'avg_tasks_per_workflow': avg_tasks,
            'avg_agents_per_workflow': avg_agents,
            'format_distribution': format_counts,
            'created_by': 'Vatsal216',
            'timestamp': time.time()
        }


class WorkflowToNLConverter:
    """
    Convert workflow definitions back to natural language descriptions.
    
    Useful for documentation generation and human-readable explanations.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:25:00 UTC
    """
    
    def __init__(self):
        """Initialize workflow to NL converter"""
        self._conversion_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        logger.info("WorkflowToNLConverter initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:25:00")
    
    def convert_to_natural_language(self, 
                                  workflow_def: Dict[str, Any],
                                  style: str = "descriptive") -> str:
        """
        Convert workflow definition to natural language.
        
        Args:
            workflow_def: Workflow definition to convert
            style: Writing style ("descriptive", "technical", "concise")
            
        Returns:
            Natural language description
        """
        with self._lock:
            start_time = time.time()
            
            # Detect workflow format
            workflow_format = workflow_def.get('format', 'unknown')
            
            # Extract workflow components
            if workflow_format == 'crewai':
                description = self._convert_crewai_to_nl(workflow_def, style)
            elif workflow_format == 'langgraph':
                description = self._convert_langgraph_to_nl(workflow_def, style)
            elif workflow_format == 'generic':
                description = self._convert_generic_to_nl(workflow_def, style)
            else:
                description = self._convert_unknown_to_nl(workflow_def, style)
            
            # Record conversion
            conversion_time = time.time() - start_time
            self._record_conversion(workflow_def, description, style, conversion_time)
            
            metrics.record_metric("workflow_to_nl_conversions_total", 1.0)
            metrics.record_metric("nl_conversion_time_seconds", conversion_time)
            
            logger.info(f"Converted {workflow_format} workflow to NL in {conversion_time:.3f}s")
            
            return description
    
    def _convert_crewai_to_nl(self, workflow_def: Dict[str, Any], style: str) -> str:
        """Convert CrewAI workflow to natural language"""
        crew = workflow_def.get('crew', {})
        agents = crew.get('agents', [])
        tasks = crew.get('tasks', [])
        config = crew.get('config', {})
        
        description_parts = []
        
        # Workflow overview
        workflow_name = config.get('workflow_name', 'Unnamed Workflow')
        description_parts.append(f"The {workflow_name} is a collaborative workflow that")
        
        if config.get('objectives'):
            objectives = config['objectives']
            if len(objectives) == 1:
                description_parts.append(f"aims to {objectives[0]}.")
            else:
                obj_list = ', '.join(objectives[:-1]) + f', and {objectives[-1]}'
                description_parts.append(f"aims to {obj_list}.")
        else:
            description_parts.append("executes a series of coordinated tasks.")
        
        # Agents description
        if agents:
            description_parts.append(f"\nThis workflow involves {len(agents)} specialized agents:")
            for i, agent in enumerate(agents, 1):
                role = agent.get('role', 'agent')
                goal = agent.get('goal', 'execute tasks')
                description_parts.append(f"{i}. A {role} agent whose goal is to {goal.lower()}")
        
        # Tasks description  
        if tasks:
            description_parts.append(f"\nThe workflow consists of {len(tasks)} main tasks:")
            for i, task in enumerate(tasks, 1):
                task_desc = task.get('description', 'perform task')
                expected_output = task.get('expected_output', 'complete the task')
                
                if style == "concise":
                    description_parts.append(f"{i}. {task_desc}")
                else:
                    description_parts.append(f"{i}. {task_desc}, with the expected outcome being {expected_output.lower()}")
        
        # Process description
        process = crew.get('process', 'sequential')
        if process == 'sequential':
            description_parts.append("\nTasks are executed sequentially, with each task building upon the results of the previous one.")
        elif process == 'hierarchical':
            description_parts.append("\nTasks are organized hierarchically, with a manager agent coordinating the execution.")
        
        # Success criteria
        if config.get('success_criteria'):
            criteria = config['success_criteria']
            description_parts.append(f"\nSuccess is measured by: {', '.join(criteria)}.")
        
        return ' '.join(description_parts)
    
    def _convert_langgraph_to_nl(self, workflow_def: Dict[str, Any], style: str) -> str:
        """Convert LangGraph workflow to natural language"""
        graph = workflow_def.get('graph', {})
        nodes = graph.get('nodes', {})
        edges = graph.get('edges', [])
        config = graph.get('config', {})
        
        description_parts = []
        
        # Workflow overview
        workflow_name = config.get('workflow_name', 'State Graph Workflow')
        description_parts.append(f"The {workflow_name} is a state-driven workflow that")
        
        if config.get('objectives'):
            objectives = config['objectives']
            description_parts.append(f"aims to {objectives[0]}.")
        else:
            description_parts.append("processes data through a series of connected nodes.")
        
        # Nodes description
        if nodes:
            description_parts.append(f"\nThe workflow contains {len(nodes)} processing nodes:")
            for i, (node_name, node_def) in enumerate(nodes.items(), 1):
                node_desc = node_def.get('metadata', {}).get('description', f'execute {node_name}')
                description_parts.append(f"{i}. {node_name}: {node_desc}")
        
        # Flow description
        if edges:
            description_parts.append(f"\nThe execution flow follows {len(edges)} defined connections:")
            for edge in edges:
                from_node = edge.get('from')
                to_node = edge.get('to')
                description_parts.append(f"- Data flows from {from_node} to {to_node}")
        
        # Entry and exit points
        entry_point = graph.get('entry_point')
        finish_points = graph.get('finish_points', [])
        
        if entry_point:
            description_parts.append(f"\nExecution begins at the {entry_point} node")
        
        if finish_points:
            if len(finish_points) == 1:
                description_parts.append(f"and concludes at the {finish_points[0]} node.")
            else:
                finish_list = ', '.join(finish_points[:-1]) + f', or {finish_points[-1]}'
                description_parts.append(f"and can conclude at {finish_list}.")
        
        return ' '.join(description_parts)
    
    def _convert_generic_to_nl(self, workflow_def: Dict[str, Any], style: str) -> str:
        """Convert generic workflow to natural language"""
        workflow = workflow_def.get('workflow', {})
        
        description_parts = []
        
        # Basic info
        name = workflow.get('name', 'Workflow')
        description = workflow.get('description', '')
        
        if description:
            description_parts.append(f"The {name} {description.lower()}")
        else:
            description_parts.append(f"The {name} is a multi-step process")
        
        # Objectives
        objectives = workflow.get('objectives', [])
        if objectives:
            if len(objectives) == 1:
                description_parts.append(f"with the goal to {objectives[0]}.")
            else:
                obj_list = ', '.join(objectives[:-1]) + f', and {objectives[-1]}'
                description_parts.append(f"with goals to {obj_list}.")
        
        # Tasks and agents
        tasks = workflow.get('tasks', [])
        agents = workflow.get('agents', [])
        
        if tasks and agents:
            description_parts.append(f"\nThis workflow involves {len(agents)} agents executing {len(tasks)} tasks.")
        elif tasks:
            description_parts.append(f"\nThe workflow consists of {len(tasks)} tasks.")
        
        # Execution plan
        execution_plan = workflow.get('execution_plan', {})
        if execution_plan:
            entry_tasks = execution_plan.get('entry_tasks', [])
            parallel_groups = execution_plan.get('parallel_groups', [])
            
            if entry_tasks:
                entry_list = ', '.join(entry_tasks)
                description_parts.append(f"Execution begins with: {entry_list}.")
            
            if parallel_groups:
                description_parts.append(f"The workflow includes {len(parallel_groups)} groups of parallel tasks for improved efficiency.")
        
        # Duration and resources
        duration = workflow.get('estimated_duration')
        if duration and duration > 0:
            if duration < 60:
                description_parts.append(f"The estimated execution time is {duration:.0f} minutes.")
            else:
                hours = duration / 60
                description_parts.append(f"The estimated execution time is {hours:.1f} hours.")
        
        return ' '.join(description_parts)
    
    def _convert_unknown_to_nl(self, workflow_def: Dict[str, Any], style: str) -> str:
        """Convert unknown format workflow to natural language"""
        description_parts = ["This workflow definition contains"]
        
        # Count components
        component_counts = []
        
        if 'tasks' in workflow_def:
            task_count = len(workflow_def['tasks'])
            component_counts.append(f"{task_count} tasks")
        
        if 'agents' in workflow_def:
            agent_count = len(workflow_def['agents'])
            component_counts.append(f"{agent_count} agents")
        
        if 'nodes' in workflow_def:
            node_count = len(workflow_def['nodes'])
            component_counts.append(f"{node_count} nodes")
        
        if component_counts:
            description_parts.append(', '.join(component_counts) + ".")
        else:
            description_parts.append("workflow components.")
        
        # Add any available description
        if 'description' in workflow_def:
            description_parts.append(f"The workflow {workflow_def['description'].lower()}")
        
        return ' '.join(description_parts)
    
    def _record_conversion(self, 
                         workflow_def: Dict[str, Any],
                         description: str,
                         style: str,
                         conversion_time: float):
        """Record conversion for analysis"""
        record = {
            'timestamp': time.time(),
            'workflow_format': workflow_def.get('format', 'unknown'),
            'workflow_size': len(json.dumps(workflow_def)),
            'description_length': len(description),
            'style': style,
            'conversion_time': conversion_time,
            'created_by': 'Vatsal216'
        }
        
        self._conversion_history.append(record)
        
        # Keep only last 100 conversions
        if len(self._conversion_history) > 100:
            self._conversion_history = self._conversion_history[-100:]
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get statistics about NL conversion performance"""
        if not self._conversion_history:
            return {'total_conversions': 0}
        
        total_conversions = len(self._conversion_history)
        avg_conversion_time = sum(c['conversion_time'] for c in self._conversion_history) / total_conversions
        avg_description_length = sum(c['description_length'] for c in self._conversion_history) / total_conversions
        
        format_counts = {}
        style_counts = {}
        
        for conversion in self._conversion_history:
            format_name = conversion['workflow_format']
            style_name = conversion['style']
            
            format_counts[format_name] = format_counts.get(format_name, 0) + 1
            style_counts[style_name] = style_counts.get(style_name, 0) + 1
        
        return {
            'total_conversions': total_conversions,
            'avg_conversion_time': avg_conversion_time,
            'avg_description_length': avg_description_length,
            'format_distribution': format_counts,
            'style_distribution': style_counts,
            'created_by': 'Vatsal216',
            'timestamp': time.time()
        }