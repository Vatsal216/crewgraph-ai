"""
<<<<<<< HEAD
Documentation and Code Generators for CrewGraph AI

Automatic generation of documentation and code from workflow definitions.

Author: Vatsal216
Created: 2025-07-23 06:25:00 UTC
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import re

from .converter import WorkflowToNLConverter
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class DocumentationType(Enum):
    """Types of documentation that can be generated"""
    USER_MANUAL = "user_manual"
    TECHNICAL_SPECS = "technical_specs"
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    README = "readme"
    DEPLOYMENT_GUIDE = "deployment_guide"


class CodeLanguage(Enum):
    """Programming languages for code generation"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    YAML = "yaml"
    JSON = "json"


@dataclass
class DocumentationSection:
    """A section of generated documentation"""
    title: str
    content: str
    section_type: str
    level: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedCode:
    """Generated code structure"""
    language: CodeLanguage
    filename: str
    content: str
    dependencies: List[str]
    description: str
    metadata: Optional[Dict[str, Any]] = None
=======
Documentation Generator for CrewGraph AI NLP Module

Generates comprehensive documentation from workflow definitions and execution data.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WorkflowDocumentation:
    """Generated documentation for a workflow."""
    title: str
    overview: str
    tasks_documentation: List[Dict[str, str]]
    dependencies_diagram: str
    usage_instructions: str
    troubleshooting: List[str]
    performance_notes: List[str]
    generated_timestamp: str
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d


class DocumentationGenerator:
    """
<<<<<<< HEAD
    Generate comprehensive documentation from workflow definitions.
    
    Creates user manuals, technical specifications, API references,
    and other documentation types automatically.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:25:00 UTC
    """
    
    def __init__(self):
        """Initialize documentation generator"""
        self.nl_converter = WorkflowToNLConverter()
        
        self._documentation_templates = self._build_documentation_templates()
        self._generation_history: List[Dict[str, Any]] = []
        
        self._lock = threading.RLock()
        
        logger.info("DocumentationGenerator initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:25:00")
    
    def generate_documentation(self, 
                             workflow_def: Dict[str, Any],
                             doc_type: DocumentationType,
                             include_examples: bool = True,
                             include_diagrams: bool = False) -> Dict[str, Any]:
        """
        Generate documentation for a workflow.
        
        Args:
            workflow_def: Workflow definition
            doc_type: Type of documentation to generate
            include_examples: Whether to include code examples
            include_diagrams: Whether to include diagrams (text-based)
            
        Returns:
            Generated documentation structure
        """
        with self._lock:
            start_time = time.time()
            
            # Generate natural language description
            workflow_description = self.nl_converter.convert_to_natural_language(
                workflow_def, style="technical"
            )
            
            # Generate documentation based on type
            if doc_type == DocumentationType.USER_MANUAL:
                doc_sections = self._generate_user_manual(workflow_def, workflow_description, include_examples)
            elif doc_type == DocumentationType.TECHNICAL_SPECS:
                doc_sections = self._generate_technical_specs(workflow_def, workflow_description, include_diagrams)
            elif doc_type == DocumentationType.API_REFERENCE:
                doc_sections = self._generate_api_reference(workflow_def, workflow_description)
            elif doc_type == DocumentationType.TUTORIAL:
                doc_sections = self._generate_tutorial(workflow_def, workflow_description, include_examples)
            elif doc_type == DocumentationType.README:
                doc_sections = self._generate_readme(workflow_def, workflow_description)
            elif doc_type == DocumentationType.DEPLOYMENT_GUIDE:
                doc_sections = self._generate_deployment_guide(workflow_def, workflow_description)
            else:
                doc_sections = self._generate_generic_documentation(workflow_def, workflow_description)
            
            # Compile into final document
            compiled_doc = self._compile_documentation(doc_sections, doc_type)
            
            # Record generation
            generation_time = time.time() - start_time
            self._record_generation(workflow_def, doc_type, compiled_doc, generation_time)
            
            metrics.record_metric("documentation_generated_total", 1.0)
            metrics.record_metric("documentation_generation_time_seconds", generation_time)
            
            logger.info(f"Generated {doc_type.value} documentation in {generation_time:.3f}s")
            
            return {
                "documentation_type": doc_type.value,
                "sections": [section.__dict__ for section in doc_sections],
                "compiled_content": compiled_doc,
                "metadata": {
                    "workflow_format": workflow_def.get('format', 'unknown'),
                    "generation_time": generation_time,
                    "include_examples": include_examples,
                    "include_diagrams": include_diagrams,
                    "created_by": "Vatsal216",
                    "created_at": time.time()
                }
            }
    
    def _generate_user_manual(self, 
                            workflow_def: Dict[str, Any], 
                            description: str,
                            include_examples: bool) -> List[DocumentationSection]:
        """Generate user manual sections"""
        sections = []
        
        # Title and overview
        workflow_name = self._extract_workflow_name(workflow_def)
        sections.append(DocumentationSection(
            title=f"{workflow_name} User Manual",
            content=f"# {workflow_name} User Manual\n\n{description}",
            section_type="title",
            level=1
        ))
        
        # Getting started
        sections.append(DocumentationSection(
            title="Getting Started",
            content=self._generate_getting_started_content(workflow_def),
            section_type="getting_started",
            level=2
        ))
        
        # Usage instructions
        sections.append(DocumentationSection(
            title="How to Use",
            content=self._generate_usage_instructions(workflow_def),
            section_type="usage",
            level=2
        ))
        
        # Configuration
        sections.append(DocumentationSection(
            title="Configuration",
            content=self._generate_configuration_guide(workflow_def),
            section_type="configuration",
            level=2
        ))
        
        # Examples
        if include_examples:
            sections.append(DocumentationSection(
                title="Examples",
                content=self._generate_usage_examples(workflow_def),
                section_type="examples",
                level=2
            ))
        
        # Troubleshooting
        sections.append(DocumentationSection(
            title="Troubleshooting",
            content=self._generate_troubleshooting_guide(workflow_def),
            section_type="troubleshooting",
            level=2
        ))
        
        return sections
    
    def _generate_technical_specs(self, 
                                workflow_def: Dict[str, Any], 
                                description: str,
                                include_diagrams: bool) -> List[DocumentationSection]:
        """Generate technical specification sections"""
        sections = []
        
        workflow_name = self._extract_workflow_name(workflow_def)
        
        # Title and overview
        sections.append(DocumentationSection(
            title=f"{workflow_name} Technical Specification",
            content=f"# {workflow_name} Technical Specification\n\n{description}",
            section_type="title",
            level=1
        ))
        
        # Architecture overview
        sections.append(DocumentationSection(
            title="Architecture Overview",
            content=self._generate_architecture_overview(workflow_def),
            section_type="architecture",
            level=2
        ))
        
        # Component specifications
        sections.append(DocumentationSection(
            title="Component Specifications",
            content=self._generate_component_specs(workflow_def),
            section_type="components",
            level=2
        ))
        
        # Data flow
        sections.append(DocumentationSection(
            title="Data Flow",
            content=self._generate_data_flow_documentation(workflow_def),
            section_type="data_flow",
            level=2
        ))
        
        # Performance specifications
        sections.append(DocumentationSection(
            title="Performance Specifications",
            content=self._generate_performance_specs(workflow_def),
            section_type="performance",
            level=2
        ))
        
        # Security considerations
        sections.append(DocumentationSection(
            title="Security Considerations",
            content=self._generate_security_documentation(workflow_def),
            section_type="security",
            level=2
        ))
        
        return sections
    
    def _generate_api_reference(self, 
                              workflow_def: Dict[str, Any], 
                              description: str) -> List[DocumentationSection]:
        """Generate API reference sections"""
        sections = []
        
        workflow_name = self._extract_workflow_name(workflow_def)
        
        # Title
        sections.append(DocumentationSection(
            title=f"{workflow_name} API Reference",
            content=f"# {workflow_name} API Reference\n\n{description}",
            section_type="title",
            level=1
        ))
        
        # API overview
        sections.append(DocumentationSection(
            title="API Overview",
            content=self._generate_api_overview(workflow_def),
            section_type="api_overview",
            level=2
        ))
        
        # Endpoints
        sections.append(DocumentationSection(
            title="Endpoints",
            content=self._generate_api_endpoints(workflow_def),
            section_type="endpoints",
            level=2
        ))
        
        # Request/Response schemas
        sections.append(DocumentationSection(
            title="Request and Response Schemas",
            content=self._generate_api_schemas(workflow_def),
            section_type="schemas",
            level=2
        ))
        
        # Error codes
        sections.append(DocumentationSection(
            title="Error Codes",
            content=self._generate_error_codes_documentation(),
            section_type="error_codes",
            level=2
        ))
        
        return sections
    
    def _generate_tutorial(self, 
                         workflow_def: Dict[str, Any], 
                         description: str,
                         include_examples: bool) -> List[DocumentationSection]:
        """Generate tutorial sections"""
        sections = []
        
        workflow_name = self._extract_workflow_name(workflow_def)
        
        # Title
        sections.append(DocumentationSection(
            title=f"{workflow_name} Tutorial",
            content=f"# {workflow_name} Tutorial\n\nLearn how to use {workflow_name} step by step.\n\n{description}",
            section_type="title",
            level=1
        ))
        
        # Prerequisites
        sections.append(DocumentationSection(
            title="Prerequisites",
            content=self._generate_prerequisites(workflow_def),
            section_type="prerequisites",
            level=2
        ))
        
        # Step-by-step guide
        tutorial_steps = self._generate_tutorial_steps(workflow_def)
        for i, step in enumerate(tutorial_steps, 1):
            sections.append(DocumentationSection(
                title=f"Step {i}: {step['title']}",
                content=step['content'],
                section_type=f"tutorial_step_{i}",
                level=3
            ))
        
        # Advanced usage
        if include_examples:
            sections.append(DocumentationSection(
                title="Advanced Usage",
                content=self._generate_advanced_usage_examples(workflow_def),
                section_type="advanced_usage",
                level=2
            ))
        
        # Next steps
        sections.append(DocumentationSection(
            title="Next Steps",
            content=self._generate_next_steps_content(),
            section_type="next_steps",
            level=2
        ))
        
        return sections
    
    def _generate_readme(self, 
                       workflow_def: Dict[str, Any], 
                       description: str) -> List[DocumentationSection]:
        """Generate README sections"""
        sections = []
        
        workflow_name = self._extract_workflow_name(workflow_def)
        
        # Title and description
        sections.append(DocumentationSection(
            title=workflow_name,
            content=f"# {workflow_name}\n\n{description}",
            section_type="title",
            level=1
        ))
        
        # Features
        sections.append(DocumentationSection(
            title="Features",
            content=self._generate_features_list(workflow_def),
            section_type="features",
            level=2
        ))
        
        # Quick start
        sections.append(DocumentationSection(
            title="Quick Start",
            content=self._generate_quick_start_guide(workflow_def),
            section_type="quick_start",
            level=2
        ))
        
        # Installation
        sections.append(DocumentationSection(
            title="Installation",
            content=self._generate_installation_instructions(workflow_def),
            section_type="installation",
            level=2
        ))
        
        # Usage
        sections.append(DocumentationSection(
            title="Usage",
            content=self._generate_basic_usage_example(workflow_def),
            section_type="usage",
            level=2
        ))
        
        # Contributing
        sections.append(DocumentationSection(
            title="Contributing",
            content=self._generate_contributing_guidelines(),
            section_type="contributing",
            level=2
        ))
        
        return sections
    
    def _generate_deployment_guide(self, 
                                 workflow_def: Dict[str, Any], 
                                 description: str) -> List[DocumentationSection]:
        """Generate deployment guide sections"""
        sections = []
        
        workflow_name = self._extract_workflow_name(workflow_def)
        
        # Title
        sections.append(DocumentationSection(
            title=f"{workflow_name} Deployment Guide",
            content=f"# {workflow_name} Deployment Guide\n\n{description}",
            section_type="title",
            level=1
        ))
        
        # System requirements
        sections.append(DocumentationSection(
            title="System Requirements",
            content=self._generate_system_requirements(workflow_def),
            section_type="requirements",
            level=2
        ))
        
        # Environment setup
        sections.append(DocumentationSection(
            title="Environment Setup",
            content=self._generate_environment_setup(workflow_def),
            section_type="environment",
            level=2
        ))
        
        # Deployment options
        sections.append(DocumentationSection(
            title="Deployment Options",
            content=self._generate_deployment_options(workflow_def),
            section_type="deployment_options",
            level=2
        ))
        
        # Configuration
        sections.append(DocumentationSection(
            title="Production Configuration",
            content=self._generate_production_configuration(workflow_def),
            section_type="production_config",
            level=2
        ))
        
        # Monitoring and maintenance
        sections.append(DocumentationSection(
            title="Monitoring and Maintenance",
            content=self._generate_monitoring_guide(workflow_def),
            section_type="monitoring",
            level=2
        ))
        
        return sections
    
    def _extract_workflow_name(self, workflow_def: Dict[str, Any]) -> str:
        """Extract workflow name from definition"""
        # Try different possible locations for the name
        name_candidates = [
            workflow_def.get('name'),
            workflow_def.get('workflow_name'),
            workflow_def.get('crew', {}).get('config', {}).get('workflow_name'),
            workflow_def.get('graph', {}).get('config', {}).get('workflow_name'),
            workflow_def.get('workflow', {}).get('name')
        ]
        
        for name in name_candidates:
            if name and isinstance(name, str):
                return name
        
        return "Workflow"
    
    def _generate_getting_started_content(self, workflow_def: Dict[str, Any]) -> str:
        """Generate getting started section content"""
        content = [
            "## Getting Started",
            "",
            "This section will help you set up and run your first workflow execution.",
            "",
            "### Prerequisites",
            "- Python 3.8 or higher",
            "- CrewGraph AI library installed",
            "- Basic understanding of AI workflows",
            "",
            "### Installation",
            "```bash",
            "pip install crewgraph-ai",
            "```",
            "",
            "### Basic Setup",
            "```python",
            "from crewgraph_ai import CrewGraph",
            "",
            "# Initialize your workflow",
            "workflow = CrewGraph('your_workflow')",
            "```"
        ]
        
        return "\n".join(content)
    
    def _generate_usage_instructions(self, workflow_def: Dict[str, Any]) -> str:
        """Generate usage instructions"""
        workflow_format = workflow_def.get('format', 'unknown')
        
        if workflow_format == 'crewai':
            return self._generate_crewai_usage_instructions(workflow_def)
        elif workflow_format == 'langgraph':
            return self._generate_langgraph_usage_instructions(workflow_def)
        else:
            return self._generate_generic_usage_instructions(workflow_def)
    
    def _generate_crewai_usage_instructions(self, workflow_def: Dict[str, Any]) -> str:
        """Generate CrewAI-specific usage instructions"""
        crew = workflow_def.get('crew', {})
        agents = crew.get('agents', [])
        tasks = crew.get('tasks', [])
        
        content = [
            "## How to Use",
            "",
            "This workflow is built using the CrewAI framework.",
            "",
            f"### Agents ({len(agents)} total)",
            "The workflow includes the following specialized agents:",
        ]
        
        for i, agent in enumerate(agents, 1):
            role = agent.get('role', 'agent')
            goal = agent.get('goal', 'execute tasks')
            content.append(f"{i}. **{role}**: {goal}")
        
        content.extend([
            "",
            f"### Tasks ({len(tasks)} total)",
            "The workflow executes these tasks:",
        ])
        
        for i, task in enumerate(tasks, 1):
            description = task.get('description', 'task execution')
            content.append(f"{i}. {description}")
        
        content.extend([
            "",
            "### Execution",
            "```python",
            "from crewai import Crew",
            "",
            "# Create and run the crew",
            "crew = Crew(",
            "    agents=agents,",
            "    tasks=tasks,",
            "    verbose=True",
            ")",
            "",
            "result = crew.kickoff()",
            "```"
        ])
        
        return "\n".join(content)
    
    def _generate_langgraph_usage_instructions(self, workflow_def: Dict[str, Any]) -> str:
        """Generate LangGraph-specific usage instructions"""
        graph = workflow_def.get('graph', {})
        nodes = graph.get('nodes', {})
        
        content = [
            "## How to Use",
            "",
            "This workflow is built using the LangGraph framework.",
            "",
            f"### Nodes ({len(nodes)} total)",
            "The workflow includes the following processing nodes:",
        ]
        
        for node_name, node_def in nodes.items():
            description = node_def.get('metadata', {}).get('description', 'process data')
            content.append(f"- **{node_name}**: {description}")
        
        content.extend([
            "",
            "### Execution",
            "```python",
            "from langgraph.graph import StateGraph",
            "",
            "# Create and compile the graph",
            "workflow = StateGraph(state_schema)",
            "",
            "# Add nodes and edges",
            "# ... (node and edge definitions)",
            "",
            "# Compile and run",
            "graph = workflow.compile()",
            "result = graph.invoke(initial_state)",
            "```"
        ])
        
        return "\n".join(content)
    
    def _generate_generic_usage_instructions(self, workflow_def: Dict[str, Any]) -> str:
        """Generate generic usage instructions"""
        return """## How to Use

This is a generic workflow definition that can be executed using CrewGraph AI.

### Basic Usage
```python
from crewgraph_ai import WorkflowExecutor

# Load the workflow
executor = WorkflowExecutor()
workflow = executor.load_workflow(workflow_definition)

# Execute the workflow
result = executor.run(workflow, inputs=your_inputs)
```

### Customization
You can customize the workflow by modifying the configuration parameters and adding your own data sources."""
    
    def _compile_documentation(self, sections: List[DocumentationSection], doc_type: DocumentationType) -> str:
        """Compile documentation sections into final document"""
        content_parts = []
        
        for section in sections:
            content_parts.append(section.content)
            content_parts.append("")  # Add spacing
        
        compiled_content = "\n".join(content_parts)
        
        # Add footer
        footer = f"""
---

*Generated by CrewGraph AI Documentation Generator*  
*Created by: Vatsal216*  
*Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return compiled_content + footer
    
    def _record_generation(self, 
                         workflow_def: Dict[str, Any],
                         doc_type: DocumentationType,
                         generated_doc: str,
                         generation_time: float):
        """Record documentation generation for analysis"""
        record = {
            'timestamp': time.time(),
            'workflow_format': workflow_def.get('format', 'unknown'),
            'documentation_type': doc_type.value,
            'workflow_size': len(json.dumps(workflow_def)),
            'documentation_length': len(generated_doc),
            'generation_time': generation_time,
            'created_by': 'Vatsal216'
        }
        
        self._generation_history.append(record)
        
        # Keep only last 100 generations
        if len(self._generation_history) > 100:
            self._generation_history = self._generation_history[-100:]
    
    def _generate_configuration_guide(self, workflow_def: Dict[str, Any]) -> str:
        """Generate configuration guide content"""
        return """## Configuration

### Environment Variables
Set the following environment variables for proper operation:

```bash
export OPENAI_API_KEY="your-api-key"
export CREWGRAPH_LOG_LEVEL="INFO"
```

### Workflow Parameters
Customize the workflow by modifying these parameters:

- `max_execution_time`: Maximum time for workflow execution
- `retry_attempts`: Number of retry attempts for failed tasks
- `parallel_execution`: Enable/disable parallel task execution

### Agent Configuration
Each agent can be configured with:

- Custom LLM models
- Specific tool sets
- Memory settings
- Execution parameters"""
    
    def _generate_troubleshooting_guide(self, workflow_def: Dict[str, Any]) -> str:
        """Generate troubleshooting guide content"""
        return """## Troubleshooting

### Common Issues

#### Workflow Fails to Start
- Check that all required dependencies are installed
- Verify API keys are properly configured
- Ensure input data format is correct

#### Tasks Time Out
- Increase the `max_execution_time` parameter
- Check network connectivity for external API calls
- Optimize complex tasks for better performance

#### Memory Issues
- Reduce batch sizes for large datasets
- Enable data streaming for memory-intensive operations
- Monitor system resources during execution

#### Agent Communication Problems
- Verify agent configurations are correct
- Check for conflicting tool assignments
- Review agent role definitions

### Getting Help
- Check the logs for detailed error messages
- Review the configuration settings
- Contact support with workflow definitions and error logs"""
    
    def _build_documentation_templates(self) -> Dict[str, Dict]:
        """Build documentation templates"""
        return {
            'user_manual': {
                'sections': ['overview', 'getting_started', 'usage', 'configuration', 'examples', 'troubleshooting'],
                'style': 'user_friendly'
            },
            'technical_specs': {
                'sections': ['architecture', 'components', 'data_flow', 'performance', 'security'],
                'style': 'technical'
            },
            'api_reference': {
                'sections': ['overview', 'endpoints', 'schemas', 'error_codes'],
                'style': 'reference'
            }
        }
    
    # Additional helper methods would be implemented here for specific content generation
    def _generate_architecture_overview(self, workflow_def: Dict[str, Any]) -> str:
        """Generate architecture overview content"""
        return "## Architecture Overview\n\nThis section describes the technical architecture of the workflow..."
    
    def _generate_component_specs(self, workflow_def: Dict[str, Any]) -> str:
        """Generate component specifications"""
        return "## Component Specifications\n\nDetailed specifications for each workflow component..."
    
    def _generate_data_flow_documentation(self, workflow_def: Dict[str, Any]) -> str:
        """Generate data flow documentation"""
        return "## Data Flow\n\nDescription of how data flows through the workflow..."
    
    def _generate_performance_specs(self, workflow_def: Dict[str, Any]) -> str:
        """Generate performance specifications"""
        return "## Performance Specifications\n\nExpected performance characteristics and benchmarks..."
    
    def _generate_security_documentation(self, workflow_def: Dict[str, Any]) -> str:
        """Generate security documentation"""
        return "## Security Considerations\n\nSecurity features and best practices..."
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about documentation generation"""
        if not self._generation_history:
            return {'total_generations': 0}
        
        total_generations = len(self._generation_history)
        avg_generation_time = sum(g['generation_time'] for g in self._generation_history) / total_generations
        avg_doc_length = sum(g['documentation_length'] for g in self._generation_history) / total_generations
        
        doc_type_counts = {}
        format_counts = {}
        
        for generation in self._generation_history:
            doc_type = generation['documentation_type']
            format_type = generation['workflow_format']
            
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            format_counts[format_type] = format_counts.get(format_type, 0) + 1
        
        return {
            'total_generations': total_generations,
            'avg_generation_time': avg_generation_time,
            'avg_document_length': avg_doc_length,
            'documentation_type_distribution': doc_type_counts,
            'workflow_format_distribution': format_counts,
            'created_by': 'Vatsal216',
            'timestamp': time.time()
        }


class CodeGenerator:
    """
    Generate executable code from workflow definitions.
    
    Creates Python, JavaScript, and other code implementations
    from workflow specifications.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:25:00 UTC
    """
    
    def __init__(self):
        """Initialize code generator"""
        self._code_templates = self._build_code_templates()
        self._generation_history: List[Dict[str, Any]] = []
        
        self._lock = threading.RLock()
        
        logger.info("CodeGenerator initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:25:00")
    
    def generate_code(self, 
                     workflow_def: Dict[str, Any],
                     language: CodeLanguage,
                     include_tests: bool = True,
                     include_documentation: bool = True) -> Dict[str, Any]:
        """
        Generate code from workflow definition.
        
        Args:
            workflow_def: Workflow definition
            language: Target programming language
            include_tests: Whether to include test files
            include_documentation: Whether to include inline documentation
            
        Returns:
            Generated code files and metadata
        """
        with self._lock:
            start_time = time.time()
            
            generated_files = []
            
            # Generate main implementation
            main_file = self._generate_main_implementation(workflow_def, language, include_documentation)
            generated_files.append(main_file)
            
            # Generate configuration files
            config_files = self._generate_configuration_files(workflow_def, language)
            generated_files.extend(config_files)
            
            # Generate test files
            if include_tests:
                test_files = self._generate_test_files(workflow_def, language)
                generated_files.extend(test_files)
            
            # Generate deployment files
            deployment_files = self._generate_deployment_files(workflow_def, language)
            generated_files.extend(deployment_files)
            
            # Record generation
            generation_time = time.time() - start_time
            self._record_code_generation(workflow_def, language, generated_files, generation_time)
            
            metrics.record_metric("code_generated_total", 1.0)
            metrics.record_metric("code_generation_time_seconds", generation_time)
            
            logger.info(f"Generated {language.value} code in {generation_time:.3f}s")
            
            return {
                'language': language.value,
                'files': [file.__dict__ for file in generated_files],
                'metadata': {
                    'workflow_format': workflow_def.get('format', 'unknown'),
                    'generation_time': generation_time,
                    'include_tests': include_tests,
                    'include_documentation': include_documentation,
                    'file_count': len(generated_files),
                    'created_by': 'Vatsal216',
                    'created_at': time.time()
                }
            }
    
    def _generate_main_implementation(self, 
                                    workflow_def: Dict[str, Any], 
                                    language: CodeLanguage,
                                    include_documentation: bool) -> GeneratedCode:
        """Generate main implementation file"""
        
        if language == CodeLanguage.PYTHON:
            return self._generate_python_implementation(workflow_def, include_documentation)
        elif language == CodeLanguage.JAVASCRIPT:
            return self._generate_javascript_implementation(workflow_def, include_documentation)
        elif language == CodeLanguage.TYPESCRIPT:
            return self._generate_typescript_implementation(workflow_def, include_documentation)
        else:
            return self._generate_generic_implementation(workflow_def, language, include_documentation)
    
    def _generate_python_implementation(self, 
                                      workflow_def: Dict[str, Any],
                                      include_documentation: bool) -> GeneratedCode:
        """Generate Python implementation"""
        
        workflow_format = workflow_def.get('format', 'generic')
        workflow_name = self._extract_workflow_name(workflow_def)
        
        if workflow_format == 'crewai':
            content = self._generate_crewai_python_code(workflow_def, include_documentation)
            dependencies = ['crewai', 'crewgraph-ai', 'langchain']
        elif workflow_format == 'langgraph':
            content = self._generate_langgraph_python_code(workflow_def, include_documentation)
            dependencies = ['langgraph', 'crewgraph-ai', 'langchain']
        else:
            content = self._generate_generic_python_code(workflow_def, include_documentation)
            dependencies = ['crewgraph-ai']
        
        filename = f"{self._sanitize_filename(workflow_name)}.py"
        
        return GeneratedCode(
            language=CodeLanguage.PYTHON,
            filename=filename,
            content=content,
            dependencies=dependencies,
            description=f"Python implementation of {workflow_name} workflow"
        )
    
    def _generate_crewai_python_code(self, 
                                   workflow_def: Dict[str, Any],
                                   include_documentation: bool) -> str:
        """Generate CrewAI Python code"""
        
        crew = workflow_def.get('crew', {})
        agents = crew.get('agents', [])
        tasks = crew.get('tasks', [])
        config = crew.get('config', {})
        
        code_parts = []
        
        # Header and imports
        if include_documentation:
            code_parts.extend([
                '"""',
                f'{config.get("workflow_name", "Workflow")} - CrewAI Implementation',
                '',
                'Auto-generated by CrewGraph AI Code Generator',
                'Created by: Vatsal216',
                f'Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}',
                '"""',
                ''
            ])
        
        code_parts.extend([
            'from crewai import Agent, Task, Crew, Process',
            'from crewgraph_ai import get_logger',
            'import os',
            '',
            'logger = get_logger(__name__)',
            ''
        ])
        
        # Agent definitions
        code_parts.append('# Agent Definitions')
        for i, agent in enumerate(agents):
            agent_var = f"agent_{i+1}"
            code_parts.extend([
                f'{agent_var} = Agent(',
                f'    role="{agent.get("role", "agent")}",',
                f'    goal="{agent.get("goal", "execute tasks")}",',
                f'    backstory="{agent.get("backstory", "I am a specialized agent")}",',
                f'    verbose={agent.get("verbose", True)},',
                f'    allow_delegation={agent.get("allow_delegation", False)},',
                f'    tools={agent.get("tools", [])},',
                ')',
                ''
            ])
        
        # Task definitions
        code_parts.append('# Task Definitions')
        for i, task in enumerate(tasks):
            task_var = f"task_{i+1}"
            agent_ref = f"agent_{i+1}" if i < len(agents) else "agent_1"
            
            code_parts.extend([
                f'{task_var} = Task(',
                f'    description="{task.get("description", "execute task")}",',
                f'    expected_output="{task.get("expected_output", "task completed")}",',
                f'    agent={agent_ref},',
                f'    tools={task.get("tools", [])},',
                ')',
                ''
            ])
        
        # Crew setup
        agent_list = ', '.join([f"agent_{i+1}" for i in range(len(agents))])
        task_list = ', '.join([f"task_{i+1}" for i in range(len(tasks))])
        
        code_parts.extend([
            '# Crew Setup',
            'crew = Crew(',
            f'    agents=[{agent_list}],',
            f'    tasks=[{task_list}],',
            f'    process=Process.{crew.get("process", "sequential").upper()},',
            f'    verbose={crew.get("verbose", 2)},',
            f'    memory={crew.get("memory", True)},',
            ')',
            '',
            '# Execution',
            'def run_workflow(inputs=None):',
            '    """Execute the workflow with optional inputs"""',
            '    try:',
            '        logger.info("Starting workflow execution")',
            '        result = crew.kickoff(inputs=inputs)',
            '        logger.info("Workflow completed successfully")',
            '        return result',
            '    except Exception as e:',
            '        logger.error(f"Workflow execution failed: {e}")',
            '        raise',
            '',
            'if __name__ == "__main__":',
            '    result = run_workflow()',
            '    print("Workflow Result:", result)'
        ])
        
        return '\n'.join(code_parts)
    
    def _generate_configuration_files(self, 
                                    workflow_def: Dict[str, Any], 
                                    language: CodeLanguage) -> List[GeneratedCode]:
        """Generate configuration files"""
        
        config_files = []
        
        if language == CodeLanguage.PYTHON:
            # requirements.txt
            workflow_format = workflow_def.get('format', 'generic')
            
            if workflow_format == 'crewai':
                requirements = ['crewai>=0.28.0', 'crewgraph-ai>=1.0.0', 'python-dotenv>=1.0.0']
            elif workflow_format == 'langgraph':
                requirements = ['langgraph>=0.0.40', 'crewgraph-ai>=1.0.0', 'python-dotenv>=1.0.0']
            else:
                requirements = ['crewgraph-ai>=1.0.0', 'python-dotenv>=1.0.0']
            
            config_files.append(GeneratedCode(
                language=CodeLanguage.PYTHON,
                filename='requirements.txt',
                content='\n'.join(requirements),
                dependencies=[],
                description='Python package requirements'
            ))
            
            # .env template
            env_content = [
                '# Environment Configuration',
                '# Copy this file to .env and fill in your values',
                '',
                'OPENAI_API_KEY=your_openai_api_key_here',
                'CREWGRAPH_LOG_LEVEL=INFO',
                'WORKFLOW_MAX_EXECUTION_TIME=3600',
                'WORKFLOW_RETRY_ATTEMPTS=3'
            ]
            
            config_files.append(GeneratedCode(
                language=CodeLanguage.PYTHON,
                filename='.env.template',
                content='\n'.join(env_content),
                dependencies=[],
                description='Environment variables template'
            ))
        
        elif language == CodeLanguage.JAVASCRIPT:
            # package.json
            package_json = {
                "name": self._sanitize_filename(self._extract_workflow_name(workflow_def)),
                "version": "1.0.0",
                "description": "Auto-generated workflow implementation",
                "main": "index.js",
                "scripts": {
                    "start": "node index.js",
                    "test": "jest",
                    "dev": "nodemon index.js"
                },
                "dependencies": {
                    "dotenv": "^16.0.0",
                    "axios": "^1.0.0"
                },
                "devDependencies": {
                    "jest": "^29.0.0",
                    "nodemon": "^2.0.0"
                }
            }
            
            config_files.append(GeneratedCode(
                language=CodeLanguage.JSON,
                filename='package.json',
                content=json.dumps(package_json, indent=2),
                dependencies=[],
                description='Node.js package configuration'
            ))
        
        return config_files
    
    def _generate_test_files(self, 
                           workflow_def: Dict[str, Any], 
                           language: CodeLanguage) -> List[GeneratedCode]:
        """Generate test files"""
        
        test_files = []
        workflow_name = self._extract_workflow_name(workflow_def)
        
        if language == CodeLanguage.PYTHON:
            test_content = self._generate_python_tests(workflow_def)
            test_files.append(GeneratedCode(
                language=CodeLanguage.PYTHON,
                filename=f'test_{self._sanitize_filename(workflow_name)}.py',
                content=test_content,
                dependencies=['pytest', 'pytest-asyncio'],
                description='Python test suite'
            ))
        
        return test_files
    
    def _generate_python_tests(self, workflow_def: Dict[str, Any]) -> str:
        """Generate Python test code"""
        
        workflow_name = self._extract_workflow_name(workflow_def)
        workflow_format = workflow_def.get('format', 'generic')
        
        test_parts = [
            '"""',
            f'Tests for {workflow_name} workflow',
            '',
            'Auto-generated by CrewGraph AI Code Generator',
            'Created by: Vatsal216',
            '"""',
            '',
            'import pytest',
            'import sys',
            'import os',
            '',
            '# Add parent directory to path',
            'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
            '',
            f'from {self._sanitize_filename(workflow_name)} import run_workflow',
            '',
            'class TestWorkflow:',
            '    """Test suite for the workflow"""',
            '',
            '    def test_workflow_import(self):',
            '        """Test that workflow can be imported"""',
            '        assert run_workflow is not None',
            '',
            '    def test_workflow_execution_with_empty_input(self):',
            '        """Test workflow execution with empty input"""',
            '        try:',
            '            result = run_workflow({})',
            '            assert result is not None',
            '        except Exception as e:',
            '            pytest.skip(f"Workflow requires specific inputs: {e}")',
            '',
            '    def test_workflow_execution_with_sample_data(self):',
            '        """Test workflow execution with sample data"""',
            '        sample_inputs = {',
            '            "test_input": "sample_value",',
            '            "data": "test_data"',
            '        }',
            '',
            '        try:',
            '            result = run_workflow(sample_inputs)',
            '            assert result is not None',
            '        except Exception as e:',
            '            pytest.skip(f"Workflow execution failed: {e}")',
            '',
            'if __name__ == "__main__":',
            '    pytest.main([__file__])'
        ]
        
        return '\n'.join(test_parts)
    
    def _generate_deployment_files(self, 
                                 workflow_def: Dict[str, Any], 
                                 language: CodeLanguage) -> List[GeneratedCode]:
        """Generate deployment files"""
        
        deployment_files = []
        
        # Docker files for containerization
        if language == CodeLanguage.PYTHON:
            dockerfile_content = [
                'FROM python:3.9-slim',
                '',
                'WORKDIR /app',
                '',
                'COPY requirements.txt .',
                'RUN pip install --no-cache-dir -r requirements.txt',
                '',
                'COPY . .',
                '',
                'CMD ["python", "workflow.py"]'
            ]
            
            deployment_files.append(GeneratedCode(
                language=CodeLanguage.PYTHON,
                filename='Dockerfile',
                content='\n'.join(dockerfile_content),
                dependencies=[],
                description='Docker container configuration'
            ))
            
            # docker-compose.yml
            compose_content = [
                'version: "3.8"',
                '',
                'services:',
                '  workflow:',
                '    build: .',
                '    environment:',
                '      - OPENAI_API_KEY=${OPENAI_API_KEY}',
                '      - CREWGRAPH_LOG_LEVEL=INFO',
                '    volumes:',
                '      - ./logs:/app/logs',
                '      - ./data:/app/data',
                '    restart: unless-stopped'
            ]
            
            deployment_files.append(GeneratedCode(
                language=CodeLanguage.YAML,
                filename='docker-compose.yml',
                content='\n'.join(compose_content),
                dependencies=[],
                description='Docker Compose configuration'
            ))
        
        return deployment_files
    
    def _extract_workflow_name(self, workflow_def: Dict[str, Any]) -> str:
        """Extract workflow name from definition"""
        # Try different possible locations for the name
        name_candidates = [
            workflow_def.get('name'),
            workflow_def.get('workflow_name'),
            workflow_def.get('crew', {}).get('config', {}).get('workflow_name'),
            workflow_def.get('graph', {}).get('config', {}).get('workflow_name'),
            workflow_def.get('workflow', {}).get('name')
        ]
        
        for name in name_candidates:
            if name and isinstance(name, str):
                return name
        
        return "workflow"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use in code generation"""
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', filename.lower())
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it's a valid Python identifier
        if not sanitized or sanitized[0].isdigit():
            sanitized = 'workflow_' + sanitized
        
        return sanitized
    
    def _build_code_templates(self) -> Dict[str, Dict]:
        """Build code generation templates"""
        return {
            'python': {
                'imports': [
                    'from crewai import Agent, Task, Crew',
                    'from crewgraph_ai import get_logger',
                    'import os'
                ],
                'agent_template': 'Agent(role="{role}", goal="{goal}", backstory="{backstory}")',
                'task_template': 'Task(description="{description}", agent={agent})'
            },
            'javascript': {
                'imports': [
                    'const { CrewAI } = require("crewai-js");',
                    'require("dotenv").config();'
                ]
            }
        }
    
    def _record_code_generation(self, 
                              workflow_def: Dict[str, Any],
                              language: CodeLanguage,
                              generated_files: List[GeneratedCode],
                              generation_time: float):
        """Record code generation for analysis"""
        record = {
            'timestamp': time.time(),
            'workflow_format': workflow_def.get('format', 'unknown'),
            'target_language': language.value,
            'file_count': len(generated_files),
            'total_code_lines': sum(file.content.count('\n') for file in generated_files),
            'generation_time': generation_time,
            'created_by': 'Vatsal216'
        }
        
        self._generation_history.append(record)
        
        # Keep only last 100 generations
        if len(self._generation_history) > 100:
            self._generation_history = self._generation_history[-100:]
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about code generation"""
        if not self._generation_history:
            return {'total_generations': 0}
        
        total_generations = len(self._generation_history)
        avg_generation_time = sum(g['generation_time'] for g in self._generation_history) / total_generations
        avg_file_count = sum(g['file_count'] for g in self._generation_history) / total_generations
        total_code_lines = sum(g['total_code_lines'] for g in self._generation_history)
        
        language_counts = {}
        format_counts = {}
        
        for generation in self._generation_history:
            language = generation['target_language']
            format_type = generation['workflow_format']
            
            language_counts[language] = language_counts.get(language, 0) + 1
            format_counts[format_type] = format_counts.get(format_type, 0) + 1
        
        return {
            'total_generations': total_generations,
            'avg_generation_time': avg_generation_time,
            'avg_files_per_generation': avg_file_count,
            'total_code_lines_generated': total_code_lines,
            'language_distribution': language_counts,
            'workflow_format_distribution': format_counts,
            'created_by': 'Vatsal216',
            'timestamp': time.time()
        }
=======
    Generates comprehensive documentation from workflow definitions,
    including user guides, technical documentation, and troubleshooting guides.
    """
    
    def __init__(self):
        """Initialize the documentation generator."""
        logger.info("DocumentationGenerator initialized")
    
    def generate_workflow_documentation(self, 
                                      workflow_definition: Dict[str, Any],
                                      execution_history: Optional[List[Dict[str, Any]]] = None) -> WorkflowDocumentation:
        """
        Generate comprehensive documentation for a workflow.
        
        Args:
            workflow_definition: Workflow structure and metadata
            execution_history: Historical execution data for insights
            
        Returns:
            Generated workflow documentation
        """
        # Generate title and overview
        title = self._generate_title(workflow_definition)
        overview = self._generate_overview(workflow_definition)
        
        # Document tasks
        tasks_docs = self._document_tasks(workflow_definition.get("tasks", []))
        
        # Create dependencies diagram
        dependencies_diagram = self._create_dependencies_diagram(
            workflow_definition.get("tasks", []),
            workflow_definition.get("dependencies", [])
        )
        
        # Generate usage instructions
        usage_instructions = self._generate_usage_instructions(workflow_definition)
        
        # Generate troubleshooting guide
        troubleshooting = self._generate_troubleshooting_guide(workflow_definition, execution_history)
        
        # Add performance notes
        performance_notes = self._generate_performance_notes(execution_history)
        
        documentation = WorkflowDocumentation(
            title=title,
            overview=overview,
            tasks_documentation=tasks_docs,
            dependencies_diagram=dependencies_diagram,
            usage_instructions=usage_instructions,
            troubleshooting=troubleshooting,
            performance_notes=performance_notes,
            generated_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Generated documentation for workflow: {title}")
        return documentation
    
    def generate_api_documentation(self, workflow_definition: Dict[str, Any]) -> str:
        """
        Generate API documentation for the workflow.
        
        Args:
            workflow_definition: Workflow definition
            
        Returns:
            API documentation in markdown format
        """
        docs = []
        
        docs.append(f"# {self._generate_title(workflow_definition)} API")
        docs.append("")
        docs.append("## Overview")
        docs.append(self._generate_overview(workflow_definition))
        docs.append("")
        
        # Input/Output specification
        docs.append("## Input Parameters")
        docs.append("```json")
        docs.append(self._generate_input_schema(workflow_definition))
        docs.append("```")
        docs.append("")
        
        docs.append("## Output Format")
        docs.append("```json")
        docs.append(self._generate_output_schema(workflow_definition))
        docs.append("```")
        docs.append("")
        
        # Usage examples
        docs.append("## Usage Examples")
        docs.append("```python")
        docs.append(self._generate_usage_examples(workflow_definition))
        docs.append("```")
        
        return "\n".join(docs)
    
    def generate_user_guide(self, workflow_definition: Dict[str, Any]) -> str:
        """
        Generate user-friendly guide for the workflow.
        
        Args:
            workflow_definition: Workflow definition
            
        Returns:
            User guide in markdown format
        """
        guide = []
        
        guide.append(f"# {self._generate_title(workflow_definition)} - User Guide")
        guide.append("")
        guide.append("## What This Workflow Does")
        guide.append(self._generate_user_friendly_overview(workflow_definition))
        guide.append("")
        
        guide.append("## How to Use")
        guide.append(self._generate_step_by_step_guide(workflow_definition))
        guide.append("")
        
        guide.append("## What to Expect")
        guide.append(self._generate_expectations_guide(workflow_definition))
        guide.append("")
        
        guide.append("## Common Issues and Solutions")
        troubleshooting = self._generate_troubleshooting_guide(workflow_definition, None)
        for issue in troubleshooting:
            guide.append(f"- {issue}")
        
        return "\n".join(guide)
    
    def _generate_title(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate a descriptive title for the workflow."""
        name = workflow_definition.get("name", "")
        if name and name != "Custom Workflow":
            return name
        
        # Generate from tasks
        tasks = workflow_definition.get("tasks", [])
        if tasks:
            first_task = tasks[0].get("description", "")
            if len(first_task) > 50:
                first_task = first_task[:47] + "..."
            return f"Workflow: {first_task}"
        
        return "Custom Workflow"
    
    def _generate_overview(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate workflow overview."""
        description = workflow_definition.get("description", "")
        if description:
            return description
        
        # Generate from tasks
        tasks = workflow_definition.get("tasks", [])
        task_count = len(tasks)
        
        if task_count == 0:
            return "This workflow contains no defined tasks."
        elif task_count == 1:
            return f"This workflow executes a single task: {tasks[0].get('description', 'Unknown task')}"
        else:
            return (f"This workflow executes {task_count} tasks in sequence, "
                   f"starting with '{tasks[0].get('description', 'Unknown task')}' "
                   f"and ending with '{tasks[-1].get('description', 'Unknown task')}'.")
    
    def _document_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate documentation for each task."""
        documented_tasks = []
        
        for i, task in enumerate(tasks, 1):
            task_doc = {
                "step": str(i),
                "name": task.get("id", f"Task {i}"),
                "description": task.get("description", "No description provided"),
                "type": task.get("type", "general"),
                "estimated_duration": f"{task.get('estimated_duration', 10)} minutes",
                "notes": self._generate_task_notes(task)
            }
            documented_tasks.append(task_doc)
        
        return documented_tasks
    
    def _generate_task_notes(self, task: Dict[str, Any]) -> str:
        """Generate additional notes for a task."""
        notes = []
        
        task_type = task.get("type", "")
        if task_type == "data_processing":
            notes.append("Requires input data to be properly formatted")
        elif task_type == "io":
            notes.append("May require file system or network access")
        elif task_type == "ml":
            notes.append("Machine learning task - may require significant compute resources")
        
        if task.get("suggested", False):
            notes.append("This task was automatically suggested for best practices")
        
        return "; ".join(notes) if notes else "No special requirements"
    
    def _create_dependencies_diagram(self, tasks: List[Dict[str, Any]], dependencies: List[Dict[str, str]]) -> str:
        """Create a text-based dependencies diagram."""
        if not tasks:
            return "No tasks defined"
        
        if not dependencies:
            # Sequential execution
            diagram = []
            for i, task in enumerate(tasks):
                task_name = task.get("id", f"Task {i+1}")
                if i == 0:
                    diagram.append(f"START → {task_name}")
                else:
                    diagram.append(f"      ↓")
                    diagram.append(f"      {task_name}")
            diagram.append("      ↓")
            diagram.append("      END")
            return "\n".join(diagram)
        
        # Build dependency graph representation
        diagram = ["Dependencies:"]
        for dep in dependencies:
            source = dep.get("source", "Unknown")
            target = dep.get("target", "Unknown")
            diagram.append(f"  {source} → {target}")
        
        return "\n".join(diagram)
    
    def _generate_usage_instructions(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate usage instructions."""
        instructions = []
        
        instructions.append("To execute this workflow:")
        instructions.append("1. Ensure all required dependencies are installed")
        instructions.append("2. Prepare input data according to the specifications")
        instructions.append("3. Execute the workflow using CrewGraph AI")
        instructions.append("4. Monitor execution progress and handle any errors")
        instructions.append("5. Review results and logs for completion")
        
        # Add specific instructions based on workflow type
        workflow_type = workflow_definition.get("workflow_type")
        if workflow_type:
            type_instructions = self._get_type_specific_instructions(workflow_type)
            if type_instructions:
                instructions.append("")
                instructions.append("Additional instructions for this workflow type:")
                instructions.extend(type_instructions)
        
        return "\n".join(instructions)
    
    def _get_type_specific_instructions(self, workflow_type: str) -> List[str]:
        """Get type-specific instructions."""
        type_guides = {
            "data_processing": [
                "- Ensure input data is accessible and in the correct format",
                "- Verify sufficient storage space for processed data",
                "- Consider data backup before processing"
            ],
            "analysis": [
                "- Prepare analysis parameters and criteria",
                "- Ensure output directory exists",
                "- Review analysis results for accuracy"
            ],
            "automation": [
                "- Verify automation triggers are properly configured",
                "- Test in a safe environment before production use",
                "- Set up monitoring and alerting"
            ]
        }
        
        return type_guides.get(workflow_type, [])
    
    def _generate_troubleshooting_guide(self, 
                                       workflow_definition: Dict[str, Any],
                                       execution_history: Optional[List[Dict[str, Any]]]) -> List[str]:
        """Generate troubleshooting guide."""
        troubleshooting = []
        
        # Common issues
        troubleshooting.extend([
            "Workflow fails to start: Check that all dependencies are installed and accessible",
            "Slow execution: Monitor system resources and consider reducing parallelism",
            "Task failures: Review task logs and verify input data format",
            "Memory errors: Reduce batch size or increase available memory",
            "Network timeouts: Check network connectivity and increase timeout values"
        ])
        
        # Add history-based troubleshooting
        if execution_history:
            common_errors = self._analyze_common_errors(execution_history)
            if common_errors:
                troubleshooting.append("Common issues based on execution history:")
                troubleshooting.extend(common_errors)
        
        return troubleshooting
    
    def _analyze_common_errors(self, execution_history: List[Dict[str, Any]]) -> List[str]:
        """Analyze execution history for common errors."""
        # This would analyze actual error patterns
        # For now, return placeholder guidance
        return [
            "Review recent execution logs for specific error patterns",
            "Check for resource constraints during peak usage times"
        ]
    
    def _generate_performance_notes(self, execution_history: Optional[List[Dict[str, Any]]]) -> List[str]:
        """Generate performance notes based on execution history."""
        if not execution_history:
            return ["No execution history available for performance analysis"]
        
        notes = []
        
        # Analyze execution times
        execution_times = [h.get("execution_time", 0) for h in execution_history if h.get("execution_time")]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            notes.append(f"Average execution time: {avg_time:.1f} seconds")
            
            if max(execution_times) > avg_time * 2:
                notes.append("Execution time varies significantly - consider optimization")
        
        # Analyze success rates
        success_rates = [h.get("success_rate", 1.0) for h in execution_history if h.get("success_rate") is not None]
        if success_rates:
            avg_success = sum(success_rates) / len(success_rates)
            notes.append(f"Average success rate: {avg_success:.1%}")
            
            if avg_success < 0.9:
                notes.append("Success rate could be improved with better error handling")
        
        return notes
    
    def _generate_user_friendly_overview(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate user-friendly overview."""
        tasks = workflow_definition.get("tasks", [])
        
        if not tasks:
            return "This workflow doesn't contain any tasks yet."
        
        overview = f"This workflow will help you complete {len(tasks)} main steps:\n"
        
        for i, task in enumerate(tasks[:3], 1):  # Show first 3 tasks
            description = task.get("description", "Unknown task")
            overview += f"{i}. {description}\n"
        
        if len(tasks) > 3:
            overview += f"... and {len(tasks) - 3} more steps"
        
        return overview
    
    def _generate_step_by_step_guide(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate step-by-step user guide."""
        guide = "Follow these steps to run the workflow:\n\n"
        guide += "1. **Prepare**: Make sure you have all necessary inputs ready\n"
        guide += "2. **Start**: Launch the workflow through CrewGraph AI\n"
        guide += "3. **Monitor**: Watch the progress and check for any alerts\n"
        guide += "4. **Complete**: Review the results when finished\n"
        
        return guide
    
    def _generate_expectations_guide(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate guide for what users should expect."""
        tasks = workflow_definition.get("tasks", [])
        estimated_duration = sum(task.get("estimated_duration", 10) for task in tasks)
        
        guide = f"**Time**: This workflow typically takes about {estimated_duration} minutes to complete.\n\n"
        guide += f"**Steps**: You'll see {len(tasks)} main steps being executed.\n\n"
        guide += "**Output**: The workflow will provide results and logs when completed.\n\n"
        
        return guide
    
    def _generate_input_schema(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate input schema for API documentation."""
        schema = {
            "input_data": "Required input data for the workflow",
            "parameters": {
                "timeout": "Maximum execution time in seconds (optional)",
                "parallel": "Enable parallel execution (optional, default: false)"
            }
        }
        
        import json
        return json.dumps(schema, indent=2)
    
    def _generate_output_schema(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate output schema for API documentation."""
        schema = {
            "status": "Execution status (success/failure)",
            "results": "Workflow execution results",
            "execution_time": "Total execution time in seconds",
            "task_results": "Individual task results",
            "logs": "Execution logs and messages"
        }
        
        import json
        return json.dumps(schema, indent=2)
    
    def _generate_usage_examples(self, workflow_definition: Dict[str, Any]) -> str:
        """Generate usage examples for API documentation."""
        workflow_name = workflow_definition.get("name", "workflow")
        
        example = f'''from crewgraph_ai import CrewGraph

# Initialize workflow
workflow = CrewGraph("{workflow_name}")

# Execute workflow
result = workflow.execute({{
    "input_data": "your_input_data_here"
}})

# Check results
if result.status == "success":
    print("Workflow completed successfully!")
    print(f"Results: {{result.results}}")
else:
    print(f"Workflow failed: {{result.error}}")'''
        
        return example
>>>>>>> origin/copilot/fix-71b1051d-a728-419e-a2a5-887da1cf638d
