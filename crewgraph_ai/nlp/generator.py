"""
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


class DocumentationGenerator:
    """
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