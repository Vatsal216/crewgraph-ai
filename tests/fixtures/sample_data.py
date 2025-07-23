"""
Sample data for CrewGraph AI tests

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

from typing import Dict, List, Any
import json
import uuid
from datetime import datetime, timedelta


def generate_sample_agents(count: int = 5) -> List[Dict[str, Any]]:
    """Generate sample agent configurations"""
    agent_templates = [
        {
            "role": "Research Analyst",
            "goal": "Conduct thorough research on assigned topics",
            "backstory": "Expert researcher with 10+ years experience in data analysis and information gathering",
            "skills": ["research", "data_analysis", "report_writing"],
            "tools": ["search_tool", "data_analyzer", "report_generator"]
        },
        {
            "role": "Content Writer",
            "goal": "Create engaging and informative content",
            "backstory": "Professional writer specializing in technical and business content creation",
            "skills": ["writing", "editing", "content_strategy"],
            "tools": ["text_processor", "grammar_checker", "style_guide"]
        },
        {
            "role": "Data Scientist", 
            "goal": "Analyze complex datasets and extract insights",
            "backstory": "PhD in Statistics with expertise in machine learning and predictive modeling",
            "skills": ["statistics", "machine_learning", "data_visualization"],
            "tools": ["python_interpreter", "ml_toolkit", "visualization_engine"]
        },
        {
            "role": "Project Manager",
            "goal": "Coordinate and manage project execution",
            "backstory": "Certified PMP with experience managing cross-functional AI projects",
            "skills": ["project_management", "coordination", "planning"],
            "tools": ["project_tracker", "scheduling_tool", "communication_hub"]
        },
        {
            "role": "Quality Assurance",
            "goal": "Ensure output quality and accuracy",
            "backstory": "Quality expert with deep understanding of AI system validation",
            "skills": ["quality_control", "testing", "validation"],
            "tools": ["quality_checker", "validation_suite", "metrics_analyzer"]
        }
    ]
    
    agents = []
    for i in range(count):
        template = agent_templates[i % len(agent_templates)]
        agent = template.copy()
        agent["id"] = f"agent_{i+1:03d}"
        agent["created_at"] = datetime.now().isoformat()
        agents.append(agent)
    
    return agents


def generate_sample_tasks(count: int = 10) -> List[Dict[str, Any]]:
    """Generate sample task configurations"""
    task_templates = [
        {
            "description": "Research the latest trends in AI and machine learning",
            "expected_output": "Comprehensive research report with key findings and trends",
            "category": "research",
            "priority": "high",
            "estimated_duration": 120  # minutes
        },
        {
            "description": "Analyze customer feedback data to identify improvement areas",
            "expected_output": "Data analysis report with actionable insights",
            "category": "analysis",
            "priority": "medium", 
            "estimated_duration": 90
        },
        {
            "description": "Write a technical blog post about the research findings",
            "expected_output": "Well-structured blog post ready for publication",
            "category": "content_creation",
            "priority": "medium",
            "estimated_duration": 60
        },
        {
            "description": "Create visualizations for the analysis results",
            "expected_output": "Interactive charts and graphs showing key metrics",
            "category": "visualization",
            "priority": "low",
            "estimated_duration": 45
        },
        {
            "description": "Validate the accuracy of generated content",
            "expected_output": "Quality assurance report with validation results",
            "category": "quality_control",
            "priority": "high",
            "estimated_duration": 30
        },
        {
            "description": "Plan the project timeline and resource allocation",
            "expected_output": "Detailed project plan with milestones and dependencies",
            "category": "planning",
            "priority": "high",
            "estimated_duration": 75
        },
        {
            "description": "Collect and preprocess training data",
            "expected_output": "Clean, preprocessed dataset ready for model training",
            "category": "data_preparation",
            "priority": "medium",
            "estimated_duration": 100
        },
        {
            "description": "Train machine learning model on prepared data",
            "expected_output": "Trained model with performance metrics",
            "category": "model_training",
            "priority": "high",
            "estimated_duration": 150
        },
        {
            "description": "Generate summary report of all findings",
            "expected_output": "Executive summary highlighting key insights and recommendations",
            "category": "reporting",
            "priority": "high",
            "estimated_duration": 40
        },
        {
            "description": "Review and optimize workflow performance",
            "expected_output": "Performance optimization recommendations",
            "category": "optimization",
            "priority": "low",
            "estimated_duration": 35
        }
    ]
    
    tasks = []
    for i in range(count):
        template = task_templates[i % len(task_templates)]
        task = template.copy()
        task["id"] = f"task_{i+1:03d}"
        task["created_at"] = datetime.now().isoformat()
        
        # Add random dependencies for some tasks
        if i > 0 and i % 3 == 0:
            num_deps = min(2, i)
            task["dependencies"] = [f"task_{j+1:03d}" for j in range(i-num_deps, i)]
        else:
            task["dependencies"] = []
        
        tasks.append(task)
    
    return tasks


# Export commonly used data
SAMPLE_AGENTS = generate_sample_agents(3)
SAMPLE_TASKS = generate_sample_tasks(5)

# Quick access to basic test data
BASIC_WORKFLOW_DATA = {
    "agents": SAMPLE_AGENTS[:2],
    "tasks": SAMPLE_TASKS[:2]
}


if __name__ == "__main__":
    print("Sample test data available:")
    print(f"- {len(SAMPLE_AGENTS)} sample agents")
    print(f"- {len(SAMPLE_TASKS)} sample tasks")