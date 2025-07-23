"""
Workflow Parser for CrewGraph AI NLP Module

This module provides natural language parsing capabilities to convert
descriptions into structured workflow definitions.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import re
import json
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


@dataclass
class ParsedWorkflow:
    """Represents a workflow parsed from natural language."""
    name: str
    description: str
    workflow_type: WorkflowType
    tasks: List[Dict[str, Any]]
    dependencies: List[Dict[str, str]]
    estimated_duration: int  # in minutes
    confidence_score: float
    metadata: Dict[str, Any]


class WorkflowParser:
    """
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
            
        Returns:
            Parsed workflow structure
        """
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
        
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
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