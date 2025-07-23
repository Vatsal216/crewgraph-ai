"""
Task Extractor for CrewGraph AI NLP Module

Extracts individual tasks and their characteristics from natural language descriptions.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedTask:
    """Represents a task extracted from natural language."""

    description: str
    task_type: str
    priority: str
    estimated_duration: int
    dependencies: List[str]
    resources_required: Dict[str, Any]
    confidence: float


class TaskExtractor:
    """
    Extracts individual tasks from natural language descriptions with
    detailed analysis of task characteristics and requirements.
    """

    def __init__(self):
        """Initialize the task extractor."""
        self.action_verbs = self._initialize_action_verbs()
        self.priority_indicators = self._initialize_priority_indicators()
        logger.info("TaskExtractor initialized")

    def extract_tasks(self, text: str) -> List[ExtractedTask]:
        """
        Extract tasks from natural language text.

        Args:
            text: Natural language description

        Returns:
            List of extracted tasks
        """
        # Split text into potential task sentences
        sentences = self._split_into_sentences(text)

        tasks = []
        for sentence in sentences:
            if self._is_task_sentence(sentence):
                task = self._analyze_task_sentence(sentence)
                if task:
                    tasks.append(task)

        logger.info(f"Extracted {len(tasks)} tasks from input")
        return tasks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for task analysis."""
        # Split on common sentence delimiters
        sentences = re.split(r"[.!?;]|\n", text)

        # Clean and filter sentences
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                cleaned.append(sentence)

        return cleaned

    def _is_task_sentence(self, sentence: str) -> bool:
        """Determine if a sentence describes a task."""
        sentence_lower = sentence.lower()

        # Check for action verbs
        has_action = any(verb in sentence_lower for verb in self.action_verbs)

        # Check for task indicators
        task_indicators = ["need to", "should", "must", "will", "then", "next", "first", "finally"]
        has_indicator = any(indicator in sentence_lower for indicator in task_indicators)

        return has_action or has_indicator

    def _analyze_task_sentence(self, sentence: str) -> Optional[ExtractedTask]:
        """Analyze a sentence to extract task details."""
        try:
            # Extract basic task description
            description = self._clean_task_description(sentence)

            # Classify task type
            task_type = self._classify_task_type(sentence)

            # Extract priority
            priority = self._extract_priority(sentence)

            # Estimate duration
            duration = self._estimate_duration(sentence)

            # Extract resource requirements
            resources = self._extract_resource_requirements(sentence)

            # Calculate confidence
            confidence = self._calculate_extraction_confidence(sentence)

            return ExtractedTask(
                description=description,
                task_type=task_type,
                priority=priority,
                estimated_duration=duration,
                dependencies=[],  # Dependencies extracted separately
                resources_required=resources,
                confidence=confidence,
            )

        except Exception as e:
            logger.warning(f"Error analyzing task sentence: {e}")
            return None

    def _clean_task_description(self, sentence: str) -> str:
        """Clean and normalize task description."""
        # Remove common prefixes
        prefixes = [
            "first",
            "then",
            "next",
            "after that",
            "finally",
            "we need to",
            "should",
            "must",
        ]

        cleaned = sentence.strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                if cleaned.startswith(","):
                    cleaned = cleaned[1:].strip()

        return cleaned

    def _classify_task_type(self, sentence: str) -> str:
        """Classify the type of task."""
        sentence_lower = sentence.lower()

        type_keywords = {
            "data_processing": ["process", "transform", "clean", "filter", "convert"],
            "analysis": ["analyze", "examine", "evaluate", "assess", "review"],
            "io": [
                "read",
                "write",
                "save",
                "load",
                "fetch",
                "download",
                "upload",
                "export",
                "import",
            ],
            "communication": ["send", "notify", "email", "message", "alert", "notify"],
            "validation": ["validate", "verify", "check", "test", "confirm"],
            "ml": ["predict", "model", "train", "classify", "recommend", "learn"],
            "monitoring": ["monitor", "track", "watch", "observe", "log"],
        }

        for task_type, keywords in type_keywords.items():
            if any(keyword in sentence_lower for keyword in keywords):
                return task_type

        return "general"

    def _extract_priority(self, sentence: str) -> str:
        """Extract task priority from sentence."""
        sentence_lower = sentence.lower()

        for priority, indicators in self.priority_indicators.items():
            if any(indicator in sentence_lower for indicator in indicators):
                return priority

        return "normal"

    def _estimate_duration(self, sentence: str) -> int:
        """Estimate task duration in minutes."""
        sentence_lower = sentence.lower()

        # Look for explicit time mentions
        time_patterns = [
            r"(\d+)\s*(?:minutes?|mins?)",
            r"(\d+)\s*(?:hours?|hrs?)",
            r"(\d+)\s*(?:seconds?|secs?)",
        ]

        for pattern in time_patterns:
            match = re.search(pattern, sentence_lower)
            if match:
                value = int(match.group(1))
                if "hour" in match.group(0):
                    return value * 60
                elif "second" in match.group(0):
                    return max(1, value // 60)
                else:
                    return value

        # Heuristic based on task complexity
        if any(word in sentence_lower for word in ["quick", "simple", "fast"]):
            return 2
        elif any(word in sentence_lower for word in ["complex", "detailed", "thorough"]):
            return 30
        elif any(word in sentence_lower for word in ["analyze", "process", "transform"]):
            return 15

        return 10  # Default

    def _extract_resource_requirements(self, sentence: str) -> Dict[str, Any]:
        """Extract resource requirements from sentence."""
        sentence_lower = sentence.lower()
        resources = {}

        # Check for memory requirements
        if any(word in sentence_lower for word in ["large", "big", "massive", "huge"]):
            resources["memory_intensive"] = True

        # Check for CPU requirements
        if any(word in sentence_lower for word in ["compute", "calculate", "process", "heavy"]):
            resources["cpu_intensive"] = True

        # Check for I/O requirements
        if any(word in sentence_lower for word in ["file", "database", "network", "api"]):
            resources["io_intensive"] = True

        return resources

    def _calculate_extraction_confidence(self, sentence: str) -> float:
        """Calculate confidence in task extraction."""
        confidence = 0.5  # Base confidence

        sentence_lower = sentence.lower()

        # Increase confidence for clear action verbs
        action_count = sum(1 for verb in self.action_verbs if verb in sentence_lower)
        confidence += min(0.3, action_count * 0.1)

        # Increase confidence for specific details
        if len(sentence.split()) > 5:
            confidence += 0.1

        # Decrease confidence for very vague sentences
        vague_words = ["something", "stuff", "things", "maybe", "possibly"]
        if any(word in sentence_lower for word in vague_words):
            confidence -= 0.2

        return max(0.1, min(0.95, confidence))

    def _initialize_action_verbs(self) -> List[str]:
        """Initialize list of action verbs that indicate tasks."""
        return [
            "analyze",
            "process",
            "create",
            "generate",
            "send",
            "receive",
            "transform",
            "convert",
            "validate",
            "verify",
            "check",
            "test",
            "save",
            "load",
            "read",
            "write",
            "update",
            "modify",
            "delete",
            "calculate",
            "compute",
            "evaluate",
            "assess",
            "review",
            "examine",
            "collect",
            "gather",
            "fetch",
            "download",
            "upload",
            "export",
            "import",
            "sync",
            "merge",
            "combine",
            "split",
            "filter",
            "sort",
            "monitor",
            "track",
            "watch",
            "log",
            "notify",
            "alert",
            "schedule",
        ]

    def _initialize_priority_indicators(self) -> Dict[str, List[str]]:
        """Initialize priority indicators."""
        return {
            "high": ["urgent", "critical", "immediately", "asap", "priority", "important"],
            "medium": ["soon", "moderately", "somewhat"],
            "low": ["later", "eventually", "when possible", "low priority", "optional"],
        }
