"""
Intent Classifier for CrewGraph AI NLP Module

Classifies user intentions from natural language input to better
understand workflow requirements and context.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowIntent(Enum):
    """Types of workflow intentions."""
    CREATE_NEW = "create_new"
    MODIFY_EXISTING = "modify_existing"
    ANALYZE_DATA = "analyze_data"
    AUTOMATE_PROCESS = "automate_process"
    GENERATE_CONTENT = "generate_content"
    PROCESS_DATA = "process_data"
    INTEGRATE_SYSTEMS = "integrate_systems"
    MONITOR_PERFORMANCE = "monitor_performance"


@dataclass
class IntentClassification:
    """Result of intent classification."""
    intent: WorkflowIntent
    confidence: float
    supporting_keywords: List[str]
    context_clues: List[str]


class IntentClassifier:
    """
    Classifies user intentions from natural language input to understand
    what type of workflow they want to create or modify.
    """
    
    def __init__(self):
        """Initialize the intent classifier."""
        self.intent_keywords = self._initialize_intent_keywords()
        logger.info("IntentClassifier initialized")
    
    def classify_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> IntentClassification:
        """
        Classify the intent from user input.
        
        Args:
            text: User input text
            context: Additional context information
            
        Returns:
            Intent classification result
        """
        text_lower = text.lower()
        intent_scores = {}
        
        # Score each intent based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Determine best intent
        if intent_scores:
            best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x])
            confidence = min(0.95, intent_scores[best_intent] / 5.0)
            supporting_keywords = [kw for kw in self.intent_keywords[best_intent] if kw in text_lower]
        else:
            best_intent = WorkflowIntent.CREATE_NEW  # Default
            confidence = 0.5
            supporting_keywords = []
        
        # Extract context clues
        context_clues = self._extract_context_clues(text_lower)
        
        return IntentClassification(
            intent=WorkflowIntent(best_intent),
            confidence=confidence,
            supporting_keywords=supporting_keywords,
            context_clues=context_clues
        )
    
    def _initialize_intent_keywords(self) -> Dict[str, List[str]]:
        """Initialize keywords for each intent type."""
        return {
            "create_new": ["create", "build", "make", "new", "develop", "design"],
            "modify_existing": ["modify", "change", "update", "edit", "improve", "enhance"],
            "analyze_data": ["analyze", "examine", "study", "investigate", "explore", "insights"],
            "automate_process": ["automate", "schedule", "trigger", "streamline", "optimize"],
            "generate_content": ["generate", "create content", "write", "produce", "compose"],
            "process_data": ["process", "transform", "clean", "convert", "pipeline"],
            "integrate_systems": ["integrate", "connect", "sync", "merge", "combine"],
            "monitor_performance": ["monitor", "track", "measure", "watch", "observe"]
        }
    
    def _extract_context_clues(self, text: str) -> List[str]:
        """Extract additional context clues from text."""
        clues = []
        
        if "real-time" in text or "live" in text:
            clues.append("real-time")
        if "batch" in text or "scheduled" in text:
            clues.append("batch_processing")
        if "urgent" in text or "asap" in text:
            clues.append("high_priority")
        if "simple" in text or "basic" in text:
            clues.append("simple_workflow")
        if "complex" in text or "advanced" in text:
            clues.append("complex_workflow")
            
        return clues