"""
Interactive Flow Selection Agent for CrewGraph AI
Intelligent agent that analyzes user requirements and recommends optimal workflow approaches.

Author: Vatsal216
Created: 2025-07-23
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import CrewGraphError
from ..config.enterprise_config import get_enterprise_config

logger = get_logger(__name__)


class WorkflowType(Enum):
    """Types of workflows"""
    AI_DRIVEN = "ai_driven"
    TRADITIONAL = "traditional"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class ComplexityLevel(Enum):
    """Workflow complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class IndustryDomain(Enum):
    """Industry domains"""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"
    GOVERNMENT = "government"
    GENERAL = "general"


@dataclass
class UserRequirements:
    """User requirements for workflow selection"""
    # Basic requirements
    use_case: str
    industry: IndustryDomain
    expected_volume: int  # requests per day
    team_size: int
    budget_tier: str  # "low", "medium", "high"
    
    # Technical requirements
    preferred_approach: Optional[str] = None  # "ai", "traditional", "no_preference"
    existing_systems: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # AI preferences
    ai_experience_level: str = "beginner"  # "beginner", "intermediate", "expert"
    llm_preferences: List[str] = field(default_factory=list)
    
    # Operational preferences
    deployment_environment: str = "cloud"  # "cloud", "on_premise", "hybrid"
    monitoring_needs: List[str] = field(default_factory=list)
    scaling_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowRecommendation:
    """Workflow recommendation with scoring"""
    workflow_type: WorkflowType
    complexity: ComplexityLevel
    confidence_score: float
    reasoning: List[str]
    estimated_setup_time: str
    estimated_cost: str
    pros: List[str]
    cons: List[str]
    recommended_components: List[str]
    configuration_template: Dict[str, Any]
    next_steps: List[str]


class FlowSelectorAgent:
    """
    Intelligent agent for workflow flow selection and recommendation.
    
    Analyzes user requirements and provides personalized recommendations for
    AI vs traditional workflow approaches with detailed reasoning and setup guidance.
    """
    
    def __init__(self, config=None):
        self.config = config or get_enterprise_config()
        self.knowledge_base = self._initialize_knowledge_base()
        self.recommendation_history: List[Tuple[UserRequirements, WorkflowRecommendation]] = []
        
        logger.info("FlowSelectorAgent initialized with knowledge base")
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base for recommendations"""
        return {
            "use_case_patterns": {
                "content_generation": {
                    "ai_score": 0.9,
                    "traditional_score": 0.3,
                    "complexity_modifier": 0.1,
                    "recommended_components": ["nlp", "llm_providers", "content_templates"]
                },
                "data_processing": {
                    "ai_score": 0.6,
                    "traditional_score": 0.8,
                    "complexity_modifier": 0.2,
                    "recommended_components": ["data_pipeline", "validation", "transformation"]
                },
                "customer_support": {
                    "ai_score": 0.8,
                    "traditional_score": 0.4,
                    "complexity_modifier": 0.3,
                    "recommended_components": ["nlp", "knowledge_base", "response_generation"]
                },
                "document_analysis": {
                    "ai_score": 0.9,
                    "traditional_score": 0.2,
                    "complexity_modifier": 0.4,
                    "recommended_components": ["document_processing", "classification", "extraction"]
                },
                "workflow_automation": {
                    "ai_score": 0.5,
                    "traditional_score": 0.9,
                    "complexity_modifier": 0.1,
                    "recommended_components": ["orchestration", "scheduling", "monitoring"]
                },
                "research_analysis": {
                    "ai_score": 0.8,
                    "traditional_score": 0.3,
                    "complexity_modifier": 0.5,
                    "recommended_components": ["research_tools", "summarization", "knowledge_graph"]
                }
            },
            "industry_modifiers": {
                IndustryDomain.FINANCE: {"compliance_weight": 0.3, "ai_caution": 0.2},
                IndustryDomain.HEALTHCARE: {"compliance_weight": 0.4, "ai_caution": 0.3},
                IndustryDomain.GOVERNMENT: {"compliance_weight": 0.5, "ai_caution": 0.4},
                IndustryDomain.TECHNOLOGY: {"compliance_weight": 0.1, "ai_caution": -0.1},
                IndustryDomain.GENERAL: {"compliance_weight": 0.1, "ai_caution": 0.0}
            },
            "volume_thresholds": {
                "low": 100,
                "medium": 1000,
                "high": 10000,
                "enterprise": 100000
            },
            "complexity_indicators": {
                "simple": {"max_components": 3, "max_integrations": 2, "max_team_size": 5},
                "moderate": {"max_components": 8, "max_integrations": 5, "max_team_size": 15},
                "complex": {"max_components": 15, "max_integrations": 10, "max_team_size": 50},
                "enterprise": {"max_components": 999, "max_integrations": 999, "max_team_size": 999}
            }
        }
    
    async def analyze_requirements(self, requirements: UserRequirements) -> WorkflowRecommendation:
        """Analyze user requirements and generate workflow recommendation"""
        logger.info(f"Analyzing requirements for use case: {requirements.use_case}")
        
        # Calculate scores for different workflow types
        ai_score = self._calculate_ai_score(requirements)
        traditional_score = self._calculate_traditional_score(requirements)
        hybrid_score = (ai_score + traditional_score) / 2
        
        # Determine recommended workflow type
        scores = {
            WorkflowType.AI_DRIVEN: ai_score,
            WorkflowType.TRADITIONAL: traditional_score,
            WorkflowType.HYBRID: hybrid_score
        }
        
        recommended_type = max(scores, key=scores.get)
        confidence_score = scores[recommended_type]
        
        # Determine complexity level
        complexity = self._determine_complexity(requirements)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(requirements, recommended_type, scores)
        
        # Create recommendation
        recommendation = WorkflowRecommendation(
            workflow_type=recommended_type,
            complexity=complexity,
            confidence_score=confidence_score,
            reasoning=reasoning,
            estimated_setup_time=self._estimate_setup_time(complexity, recommended_type),
            estimated_cost=self._estimate_cost(requirements, recommended_type),
            pros=self._get_pros(recommended_type, requirements),
            cons=self._get_cons(recommended_type, requirements),
            recommended_components=self._get_recommended_components(requirements, recommended_type),
            configuration_template=self._generate_configuration_template(requirements, recommended_type),
            next_steps=self._generate_next_steps(requirements, recommended_type)
        )
        
        # Store in history for learning
        self.recommendation_history.append((requirements, recommendation))
        
        logger.info(f"Generated recommendation: {recommended_type.value} with {confidence_score:.2f} confidence")
        
        return recommendation
    
    def _calculate_ai_score(self, requirements: UserRequirements) -> float:
        """Calculate score for AI-driven approach"""
        base_score = 0.5
        
        # Use case pattern matching
        use_case_lower = requirements.use_case.lower()
        for pattern, data in self.knowledge_base["use_case_patterns"].items():
            if pattern in use_case_lower:
                base_score = data["ai_score"]
                break
        
        # Industry modifiers
        industry_data = self.knowledge_base["industry_modifiers"].get(
            requirements.industry, {"ai_caution": 0.0}
        )
        base_score -= industry_data.get("ai_caution", 0.0)
        
        # Team experience modifier
        experience_modifiers = {
            "beginner": -0.2,
            "intermediate": 0.0,
            "expert": 0.2
        }
        base_score += experience_modifiers.get(requirements.ai_experience_level, 0.0)
        
        # Budget modifier
        budget_modifiers = {
            "low": -0.1,
            "medium": 0.0,
            "high": 0.1
        }
        base_score += budget_modifiers.get(requirements.budget_tier, 0.0)
        
        # Volume modifier (AI scales better with high volume)
        if requirements.expected_volume > 10000:
            base_score += 0.2
        elif requirements.expected_volume > 1000:
            base_score += 0.1
        
        # Compliance requirements (reduce AI score if many compliance needs)
        if len(requirements.compliance_requirements) > 3:
            base_score -= 0.2
        elif len(requirements.compliance_requirements) > 1:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_traditional_score(self, requirements: UserRequirements) -> float:
        """Calculate score for traditional approach"""
        base_score = 0.5
        
        # Use case pattern matching
        use_case_lower = requirements.use_case.lower()
        for pattern, data in self.knowledge_base["use_case_patterns"].items():
            if pattern in use_case_lower:
                base_score = data["traditional_score"]
                break
        
        # Industry modifiers (traditional is safer for regulated industries)
        industry_data = self.knowledge_base["industry_modifiers"].get(
            requirements.industry, {"compliance_weight": 0.1}
        )
        compliance_weight = industry_data.get("compliance_weight", 0.1)
        base_score += compliance_weight * len(requirements.compliance_requirements)
        
        # Existing systems favor traditional (easier integration)
        if len(requirements.existing_systems) > 2:
            base_score += 0.2
        elif len(requirements.existing_systems) > 0:
            base_score += 0.1
        
        # Team size (traditional workflows are easier to manage with small teams)
        if requirements.team_size < 5:
            base_score += 0.1
        elif requirements.team_size > 20:
            base_score -= 0.1
        
        # Budget considerations (traditional often has lower ongoing costs)
        if requirements.budget_tier == "low":
            base_score += 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def _determine_complexity(self, requirements: UserRequirements) -> ComplexityLevel:
        """Determine workflow complexity based on requirements"""
        complexity_factors = {
            "integrations": len(requirements.existing_systems),
            "compliance": len(requirements.compliance_requirements),
            "team_size": requirements.team_size,
            "volume": requirements.expected_volume,
            "monitoring": len(requirements.monitoring_needs)
        }
        
        # Calculate complexity score
        score = 0
        
        # Integration complexity
        if complexity_factors["integrations"] > 5:
            score += 3
        elif complexity_factors["integrations"] > 2:
            score += 2
        elif complexity_factors["integrations"] > 0:
            score += 1
        
        # Compliance complexity
        if complexity_factors["compliance"] > 3:
            score += 3
        elif complexity_factors["compliance"] > 1:
            score += 2
        elif complexity_factors["compliance"] > 0:
            score += 1
        
        # Team size complexity
        if complexity_factors["team_size"] > 50:
            score += 3
        elif complexity_factors["team_size"] > 15:
            score += 2
        elif complexity_factors["team_size"] > 5:
            score += 1
        
        # Volume complexity
        if complexity_factors["volume"] > 100000:
            score += 3
        elif complexity_factors["volume"] > 10000:
            score += 2
        elif complexity_factors["volume"] > 1000:
            score += 1
        
        # Monitoring complexity
        if complexity_factors["monitoring"] > 5:
            score += 2
        elif complexity_factors["monitoring"] > 2:
            score += 1
        
        # Map score to complexity level
        if score >= 10:
            return ComplexityLevel.ENTERPRISE
        elif score >= 6:
            return ComplexityLevel.COMPLEX
        elif score >= 3:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE
    
    def _generate_reasoning(self, requirements: UserRequirements, recommended_type: WorkflowType, scores: Dict[WorkflowType, float]) -> List[str]:
        """Generate reasoning for the recommendation"""
        reasoning = []
        
        # Primary recommendation reasoning
        if recommended_type == WorkflowType.AI_DRIVEN:
            reasoning.append(f"AI-driven approach scored highest ({scores[recommended_type]:.2f}) for your use case")
            
            if "content" in requirements.use_case.lower():
                reasoning.append("Content generation tasks benefit significantly from AI capabilities")
            if "analysis" in requirements.use_case.lower():
                reasoning.append("Analysis workflows leverage AI's pattern recognition strengths")
            if requirements.expected_volume > 1000:
                reasoning.append("High volume requirements favor AI automation and scaling")
            
        elif recommended_type == WorkflowType.TRADITIONAL:
            reasoning.append(f"Traditional approach scored highest ({scores[recommended_type]:.2f}) for your requirements")
            
            if len(requirements.compliance_requirements) > 2:
                reasoning.append("Multiple compliance requirements favor traditional, well-tested approaches")
            if len(requirements.existing_systems) > 1:
                reasoning.append("Existing system integrations are often easier with traditional workflows")
            if requirements.budget_tier == "low":
                reasoning.append("Budget constraints favor traditional approaches with lower ongoing costs")
        
        elif recommended_type == WorkflowType.HYBRID:
            reasoning.append(f"Hybrid approach offers the best balance ({scores[recommended_type]:.2f}) for your needs")
            reasoning.append("Combines AI strengths with traditional reliability and control")
        
        # Industry-specific reasoning
        if requirements.industry in [IndustryDomain.FINANCE, IndustryDomain.HEALTHCARE]:
            reasoning.append(f"{requirements.industry.value} industry requires careful consideration of AI vs traditional approaches")
        
        # Experience level reasoning
        if requirements.ai_experience_level == "beginner":
            reasoning.append("Beginner AI experience suggests starting with simpler, more traditional approaches")
        elif requirements.ai_experience_level == "expert":
            reasoning.append("Expert AI experience enables adoption of advanced AI-driven workflows")
        
        return reasoning
    
    def _estimate_setup_time(self, complexity: ComplexityLevel, workflow_type: WorkflowType) -> str:
        """Estimate setup time for the recommended workflow"""
        base_times = {
            ComplexityLevel.SIMPLE: {
                WorkflowType.AI_DRIVEN: "1-2 weeks",
                WorkflowType.TRADITIONAL: "1 week",
                WorkflowType.HYBRID: "2-3 weeks"
            },
            ComplexityLevel.MODERATE: {
                WorkflowType.AI_DRIVEN: "3-6 weeks",
                WorkflowType.TRADITIONAL: "2-4 weeks",
                WorkflowType.HYBRID: "4-8 weeks"
            },
            ComplexityLevel.COMPLEX: {
                WorkflowType.AI_DRIVEN: "2-4 months",
                WorkflowType.TRADITIONAL: "1-3 months",
                WorkflowType.HYBRID: "3-6 months"
            },
            ComplexityLevel.ENTERPRISE: {
                WorkflowType.AI_DRIVEN: "6-12 months",
                WorkflowType.TRADITIONAL: "3-9 months",
                WorkflowType.HYBRID: "9-18 months"
            }
        }
        
        return base_times[complexity][workflow_type]
    
    def _estimate_cost(self, requirements: UserRequirements, workflow_type: WorkflowType) -> str:
        """Estimate cost tier for the recommended workflow"""
        volume_multiplier = {
            "low": 1.0,
            "medium": 1.5,
            "high": 2.5
        }
        
        base_cost_tiers = {
            WorkflowType.AI_DRIVEN: {
                "low": "$100-500/month",
                "medium": "$500-2000/month", 
                "high": "$2000+/month"
            },
            WorkflowType.TRADITIONAL: {
                "low": "$50-200/month",
                "medium": "$200-800/month",
                "high": "$800+/month"
            },
            WorkflowType.HYBRID: {
                "low": "$200-800/month",
                "medium": "$800-3000/month",
                "high": "$3000+/month"
            }
        }
        
        volume_tier = "low"
        if requirements.expected_volume > 10000:
            volume_tier = "high"
        elif requirements.expected_volume > 1000:
            volume_tier = "medium"
        
        return base_cost_tiers[workflow_type][volume_tier]
    
    def _get_pros(self, workflow_type: WorkflowType, requirements: UserRequirements) -> List[str]:
        """Get pros for the recommended workflow type"""
        pros_map = {
            WorkflowType.AI_DRIVEN: [
                "Highly automated and scalable",
                "Adapts and improves over time",
                "Handles complex, unstructured data well",
                "Reduces manual intervention",
                "State-of-the-art capabilities in language and reasoning"
            ],
            WorkflowType.TRADITIONAL: [
                "Predictable and reliable behavior",
                "Easier to debug and troubleshoot",
                "Lower ongoing operational costs",
                "Better compliance and audit trails",
                "Simpler team training requirements"
            ],
            WorkflowType.HYBRID: [
                "Combines best of both approaches",
                "Flexible and adaptable to changing needs",
                "Risk mitigation through diversification",
                "Gradual AI adoption path",
                "Balanced cost and capability profile"
            ]
        }
        
        return pros_map[workflow_type]
    
    def _get_cons(self, workflow_type: WorkflowType, requirements: UserRequirements) -> List[str]:
        """Get cons for the recommended workflow type"""
        cons_map = {
            WorkflowType.AI_DRIVEN: [
                "Higher setup complexity and costs",
                "Requires AI expertise on team",
                "Less predictable behavior",
                "Potential compliance challenges",
                "Dependency on external AI services"
            ],
            WorkflowType.TRADITIONAL: [
                "Limited adaptability to new scenarios",
                "Requires more manual configuration",
                "May not scale as efficiently",
                "Misses advanced AI capabilities",
                "Higher maintenance overhead for complex logic"
            ],
            WorkflowType.HYBRID: [
                "Increased overall complexity",
                "Requires expertise in both approaches",
                "Potentially higher total costs",
                "More integration points to maintain",
                "Longer initial setup time"
            ]
        }
        
        return cons_map[workflow_type]
    
    def _get_recommended_components(self, requirements: UserRequirements, workflow_type: WorkflowType) -> List[str]:
        """Get recommended components for the workflow"""
        base_components = ["orchestration", "monitoring", "logging"]
        
        # Use case specific components
        use_case_lower = requirements.use_case.lower()
        for pattern, data in self.knowledge_base["use_case_patterns"].items():
            if pattern in use_case_lower:
                base_components.extend(data["recommended_components"])
                break
        
        # Workflow type specific components
        if workflow_type == WorkflowType.AI_DRIVEN:
            base_components.extend(["llm_providers", "ai_agents", "embedding_store"])
        elif workflow_type == WorkflowType.TRADITIONAL:
            base_components.extend(["rule_engine", "data_validation", "scheduling"])
        elif workflow_type == WorkflowType.HYBRID:
            base_components.extend(["ai_agents", "rule_engine", "smart_routing"])
        
        # Requirements specific components
        if len(requirements.compliance_requirements) > 0:
            base_components.extend(["audit_logging", "security_manager"])
        
        if requirements.expected_volume > 1000:
            base_components.extend(["auto_scaling", "load_balancer"])
        
        if requirements.deployment_environment == "on_premise":
            base_components.extend(["local_deployment", "data_encryption"])
        
        return list(set(base_components))  # Remove duplicates
    
    def _generate_configuration_template(self, requirements: UserRequirements, workflow_type: WorkflowType) -> Dict[str, Any]:
        """Generate configuration template for the recommended workflow"""
        template = {
            "workflow_type": workflow_type.value,
            "environment": requirements.deployment_environment,
            "scaling": {
                "max_concurrent_workflows": min(10, requirements.expected_volume // 100),
                "auto_scaling_enabled": requirements.expected_volume > 1000,
                "target_cpu_utilization": 70
            },
            "monitoring": {
                "metrics_enabled": True,
                "logging_level": "INFO",
                "alert_webhooks": []
            },
            "security": {
                "encryption_enabled": len(requirements.compliance_requirements) > 0,
                "audit_logging": len(requirements.compliance_requirements) > 0,
                "rate_limiting": True
            }
        }
        
        # AI-specific configuration
        if workflow_type in [WorkflowType.AI_DRIVEN, WorkflowType.HYBRID]:
            template["llm_providers"] = {
                "default_provider": "openai",
                "fallback_enabled": True,
                "rate_limits": {
                    "requests_per_minute": 60,
                    "tokens_per_minute": 1000
                }
            }
        
        # Industry-specific configuration
        if requirements.industry in [IndustryDomain.FINANCE, IndustryDomain.HEALTHCARE]:
            template["security"]["encryption_enabled"] = True
            template["security"]["audit_logging"] = True
            template["monitoring"]["logging_level"] = "DEBUG"
        
        return template
    
    def _generate_next_steps(self, requirements: UserRequirements, workflow_type: WorkflowType) -> List[str]:
        """Generate next steps for implementation"""
        steps = [
            "Review and validate the recommended configuration",
            "Set up development environment",
            "Configure basic workflow structure"
        ]
        
        if workflow_type in [WorkflowType.AI_DRIVEN, WorkflowType.HYBRID]:
            steps.extend([
                "Obtain API keys for chosen LLM providers",
                "Set up AI agent configurations",
                "Test AI components with sample data"
            ])
        
        if len(requirements.existing_systems) > 0:
            steps.append("Plan integration with existing systems")
        
        if len(requirements.compliance_requirements) > 0:
            steps.extend([
                "Implement security and compliance measures",
                "Set up audit logging and monitoring"
            ])
        
        steps.extend([
            "Create initial workflow templates",
            "Set up monitoring and alerting",
            "Plan rollout and team training",
            "Schedule regular review and optimization"
        ])
        
        return steps
    
    def get_interactive_questionnaire(self) -> Dict[str, Any]:
        """Get interactive questionnaire for gathering requirements"""
        return {
            "sections": [
                {
                    "title": "Basic Information",
                    "questions": [
                        {
                            "id": "use_case",
                            "type": "text",
                            "question": "What is your primary use case or business problem?",
                            "required": True,
                            "placeholder": "e.g., Content generation, Customer support automation, Document analysis"
                        },
                        {
                            "id": "industry",
                            "type": "select",
                            "question": "What industry are you in?",
                            "required": True,
                            "options": [domain.value for domain in IndustryDomain]
                        },
                        {
                            "id": "team_size",
                            "type": "number",
                            "question": "How many people are on your team?",
                            "required": True,
                            "min": 1,
                            "max": 1000
                        },
                        {
                            "id": "expected_volume",
                            "type": "number",
                            "question": "How many requests/operations do you expect per day?",
                            "required": True,
                            "min": 1,
                            "help": "This helps determine scaling requirements"
                        }
                    ]
                },
                {
                    "title": "Technical Preferences",
                    "questions": [
                        {
                            "id": "preferred_approach",
                            "type": "select",
                            "question": "Do you have a preference for AI vs traditional approaches?",
                            "required": False,
                            "options": ["ai", "traditional", "no_preference"],
                            "labels": ["Prefer AI-driven", "Prefer traditional", "No preference"]
                        },
                        {
                            "id": "ai_experience_level",
                            "type": "select",
                            "question": "What is your team's AI/ML experience level?",
                            "required": True,
                            "options": ["beginner", "intermediate", "expert"]
                        },
                        {
                            "id": "deployment_environment",
                            "type": "select",
                            "question": "Where do you plan to deploy the solution?",
                            "required": True,
                            "options": ["cloud", "on_premise", "hybrid"]
                        }
                    ]
                },
                {
                    "title": "Requirements & Constraints",
                    "questions": [
                        {
                            "id": "budget_tier",
                            "type": "select",
                            "question": "What is your budget tier?",
                            "required": True,
                            "options": ["low", "medium", "high"],
                            "labels": ["Low (<$1K/month)", "Medium ($1K-10K/month)", "High (>$10K/month)"]
                        },
                        {
                            "id": "compliance_requirements",
                            "type": "multiselect",
                            "question": "What compliance requirements do you have?",
                            "required": False,
                            "options": ["GDPR", "HIPAA", "SOX", "PCI-DSS", "SOC2", "ISO27001", "None"]
                        },
                        {
                            "id": "existing_systems",
                            "type": "text",
                            "question": "List any existing systems you need to integrate with (comma-separated)",
                            "required": False,
                            "placeholder": "e.g., Salesforce, SAP, Custom API"
                        }
                    ]
                }
            ]
        }
    
    async def process_questionnaire_responses(self, responses: Dict[str, Any]) -> WorkflowRecommendation:
        """Process questionnaire responses and generate recommendation"""
        # Parse responses into UserRequirements
        requirements = UserRequirements(
            use_case=responses.get("use_case", ""),
            industry=IndustryDomain(responses.get("industry", "general")),
            team_size=int(responses.get("team_size", 1)),
            expected_volume=int(responses.get("expected_volume", 1)),
            budget_tier=responses.get("budget_tier", "low"),
            preferred_approach=responses.get("preferred_approach"),
            ai_experience_level=responses.get("ai_experience_level", "beginner"),
            deployment_environment=responses.get("deployment_environment", "cloud"),
            existing_systems=responses.get("existing_systems", "").split(",") if responses.get("existing_systems") else [],
            compliance_requirements=responses.get("compliance_requirements", []) if isinstance(responses.get("compliance_requirements"), list) else []
        )
        
        return await self.analyze_requirements(requirements)
    
    def get_recommendation_summary(self, recommendation: WorkflowRecommendation) -> Dict[str, Any]:
        """Get a summary of the recommendation for display"""
        return {
            "recommendation": {
                "workflow_type": recommendation.workflow_type.value,
                "complexity": recommendation.complexity.value,
                "confidence_score": round(recommendation.confidence_score, 2)
            },
            "estimates": {
                "setup_time": recommendation.estimated_setup_time,
                "cost": recommendation.estimated_cost
            },
            "key_points": {
                "top_pros": recommendation.pros[:3],
                "main_concerns": recommendation.cons[:2],
                "essential_components": recommendation.recommended_components[:5]
            },
            "immediate_actions": recommendation.next_steps[:3]
        }


# Global flow selector agent
_global_flow_selector: Optional[FlowSelectorAgent] = None


def get_flow_selector() -> FlowSelectorAgent:
    """Get global flow selector agent"""
    global _global_flow_selector
    
    if _global_flow_selector is None:
        _global_flow_selector = FlowSelectorAgent()
    
    return _global_flow_selector


# Convenience functions
async def analyze_workflow_requirements(requirements: UserRequirements) -> WorkflowRecommendation:
    """Analyze workflow requirements using global flow selector"""
    selector = get_flow_selector()
    return await selector.analyze_requirements(requirements)


def get_workflow_questionnaire() -> Dict[str, Any]:
    """Get interactive questionnaire for workflow analysis"""
    selector = get_flow_selector()
    return selector.get_interactive_questionnaire()