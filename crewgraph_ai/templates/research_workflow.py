"""
Research Workflow Template - Pre-built template for research automation

This template provides a comprehensive framework for automating research processes with:
- Topic research and source identification
- Information gathering and validation
- Analysis and synthesis
- Report generation and documentation

Features:
- Multi-source research capabilities
- Automated fact-checking and validation
- Comprehensive analysis and insights
- Professional research documentation
- Citation management and formatting

Created by: Vatsal216
Date: 2025-07-23
"""

from typing import Dict, List, Any
from crewai import Agent

from .workflow_templates import (
    WorkflowTemplate, 
    TemplateMetadata, 
    TemplateParameter, 
    TemplateStep,
    TemplateCategory
)
from ..core.agents import AgentWrapper
from ..core.tasks import TaskWrapper
from ..core.orchestrator import GraphOrchestrator
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResearchWorkflowTemplate(WorkflowTemplate):
    """
    Template for creating automated research workflows.
    
    This template provides a comprehensive framework for conducting automated
    research with information gathering, validation, analysis, and documentation.
    """
    
    def _define_metadata(self) -> TemplateMetadata:
        """Define template metadata."""
        return TemplateMetadata(
            name="research_workflow",
            description="Comprehensive research automation with information gathering, validation, analysis, and reporting",
            version="1.0.0",
            category=TemplateCategory.RESEARCH,
            author="Vatsal216",
            tags=["research", "automation", "analysis", "documentation", "investigation"],
            complexity="complex",
            estimated_time="30-60 minutes",
            requirements=[
                "Research topic definition",
                "Access to information sources",
                "Quality validation criteria",
                "Output format specification"
            ],
            examples=[
                {
                    "name": "Market Research",
                    "description": "Comprehensive market analysis for a new product",
                    "parameters": {
                        "research_topic": "AI chatbot market trends 2024",
                        "research_depth": "comprehensive",
                        "source_types": ["academic", "industry", "news"],
                        "analysis_focus": "market_trends"
                    }
                },
                {
                    "name": "Competitive Analysis",
                    "description": "Research competitors in a specific industry",
                    "parameters": {
                        "research_topic": "cloud computing providers comparison",
                        "research_depth": "detailed",
                        "source_types": ["industry", "reviews", "financial"],
                        "analysis_focus": "competitive_landscape"
                    }
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        """Define template parameters."""
        return [
            TemplateParameter(
                name="research_topic",
                description="Main research topic or question to investigate",
                param_type="str",
                required=True,
                examples=["AI in healthcare", "renewable energy trends", "fintech innovations"]
            ),
            TemplateParameter(
                name="research_depth",
                description="Depth of research to conduct",
                param_type="str",
                required=True,
                default_value="detailed",
                validation_rules={"allowed_values": ["overview", "detailed", "comprehensive", "expert"]},
                examples=["overview", "detailed", "comprehensive"]
            ),
            TemplateParameter(
                name="source_types",
                description="Types of sources to include in research",
                param_type="list",
                required=True,
                default_value=["academic", "industry", "news"],
                validation_rules={"allowed_values": ["academic", "industry", "news", "reports", "reviews", "financial", "government", "social"]},
                examples=[["academic", "industry"], ["news", "reports", "reviews"]]
            ),
            TemplateParameter(
                name="analysis_focus",
                description="Primary focus of the analysis",
                param_type="str",
                required=True,
                default_value="general_insights",
                validation_rules={"allowed_values": [
                    "general_insights", "market_trends", "competitive_landscape", 
                    "technology_assessment", "risk_analysis", "opportunity_identification"
                ]},
                examples=["market_trends", "competitive_landscape", "technology_assessment"]
            ),
            TemplateParameter(
                name="time_horizon",
                description="Time horizon for the research (historical and future)",
                param_type="str",
                required=False,
                default_value="current",
                validation_rules={"allowed_values": ["historical", "current", "future", "comprehensive"]},
                examples=["current", "future", "comprehensive"]
            ),
            TemplateParameter(
                name="geographic_scope",
                description="Geographic scope of the research",
                param_type="str",
                required=False,
                default_value="global",
                validation_rules={"allowed_values": ["local", "national", "regional", "global"]},
                examples=["national", "global", "regional"]
            ),
            TemplateParameter(
                name="language_preferences",
                description="Preferred languages for source materials",
                param_type="list",
                required=False,
                default_value=["english"],
                examples=[["english"], ["english", "spanish"], ["english", "chinese", "japanese"]]
            ),
            TemplateParameter(
                name="fact_checking",
                description="Enable comprehensive fact-checking and validation",
                param_type="bool",
                required=False,
                default_value=True
            ),
            TemplateParameter(
                name="citation_style",
                description="Citation style for references",
                param_type="str",
                required=False,
                default_value="apa",
                validation_rules={"allowed_values": ["apa", "mla", "chicago", "harvard", "ieee"]},
                examples=["apa", "mla", "chicago"]
            ),
            TemplateParameter(
                name="max_sources",
                description="Maximum number of sources to research",
                param_type="int",
                required=False,
                default_value=50,
                validation_rules={"min_value": 10, "max_value": 500}
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        """Define workflow steps."""
        return [
            TemplateStep(
                step_id="topic_analysis",
                name="Topic Analysis",
                description="Analyze research topic and define research strategy",
                agent_role="Research Coordinator",
                task_description="Break down research topic, identify key concepts, and plan research strategy",
                inputs=["research_topic", "research_depth", "analysis_focus"],
                outputs=["research_plan", "key_concepts", "search_strategies"],
                tools=["topic_analyzer", "concept_extractor"],
                configuration={"analysis_depth": "comprehensive"}
            ),
            TemplateStep(
                step_id="source_identification",
                name="Source Identification",
                description="Identify and catalog relevant information sources",
                agent_role="Information Specialist",
                task_description="Find and evaluate potential information sources based on research plan",
                inputs=["research_plan", "source_types", "geographic_scope"],
                outputs=["source_catalog", "source_priorities"],
                dependencies=["topic_analysis"],
                tools=["source_finder", "credibility_assessor"],
                configuration={"source_diversity": True}
            ),
            TemplateStep(
                step_id="information_gathering",
                name="Information Gathering",
                description="Collect information from identified sources",
                agent_role="Research Assistant",
                task_description="Systematically gather information from prioritized sources",
                inputs=["source_catalog", "key_concepts", "max_sources"],
                outputs=["raw_information", "source_metadata"],
                dependencies=["source_identification"],
                tools=["web_scraper", "document_extractor", "api_connector"],
                configuration={"parallel_processing": True}
            ),
            TemplateStep(
                step_id="fact_validation",
                name="Fact Validation",
                description="Validate and verify collected information",
                agent_role="Fact Checker",
                task_description="Cross-reference facts, identify contradictions, and assess reliability",
                inputs=["raw_information", "source_metadata"],
                outputs=["validated_information", "credibility_scores"],
                dependencies=["information_gathering"],
                tools=["fact_checker", "source_validator", "contradiction_detector"],
                configuration={"validation_threshold": "high"},
                optional=False  # Can be skipped if fact_checking=False
            ),
            TemplateStep(
                step_id="content_analysis",
                name="Content Analysis",
                description="Analyze and synthesize the validated information",
                agent_role="Research Analyst",
                task_description="Identify patterns, trends, and insights from the collected information",
                inputs=["validated_information", "analysis_focus", "time_horizon"],
                outputs=["analysis_results", "key_insights", "trends"],
                dependencies=["fact_validation"],
                tools=["content_analyzer", "trend_detector", "pattern_recognizer"],
                configuration={"analysis_type": "comprehensive"}
            ),
            TemplateStep(
                step_id="synthesis_documentation",
                name="Synthesis & Documentation",
                description="Synthesize findings into comprehensive research documentation",
                agent_role="Research Writer",
                task_description="Create structured research report with findings, analysis, and conclusions",
                inputs=["analysis_results", "key_insights", "citation_style"],
                outputs=["research_report", "executive_summary"],
                dependencies=["content_analysis"],
                tools=["report_writer", "citation_manager", "document_formatter"],
                configuration={"include_methodology": True}
            ),
            TemplateStep(
                step_id="quality_review",
                name="Quality Review",
                description="Review and validate the research output quality",
                agent_role="Senior Researcher",
                task_description="Conduct comprehensive quality review and suggest improvements",
                inputs=["research_report", "executive_summary"],
                outputs=["quality_assessment", "final_report"],
                dependencies=["synthesis_documentation"],
                tools=["quality_checker", "peer_reviewer"],
                configuration={"review_criteria": "academic_standards"}
            )
        ]
    
    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, AgentWrapper]:
        """Create agents for the research workflow."""
        agents = {}
        
        # Research Coordinator Agent
        research_coordinator = Agent(
            role="Research Coordinator",
            goal="Plan and coordinate comprehensive research activities",
            backstory="""You are an experienced research coordinator with expertise in 
            planning complex research projects. You excel at breaking down research topics, 
            identifying key concepts, and developing systematic research strategies.""",
            verbose=True,
            allow_delegation=True
        )
        agents["research_coordinator"] = AgentWrapper(
            name="ResearchCoordinator",
            role="Research Coordinator",
            crew_agent=research_coordinator
        )
        
        # Information Specialist Agent
        information_specialist = Agent(
            role="Information Specialist",
            goal="Identify and evaluate the best information sources",
            backstory="""You are a skilled information specialist with deep knowledge of 
            information sources across various domains. You excel at finding credible, 
            relevant sources and assessing their quality and reliability.""",
            verbose=True,
            allow_delegation=False
        )
        agents["information_specialist"] = AgentWrapper(
            name="InformationSpecialist",
            role="Information Specialist",
            crew_agent=information_specialist
        )
        
        # Research Assistant Agent
        research_assistant = Agent(
            role="Research Assistant",
            goal="Efficiently gather information from multiple sources",
            backstory="""You are a diligent research assistant with expertise in information 
            gathering from diverse sources. You are systematic, thorough, and skilled at 
            extracting relevant information while maintaining detailed records.""",
            verbose=True,
            allow_delegation=False
        )
        agents["research_assistant"] = AgentWrapper(
            name="ResearchAssistant",
            role="Research Assistant",
            crew_agent=research_assistant
        )
        
        # Fact Checker Agent
        fact_checker = Agent(
            role="Fact Checker",
            goal="Ensure accuracy and reliability of collected information",
            backstory="""You are a meticulous fact checker with expertise in validating 
            information from multiple sources. You excel at cross-referencing facts, 
            identifying contradictions, and assessing source credibility.""",
            verbose=True,
            allow_delegation=False
        )
        agents["fact_checker"] = AgentWrapper(
            name="FactChecker",
            role="Fact Checker",
            crew_agent=fact_checker
        )
        
        # Research Analyst Agent
        research_analyst = Agent(
            role="Research Analyst",
            goal="Analyze information and extract meaningful insights",
            backstory="""You are a skilled research analyst with expertise in data analysis, 
            pattern recognition, and insight generation. You excel at synthesizing complex 
            information and identifying trends and relationships.""",
            verbose=True,
            allow_delegation=False
        )
        agents["research_analyst"] = AgentWrapper(
            name="ResearchAnalyst",
            role="Research Analyst",
            crew_agent=research_analyst
        )
        
        # Research Writer Agent
        research_writer = Agent(
            role="Research Writer",
            goal="Create clear, comprehensive research documentation",
            backstory="""You are an experienced research writer with expertise in academic 
            and professional writing. You excel at organizing complex information into 
            clear, well-structured reports with proper citations and formatting.""",
            verbose=True,
            allow_delegation=False
        )
        agents["research_writer"] = AgentWrapper(
            name="ResearchWriter",
            role="Research Writer",
            crew_agent=research_writer
        )
        
        # Senior Researcher Agent
        senior_researcher = Agent(
            role="Senior Researcher",
            goal="Ensure research quality and provide expert review",
            backstory="""You are a senior researcher with extensive experience in research 
            methodology and quality assurance. You provide expert review and ensure that 
            research meets the highest academic and professional standards.""",
            verbose=True,
            allow_delegation=True
        )
        agents["senior_researcher"] = AgentWrapper(
            name="SeniorResearcher",
            role="Senior Researcher",
            crew_agent=senior_researcher
        )
        
        logger.info(f"Created {len(agents)} agents for research workflow")
        return agents
    
    def _create_tasks(self, 
                     params: Dict[str, Any], 
                     agents: Dict[str, AgentWrapper]) -> Dict[str, TaskWrapper]:
        """Create tasks for the research workflow."""
        tasks = {}
        
        # Topic Analysis Task
        tasks["topic_analysis"] = TaskWrapper(
            task_id="topic_analysis",
            description=f"""
            Analyze the research topic: "{params['research_topic']}"
            
            Requirements:
            - Break down the topic into key concepts and sub-topics
            - Define research scope for {params['research_depth']} analysis
            - Focus on {params['analysis_focus']} perspective
            - Time horizon: {params.get('time_horizon', 'current')}
            - Geographic scope: {params.get('geographic_scope', 'global')}
            - Create comprehensive research plan and strategy
            
            Output: Detailed research plan with key concepts and search strategies.
            """,
            agent=agents["research_coordinator"],
            expected_output="Comprehensive research plan with clearly defined scope and methodology"
        )
        
        # Source Identification Task
        tasks["source_identification"] = TaskWrapper(
            task_id="source_identification",
            description=f"""
            Identify and catalog relevant information sources for the research.
            
            Requirements:
            - Source types to include: {', '.join(params['source_types'])}
            - Language preferences: {', '.join(params.get('language_preferences', ['english']))}
            - Geographic scope: {params.get('geographic_scope', 'global')}
            - Maximum sources: {params.get('max_sources', 50)}
            - Assess source credibility and relevance
            - Prioritize sources based on quality and relevance
            
            Output: Catalog of prioritized sources with credibility assessments.
            """,
            agent=agents["information_specialist"],
            expected_output="Comprehensive source catalog with priority rankings and credibility scores"
        )
        
        # Information Gathering Task
        tasks["information_gathering"] = TaskWrapper(
            task_id="information_gathering",
            description="""
            Systematically gather information from identified sources.
            
            Requirements:
            - Collect information from prioritized sources
            - Extract relevant content based on key concepts
            - Maintain detailed source metadata and citations
            - Organize information by topic and relevance
            - Ensure comprehensive coverage of research scope
            
            Output: Organized collection of information with complete source metadata.
            """,
            agent=agents["research_assistant"],
            expected_output="Comprehensive information collection with detailed source tracking"
        )
        
        # Fact Validation Task (conditional)
        if params.get('fact_checking', True):
            tasks["fact_validation"] = TaskWrapper(
                task_id="fact_validation",
                description="""
                Validate and verify the collected information for accuracy and reliability.
                
                Requirements:
                - Cross-reference facts across multiple sources
                - Identify contradictions and inconsistencies
                - Assess source credibility and bias
                - Flag questionable or unverified claims
                - Generate credibility scores for information
                
                Output: Validated information with credibility assessments and conflict resolution.
                """,
                agent=agents["fact_checker"],
                expected_output="Validated information with detailed credibility analysis and conflict resolution"
            )
        
        # Content Analysis Task
        tasks["content_analysis"] = TaskWrapper(
            task_id="content_analysis",
            description=f"""
            Analyze and synthesize the validated information to generate insights.
            
            Requirements:
            - Focus on {params['analysis_focus']} analysis
            - Identify patterns, trends, and relationships
            - Generate key insights and findings
            - Consider time horizon: {params.get('time_horizon', 'current')}
            - Assess implications and significance
            - Support findings with evidence
            
            Output: Comprehensive analysis with key insights and supporting evidence.
            """,
            agent=agents["research_analyst"],
            expected_output="Detailed analysis with clear insights and evidence-based conclusions"
        )
        
        # Synthesis Documentation Task
        tasks["synthesis_documentation"] = TaskWrapper(
            task_id="synthesis_documentation",
            description=f"""
            Create comprehensive research documentation with findings and analysis.
            
            Requirements:
            - Use {params.get('citation_style', 'apa')} citation style
            - Include executive summary with key findings
            - Provide detailed methodology section
            - Present analysis results clearly
            - Include recommendations and implications
            - Ensure professional formatting and presentation
            
            Output: Complete research report with executive summary.
            """,
            agent=agents["research_writer"],
            expected_output="Professional research report with executive summary and proper citations"
        )
        
        # Quality Review Task
        tasks["quality_review"] = TaskWrapper(
            task_id="quality_review",
            description="""
            Conduct comprehensive quality review of the research output.
            
            Requirements:
            - Review methodology and approach
            - Validate findings and conclusions
            - Check citation accuracy and completeness
            - Assess report clarity and organization
            - Identify areas for improvement
            - Ensure adherence to research standards
            
            Output: Quality assessment and final reviewed report.
            """,
            agent=agents["senior_researcher"],
            expected_output="Final high-quality research report with comprehensive quality assessment"
        )
        
        logger.info(f"Created {len(tasks)} tasks for research workflow")
        return tasks
    
    def _configure_workflow_structure(self,
                                    orchestrator: GraphOrchestrator,
                                    params: Dict[str, Any],
                                    agents: Dict[str, AgentWrapper],
                                    tasks: Dict[str, TaskWrapper]):
        """Configure the workflow structure and dependencies."""
        
        # Set the entry point
        orchestrator.set_entry_point("topic_analysis")
        
        # Build the linear workflow with dependencies
        orchestrator.add_edge("topic_analysis", "source_identification")
        orchestrator.add_edge("source_identification", "information_gathering")
        
        # Add fact validation if enabled
        if params.get('fact_checking', True) and 'fact_validation' in tasks:
            orchestrator.add_edge("information_gathering", "fact_validation")
            orchestrator.add_edge("fact_validation", "content_analysis")
        else:
            orchestrator.add_edge("information_gathering", "content_analysis")
        
        # Continue with analysis and documentation
        orchestrator.add_edge("content_analysis", "synthesis_documentation")
        orchestrator.add_edge("synthesis_documentation", "quality_review")
        
        # Add error handling for critical steps
        for task_id in tasks.keys():
            orchestrator.add_error_handler(task_id, self._create_error_handler(task_id, params))
        
        # Add conditional logic for quality gates
        orchestrator.add_conditional_edge(
            "quality_review",
            self._quality_gate_condition,
            {"approved": "END", "revision_needed": "synthesis_documentation"}
        )
        
        logger.info("Research workflow structure configured")
    
    def _create_error_handler(self, task_id: str, params: Dict[str, Any]) -> callable:
        """Create error handler for specific task."""
        def error_handler(error, context):
            logger.error(f"Error in {task_id}: {error}")
            
            # Implement retry logic for information gathering tasks
            if task_id in ["source_identification", "information_gathering"]:
                retry_count = context.get("retry_count", 0)
                if retry_count < 2:
                    context["retry_count"] = retry_count + 1
                    logger.info(f"Retrying {task_id} (attempt {retry_count + 1})")
                    return "retry"
            
            # For critical failures, attempt recovery
            if task_id == "fact_validation" and not params.get('fact_checking', True):
                logger.warning("Fact validation failed, continuing without validation")
                return "skip"
            
            return "continue"
        
        return error_handler
    
    def _quality_gate_condition(self, state: Dict[str, Any]) -> str:
        """Determine if quality review passed or needs revision."""
        quality_assessment = state.get("quality_assessment", {})
        quality_score = quality_assessment.get("overall_score", 0)
        
        # If quality score is acceptable, approve
        if quality_score >= 85:
            return "approved"
        
        # Check revision count to avoid infinite loops
        revision_count = state.get("revision_count", 0)
        if revision_count >= 2:
            logger.warning("Maximum revisions reached, proceeding with current version")
            return "approved"
        
        state["revision_count"] = revision_count + 1
        return "revision_needed"