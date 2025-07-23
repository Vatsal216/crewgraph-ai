"""
Content Generation Template - Pre-built template for content creation workflows

This template provides a comprehensive framework for automated content creation with:
- Content planning and strategy
- Research and information gathering
- Content creation and development
- Review and quality assurance
- Publication and distribution

Features:
- Multi-format content creation
- Automated research integration
- Quality review processes
- SEO optimization
- Brand consistency checks

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


class ContentGenerationTemplate(WorkflowTemplate):
    """
    Template for creating automated content generation workflows.
    
    This template provides a comprehensive framework for content creation
    with planning, research, writing, review, and optimization processes.
    """
    
    def _define_metadata(self) -> TemplateMetadata:
        """Define template metadata."""
        return TemplateMetadata(
            name="content_generation",
            description="Comprehensive content creation workflow with planning, research, writing, review, and optimization",
            version="1.0.0",
            category=TemplateCategory.CONTENT_GENERATION,
            author="Vatsal216",
            tags=["content", "writing", "marketing", "seo", "creation", "automation"],
            complexity="medium",
            estimated_time="20-45 minutes",
            requirements=[
                "Content topic and objectives",
                "Target audience definition",
                "Brand guidelines (optional)",
                "Distribution channel requirements"
            ],
            examples=[
                {
                    "name": "Blog Post Creation",
                    "description": "Create an SEO-optimized blog post with research",
                    "parameters": {
                        "content_topic": "Benefits of AI in small business",
                        "content_type": "blog_post",
                        "target_audience": "small business owners",
                        "content_length": "medium"
                    }
                },
                {
                    "name": "Social Media Campaign",
                    "description": "Create multi-platform social media content",
                    "parameters": {
                        "content_topic": "Product launch announcement",
                        "content_type": "social_media",
                        "target_audience": "technology enthusiasts",
                        "platforms": ["twitter", "linkedin", "instagram"]
                    }
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        """Define template parameters."""
        return [
            TemplateParameter(
                name="content_topic",
                description="Main topic or subject for the content",
                param_type="str",
                required=True,
                examples=["AI in healthcare", "sustainable living tips", "digital marketing trends"]
            ),
            TemplateParameter(
                name="content_type",
                description="Type of content to create",
                param_type="str",
                required=True,
                default_value="blog_post",
                validation_rules={"allowed_values": [
                    "blog_post", "article", "social_media", "newsletter", 
                    "white_paper", "case_study", "product_description", "email_campaign"
                ]},
                examples=["blog_post", "article", "social_media"]
            ),
            TemplateParameter(
                name="target_audience",
                description="Primary target audience for the content",
                param_type="str",
                required=True,
                examples=["business professionals", "students", "technology enthusiasts", "general public"]
            ),
            TemplateParameter(
                name="content_length",
                description="Desired length of the content",
                param_type="str",
                required=False,
                default_value="medium",
                validation_rules={"allowed_values": ["short", "medium", "long", "comprehensive"]},
                examples=["short", "medium", "long"]
            ),
            TemplateParameter(
                name="tone",
                description="Tone and style for the content",
                param_type="str",
                required=False,
                default_value="professional",
                validation_rules={"allowed_values": [
                    "professional", "casual", "friendly", "authoritative", 
                    "conversational", "technical", "persuasive"
                ]},
                examples=["professional", "casual", "friendly"]
            ),
            TemplateParameter(
                name="content_goals",
                description="Primary goals for the content",
                param_type="list",
                required=False,
                default_value=["inform", "engage"],
                validation_rules={"allowed_values": [
                    "inform", "engage", "persuade", "educate", "entertain", 
                    "convert", "build_awareness", "drive_traffic"
                ]},
                examples=[["inform", "engage"], ["persuade", "convert"]]
            ),
            TemplateParameter(
                name="seo_optimization",
                description="Enable SEO optimization for the content",
                param_type="bool",
                required=False,
                default_value=True
            ),
            TemplateParameter(
                name="include_research",
                description="Include background research in content creation",
                param_type="bool",
                required=False,
                default_value=True
            ),
            TemplateParameter(
                name="review_cycles",
                description="Number of review and revision cycles",
                param_type="int",
                required=False,
                default_value=2,
                validation_rules={"min_value": 1, "max_value": 5}
            ),
            TemplateParameter(
                name="brand_guidelines",
                description="Brand guidelines and style requirements",
                param_type="dict",
                required=False,
                default_value={},
                examples=[{"brand_voice": "friendly", "avoid_terms": ["cheap", "discount"]}]
            ),
            TemplateParameter(
                name="platforms",
                description="Target platforms for content distribution (for social media content)",
                param_type="list",
                required=False,
                default_value=["general"],
                validation_rules={"allowed_values": [
                    "general", "twitter", "linkedin", "facebook", "instagram", 
                    "youtube", "tiktok", "blog", "website", "email"
                ]},
                examples=[["twitter", "linkedin"], ["instagram", "facebook"]]
            ),
            TemplateParameter(
                name="call_to_action",
                description="Include call-to-action in the content",
                param_type="str",
                required=False,
                default_value="",
                examples=["Visit our website", "Contact us today", "Download our guide"]
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        """Define workflow steps."""
        return [
            TemplateStep(
                step_id="content_planning",
                name="Content Planning",
                description="Plan content structure, strategy, and approach",
                agent_role="Content Strategist",
                task_description="Develop comprehensive content plan and strategy",
                inputs=["content_topic", "content_type", "target_audience", "content_goals"],
                outputs=["content_plan", "content_outline", "strategy_brief"],
                tools=["content_planner", "audience_analyzer"],
                configuration={"planning_depth": "comprehensive"}
            ),
            TemplateStep(
                step_id="research_gathering",
                name="Research Gathering",
                description="Gather relevant information and insights for content creation",
                agent_role="Content Researcher",
                task_description="Research topic, gather supporting information and current insights",
                inputs=["content_topic", "content_plan", "target_audience"],
                outputs=["research_findings", "key_insights", "supporting_data"],
                dependencies=["content_planning"],
                tools=["research_tools", "trend_analyzer"],
                configuration={"research_depth": "focused"},
                optional=False  # Can be skipped if include_research=False
            ),
            TemplateStep(
                step_id="content_creation",
                name="Content Creation",
                description="Create the primary content based on plan and research",
                agent_role="Content Writer",
                task_description="Write engaging, high-quality content according to specifications",
                inputs=["content_outline", "research_findings", "tone", "content_length"],
                outputs=["draft_content", "content_structure"],
                dependencies=["research_gathering"],
                tools=["content_writer", "grammar_checker"],
                configuration={"writing_style": "adaptive"}
            ),
            TemplateStep(
                step_id="seo_optimization",
                name="SEO Optimization",
                description="Optimize content for search engines and discoverability",
                agent_role="SEO Specialist",
                task_description="Optimize content for SEO with keywords, meta descriptions, and structure",
                inputs=["draft_content", "content_topic", "target_audience"],
                outputs=["optimized_content", "seo_recommendations"],
                dependencies=["content_creation"],
                tools=["seo_analyzer", "keyword_optimizer"],
                configuration={"optimization_level": "comprehensive"},
                optional=False  # Can be skipped if seo_optimization=False
            ),
            TemplateStep(
                step_id="quality_review",
                name="Quality Review",
                description="Review content for quality, accuracy, and brand alignment",
                agent_role="Content Editor",
                task_description="Comprehensive review for quality, accuracy, and brand consistency",
                inputs=["optimized_content", "brand_guidelines", "content_goals"],
                outputs=["reviewed_content", "feedback_report"],
                dependencies=["seo_optimization"],
                tools=["quality_checker", "brand_validator"],
                configuration={"review_criteria": "comprehensive"}
            ),
            TemplateStep(
                step_id="content_refinement",
                name="Content Refinement",
                description="Refine and polish content based on review feedback",
                agent_role="Content Writer",
                task_description="Revise and refine content based on editor feedback and suggestions",
                inputs=["reviewed_content", "feedback_report", "call_to_action"],
                outputs=["final_content", "revision_log"],
                dependencies=["quality_review"],
                tools=["content_editor", "style_checker"],
                configuration={"refinement_focus": "quality"}
            ),
            TemplateStep(
                step_id="format_adaptation",
                name="Format Adaptation",
                description="Adapt content for specific platforms and formats",
                agent_role="Content Formatter",
                task_description="Format content for target platforms and distribution channels",
                inputs=["final_content", "platforms", "content_type"],
                outputs=["formatted_content", "platform_versions"],
                dependencies=["content_refinement"],
                tools=["format_converter", "platform_optimizer"],
                configuration={"multi_platform": True}
            ),
            TemplateStep(
                step_id="final_approval",
                name="Final Approval",
                description="Final review and approval for publication",
                agent_role="Content Manager",
                task_description="Final approval check and preparation for publication",
                inputs=["formatted_content", "platform_versions"],
                outputs=["approved_content", "publication_package"],
                dependencies=["format_adaptation"],
                tools=["approval_checker", "publication_prep"],
                configuration={"approval_criteria": "standard"}
            )
        ]
    
    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, AgentWrapper]:
        """Create agents for the content generation workflow."""
        agents = {}
        
        # Content Strategist Agent
        content_strategist = Agent(
            role="Content Strategist",
            goal="Develop comprehensive content strategy and planning",
            backstory="""You are an experienced content strategist with expertise in content 
            planning, audience analysis, and strategic thinking. You excel at creating 
            content plans that align with business goals and audience needs.""",
            verbose=True,
            allow_delegation=True
        )
        agents["content_strategist"] = AgentWrapper(
            name="ContentStrategist",
            role="Content Strategist",
            crew_agent=content_strategist
        )
        
        # Content Researcher Agent
        content_researcher = Agent(
            role="Content Researcher",
            goal="Gather relevant information and insights for content creation",
            backstory="""You are a skilled content researcher with expertise in finding 
            relevant, up-to-date information and insights. You excel at identifying 
            key trends, data, and supporting information that makes content valuable.""",
            verbose=True,
            allow_delegation=False
        )
        agents["content_researcher"] = AgentWrapper(
            name="ContentResearcher",
            role="Content Researcher",
            crew_agent=content_researcher
        )
        
        # Content Writer Agent
        content_writer = Agent(
            role="Content Writer",
            goal="Create engaging, high-quality content that resonates with the audience",
            backstory="""You are a talented content writer with expertise in creating 
            engaging, well-structured content across various formats. You excel at 
            adapting your writing style to different audiences and purposes.""",
            verbose=True,
            allow_delegation=False
        )
        agents["content_writer"] = AgentWrapper(
            name="ContentWriter",
            role="Content Writer",
            crew_agent=content_writer
        )
        
        # SEO Specialist Agent
        seo_specialist = Agent(
            role="SEO Specialist",
            goal="Optimize content for search engines and discoverability",
            backstory="""You are an experienced SEO specialist with deep knowledge of 
            search engine optimization techniques. You excel at optimizing content 
            for better visibility while maintaining quality and readability.""",
            verbose=True,
            allow_delegation=False
        )
        agents["seo_specialist"] = AgentWrapper(
            name="SEOSpecialist",
            role="SEO Specialist",
            crew_agent=seo_specialist
        )
        
        # Content Editor Agent
        content_editor = Agent(
            role="Content Editor",
            goal="Ensure content quality, accuracy, and brand consistency",
            backstory="""You are a meticulous content editor with expertise in quality 
            assurance, fact-checking, and brand consistency. You excel at identifying 
            areas for improvement and ensuring content meets high standards.""",
            verbose=True,
            allow_delegation=False
        )
        agents["content_editor"] = AgentWrapper(
            name="ContentEditor",
            role="Content Editor",
            crew_agent=content_editor
        )
        
        # Content Formatter Agent
        content_formatter = Agent(
            role="Content Formatter",
            goal="Adapt content for different platforms and formats",
            backstory="""You are a skilled content formatter with expertise in adapting 
            content for various platforms and distribution channels. You understand 
            platform-specific requirements and optimization techniques.""",
            verbose=True,
            allow_delegation=False
        )
        agents["content_formatter"] = AgentWrapper(
            name="ContentFormatter",
            role="Content Formatter",
            crew_agent=content_formatter
        )
        
        # Content Manager Agent
        content_manager = Agent(
            role="Content Manager",
            goal="Oversee content production and ensure final quality",
            backstory="""You are an experienced content manager with expertise in content 
            production oversight and quality control. You ensure that all content meets 
            organizational standards and is ready for publication.""",
            verbose=True,
            allow_delegation=True
        )
        agents["content_manager"] = AgentWrapper(
            name="ContentManager",
            role="Content Manager",
            crew_agent=content_manager
        )
        
        logger.info(f"Created {len(agents)} agents for content generation workflow")
        return agents
    
    def _create_tasks(self, 
                     params: Dict[str, Any], 
                     agents: Dict[str, AgentWrapper]) -> Dict[str, TaskWrapper]:
        """Create tasks for the content generation workflow."""
        tasks = {}
        
        # Content Planning Task
        tasks["content_planning"] = TaskWrapper(
            task_id="content_planning",
            description=f"""
            Develop a comprehensive content plan for: "{params['content_topic']}"
            
            Requirements:
            - Content type: {params['content_type']}
            - Target audience: {params['target_audience']}
            - Content goals: {', '.join(params.get('content_goals', ['inform', 'engage']))}
            - Content length: {params.get('content_length', 'medium')}
            - Tone: {params.get('tone', 'professional')}
            - Create detailed content outline and structure
            - Define key messages and value propositions
            
            Output: Comprehensive content plan with detailed outline and strategy.
            """,
            agent=agents["content_strategist"],
            expected_output="Detailed content plan with clear outline, strategy, and key messages"
        )
        
        # Research Gathering Task (conditional)
        if params.get('include_research', True):
            tasks["research_gathering"] = TaskWrapper(
                task_id="research_gathering",
                description=f"""
                Conduct focused research for the content topic: "{params['content_topic']}"
                
                Requirements:
                - Research current trends and insights related to the topic
                - Gather supporting data, statistics, and examples
                - Identify expert opinions and authoritative sources
                - Focus on information relevant to {params['target_audience']}
                - Ensure information is current and accurate
                
                Output: Comprehensive research findings with key insights and supporting data.
                """,
                agent=agents["content_researcher"],
                expected_output="Well-organized research findings with relevant insights and credible sources"
            )
        
        # Content Creation Task
        tasks["content_creation"] = TaskWrapper(
            task_id="content_creation",
            description=f"""
            Create {params['content_type']} content based on the plan and research.
            
            Requirements:
            - Follow the content outline and structure
            - Write in {params.get('tone', 'professional')} tone
            - Target length: {params.get('content_length', 'medium')}
            - Focus on {params['target_audience']} audience
            - Incorporate research findings and insights
            - Ensure engaging and valuable content
            - Include proper headings and structure
            
            Output: Well-written draft content that engages the target audience.
            """,
            agent=agents["content_writer"],
            expected_output="High-quality draft content that is engaging, well-structured, and audience-appropriate"
        )
        
        # SEO Optimization Task (conditional)
        if params.get('seo_optimization', True):
            tasks["seo_optimization"] = TaskWrapper(
                task_id="seo_optimization",
                description=f"""
                Optimize the content for search engines and discoverability.
                
                Requirements:
                - Identify relevant keywords for "{params['content_topic']}"
                - Optimize title, headings, and meta descriptions
                - Ensure proper keyword density and placement
                - Improve content structure for SEO
                - Add internal and external linking opportunities
                - Create SEO recommendations report
                
                Output: SEO-optimized content with keyword integration and recommendations.
                """,
                agent=agents["seo_specialist"],
                expected_output="SEO-optimized content with improved discoverability and search ranking potential"
            )
        
        # Quality Review Task
        tasks["quality_review"] = TaskWrapper(
            task_id="quality_review",
            description=f"""
            Conduct comprehensive quality review of the content.
            
            Requirements:
            - Review for accuracy, clarity, and quality
            - Check alignment with brand guidelines: {params.get('brand_guidelines', {})}
            - Verify content meets stated goals: {', '.join(params.get('content_goals', ['inform', 'engage']))}
            - Check grammar, spelling, and style
            - Assess audience appropriateness
            - Provide specific feedback and improvement suggestions
            
            Output: Reviewed content with detailed feedback report and improvement recommendations.
            """,
            agent=agents["content_editor"],
            expected_output="Quality-reviewed content with comprehensive feedback and improvement recommendations"
        )
        
        # Content Refinement Task
        cta_text = f"Include call-to-action: {params['call_to_action']}" if params.get('call_to_action') else "No specific call-to-action required"
        tasks["content_refinement"] = TaskWrapper(
            task_id="content_refinement",
            description=f"""
            Refine and polish the content based on editor feedback.
            
            Requirements:
            - Address all feedback points from quality review
            - Improve content clarity and flow
            - {cta_text}
            - Ensure final polish and professional presentation
            - Maintain consistency in tone and style
            - Create revision log documenting changes
            
            Output: Final refined content ready for formatting and publication.
            """,
            agent=agents["content_writer"],
            expected_output="Polished, refined content that addresses all feedback and is ready for publication"
        )
        
        # Format Adaptation Task
        tasks["format_adaptation"] = TaskWrapper(
            task_id="format_adaptation",
            description=f"""
            Adapt content for target platforms and distribution channels.
            
            Requirements:
            - Content type: {params['content_type']}
            - Target platforms: {', '.join(params.get('platforms', ['general']))}
            - Create platform-specific versions if needed
            - Ensure proper formatting for each platform
            - Optimize for platform-specific requirements
            - Maintain content integrity across formats
            
            Output: Platform-optimized content versions ready for distribution.
            """,
            agent=agents["content_formatter"],
            expected_output="Platform-optimized content formatted for all target distribution channels"
        )
        
        # Final Approval Task
        tasks["final_approval"] = TaskWrapper(
            task_id="final_approval",
            description="""
            Conduct final approval review and prepare for publication.
            
            Requirements:
            - Final quality check and approval
            - Verify all requirements have been met
            - Prepare publication package with all assets
            - Ensure content is ready for distribution
            - Document final approval and publication readiness
            
            Output: Approved content package ready for publication and distribution.
            """,
            agent=agents["content_manager"],
            expected_output="Final approved content package with all materials ready for publication"
        )
        
        logger.info(f"Created {len(tasks)} tasks for content generation workflow")
        return tasks
    
    def _configure_workflow_structure(self,
                                    orchestrator: GraphOrchestrator,
                                    params: Dict[str, Any],
                                    agents: Dict[str, AgentWrapper],
                                    tasks: Dict[str, TaskWrapper]):
        """Configure the workflow structure and dependencies."""
        
        # Set the entry point
        orchestrator.set_entry_point("content_planning")
        
        # Build the workflow structure
        current_step = "content_planning"
        
        # Add research if enabled
        if params.get('include_research', True) and 'research_gathering' in tasks:
            orchestrator.add_edge(current_step, "research_gathering")
            current_step = "research_gathering"
        
        # Continue with content creation
        orchestrator.add_edge(current_step, "content_creation")
        current_step = "content_creation"
        
        # Add SEO optimization if enabled
        if params.get('seo_optimization', True) and 'seo_optimization' in tasks:
            orchestrator.add_edge(current_step, "seo_optimization")
            current_step = "seo_optimization"
        
        # Continue with review and refinement
        orchestrator.add_edge(current_step, "quality_review")
        orchestrator.add_edge("quality_review", "content_refinement")
        orchestrator.add_edge("content_refinement", "format_adaptation")
        orchestrator.add_edge("format_adaptation", "final_approval")
        
        # Add review cycle logic
        review_cycles = params.get('review_cycles', 2)
        if review_cycles > 1:
            orchestrator.add_conditional_edge(
                "quality_review",
                self._review_cycle_condition,
                {"continue": "content_refinement", "revise": "content_creation"}
            )
        
        # Add error handling
        for task_id in tasks.keys():
            orchestrator.add_error_handler(task_id, self._create_error_handler(task_id, params))
        
        logger.info("Content generation workflow structure configured")
    
    def _create_error_handler(self, task_id: str, params: Dict[str, Any]) -> callable:
        """Create error handler for specific task."""
        def error_handler(error, context):
            logger.error(f"Error in {task_id}: {error}")
            
            # Retry logic for certain tasks
            if task_id in ["research_gathering", "seo_optimization"]:
                retry_count = context.get("retry_count", 0)
                if retry_count < 2:
                    context["retry_count"] = retry_count + 1
                    logger.info(f"Retrying {task_id} (attempt {retry_count + 1})")
                    return "retry"
            
            # Skip optional steps if they fail
            if task_id == "seo_optimization" and not params.get('seo_optimization', True):
                logger.warning("SEO optimization failed, continuing without optimization")
                return "skip"
            
            return "continue"
        
        return error_handler
    
    def _review_cycle_condition(self, state: Dict[str, Any]) -> str:
        """Determine if additional review cycles are needed."""
        feedback_report = state.get("feedback_report", {})
        cycle_count = state.get("review_cycle_count", 0)
        max_cycles = state.get("max_review_cycles", 2)
        
        # Check if major revisions are needed
        major_issues = feedback_report.get("major_issues", 0)
        
        # If no major issues or max cycles reached, continue
        if major_issues == 0 or cycle_count >= max_cycles:
            return "continue"
        
        # Otherwise, request revision
        state["review_cycle_count"] = cycle_count + 1
        return "revise"