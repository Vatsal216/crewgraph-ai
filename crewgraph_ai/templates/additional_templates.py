"""
Additional Built-in Workflow Templates for CrewGraph AI

Provides more pre-built workflow templates for common use cases:
- Customer Support Automation
- Marketing Campaign Management  
- Quality Assurance Testing
- Financial Analysis
- Social Media Management
- Email Processing
- Document Processing

Created by: Vatsal216
Date: 2025-07-23
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from ..core.orchestrator import GraphOrchestrator
from .workflow_templates import (
    TemplateCategory,
    TemplateMetadata,
    TemplateParameter,
    TemplateStep,
    WorkflowTemplate,
)


class CustomerSupportTemplate(WorkflowTemplate):
    """Template for customer support automation workflows"""
    
    def _define_metadata(self) -> TemplateMetadata:
        return TemplateMetadata(
            name="Customer Support Automation",
            description="Automated customer support workflow with ticket classification, response generation, and escalation",
            version="1.0.0",
            category=TemplateCategory.CUSTOMER_SERVICE,
            author="Vatsal216",
            tags=["customer_service", "automation", "support", "classification"],
            complexity="medium",
            estimated_time="3-5 minutes",
            requirements=["nlp_tools", "email_integration", "knowledge_base"],
            examples=[
                {
                    "ticket_content": "My order hasn't arrived yet and I need it urgently",
                    "customer_type": "premium",
                    "expected_priority": "high"
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        return [
            TemplateParameter(
                name="ticket_content",
                description="Customer support ticket content",
                param_type="str",
                required=True,
                examples=["Product not working", "Billing question", "Feature request"]
            ),
            TemplateParameter(
                name="customer_type",
                description="Type of customer (basic, premium, enterprise)",
                param_type="str",
                required=False,
                default_value="basic",
                validation_rules={"choices": ["basic", "premium", "enterprise"]}
            ),
            TemplateParameter(
                name="priority_threshold",
                description="Priority threshold for escalation",
                param_type="int",
                required=False,
                default_value=8,
                validation_rules={"min": 1, "max": 10}
            ),
            TemplateParameter(
                name="response_tone",
                description="Tone for automated responses",
                param_type="str",
                required=False,
                default_value="professional",
                validation_rules={"choices": ["professional", "friendly", "formal"]}
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        return [
            TemplateStep(
                step_id="classify_ticket",
                name="Classify Support Ticket",
                description="Classify the support ticket by category and priority",
                agent_role="Support Classifier",
                task_description="Analyze the support ticket content and classify it by category (technical, billing, general) and priority (1-10)",
                inputs=["ticket_content", "customer_type"],
                outputs=["category", "priority", "urgency"],
                tools=["nlp_classifier", "sentiment_analyzer"]
            ),
            TemplateStep(
                step_id="check_knowledge_base",
                name="Search Knowledge Base",
                description="Search knowledge base for relevant solutions",
                agent_role="Knowledge Retriever",
                task_description="Search the knowledge base for solutions related to the classified ticket category",
                inputs=["category", "ticket_content"],
                outputs=["relevant_articles", "solution_confidence"],
                dependencies=["classify_ticket"],
                tools=["knowledge_search", "similarity_matcher"]
            ),
            TemplateStep(
                step_id="generate_response",
                name="Generate Response",
                description="Generate appropriate response based on knowledge base results",
                agent_role="Response Generator",
                task_description="Generate a helpful response based on knowledge base articles and customer context",
                inputs=["relevant_articles", "customer_type", "response_tone"],
                outputs=["response_text", "confidence_score"],
                dependencies=["check_knowledge_base"],
                tools=["response_generator", "template_engine"]
            ),
            TemplateStep(
                step_id="escalation_check",
                name="Check for Escalation",
                description="Determine if ticket needs human escalation",
                agent_role="Escalation Manager",
                task_description="Determine if the ticket should be escalated to human support based on priority and confidence",
                inputs=["priority", "confidence_score", "priority_threshold"],
                outputs=["needs_escalation", "escalation_reason"],
                dependencies=["classify_ticket", "generate_response"],
                tools=["decision_engine"]
            ),
            TemplateStep(
                step_id="finalize_response",
                name="Finalize Response",
                description="Finalize and send the response or escalate to human",
                agent_role="Response Coordinator",
                task_description="Either send the automated response or route to human support team",
                inputs=["response_text", "needs_escalation", "escalation_reason"],
                outputs=["final_action", "response_sent"],
                dependencies=["generate_response", "escalation_check"],
                tools=["email_sender", "ticket_router"]
            )
        ]

    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create agents for customer support workflow"""
        return {
            "support_classifier": {"role": "Support Classifier", "goal": "Classify support tickets"},
            "knowledge_retriever": {"role": "Knowledge Retriever", "goal": "Find relevant solutions"},
            "response_generator": {"role": "Response Generator", "goal": "Generate helpful responses"},
            "escalation_manager": {"role": "Escalation Manager", "goal": "Manage escalations"},
            "response_coordinator": {"role": "Response Coordinator", "goal": "Coordinate final response"}
        }

    def _create_tasks(self, params: Dict[str, Any], agents: Dict[str, Any]) -> Dict[str, Any]:
        """Create tasks for customer support workflow"""
        return {
            "classify_ticket": {"description": "Classify the support ticket", "agent": "support_classifier"},
            "check_knowledge_base": {"description": "Search knowledge base", "agent": "knowledge_retriever"},
            "generate_response": {"description": "Generate response", "agent": "response_generator"},
            "escalation_check": {"description": "Check for escalation", "agent": "escalation_manager"},
            "finalize_response": {"description": "Finalize response", "agent": "response_coordinator"}
        }

    def _configure_workflow_structure(self, orchestrator: Any, params: Dict[str, Any], agents: Dict[str, Any], tasks: Dict[str, Any]) -> None:
        """Configure workflow structure for customer support"""
        pass


class MarketingCampaignTemplate(WorkflowTemplate):
    """Template for marketing campaign management workflows"""
    
    def _define_metadata(self) -> TemplateMetadata:
        return TemplateMetadata(
            name="Marketing Campaign Management",
            description="End-to-end marketing campaign creation, execution, and analysis workflow",
            version="1.0.0",
            category=TemplateCategory.AUTOMATION,
            author="Vatsal216",
            tags=["marketing", "campaign", "automation", "analytics"],
            complexity="complex",
            estimated_time="15-30 minutes",
            requirements=["social_media_apis", "analytics_tools", "content_generation"],
            examples=[
                {
                    "campaign_type": "product_launch",
                    "target_audience": "tech_professionals",
                    "budget": 5000
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        return [
            TemplateParameter(
                name="campaign_type",
                description="Type of marketing campaign",
                param_type="str",
                required=True,
                validation_rules={"choices": ["product_launch", "brand_awareness", "lead_generation", "retention"]}
            ),
            TemplateParameter(
                name="target_audience",
                description="Target audience segment",
                param_type="str",
                required=True,
                examples=["tech_professionals", "small_business_owners", "students"]
            ),
            TemplateParameter(
                name="budget",
                description="Campaign budget in USD",
                param_type="float",
                required=True,
                validation_rules={"min": 100}
            ),
            TemplateParameter(
                name="channels",
                description="Marketing channels to use",
                param_type="list",
                required=False,
                default_value=["email", "social_media"],
                validation_rules={"choices": ["email", "social_media", "paid_ads", "content_marketing"]}
            ),
            TemplateParameter(
                name="campaign_duration",
                description="Campaign duration in days",
                param_type="int",
                required=False,
                default_value=30,
                validation_rules={"min": 1, "max": 365}
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        return [
            TemplateStep(
                step_id="audience_analysis",
                name="Analyze Target Audience",
                description="Analyze target audience characteristics and preferences",
                agent_role="Market Researcher",
                task_description="Research and analyze the target audience to understand demographics, preferences, and behavior patterns",
                inputs=["target_audience", "campaign_type"],
                outputs=["audience_profile", "content_preferences", "optimal_channels"],
                tools=["demographic_analyzer", "trend_analyzer"]
            ),
            TemplateStep(
                step_id="content_strategy",
                name="Develop Content Strategy",
                description="Create content strategy based on audience analysis",
                agent_role="Content Strategist",
                task_description="Develop comprehensive content strategy including messaging, themes, and content types",
                inputs=["audience_profile", "campaign_type", "channels"],
                outputs=["content_strategy", "messaging_framework", "content_calendar"],
                dependencies=["audience_analysis"],
                tools=["content_planner", "message_optimizer"]
            ),
            TemplateStep(
                step_id="create_content",
                name="Create Campaign Content",
                description="Generate campaign content across all channels",
                agent_role="Content Creator",
                task_description="Create engaging content for each channel including copy, visuals, and calls-to-action",
                inputs=["content_strategy", "messaging_framework", "channels"],
                outputs=["email_content", "social_content", "ad_copy", "landing_page_content"],
                dependencies=["content_strategy"],
                tools=["content_generator", "image_creator", "copy_writer"]
            ),
            TemplateStep(
                step_id="schedule_campaign",
                name="Schedule Campaign Activities",
                description="Schedule and coordinate campaign activities across channels",
                agent_role="Campaign Manager",
                task_description="Schedule content publication, ad campaigns, and email sends based on optimal timing",
                inputs=["content_calendar", "channels", "campaign_duration"],
                outputs=["execution_schedule", "channel_coordination"],
                dependencies=["create_content"],
                tools=["scheduler", "channel_integrator"]
            ),
            TemplateStep(
                step_id="monitor_performance",
                name="Monitor Campaign Performance",
                description="Monitor campaign performance and make real-time optimizations",
                agent_role="Performance Analyst",
                task_description="Track key metrics, analyze performance, and suggest optimizations",
                inputs=["execution_schedule", "budget"],
                outputs=["performance_metrics", "optimization_suggestions"],
                dependencies=["schedule_campaign"],
                tools=["analytics_tracker", "performance_monitor"],
                optional=True
            )
        ]

    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create agents for marketing campaign workflow"""
        return {
            "market_researcher": {"role": "Market Researcher", "goal": "Research target audience"},
            "content_strategist": {"role": "Content Strategist", "goal": "Develop content strategy"},
            "content_creator": {"role": "Content Creator", "goal": "Create campaign content"},
            "campaign_manager": {"role": "Campaign Manager", "goal": "Manage campaign execution"},
            "performance_analyst": {"role": "Performance Analyst", "goal": "Analyze campaign performance"}
        }

    def _create_tasks(self, params: Dict[str, Any], agents: Dict[str, Any]) -> Dict[str, Any]:
        """Create tasks for marketing campaign workflow"""
        return {
            "audience_analysis": {"description": "Analyze target audience", "agent": "market_researcher"},
            "content_strategy": {"description": "Develop content strategy", "agent": "content_strategist"},
            "create_content": {"description": "Create campaign content", "agent": "content_creator"},
            "schedule_campaign": {"description": "Schedule campaign activities", "agent": "campaign_manager"},
            "monitor_performance": {"description": "Monitor campaign performance", "agent": "performance_analyst"}
        }

    def _configure_workflow_structure(self, orchestrator: Any, params: Dict[str, Any], agents: Dict[str, Any], tasks: Dict[str, Any]) -> None:
        """Configure workflow structure for marketing campaign"""
        pass


class QualityAssuranceTemplate(WorkflowTemplate):
    """Template for quality assurance testing workflows"""
    
    def _define_metadata(self) -> TemplateMetadata:
        return TemplateMetadata(
            name="Quality Assurance Testing",
            description="Comprehensive QA testing workflow with automated and manual testing phases",
            version="1.0.0",
            category=TemplateCategory.AUTOMATION,
            author="Vatsal216",
            tags=["qa", "testing", "automation", "quality", "software"],
            complexity="complex",
            estimated_time="20-45 minutes",
            requirements=["test_frameworks", "ci_cd_integration", "bug_tracking"],
            examples=[
                {
                    "application_type": "web_application",
                    "test_environment": "staging",
                    "test_level": "comprehensive"
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        return [
            TemplateParameter(
                name="application_type",
                description="Type of application to test",
                param_type="str",
                required=True,
                validation_rules={"choices": ["web_application", "mobile_app", "api", "desktop_app"]}
            ),
            TemplateParameter(
                name="test_environment",
                description="Testing environment",
                param_type="str",
                required=True,
                validation_rules={"choices": ["development", "staging", "production"]}
            ),
            TemplateParameter(
                name="test_level",
                description="Level of testing depth",
                param_type="str",
                required=False,
                default_value="standard",
                validation_rules={"choices": ["basic", "standard", "comprehensive"]}
            ),
            TemplateParameter(
                name="test_types",
                description="Types of tests to run",
                param_type="list",
                required=False,
                default_value=["functional", "performance", "security"],
                validation_rules={"choices": ["functional", "performance", "security", "usability", "compatibility"]}
            ),
            TemplateParameter(
                name="critical_features",
                description="Critical features that must be tested",
                param_type="list",
                required=False,
                default_value=[],
                examples=[["login", "payment", "data_export"]]
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        return [
            TemplateStep(
                step_id="test_planning",
                name="Create Test Plan",
                description="Create comprehensive test plan based on requirements",
                agent_role="Test Planner",
                task_description="Analyze requirements and create detailed test plan covering all specified test types",
                inputs=["application_type", "test_types", "critical_features"],
                outputs=["test_plan", "test_cases", "test_data_requirements"],
                tools=["test_case_generator", "requirement_analyzer"]
            ),
            TemplateStep(
                step_id="setup_environment",
                name="Setup Test Environment",
                description="Prepare and configure test environment",
                agent_role="Environment Manager",
                task_description="Setup and configure the test environment with required test data and configurations",
                inputs=["test_environment", "test_data_requirements"],
                outputs=["environment_status", "test_data_prepared"],
                dependencies=["test_planning"],
                tools=["environment_provisioner", "data_generator"]
            ),
            TemplateStep(
                step_id="automated_testing",
                name="Execute Automated Tests",
                description="Run automated test suites",
                agent_role="Test Executor",
                task_description="Execute automated tests including unit, integration, and regression tests",
                inputs=["test_plan", "environment_status"],
                outputs=["automated_test_results", "test_coverage", "failed_tests"],
                dependencies=["setup_environment"],
                tools=["test_runner", "coverage_analyzer"]
            ),
            TemplateStep(
                step_id="manual_testing",
                name="Execute Manual Tests",
                description="Perform manual testing for complex scenarios",
                agent_role="Manual Tester",
                task_description="Execute manual test cases for usability, edge cases, and exploratory testing",
                inputs=["test_cases", "critical_features"],
                outputs=["manual_test_results", "usability_issues", "edge_case_findings"],
                dependencies=["automated_testing"],
                tools=["manual_test_tracker", "bug_reporter"]
            ),
            TemplateStep(
                step_id="performance_testing",
                name="Performance Testing",
                description="Execute performance and load testing",
                agent_role="Performance Tester",
                task_description="Run performance tests to identify bottlenecks and scalability issues",
                inputs=["test_environment", "test_level"],
                outputs=["performance_metrics", "bottlenecks", "scalability_report"],
                dependencies=["setup_environment"],
                tools=["load_tester", "performance_profiler"],
                optional=True
            ),
            TemplateStep(
                step_id="security_testing",
                name="Security Testing",
                description="Perform security vulnerability testing",
                agent_role="Security Tester",
                task_description="Execute security tests to identify vulnerabilities and security issues",
                inputs=["application_type", "test_environment"],
                outputs=["security_report", "vulnerabilities", "risk_assessment"],
                dependencies=["setup_environment"],
                tools=["security_scanner", "penetration_tester"],
                optional=True
            ),
            TemplateStep(
                step_id="generate_report",
                name="Generate Test Report",
                description="Compile comprehensive test report",
                agent_role="Test Reporter",
                task_description="Compile all test results into comprehensive report with recommendations",
                inputs=["automated_test_results", "manual_test_results", "performance_metrics", "security_report"],
                outputs=["comprehensive_report", "bug_summary", "recommendations"],
                dependencies=["automated_testing", "manual_testing"],
                tools=["report_generator", "trend_analyzer"]
            )
        ]

    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create agents for QA testing workflow"""
        return {
            "test_planner": {"role": "Test Planner", "goal": "Create comprehensive test plans"},
            "environment_manager": {"role": "Environment Manager", "goal": "Manage test environments"},
            "test_executor": {"role": "Test Executor", "goal": "Execute automated tests"},
            "manual_tester": {"role": "Manual Tester", "goal": "Perform manual testing"},
            "performance_tester": {"role": "Performance Tester", "goal": "Execute performance tests"},
            "security_tester": {"role": "Security Tester", "goal": "Perform security testing"},
            "test_reporter": {"role": "Test Reporter", "goal": "Generate test reports"}
        }

    def _create_tasks(self, params: Dict[str, Any], agents: Dict[str, Any]) -> Dict[str, Any]:
        """Create tasks for QA testing workflow"""
        return {
            "test_planning": {"description": "Create test plan", "agent": "test_planner"},
            "setup_environment": {"description": "Setup test environment", "agent": "environment_manager"},
            "automated_testing": {"description": "Execute automated tests", "agent": "test_executor"},
            "manual_testing": {"description": "Execute manual tests", "agent": "manual_tester"},
            "performance_testing": {"description": "Performance testing", "agent": "performance_tester"},
            "security_testing": {"description": "Security testing", "agent": "security_tester"},
            "generate_report": {"description": "Generate test report", "agent": "test_reporter"}
        }

    def _configure_workflow_structure(self, orchestrator: Any, params: Dict[str, Any], agents: Dict[str, Any], tasks: Dict[str, Any]) -> None:
        """Configure workflow structure for QA testing"""
        pass


class FinancialAnalysisTemplate(WorkflowTemplate):
    """Template for financial analysis workflows"""
    
    def _define_metadata(self) -> TemplateMetadata:
        return TemplateMetadata(
            name="Financial Analysis",
            description="Comprehensive financial analysis workflow with data processing, modeling, and reporting",
            version="1.0.0",
            category=TemplateCategory.ANALYSIS,
            author="Vatsal216",
            tags=["finance", "analysis", "modeling", "reporting", "data"],
            complexity="complex",
            estimated_time="25-40 minutes",
            requirements=["financial_data_sources", "analytical_tools", "reporting_engine"],
            examples=[
                {
                    "analysis_type": "portfolio_performance",
                    "time_period": "quarterly",
                    "include_benchmarks": True
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        return [
            TemplateParameter(
                name="analysis_type",
                description="Type of financial analysis",
                param_type="str",
                required=True,
                validation_rules={"choices": ["portfolio_performance", "risk_assessment", "valuation", "trend_analysis"]}
            ),
            TemplateParameter(
                name="time_period",
                description="Analysis time period",
                param_type="str",
                required=True,
                validation_rules={"choices": ["monthly", "quarterly", "yearly", "custom"]}
            ),
            TemplateParameter(
                name="data_sources",
                description="Financial data sources to use",
                param_type="list",
                required=False,
                default_value=["market_data", "company_financials"],
                validation_rules={"choices": ["market_data", "company_financials", "economic_indicators", "alternative_data"]}
            ),
            TemplateParameter(
                name="include_benchmarks",
                description="Include benchmark comparisons",
                param_type="bool",
                required=False,
                default_value=True
            ),
            TemplateParameter(
                name="risk_tolerance",
                description="Risk tolerance level for analysis",
                param_type="str",
                required=False,
                default_value="moderate",
                validation_rules={"choices": ["conservative", "moderate", "aggressive"]}
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        return [
            TemplateStep(
                step_id="data_collection",
                name="Collect Financial Data",
                description="Gather financial data from specified sources",
                agent_role="Data Collector",
                task_description="Collect and aggregate financial data from various sources for the specified time period",
                inputs=["data_sources", "time_period"],
                outputs=["raw_data", "data_quality_report"],
                tools=["data_fetcher", "data_validator"]
            ),
            TemplateStep(
                step_id="data_processing",
                name="Process and Clean Data",
                description="Clean, normalize, and process financial data",
                agent_role="Data Processor",
                task_description="Clean, normalize, and structure the financial data for analysis",
                inputs=["raw_data", "data_quality_report"],
                outputs=["processed_data", "data_summary"],
                dependencies=["data_collection"],
                tools=["data_cleaner", "normalizer"]
            ),
            TemplateStep(
                step_id="financial_modeling",
                name="Build Financial Models",
                description="Create financial models based on analysis type",
                agent_role="Financial Modeler",
                task_description="Build appropriate financial models for the specified analysis type",
                inputs=["processed_data", "analysis_type", "risk_tolerance"],
                outputs=["financial_models", "model_parameters"],
                dependencies=["data_processing"],
                tools=["model_builder", "statistical_analyzer"]
            ),
            TemplateStep(
                step_id="run_analysis",
                name="Execute Financial Analysis",
                description="Run financial analysis using the built models",
                agent_role="Financial Analyst",
                task_description="Execute comprehensive financial analysis using the constructed models",
                inputs=["financial_models", "processed_data"],
                outputs=["analysis_results", "key_metrics", "insights"],
                dependencies=["financial_modeling"],
                tools=["analysis_engine", "metrics_calculator"]
            ),
            TemplateStep(
                step_id="benchmark_comparison",
                name="Benchmark Comparison",
                description="Compare results against relevant benchmarks",
                agent_role="Benchmark Analyst",
                task_description="Compare analysis results against industry and market benchmarks",
                inputs=["analysis_results", "include_benchmarks"],
                outputs=["benchmark_comparison", "relative_performance"],
                dependencies=["run_analysis"],
                tools=["benchmark_fetcher", "comparison_analyzer"],
                optional=True
            ),
            TemplateStep(
                step_id="generate_insights",
                name="Generate Insights and Recommendations",
                description="Generate actionable insights and recommendations",
                agent_role="Investment Advisor",
                task_description="Generate actionable insights and investment recommendations based on analysis",
                inputs=["analysis_results", "benchmark_comparison", "risk_tolerance"],
                outputs=["insights", "recommendations", "action_items"],
                dependencies=["run_analysis"],
                tools=["insight_generator", "recommendation_engine"]
            ),
            TemplateStep(
                step_id="create_report",
                name="Create Financial Report",
                description="Generate comprehensive financial report",
                agent_role="Report Writer",
                task_description="Create comprehensive financial report with visualizations and executive summary",
                inputs=["analysis_results", "insights", "recommendations"],
                outputs=["financial_report", "executive_summary", "visualizations"],
                dependencies=["generate_insights"],
                tools=["report_generator", "chart_creator"]
            )
        ]

    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create agents for financial analysis workflow"""
        return {
            "data_collector": {"role": "Data Collector", "goal": "Collect financial data"},
            "data_processor": {"role": "Data Processor", "goal": "Process and clean data"},
            "financial_modeler": {"role": "Financial Modeler", "goal": "Build financial models"},
            "financial_analyst": {"role": "Financial Analyst", "goal": "Execute financial analysis"},
            "benchmark_analyst": {"role": "Benchmark Analyst", "goal": "Compare against benchmarks"},
            "investment_advisor": {"role": "Investment Advisor", "goal": "Generate insights and recommendations"},
            "report_writer": {"role": "Report Writer", "goal": "Create financial reports"}
        }

    def _create_tasks(self, params: Dict[str, Any], agents: Dict[str, Any]) -> Dict[str, Any]:
        """Create tasks for financial analysis workflow"""
        return {
            "data_collection": {"description": "Collect financial data", "agent": "data_collector"},
            "data_processing": {"description": "Process and clean data", "agent": "data_processor"},
            "financial_modeling": {"description": "Build financial models", "agent": "financial_modeler"},
            "run_analysis": {"description": "Execute financial analysis", "agent": "financial_analyst"},
            "benchmark_comparison": {"description": "Benchmark comparison", "agent": "benchmark_analyst"},
            "generate_insights": {"description": "Generate insights and recommendations", "agent": "investment_advisor"},
            "create_report": {"description": "Create financial report", "agent": "report_writer"}
        }

    def _configure_workflow_structure(self, orchestrator: Any, params: Dict[str, Any], agents: Dict[str, Any], tasks: Dict[str, Any]) -> None:
        """Configure workflow structure for financial analysis"""
        pass


class DocumentProcessingTemplate(WorkflowTemplate):
    """Template for document processing workflows"""
    
    def _define_metadata(self) -> TemplateMetadata:
        return TemplateMetadata(
            name="Document Processing",
            description="Automated document processing workflow with OCR, extraction, and classification",
            version="1.0.0",
            category=TemplateCategory.DATA_PROCESSING,
            author="Vatsal216",
            tags=["document", "ocr", "extraction", "classification", "automation"],
            complexity="medium",
            estimated_time="5-15 minutes",
            requirements=["ocr_engine", "nlp_tools", "document_storage"],
            examples=[
                {
                    "document_types": ["invoice", "contract", "receipt"],
                    "extract_structured_data": True,
                    "language": "en"
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        return [
            TemplateParameter(
                name="document_source",
                description="Source of documents to process",
                param_type="str",
                required=True,
                examples=["file_upload", "email_attachment", "folder_path", "cloud_storage"]
            ),
            TemplateParameter(
                name="document_types",
                description="Expected document types",
                param_type="list",
                required=False,
                default_value=["general"],
                validation_rules={"choices": ["invoice", "contract", "receipt", "form", "report", "letter", "general"]}
            ),
            TemplateParameter(
                name="extract_structured_data",
                description="Extract structured data from documents",
                param_type="bool",
                required=False,
                default_value=True
            ),
            TemplateParameter(
                name="language",
                description="Primary language of documents",
                param_type="str",
                required=False,
                default_value="en",
                validation_rules={"choices": ["en", "es", "fr", "de", "it", "pt", "auto"]}
            ),
            TemplateParameter(
                name="output_format",
                description="Output format for processed data",
                param_type="str",
                required=False,
                default_value="json",
                validation_rules={"choices": ["json", "csv", "xml", "pdf"]}
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        return [
            TemplateStep(
                step_id="document_ingestion",
                name="Ingest Documents",
                description="Collect and prepare documents for processing",
                agent_role="Document Collector",
                task_description="Collect documents from specified source and prepare them for processing",
                inputs=["document_source"],
                outputs=["document_list", "metadata"],
                tools=["file_reader", "metadata_extractor"]
            ),
            TemplateStep(
                step_id="document_classification",
                name="Classify Documents",
                description="Classify documents by type and category",
                agent_role="Document Classifier",
                task_description="Classify documents into categories based on content and structure",
                inputs=["document_list", "document_types"],
                outputs=["classified_documents", "confidence_scores"],
                dependencies=["document_ingestion"],
                tools=["document_classifier", "content_analyzer"]
            ),
            TemplateStep(
                step_id="ocr_processing",
                name="OCR Text Extraction",
                description="Extract text content using OCR",
                agent_role="OCR Processor",
                task_description="Extract text content from images and scanned documents using OCR",
                inputs=["classified_documents", "language"],
                outputs=["extracted_text", "ocr_confidence"],
                dependencies=["document_classification"],
                tools=["ocr_engine", "text_cleaner"]
            ),
            TemplateStep(
                step_id="data_extraction",
                name="Extract Structured Data",
                description="Extract structured data from documents",
                agent_role="Data Extractor",
                task_description="Extract structured data fields based on document type",
                inputs=["extracted_text", "classified_documents", "extract_structured_data"],
                outputs=["structured_data", "extracted_fields"],
                dependencies=["ocr_processing"],
                tools=["field_extractor", "pattern_matcher"]
            ),
            TemplateStep(
                step_id="validation",
                name="Validate Extracted Data",
                description="Validate and verify extracted data",
                agent_role="Data Validator",
                task_description="Validate extracted data for accuracy and completeness",
                inputs=["structured_data", "extracted_fields"],
                outputs=["validated_data", "validation_errors"],
                dependencies=["data_extraction"],
                tools=["data_validator", "rule_engine"]
            ),
            TemplateStep(
                step_id="output_generation",
                name="Generate Output",
                description="Generate final output in specified format",
                agent_role="Output Generator",
                task_description="Generate final processed output in the specified format",
                inputs=["validated_data", "output_format"],
                outputs=["final_output", "processing_summary"],
                dependencies=["validation"],
                tools=["format_converter", "report_generator"]
            )
        ]

    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create agents for document processing workflow"""
        return {
            "document_collector": {"role": "Document Collector", "goal": "Collect and prepare documents"},
            "document_classifier": {"role": "Document Classifier", "goal": "Classify document types"},
            "ocr_processor": {"role": "OCR Processor", "goal": "Extract text from documents"},
            "data_extractor": {"role": "Data Extractor", "goal": "Extract structured data"},
            "data_validator": {"role": "Data Validator", "goal": "Validate extracted data"},
            "output_generator": {"role": "Output Generator", "goal": "Generate final output"}
        }

    def _create_tasks(self, params: Dict[str, Any], agents: Dict[str, Any]) -> Dict[str, Any]:
        """Create tasks for document processing workflow"""
        return {
            "document_ingestion": {"description": "Ingest documents", "agent": "document_collector"},
            "document_classification": {"description": "Classify documents", "agent": "document_classifier"},
            "ocr_processing": {"description": "OCR text extraction", "agent": "ocr_processor"},
            "data_extraction": {"description": "Extract structured data", "agent": "data_extractor"},
            "validation": {"description": "Validate extracted data", "agent": "data_validator"},
            "output_generation": {"description": "Generate output", "agent": "output_generator"}
        }

    def _configure_workflow_structure(self, orchestrator: Any, params: Dict[str, Any], agents: Dict[str, Any], tasks: Dict[str, Any]) -> None:
        """Configure workflow structure for document processing"""
        pass