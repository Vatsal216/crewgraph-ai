"""
Data Pipeline Template - Pre-built template for data processing workflows

This template provides a standardized approach to building data processing pipelines with:
- Data ingestion and validation
- Data cleaning and transformation
- Data analysis and insights
- Report generation and output

Features:
- Configurable data sources and formats
- Flexible processing steps
- Quality checks and validation
- Multiple output formats
- Error handling and recovery

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


class DataPipelineTemplate(WorkflowTemplate):
    """
    Template for creating data processing pipelines.
    
    This template provides a comprehensive framework for building data processing
    workflows with standardized steps for ingestion, validation, cleaning,
    transformation, analysis, and reporting.
    """
    
    def _define_metadata(self) -> TemplateMetadata:
        """Define template metadata."""
        return TemplateMetadata(
            name="data_pipeline",
            description="Comprehensive data processing pipeline with ingestion, cleaning, analysis, and reporting",
            version="1.0.0",
            category=TemplateCategory.DATA_PROCESSING,
            author="Vatsal216",
            tags=["data", "pipeline", "etl", "analysis", "processing"],
            complexity="medium",
            estimated_time="15-30 minutes",
            requirements=[
                "Data source configuration",
                "Output destination setup",
                "Processing rules definition"
            ],
            examples=[
                {
                    "name": "CSV Analysis Pipeline",
                    "description": "Process CSV files with sales data analysis",
                    "parameters": {
                        "data_source": "sales_data.csv",
                        "source_format": "csv",
                        "analysis_type": "sales_trends",
                        "output_format": "dashboard"
                    }
                },
                {
                    "name": "API Data Processing",
                    "description": "Process data from REST API with real-time updates",
                    "parameters": {
                        "data_source": "https://api.example.com/data",
                        "source_format": "json",
                        "analysis_type": "real_time_metrics",
                        "output_format": "alerts"
                    }
                }
            ]
        )
    
    def _define_parameters(self) -> List[TemplateParameter]:
        """Define template parameters."""
        return [
            TemplateParameter(
                name="data_source",
                description="Data source location (file path, URL, database connection)",
                param_type="str",
                required=True,
                examples=["data/input.csv", "https://api.example.com/data", "postgresql://user:pass@host/db"]
            ),
            TemplateParameter(
                name="source_format",
                description="Format of the source data",
                param_type="str",
                required=True,
                default_value="csv",
                validation_rules={"allowed_values": ["csv", "json", "xml", "parquet", "database", "api"]},
                examples=["csv", "json", "parquet"]
            ),
            TemplateParameter(
                name="analysis_type",
                description="Type of analysis to perform on the data",
                param_type="str",
                required=True,
                default_value="general_insights",
                validation_rules={"allowed_values": [
                    "general_insights", "sales_trends", "user_behavior", 
                    "financial_analysis", "operational_metrics", "real_time_metrics"
                ]},
                examples=["sales_trends", "user_behavior", "financial_analysis"]
            ),
            TemplateParameter(
                name="output_format",
                description="Format for the analysis output",
                param_type="str",
                required=True,
                default_value="report",
                validation_rules={"allowed_values": ["report", "dashboard", "alerts", "csv", "json"]},
                examples=["report", "dashboard", "csv"]
            ),
            TemplateParameter(
                name="quality_checks",
                description="Enable data quality validation checks",
                param_type="bool",
                required=False,
                default_value=True
            ),
            TemplateParameter(
                name="data_cleaning",
                description="Enable automatic data cleaning and preprocessing",
                param_type="bool",
                required=False,
                default_value=True
            ),
            TemplateParameter(
                name="batch_size",
                description="Processing batch size for large datasets",
                param_type="int",
                required=False,
                default_value=1000,
                validation_rules={"min_value": 1, "max_value": 100000}
            ),
            TemplateParameter(
                name="error_threshold",
                description="Maximum allowed error rate (percentage)",
                param_type="float",
                required=False,
                default_value=5.0,
                validation_rules={"min_value": 0.0, "max_value": 100.0}
            ),
            TemplateParameter(
                name="custom_rules",
                description="Custom processing rules in JSON format",
                param_type="dict",
                required=False,
                default_value={},
                examples=[{"remove_duplicates": True, "fill_missing": "mean"}]
            )
        ]
    
    def _define_steps(self) -> List[TemplateStep]:
        """Define workflow steps."""
        return [
            TemplateStep(
                step_id="data_ingestion",
                name="Data Ingestion",
                description="Load and validate data from the specified source",
                agent_role="Data Engineer",
                task_description="Connect to data source and ingest data with initial validation",
                inputs=["data_source", "source_format"],
                outputs=["raw_data", "ingestion_report"],
                tools=["data_loader", "format_validator"],
                configuration={"retry_attempts": 3, "timeout": 300}
            ),
            TemplateStep(
                step_id="data_validation",
                name="Data Quality Check",
                description="Perform comprehensive data quality validation",
                agent_role="Data Analyst",
                task_description="Validate data quality, identify issues, and generate quality report",
                inputs=["raw_data"],
                outputs=["validation_report", "quality_metrics"],
                dependencies=["data_ingestion"],
                tools=["quality_checker", "schema_validator"],
                configuration={"validation_rules": "standard"},
                optional=False  # Can be skipped if quality_checks=False
            ),
            TemplateStep(
                step_id="data_cleaning",
                name="Data Cleaning",
                description="Clean and preprocess the data based on quality findings",
                agent_role="Data Engineer",
                task_description="Apply cleaning rules, handle missing values, remove duplicates",
                inputs=["raw_data", "validation_report"],
                outputs=["cleaned_data", "cleaning_report"],
                dependencies=["data_validation"],
                tools=["data_cleaner", "preprocessor"],
                configuration={"cleaning_strategy": "auto"},
                optional=False  # Can be skipped if data_cleaning=False
            ),
            TemplateStep(
                step_id="data_transformation",
                name="Data Transformation",
                description="Transform data according to analysis requirements",
                agent_role="Data Engineer",
                task_description="Apply transformations, aggregations, and feature engineering",
                inputs=["cleaned_data", "custom_rules"],
                outputs=["transformed_data", "transformation_log"],
                dependencies=["data_cleaning"],
                tools=["transformer", "aggregator"],
                configuration={"transformation_type": "auto"}
            ),
            TemplateStep(
                step_id="data_analysis",
                name="Data Analysis",
                description="Perform the specified type of analysis on the transformed data",
                agent_role="Data Scientist",
                task_description="Execute analysis algorithms and generate insights",
                inputs=["transformed_data", "analysis_type"],
                outputs=["analysis_results", "insights"],
                dependencies=["data_transformation"],
                tools=["analyzer", "statistical_tools", "ml_tools"],
                configuration={"analysis_depth": "comprehensive"}
            ),
            TemplateStep(
                step_id="report_generation",
                name="Report Generation",
                description="Generate final reports and outputs in the specified format",
                agent_role="Business Analyst",
                task_description="Create comprehensive reports with visualizations and recommendations",
                inputs=["analysis_results", "insights", "output_format"],
                outputs=["final_report", "visualizations"],
                dependencies=["data_analysis"],
                tools=["report_generator", "visualization_tools"],
                configuration={"include_recommendations": True}
            ),
            TemplateStep(
                step_id="output_delivery",
                name="Output Delivery",
                description="Deliver results to specified destinations",
                agent_role="Data Engineer",
                task_description="Package and deliver outputs to configured destinations",
                inputs=["final_report", "visualizations"],
                outputs=["delivery_confirmation"],
                dependencies=["report_generation"],
                tools=["output_handler", "notification_service"],
                configuration={"delivery_method": "auto"}
            )
        ]
    
    def _create_agents(self, params: Dict[str, Any]) -> Dict[str, AgentWrapper]:
        """Create agents for the data pipeline workflow."""
        agents = {}
        
        # Data Engineer Agent
        data_engineer = Agent(
            role="Data Engineer",
            goal="Efficiently process and manage data through the pipeline",
            backstory="""You are an experienced data engineer specializing in building robust 
            data pipelines. You excel at data ingestion, cleaning, transformation, and ensuring 
            data quality throughout the process.""",
            verbose=True,
            allow_delegation=False
        )
        agents["data_engineer"] = AgentWrapper(
            name="DataEngineer",
            role="Data Engineer", 
            crew_agent=data_engineer
        )
        
        # Data Analyst Agent
        data_analyst = Agent(
            role="Data Analyst",
            goal="Ensure data quality and provide analytical insights",
            backstory="""You are a meticulous data analyst with expertise in data validation 
            and quality assessment. You have a keen eye for identifying data issues and 
            ensuring the integrity of analytical processes.""",
            verbose=True,
            allow_delegation=False
        )
        agents["data_analyst"] = AgentWrapper(
            name="DataAnalyst",
            role="Data Analyst",
            crew_agent=data_analyst
        )
        
        # Data Scientist Agent
        data_scientist = Agent(
            role="Data Scientist",
            goal="Extract meaningful insights and patterns from data",
            backstory="""You are a skilled data scientist with expertise in statistical analysis, 
            machine learning, and pattern recognition. You excel at turning raw data into 
            actionable insights and business value.""",
            verbose=True,
            allow_delegation=False
        )
        agents["data_scientist"] = AgentWrapper(
            name="DataScientist",
            role="Data Scientist",
            crew_agent=data_scientist
        )
        
        # Business Analyst Agent
        business_analyst = Agent(
            role="Business Analyst",
            goal="Translate technical findings into business insights and recommendations",
            backstory="""You are a business analyst who bridges the gap between technical 
            analysis and business strategy. You excel at creating clear, actionable reports 
            that drive business decisions.""",
            verbose=True,
            allow_delegation=False
        )
        agents["business_analyst"] = AgentWrapper(
            name="BusinessAnalyst",
            role="Business Analyst",
            crew_agent=business_analyst
        )
        
        logger.info(f"Created {len(agents)} agents for data pipeline")
        return agents
    
    def _create_tasks(self, 
                     params: Dict[str, Any], 
                     agents: Dict[str, AgentWrapper]) -> Dict[str, TaskWrapper]:
        """Create tasks for the data pipeline workflow."""
        tasks = {}
        
        # Data Ingestion Task
        tasks["data_ingestion"] = TaskWrapper(
            task_id="data_ingestion",
            description=f"""
            Load data from {params['data_source']} in {params['source_format']} format.
            
            Requirements:
            - Validate data source accessibility
            - Handle large datasets efficiently using batch size: {params.get('batch_size', 1000)}
            - Perform initial format validation
            - Generate ingestion summary report
            
            Output: Raw data and ingestion report with statistics and any issues found.
            """,
            agent=agents["data_engineer"],
            expected_output="Raw data loaded successfully with ingestion report"
        )
        
        # Data Validation Task (conditional)
        if params.get('quality_checks', True):
            tasks["data_validation"] = TaskWrapper(
                task_id="data_validation",
                description=f"""
                Perform comprehensive data quality validation on the ingested data.
                
                Requirements:
                - Check for missing values, duplicates, and anomalies
                - Validate data types and formats
                - Assess data completeness and consistency
                - Error threshold: {params.get('error_threshold', 5.0)}%
                - Generate detailed quality report
                
                Output: Validation report with quality metrics and recommendations.
                """,
                agent=agents["data_analyst"],
                expected_output="Comprehensive data quality report with actionable recommendations"
            )
        
        # Data Cleaning Task (conditional)
        if params.get('data_cleaning', True):
            tasks["data_cleaning"] = TaskWrapper(
                task_id="data_cleaning",
                description="""
                Clean and preprocess the data based on quality assessment findings.
                
                Requirements:
                - Handle missing values appropriately
                - Remove or flag duplicates
                - Address data type inconsistencies
                - Apply data standardization
                - Document all cleaning operations
                
                Output: Cleaned dataset with comprehensive cleaning report.
                """,
                agent=agents["data_engineer"],
                expected_output="Cleaned and preprocessed data with detailed cleaning log"
            )
        
        # Data Transformation Task
        tasks["data_transformation"] = TaskWrapper(
            task_id="data_transformation",
            description=f"""
            Transform the data according to analysis requirements and custom rules.
            
            Requirements:
            - Apply necessary aggregations and calculations
            - Implement custom transformation rules: {params.get('custom_rules', {})}
            - Create derived features if needed
            - Optimize data structure for analysis
            - Log all transformation steps
            
            Output: Transformed data ready for analysis with transformation log.
            """,
            agent=agents["data_engineer"],
            expected_output="Transformed data optimized for the specified analysis type"
        )
        
        # Data Analysis Task
        tasks["data_analysis"] = TaskWrapper(
            task_id="data_analysis",
            description=f"""
            Perform {params['analysis_type']} analysis on the transformed data.
            
            Requirements:
            - Execute appropriate analytical algorithms
            - Generate statistical summaries and insights
            - Identify patterns, trends, and anomalies
            - Create preliminary visualizations
            - Validate findings and results
            
            Output: Comprehensive analysis results with key insights and findings.
            """,
            agent=agents["data_scientist"],
            expected_output="Detailed analysis results with actionable insights and statistical evidence"
        )
        
        # Report Generation Task
        tasks["report_generation"] = TaskWrapper(
            task_id="report_generation",
            description=f"""
            Generate comprehensive reports in {params['output_format']} format.
            
            Requirements:
            - Create executive summary with key findings
            - Include detailed methodology and results
            - Generate appropriate visualizations
            - Provide actionable recommendations
            - Ensure report clarity and professional presentation
            
            Output: Professional report with visualizations and recommendations.
            """,
            agent=agents["business_analyst"],
            expected_output=f"Professional {params['output_format']} report with clear insights and recommendations"
        )
        
        # Output Delivery Task
        tasks["output_delivery"] = TaskWrapper(
            task_id="output_delivery",
            description="""
            Package and deliver the final outputs to specified destinations.
            
            Requirements:
            - Organize all outputs (reports, data, visualizations)
            - Package in appropriate formats
            - Deliver to configured destinations
            - Send notifications to stakeholders
            - Confirm successful delivery
            
            Output: Delivery confirmation with access details.
            """,
            agent=agents["data_engineer"],
            expected_output="Confirmation of successful delivery with access information"
        )
        
        logger.info(f"Created {len(tasks)} tasks for data pipeline")
        return tasks
    
    def _configure_workflow_structure(self,
                                    orchestrator: GraphOrchestrator,
                                    params: Dict[str, Any],
                                    agents: Dict[str, AgentWrapper],
                                    tasks: Dict[str, TaskWrapper]):
        """Configure the workflow structure and dependencies."""
        
        # Build the workflow graph based on enabled steps
        current_step = "data_ingestion"
        
        # Always start with data ingestion
        orchestrator.set_entry_point(current_step)
        
        # Add data validation if enabled
        if params.get('quality_checks', True) and 'data_validation' in tasks:
            orchestrator.add_edge(current_step, "data_validation")
            current_step = "data_validation"
        
        # Add data cleaning if enabled
        if params.get('data_cleaning', True) and 'data_cleaning' in tasks:
            orchestrator.add_edge(current_step, "data_cleaning")
            current_step = "data_cleaning"
        
        # Always continue with transformation, analysis, and reporting
        orchestrator.add_edge(current_step, "data_transformation")
        orchestrator.add_edge("data_transformation", "data_analysis")
        orchestrator.add_edge("data_analysis", "report_generation")
        orchestrator.add_edge("report_generation", "output_delivery")
        
        # Configure error handling
        for task_id in tasks.keys():
            orchestrator.add_error_handler(task_id, self._create_error_handler(task_id, params))
        
        # Add conditional logic for quality checks
        if params.get('quality_checks', True):
            orchestrator.add_conditional_edge(
                "data_validation",
                self._quality_check_condition,
                {"passed": "data_cleaning" if params.get('data_cleaning', True) else "data_transformation",
                 "failed": "END"}
            )
        
        logger.info("Data pipeline workflow structure configured")
    
    def _create_error_handler(self, task_id: str, params: Dict[str, Any]) -> callable:
        """Create error handler for specific task."""
        def error_handler(error, context):
            logger.error(f"Error in {task_id}: {error}")
            
            # Implement retry logic for certain tasks
            if task_id in ["data_ingestion", "output_delivery"]:
                retry_count = context.get("retry_count", 0)
                if retry_count < 3:
                    context["retry_count"] = retry_count + 1
                    logger.info(f"Retrying {task_id} (attempt {retry_count + 1})")
                    return "retry"
            
            # For critical errors, stop the workflow
            error_threshold = params.get('error_threshold', 5.0)
            if context.get("error_rate", 0) > error_threshold:
                logger.error(f"Error rate exceeded threshold: {error_threshold}%")
                return "stop"
            
            return "continue"
        
        return error_handler
    
    def _quality_check_condition(self, state: Dict[str, Any]) -> str:
        """Determine next step based on quality check results."""
        validation_report = state.get("validation_report", {})
        quality_score = validation_report.get("quality_score", 100)
        
        # If quality score is too low, stop the workflow
        if quality_score < 70:
            logger.warning(f"Data quality score too low: {quality_score}%")
            return "failed"
        
        return "passed"