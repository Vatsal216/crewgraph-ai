"""
Unit tests for the Templates System

Tests for template functionality including:
- Template creation and validation
- Template registry operations
- Template marketplace functionality
- Template format conversion (YAML/JSON)
- Template inheritance and composition
- Template CLI operations

Created by: Vatsal216
Date: 2025-07-23
"""

import json
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from crewgraph_ai.templates import (
    WorkflowTemplate,
    TemplateRegistry,
    TemplateMarketplace,
    TemplateCategory,
    TemplateMetadata,
    TemplateParameter,
    TemplateStep,
    get_template_registry,
    get_template_marketplace,
    CustomerSupportTemplate,
    DataPipelineTemplate,
    TemplateSerializer,
    TemplateDeserializer,
    DynamicWorkflowTemplate,
    TemplateInheritance,
    ParameterInjector,
    TemplateSource,
    TemplateStatus,
)


class TestTemplateMetadata:
    """Test template metadata functionality"""

    def test_template_metadata_creation(self):
        """Test creating template metadata"""
        metadata = TemplateMetadata(
            name="Test Template",
            description="A test template",
            version="1.0.0",
            category=TemplateCategory.AUTOMATION,
            author="Test Author",
            tags=["test", "automation"],
            complexity="simple",
            estimated_time="2-3 minutes"
        )
        
        assert metadata.name == "Test Template"
        assert metadata.description == "A test template"
        assert metadata.version == "1.0.0"
        assert metadata.category == TemplateCategory.AUTOMATION
        assert metadata.author == "Test Author"
        assert "test" in metadata.tags
        assert "automation" in metadata.tags
        assert metadata.complexity == "simple"
        assert metadata.estimated_time == "2-3 minutes"

    def test_template_metadata_defaults(self):
        """Test template metadata default values"""
        metadata = TemplateMetadata(
            name="Test Template",
            description="A test template"
        )
        
        assert metadata.version == "1.0.0"
        assert metadata.category == TemplateCategory.CUSTOM
        assert metadata.author == "CrewGraph AI"
        assert metadata.tags == []
        assert metadata.complexity == "medium"
        assert metadata.estimated_time == "5-10 minutes"


class TestTemplateParameter:
    """Test template parameter functionality"""

    def test_template_parameter_creation(self):
        """Test creating template parameters"""
        param = TemplateParameter(
            name="test_param",
            description="A test parameter",
            param_type="str",
            required=True,
            default_value="default",
            validation_rules={"min_length": 3},
            examples=["example1", "example2"]
        )
        
        assert param.name == "test_param"
        assert param.description == "A test parameter"
        assert param.param_type == "str"
        assert param.required is True
        assert param.default_value == "default"
        assert param.validation_rules == {"min_length": 3}
        assert param.examples == ["example1", "example2"]

    def test_template_parameter_defaults(self):
        """Test template parameter default values"""
        param = TemplateParameter(
            name="test_param",
            description="A test parameter"
        )
        
        assert param.param_type == "str"
        assert param.required is True
        assert param.default_value is None
        assert param.validation_rules == {}
        assert param.examples == []


class TestTemplateStep:
    """Test template step functionality"""

    def test_template_step_creation(self):
        """Test creating template steps"""
        step = TemplateStep(
            step_id="test_step",
            name="Test Step",
            description="A test step",
            agent_role="Test Agent",
            task_description="Perform test task",
            inputs=["input1", "input2"],
            outputs=["output1"],
            dependencies=["prev_step"],
            tools=["test_tool"],
            configuration={"param": "value"},
            optional=True
        )
        
        assert step.step_id == "test_step"
        assert step.name == "Test Step"
        assert step.description == "A test step"
        assert step.agent_role == "Test Agent"
        assert step.task_description == "Perform test task"
        assert step.inputs == ["input1", "input2"]
        assert step.outputs == ["output1"]
        assert step.dependencies == ["prev_step"]
        assert step.tools == ["test_tool"]
        assert step.configuration == {"param": "value"}
        assert step.optional is True

    def test_template_step_defaults(self):
        """Test template step default values"""
        step = TemplateStep(
            step_id="test_step",
            name="Test Step",
            description="A test step",
            agent_role="Test Agent",
            task_description="Perform test task"
        )
        
        assert step.inputs == []
        assert step.outputs == []
        assert step.dependencies == []
        assert step.tools == []
        assert step.configuration == {}
        assert step.optional is False


@pytest.mark.templates
class TestTemplateRegistry:
    """Test template registry functionality"""

    def test_registry_initialization(self):
        """Test template registry initialization"""
        registry = TemplateRegistry()
        assert isinstance(registry, TemplateRegistry)
        assert len(registry.list_templates()) >= 0

    def test_template_registration(self):
        """Test registering templates"""
        registry = TemplateRegistry()
        template = CustomerSupportTemplate()
        
        success = registry.register_template(template)
        assert success is True
        
        # Check template is registered
        templates = registry.list_templates()
        assert template.metadata.name in templates

    def test_template_retrieval(self):
        """Test retrieving templates"""
        registry = TemplateRegistry()
        template = CustomerSupportTemplate()
        registry.register_template(template)
        
        retrieved = registry.get_template(template.metadata.name)
        assert retrieved is not None
        assert retrieved.metadata.name == template.metadata.name

    def test_template_search(self):
        """Test searching templates"""
        registry = TemplateRegistry()
        template = CustomerSupportTemplate()
        registry.register_template(template)
        
        # Search by name
        results = registry.search_templates("Customer")
        assert len(results) >= 1
        assert template.metadata.name in results

    def test_template_listing_by_category(self):
        """Test listing templates by category"""
        registry = TemplateRegistry()
        template = CustomerSupportTemplate()
        registry.register_template(template)
        
        templates = registry.list_templates(category=TemplateCategory.CUSTOMER_SERVICE)
        assert len(templates) >= 1
        assert template.metadata.name in templates

    def test_duplicate_template_registration(self):
        """Test handling duplicate template registration"""
        registry = TemplateRegistry()
        template = CustomerSupportTemplate()
        
        # Register once
        success1 = registry.register_template(template)
        assert success1 is True
        
        # Try to register again
        success2 = registry.register_template(template)
        assert success2 is False  # Should fail for duplicate


@pytest.mark.marketplace
class TestTemplateMarketplace:
    """Test template marketplace functionality"""

    def test_marketplace_initialization(self):
        """Test marketplace initialization"""
        marketplace = TemplateMarketplace()
        assert isinstance(marketplace, TemplateMarketplace)
        assert len(marketplace.templates) == 0

    def test_template_addition(self):
        """Test adding templates to marketplace"""
        marketplace = TemplateMarketplace()
        template = CustomerSupportTemplate()
        
        template_id = marketplace.add_template(template, TemplateSource.OFFICIAL)
        assert template_id is not None
        assert template_id in marketplace.templates
        
        marketplace_template = marketplace.templates[template_id]
        assert marketplace_template.template.metadata.name == template.metadata.name
        assert marketplace_template.source == TemplateSource.OFFICIAL

    def test_template_search(self):
        """Test searching templates in marketplace"""
        marketplace = TemplateMarketplace()
        template = CustomerSupportTemplate()
        marketplace.add_template(template, TemplateSource.OFFICIAL)
        
        # Search by query
        results = marketplace.search_templates(query="Customer")
        assert len(results) >= 1
        
        # Search by category
        results = marketplace.search_templates(category="customer_service")
        assert len(results) >= 1
        
        # Search by tags
        results = marketplace.search_templates(tags=["customer_service"])
        assert len(results) >= 1

    def test_template_rating(self):
        """Test rating templates"""
        marketplace = TemplateMarketplace()
        template = CustomerSupportTemplate()
        template_id = marketplace.add_template(template)
        
        # Rate template
        success = marketplace.rate_template(template_id, 5, "Excellent template!")
        assert success is True
        
        marketplace_template = marketplace.templates[template_id]
        assert marketplace_template.rating.average_rating == 5.0
        assert marketplace_template.rating.total_ratings == 1
        assert len(marketplace_template.rating.reviews) == 1

    def test_template_download(self):
        """Test downloading templates"""
        marketplace = TemplateMarketplace()
        template = CustomerSupportTemplate()
        template_id = marketplace.add_template(template)
        
        downloaded = marketplace.download_template(template_id)
        assert downloaded is not None
        assert downloaded.metadata.name == template.metadata.name
        
        # Check statistics updated
        marketplace_template = marketplace.templates[template_id]
        assert marketplace_template.stats.downloads == 1

    def test_marketplace_stats(self):
        """Test marketplace statistics"""
        marketplace = TemplateMarketplace()
        template = CustomerSupportTemplate()
        marketplace.add_template(template)
        
        stats = marketplace.get_stats()
        assert stats["total_templates"] >= 1
        assert "total_downloads" in stats
        assert "average_rating" in stats
        assert "categories" in stats

    def test_featured_templates(self):
        """Test featured templates functionality"""
        marketplace = TemplateMarketplace()
        template = CustomerSupportTemplate()
        template_id = marketplace.add_template(template)
        
        # Mark as featured
        marketplace.templates[template_id].featured = True
        
        featured = marketplace.get_featured_templates()
        assert len(featured) >= 1
        assert featured[0].template.metadata.name == template.metadata.name


@pytest.mark.templates
class TestTemplateFormats:
    """Test template format conversion (YAML/JSON)"""

    def test_template_serialization_to_dict(self):
        """Test serializing template to dictionary"""
        template = CustomerSupportTemplate()
        template_dict = TemplateSerializer.to_dict(template)
        
        assert "metadata" in template_dict
        assert "parameters" in template_dict
        assert "steps" in template_dict
        
        assert template_dict["metadata"]["name"] == template.metadata.name
        assert len(template_dict["parameters"]) == len(template.parameters)
        assert len(template_dict["steps"]) == len(template.steps)

    def test_template_serialization_to_json(self):
        """Test serializing template to JSON"""
        template = CustomerSupportTemplate()
        json_str = TemplateSerializer.to_json(template)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "parameters" in parsed
        assert "steps" in parsed

    @pytest.mark.skipif(not hasattr(TemplateSerializer, 'to_yaml'), 
                       reason="YAML support requires PyYAML")
    def test_template_serialization_to_yaml(self):
        """Test serializing template to YAML"""
        template = CustomerSupportTemplate()
        
        try:
            yaml_str = TemplateSerializer.to_yaml(template)
            assert isinstance(yaml_str, str)
            assert "metadata:" in yaml_str
            assert "parameters:" in yaml_str
            assert "steps:" in yaml_str
        except ImportError:
            pytest.skip("PyYAML not available")

    def test_template_deserialization_from_dict(self):
        """Test deserializing template from dictionary"""
        template = CustomerSupportTemplate()
        template_dict = TemplateSerializer.to_dict(template)
        
        deserialized = TemplateDeserializer.from_dict(template_dict)
        assert isinstance(deserialized, DynamicWorkflowTemplate)
        assert deserialized.metadata.name == template.metadata.name
        assert len(deserialized.parameters) == len(template.parameters)
        assert len(deserialized.steps) == len(template.steps)

    def test_template_deserialization_from_json(self):
        """Test deserializing template from JSON"""
        template = CustomerSupportTemplate()
        json_str = TemplateSerializer.to_json(template)
        
        deserialized = TemplateDeserializer.from_json(json_str)
        assert isinstance(deserialized, DynamicWorkflowTemplate)
        assert deserialized.metadata.name == template.metadata.name


@pytest.mark.templates
class TestTemplateInheritance:
    """Test template inheritance and composition"""

    def test_template_extension(self):
        """Test extending a base template"""
        base_template = CustomerSupportTemplate()
        
        extensions = {
            "metadata": {
                "name": "Extended Customer Support",
                "version": "2.0.0"
            },
            "parameters": [
                {
                    "name": "new_param",
                    "description": "A new parameter",
                    "param_type": "str",
                    "required": False,
                    "default_value": "default"
                }
            ]
        }
        
        extended = TemplateInheritance.extend_template(base_template, extensions)
        assert isinstance(extended, DynamicWorkflowTemplate)
        assert extended.metadata.name == "Extended Customer Support"
        assert extended.metadata.version == "2.0.0"
        assert len(extended.parameters) == len(base_template.parameters) + 1

    def test_template_composition(self):
        """Test composing multiple templates"""
        template1 = CustomerSupportTemplate()
        template2 = DataPipelineTemplate()
        
        composed = TemplateInheritance.compose_templates(
            [template1, template2],
            {"name": "Composed Template", "description": "A composed template"}
        )
        
        assert isinstance(composed, DynamicWorkflowTemplate)
        assert composed.metadata.name == "Composed Template"
        # Should have parameters and steps from both templates
        total_params = len(template1.parameters) + len(template2.parameters)
        total_steps = len(template1.steps) + len(template2.steps)
        assert len(composed.parameters) <= total_params  # May have fewer due to duplicates
        assert len(composed.steps) <= total_steps  # May have fewer due to duplicates


@pytest.mark.templates
class TestParameterInjection:
    """Test parameter injection functionality"""

    def test_parameter_injection(self):
        """Test injecting parameters into template data"""
        template_data = {
            "metadata": {
                "name": "Test Template for ${environment}",
                "description": "A template for ${use_case} in ${environment}"
            },
            "parameters": [],
            "steps": [
                {
                    "step_id": "process_${data_type}",
                    "name": "Process ${data_type}",
                    "description": "Process ${data_type} data in ${environment}",
                    "agent_role": "Data Processor",
                    "task_description": "Process the ${data_type} data"
                }
            ]
        }
        
        parameters = {
            "environment": "production",
            "use_case": "analytics",
            "data_type": "customer"
        }
        
        injected = ParameterInjector.inject_parameters(template_data, parameters)
        
        assert injected["metadata"]["name"] == "Test Template for production"
        assert injected["metadata"]["description"] == "A template for analytics in production"
        assert injected["steps"][0]["step_id"] == "process_customer"
        assert injected["steps"][0]["name"] == "Process customer"
        assert "production" in injected["steps"][0]["description"]


@pytest.mark.templates
class TestBuiltinTemplates:
    """Test built-in template functionality"""

    def test_customer_support_template(self):
        """Test CustomerSupportTemplate"""
        template = CustomerSupportTemplate()
        
        assert template.metadata.name == "Customer Support Automation"
        assert template.metadata.category == TemplateCategory.CUSTOMER_SERVICE
        assert len(template.parameters) > 0
        assert len(template.steps) > 0
        
        # Test parameter validation
        params = {
            "ticket_content": "Test ticket content",
            "customer_type": "premium",
            "priority_threshold": 8,
            "response_tone": "professional"
        }
        validated = template.validate_parameters(params)
        assert validated["ticket_content"] == "Test ticket content"
        assert validated["customer_type"] == "premium"

    def test_data_pipeline_template(self):
        """Test DataPipelineTemplate"""
        template = DataPipelineTemplate()
        
        assert template.metadata.category == TemplateCategory.DATA_PROCESSING
        assert len(template.parameters) > 0
        assert len(template.steps) > 0

    def test_template_workflow_creation(self):
        """Test creating workflow from template"""
        template = CustomerSupportTemplate()
        
        params = {
            "ticket_content": "Test ticket content",
            "customer_type": "basic"
        }
        
        # Mock the workflow creation since we don't have full orchestrator setup
        with patch.object(template, '_create_agents') as mock_agents, \
             patch.object(template, '_create_tasks') as mock_tasks, \
             patch.object(template, '_configure_workflow_structure') as mock_config:
            
            mock_agents.return_value = {"agent1": "mock_agent"}
            mock_tasks.return_value = {"task1": "mock_task"}
            mock_config.return_value = None
            
            workflow = template.create_workflow(params, "test_workflow")
            assert workflow is not None
            assert workflow.name == "test_workflow"


@pytest.mark.cli
class TestTemplateCLI:
    """Test template CLI functionality"""

    @pytest.fixture
    def cli_instance(self):
        """Create CLI instance for testing"""
        from crewgraph_ai.templates.cli import TemplateCLI
        return TemplateCLI()

    def test_cli_initialization(self, cli_instance):
        """Test CLI initialization"""
        assert cli_instance.registry is not None
        assert cli_instance.marketplace is not None

    @patch('builtins.print')
    def test_cli_list_templates(self, mock_print, cli_instance):
        """Test CLI list templates command"""
        cli_instance.list_templates()
        mock_print.assert_called()

    @patch('builtins.print')
    def test_cli_search_templates(self, mock_print, cli_instance):
        """Test CLI search templates command"""
        cli_instance.search_templates("customer")
        mock_print.assert_called()

    @patch('builtins.print')
    def test_cli_show_template_details(self, mock_print, cli_instance):
        """Test CLI show template details command"""
        # Should handle non-existent template gracefully
        cli_instance.show_template_details("NonExistentTemplate")
        mock_print.assert_called()

    @patch('builtins.print')  
    def test_cli_marketplace_stats(self, mock_print, cli_instance):
        """Test CLI marketplace stats command"""
        cli_instance.show_marketplace_stats()
        mock_print.assert_called()


@pytest.mark.performance
class TestTemplatePerformance:
    """Test template system performance"""

    def test_template_creation_performance(self):
        """Test template creation performance"""
        import time
        
        start_time = time.time()
        
        # Create multiple templates
        templates = []
        for i in range(10):
            template = CustomerSupportTemplate()
            templates.append(template)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 10 templates in reasonable time
        assert creation_time < 5.0  # Less than 5 seconds
        assert len(templates) == 10

    def test_registry_performance(self):
        """Test registry performance with multiple templates"""
        registry = TemplateRegistry()
        
        # Register multiple templates
        start_time = time.time()
        
        for i in range(20):
            template = CustomerSupportTemplate()
            # Modify name to avoid duplicates
            template.metadata.name = f"Customer Support {i}"
            registry.register_template(template)
        
        end_time = time.time()
        registration_time = end_time - start_time
        
        # Should register 20 templates in reasonable time
        assert registration_time < 2.0  # Less than 2 seconds
        assert len(registry.list_templates()) >= 20

    def test_search_performance(self):
        """Test search performance"""
        marketplace = TemplateMarketplace()
        
        # Add multiple templates
        for i in range(50):
            template = CustomerSupportTemplate()
            template.metadata.name = f"Template {i}"
            template.metadata.tags = [f"tag_{i % 5}", "common_tag"]
            marketplace.add_template(template)
        
        # Test search performance
        start_time = time.time()
        
        results = marketplace.search_templates(query="Template", limit=10)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Should search through 50 templates quickly
        assert search_time < 1.0  # Less than 1 second
        assert len(results) >= 10


@pytest.mark.regression
class TestTemplateRegression:
    """Regression tests for template system"""

    def test_template_backwards_compatibility(self):
        """Test backwards compatibility of template API"""
        # Test that old template creation still works
        template = CustomerSupportTemplate()
        
        # Essential attributes should exist
        assert hasattr(template, 'metadata')
        assert hasattr(template, 'parameters')
        assert hasattr(template, 'steps')
        assert hasattr(template, 'template_id')
        
        # Essential methods should exist
        assert hasattr(template, 'validate_parameters')
        assert hasattr(template, 'create_workflow')
        assert hasattr(template, 'get_info')

    def test_registry_backwards_compatibility(self):
        """Test backwards compatibility of registry API"""
        registry = get_template_registry()
        
        # Essential methods should exist
        assert hasattr(registry, 'register_template')
        assert hasattr(registry, 'get_template')
        assert hasattr(registry, 'list_templates')
        assert hasattr(registry, 'search_templates')

    def test_marketplace_backwards_compatibility(self):
        """Test backwards compatibility of marketplace API"""
        marketplace = get_template_marketplace()
        
        # Essential methods should exist
        assert hasattr(marketplace, 'add_template')
        assert hasattr(marketplace, 'search_templates')
        assert hasattr(marketplace, 'download_template')
        assert hasattr(marketplace, 'get_stats')


if __name__ == "__main__":
    pytest.main([__file__])