"""
CLI Tool for Template Management in CrewGraph AI

Provides command-line interface for managing workflow templates:
- List available templates
- Search templates by category/tags
- Create workflows from templates
- Import/export templates
- Marketplace operations

Created by: Vatsal216
Date: 2025-07-23
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..templates import (
    get_template_marketplace,
    get_template_registry,
    load_template_from_file,
    save_template_as_json,
    save_template_as_yaml,
    search_marketplace_templates,
    TemplateCategory,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TemplateCLI:
    """Command-line interface for template management"""
    
    def __init__(self):
        self.registry = get_template_registry()
        self.marketplace = get_template_marketplace()
    
    def list_templates(self, category: Optional[str] = None, show_details: bool = False) -> None:
        """List available templates"""
        templates = list(self.registry.templates.values())
        
        if category:
            try:
                cat_enum = TemplateCategory(category.lower())
                templates = [t for t in templates if t.metadata.category == cat_enum]
            except ValueError:
                print(f"Invalid category: {category}")
                return
        
        if not templates:
            print("No templates found.")
            return
        
        print(f"\nFound {len(templates)} template(s):")
        print("-" * 80)
        
        for template in templates:
            print(f"Name: {template.metadata.name}")
            print(f"Category: {template.metadata.category.value}")
            print(f"Author: {template.metadata.author}")
            
            if show_details:
                print(f"Description: {template.metadata.description}")
                print(f"Version: {template.metadata.version}")
                print(f"Complexity: {template.metadata.complexity}")
                print(f"Estimated Time: {template.metadata.estimated_time}")
                print(f"Tags: {', '.join(template.metadata.tags)}")
                print(f"Parameters: {len(template.parameters)}")
                print(f"Steps: {len(template.steps)}")
            
            print("-" * 80)
    
    def search_templates(self, query: str, category: Optional[str] = None, 
                        tags: Optional[List[str]] = None) -> None:
        """Search templates in marketplace"""
        results = search_marketplace_templates(
            query=query,
            category=category,
            tags=tags,
            limit=20
        )
        
        if not results:
            print("No templates found matching the search criteria.")
            return
        
        print(f"\nFound {len(results)} template(s):")
        print("-" * 80)
        
        for marketplace_template in results:
            template = marketplace_template.template
            print(f"Name: {template.metadata.name}")
            print(f"Description: {template.metadata.description}")
            print(f"Category: {template.metadata.category.value}")
            print(f"Rating: {marketplace_template.rating.average_rating:.1f}/5.0 ({marketplace_template.rating.total_ratings} reviews)")
            print(f"Downloads: {marketplace_template.stats.downloads}")
            print(f"Source: {marketplace_template.source.value}")
            if marketplace_template.featured:
                print("⭐ FEATURED")
            if marketplace_template.verified:
                print("✓ VERIFIED")
            print("-" * 80)
    
    def show_template_details(self, template_name: str) -> None:
        """Show detailed information about a specific template"""
        template = self.registry.get_template(template_name)
        
        if not template:
            print(f"Template '{template_name}' not found.")
            return
        
        print(f"\nTemplate Details: {template.metadata.name}")
        print("=" * 80)
        print(f"Description: {template.metadata.description}")
        print(f"Version: {template.metadata.version}")
        print(f"Category: {template.metadata.category.value}")
        print(f"Author: {template.metadata.author}")
        print(f"Complexity: {template.metadata.complexity}")
        print(f"Estimated Time: {template.metadata.estimated_time}")
        print(f"Tags: {', '.join(template.metadata.tags)}")
        
        if template.metadata.requirements:
            print(f"Requirements: {', '.join(template.metadata.requirements)}")
        
        print(f"\nParameters ({len(template.parameters)}):")
        for param in template.parameters:
            required = "required" if param.required else "optional"
            default = f", default: {param.default_value}" if param.default_value is not None else ""
            print(f"  - {param.name} ({param.param_type}, {required}{default}): {param.description}")
        
        print(f"\nWorkflow Steps ({len(template.steps)}):")
        for i, step in enumerate(template.steps, 1):
            deps = f" (depends on: {', '.join(step.dependencies)})" if step.dependencies else ""
            optional = " [OPTIONAL]" if step.optional else ""
            print(f"  {i}. {step.name}{optional}")
            print(f"     Agent: {step.agent_role}")
            print(f"     Task: {step.task_description}")
            if step.tools:
                print(f"     Tools: {', '.join(step.tools)}")
            print(f"     {deps}")
    
    def export_template(self, template_name: str, output_path: str, format: str = "auto") -> None:
        """Export template to file"""
        template = self.registry.get_template(template_name)
        
        if not template:
            print(f"Template '{template_name}' not found.")
            return
        
        try:
            if format == "yaml" or (format == "auto" and output_path.endswith(('.yaml', '.yml'))):
                save_template_as_yaml(template, output_path)
            else:
                save_template_as_json(template, output_path)
            
            print(f"Template exported to: {output_path}")
        except Exception as e:
            print(f"Error exporting template: {e}")
    
    def import_template(self, file_path: str) -> None:
        """Import template from file"""
        try:
            template = load_template_from_file(file_path)
            success = self.registry.register_template(template)
            
            if success:
                print(f"Template '{template.metadata.name}' imported successfully.")
            else:
                print(f"Failed to import template from {file_path}")
        except Exception as e:
            print(f"Error importing template: {e}")
    
    def create_workflow(self, template_name: str, parameters_file: Optional[str] = None,
                       workflow_name: Optional[str] = None, output_path: Optional[str] = None) -> None:
        """Create workflow from template"""
        template = self.registry.get_template(template_name)
        
        if not template:
            print(f"Template '{template_name}' not found.")
            return
        
        # Load parameters
        params = {}
        if parameters_file:
            try:
                with open(parameters_file, 'r') as f:
                    params = json.load(f)
            except Exception as e:
                print(f"Error loading parameters file: {e}")
                return
        
        try:
            # Validate parameters
            validated_params = template.validate_parameters(params)
            
            # Create workflow
            workflow = template.create_workflow(validated_params, workflow_name)
            
            print(f"Workflow '{workflow.name}' created successfully from template '{template_name}'")
            
            if output_path:
                # Save workflow configuration
                workflow_config = {
                    "name": workflow.name,
                    "template": template_name,
                    "parameters": validated_params,
                    "created_at": "2025-07-23T16:30:00Z"
                }
                
                with open(output_path, 'w') as f:
                    json.dump(workflow_config, f, indent=2)
                
                print(f"Workflow configuration saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating workflow: {e}")
    
    def show_marketplace_stats(self) -> None:
        """Show marketplace statistics"""
        stats = self.marketplace.get_stats()
        
        print("\nTemplate Marketplace Statistics:")
        print("=" * 50)
        print(f"Total Templates: {stats['total_templates']}")
        print(f"Total Downloads: {stats['total_downloads']}")
        print(f"Average Rating: {stats['average_rating']:.2f}/5.0")
        print(f"Categories: {stats['categories']}")
        print(f"Tags: {stats['tags']}")
        print(f"Featured Templates: {stats['featured_templates']}")
        print(f"Verified Templates: {stats['verified_templates']}")
        
        # Show categories
        categories = self.marketplace.get_categories()
        if categories:
            print(f"\nAvailable Categories: {', '.join(categories)}")
        
        # Show popular templates
        popular = self.marketplace.get_popular_templates(5)
        if popular:
            print("\nMost Popular Templates:")
            for i, mt in enumerate(popular, 1):
                print(f"  {i}. {mt.template.metadata.name} ({mt.stats.downloads} downloads)")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CrewGraph AI Template Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list --category data_processing
  %(prog)s search --query "data analysis" --tags analytics
  %(prog)s show "Data Processing Pipeline"
  %(prog)s export "Research Workflow" research_template.yaml
  %(prog)s import my_template.json
  %(prog)s create "Data Pipeline" --params params.json --output workflow.json
  %(prog)s stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available templates')
    list_parser.add_argument('--category', help='Filter by category')
    list_parser.add_argument('--details', action='store_true', help='Show detailed information')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search templates')
    search_parser.add_argument('--query', help='Search query')
    search_parser.add_argument('--category', help='Filter by category')
    search_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show template details')
    show_parser.add_argument('template_name', help='Name of template to show')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export template to file')
    export_parser.add_argument('template_name', help='Name of template to export')
    export_parser.add_argument('output_path', help='Output file path')
    export_parser.add_argument('--format', choices=['json', 'yaml', 'auto'], default='auto', help='Output format')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import template from file')
    import_parser.add_argument('file_path', help='Path to template file')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create workflow from template')
    create_parser.add_argument('template_name', help='Name of template to use')
    create_parser.add_argument('--params', help='JSON file with parameters')
    create_parser.add_argument('--name', help='Workflow name')
    create_parser.add_argument('--output', help='Output path for workflow configuration')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show marketplace statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = TemplateCLI()
    
    try:
        if args.command == 'list':
            cli.list_templates(args.category, args.details)
        elif args.command == 'search':
            cli.search_templates(args.query, args.category, args.tags)
        elif args.command == 'show':
            cli.show_template_details(args.template_name)
        elif args.command == 'export':
            cli.export_template(args.template_name, args.output_path, args.format)
        elif args.command == 'import':
            cli.import_template(args.file_path)
        elif args.command == 'create':
            cli.create_workflow(args.template_name, args.params, args.name, args.output)
        elif args.command == 'stats':
            cli.show_marketplace_stats()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()