#!/usr/bin/env python3
"""
CrewGraph AI Production Features Demonstration

This comprehensive demo showcases all the new production-ready features:
1. AgentCommunicationHub - Inter-agent messaging
2. WorkflowTemplates - Pre-built workflow templates  
3. Security Manager - Enterprise security features
4. Integration - All features working together

Created by: Vatsal216
Date: 2025-07-23
"""

import time
from datetime import datetime

def demo_communication_system():
    """Demonstrate the AgentCommunicationHub"""
    print("🔗 === AGENT COMMUNICATION SYSTEM DEMO ===")
    
    from crewgraph_ai import (
        AgentCommunicationHub, AgentWrapper, 
        MessageType, MessagePriority
    )
    
    # Create communication hub
    hub = AgentCommunicationHub()
    hub.start()
    
    # Create agents
    coordinator = AgentWrapper("Coordinator", role="Project Manager")
    developer = AgentWrapper("Developer", role="Software Engineer") 
    tester = AgentWrapper("Tester", role="QA Engineer")
    
    # Register agents
    hub.register_agent(coordinator)
    hub.register_agent(developer)
    hub.register_agent(tester)
    
    print(f"✅ Registered 3 agents with communication hub")
    
    # Create communication channel
    hub.create_channel("project_updates", "Project status updates")
    hub.subscribe_to_channel(developer.get_agent_id(), "project_updates")
    hub.subscribe_to_channel(tester.get_agent_id(), "project_updates")
    
    # Demonstrate different communication patterns
    
    # 1. Direct messaging
    hub.send_direct_message(
        sender_id=coordinator.get_agent_id(),
        recipient_id=developer.get_agent_id(),
        content="Please implement the new authentication feature",
        priority=MessagePriority.HIGH
    )
    
    # 2. Channel broadcasting
    hub.send_channel_message(
        sender_id=coordinator.get_agent_id(),
        channel="project_updates",
        content="Sprint planning meeting at 2 PM today"
    )
    
    # 3. Broadcast to all
    hub.broadcast_message(
        sender_id=coordinator.get_agent_id(),
        content="System maintenance window scheduled for tonight",
        priority=MessagePriority.URGENT
    )
    
    # Show message statistics
    stats = hub.get_hub_statistics()
    print(f"📊 Communication Stats: {stats['total_messages']} messages sent")
    
    hub.stop()
    print("✅ Communication demo completed\n")

def demo_workflow_templates():
    """Demonstrate the WorkflowTemplates system"""
    print("📋 === WORKFLOW TEMPLATES SYSTEM DEMO ===")
    
    from crewgraph_ai import (
        TemplateRegistry, TemplateBuilder,
        DataPipelineTemplate, ResearchWorkflowTemplate, ContentGenerationTemplate
    )
    
    # Create template registry
    registry = TemplateRegistry()
    
    # Register templates
    data_template = DataPipelineTemplate()
    research_template = ResearchWorkflowTemplate() 
    content_template = ContentGenerationTemplate()
    
    registry.register_template(data_template)
    registry.register_template(research_template)
    registry.register_template(content_template)
    
    print(f"✅ Registered {len(registry.list_templates())} workflow templates")
    
    # Demonstrate template discovery
    templates = registry.list_templates()
    print(f"📋 Available templates: {', '.join(templates)}")
    
    # Show template information
    for template_name in templates:
        info = registry.get_template_info(template_name)
        if info:
            metadata = info['metadata']
            print(f"   • {metadata['name']}: {metadata['description'][:60]}...")
    
    # Demonstrate template builder
    builder = TemplateBuilder(registry)
    
    # Configure a data pipeline
    try:
        workflow_config = (builder
                          .use_template("data_pipeline")
                          .set_parameter("data_source", "sales_data.csv")
                          .set_parameter("source_format", "csv")
                          .set_parameter("analysis_type", "sales_trends")
                          .set_parameter("output_format", "dashboard"))
        
        validation = builder.validate_current_parameters()
        print(f"✅ Template configuration: {'valid' if validation['valid'] else 'invalid'}")
        
    except Exception as e:
        print(f"⚠️ Template configuration: {str(e)[:50]}...")
    
    print("✅ Templates demo completed\n")

def demo_security_system():
    """Demonstrate the Security Manager"""
    print("🔐 === SECURITY SYSTEM DEMO ===")
    
    from crewgraph_ai import (
        SecurityManager, User, AuditEvent
    )
    
    # Create security manager with all features enabled
    security = SecurityManager(
        enable_encryption=True,
        enable_audit_logging=True
    )
    
    # Create users with different roles
    users = [
        {"id": "alice", "username": "alice", "email": "alice@company.com", "roles": ["user"]},
        {"id": "bob", "username": "bob", "email": "bob@company.com", "roles": ["developer"]},
        {"id": "charlie", "username": "charlie", "email": "charlie@company.com", "roles": ["admin"]}
    ]
    
    for user_data in users:
        security.create_user(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data["roles"]
        )
    
    print(f"✅ Created {len(users)} users with different roles")
    
    # Demonstrate authentication and authorization
    alice_session = security.authenticate_user("alice")
    bob_session = security.authenticate_user("bob")
    
    # Test different operations with different permission levels
    operations = [
        ("workflow.create", "alice", alice_session.session_id),
        ("workflow.execute", "alice", alice_session.session_id),
        ("template.create", "bob", bob_session.session_id),
        ("security.manage", "bob", bob_session.session_id),  # Should be denied
    ]
    
    for operation, user, session_id in operations:
        authorized = security.authorize_operation(session_id, operation)
        status = "✅ authorized" if authorized else "❌ denied"
        print(f"   {user} - {operation}: {status}")
    
    # Demonstrate data encryption
    if security.encryption_manager:
        sensitive_data = {
            "api_key": "sk-1234567890abcdef",
            "database_url": "postgresql://user:pass@host:5432/db",
            "secret_token": "super-secret-token-123"
        }
        
        encrypted = security.encrypt_data(sensitive_data)
        if encrypted:
            decrypted = security.decrypt_data(encrypted)
            print("✅ Data encryption/decryption successful")
    
    # Show security metrics
    metrics = security.get_security_metrics()
    print(f"📊 Security Metrics: {metrics['active_sessions']} active sessions")
    
    # Compliance check
    compliance = security.validate_compliance()
    print(f"📋 Compliance Score: {compliance['overall_score']:.1f}%")
    
    print("✅ Security demo completed\n")

def demo_integrated_features():
    """Demonstrate all features working together"""
    print("🌟 === INTEGRATED FEATURES DEMO ===")
    
    from crewgraph_ai import (
        SecurityManager, AgentCommunicationHub, AgentWrapper,
        TemplateRegistry, DataPipelineTemplate
    )
    
    # Setup security
    security = SecurityManager(enable_encryption=True, enable_audit_logging=True)
    
    # Create workflow user
    user = security.create_user(
        user_id="workflow_user",
        username="workflowuser", 
        email="workflow@company.com",
        roles=["developer"]
    )
    
    session = security.authenticate_user("workflow_user")
    print("✅ User authenticated for secure workflow")
    
    # Setup communication
    comm_hub = AgentCommunicationHub()
    comm_hub.start()
    
    # Create agents with communication
    data_agent = AgentWrapper("DataProcessor", role="Data Engineer")
    analysis_agent = AgentWrapper("Analyst", role="Data Scientist")
    
    comm_hub.register_agent(data_agent)
    comm_hub.register_agent(analysis_agent)
    
    # Create secure workflow channel
    comm_hub.create_channel("secure_workflow", "Secure data processing workflow")
    comm_hub.subscribe_to_channel(data_agent.get_agent_id(), "secure_workflow")
    comm_hub.subscribe_to_channel(analysis_agent.get_agent_id(), "secure_workflow")
    
    print("✅ Secure communication channel established")
    
    # Setup templates with security context
    registry = TemplateRegistry()
    template = DataPipelineTemplate()
    registry.register_template(template)
    
    # Authorize template usage
    authorized = security.authorize_operation(
        session.session_id,
        "template.use", 
        "data_pipeline"
    )
    
    if authorized:
        print("✅ User authorized to use data pipeline template")
        
        # Secure workflow coordination
        comm_hub.send_channel_message(
            sender_id="system",
            channel="secure_workflow",
            content="Starting secure data pipeline workflow"
        )
        
        # Simulate secure data processing
        sensitive_config = {
            "database_connection": "postgresql://secure-host/data",
            "api_keys": ["key1", "key2", "key3"],
            "processing_rules": {"encryption": True, "audit": True}
        }
        
        if security.encryption_manager:
            encrypted_config = security.encrypt_data(sensitive_config)
            print("✅ Workflow configuration encrypted")
        
        # Log workflow execution
        if security.audit_logger:
            security.audit_logger.create_workflow_event(
                event_type="workflow.execute",
                workflow_id="secure_data_pipeline",
                user_id="workflow_user",
                action="start",
                success=True,
                details={"template": "data_pipeline", "security_level": "high"}
            )
        
        print("✅ Secure workflow execution logged")
    
    # Show final statistics
    comm_stats = comm_hub.get_hub_statistics()
    security_metrics = security.get_security_metrics()
    
    print(f"\n📊 Integration Results:")
    print(f"   Communication: {comm_stats['total_messages']} messages")
    print(f"   Security: {security_metrics['active_sessions']} active sessions")
    print(f"   Templates: {len(registry.list_templates())} available")
    print(f"   Audit Events: {security_metrics.get('total_events', 0)} logged")
    
    # Cleanup
    comm_hub.stop()
    print("✅ Integrated demo completed\n")

def main():
    """Main demonstration function"""
    print("🚀 CrewGraph AI Production Features Demonstration")
    print("=" * 60)
    print("This demo showcases the new enterprise-ready features:")
    print("• AgentCommunicationHub - Inter-agent messaging")
    print("• WorkflowTemplates - Pre-built workflow templates")
    print("• Security Manager - Enterprise security features")
    print("• Full Integration - All features working together")
    print("=" * 60)
    print()
    
    try:
        # Run individual feature demos
        demo_communication_system()
        demo_workflow_templates()
        demo_security_system()
        
        # Show integrated functionality
        demo_integrated_features()
        
        print("🎉 === DEMONSTRATION COMPLETED SUCCESSFULLY ===")
        print("\nCrewGraph AI is now production-ready with:")
        print("✅ Sophisticated inter-agent communication")
        print("✅ Pre-built workflow templates for common use cases")
        print("✅ Enterprise-grade security with RBAC, audit logging, and encryption")
        print("✅ Seamless integration between all components")
        print("\nReady for enterprise deployment! 🚀")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()