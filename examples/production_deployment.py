"""
CrewGraph AI - Production Deployment Example
Demonstrates enterprise-grade deployment with monitoring, scaling, and management
"""

import os
import asyncio
import signal
import sys
from typing import Dict, Any, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time

from crewai import Agent, Tool
from crewgraph_ai import CrewGraph, CrewGraphConfig
from crewgraph_ai.memory import RedisMemory, MemoryConfig, MemoryType
from crewgraph_ai.tools import ToolRegistry
from crewgraph_ai.planning import DynamicPlanner, PlannerConfig
from crewgraph_ai.utils import (
    setup_logging, get_logger, LoggerConfig, LogLevel,
    MetricsCollector, PerformanceMonitor, SecurityManager
)
from crewgraph_ai.utils.config import CrewGraphConfig as FullConfig, load_config, save_config

# Setup production logging
production_logging_config = LoggerConfig(
    level=LogLevel.INFO,
    enable_file_logging=True,
    log_file="crewgraph_production.log",
    max_file_size=100 * 1024 * 1024,  # 100MB
    backup_count=10,
    enable_structured_logging=True,
    include_process_info=True,
    log_dir="/var/log/crewgraph",
    custom_fields={
        "environment": "production",
        "version": "1.0.0",
        "deployment": "k8s-cluster-01"
    }
)

setup_logging(production_logging_config)
logger = get_logger(__name__)

@dataclass
class ProductionConfig:
    """Production deployment configuration"""
    environment: str = "production"
    redis_host: str = "redis-cluster.internal"
    redis_port: int = 6379
    redis_password: str = ""
    max_workers: int = 10
    max_concurrent_workflows: int = 50
    health_check_interval: int = 30
    metrics_port: int = 8080
    api_port: int = 8000
    enable_security: bool = True
    enable_monitoring: bool = True
    log_level: str = "INFO"

# Production tools with error handling and monitoring
def production_data_processor(data: str, processing_type: str = "standard") -> Dict[str, Any]:
    """Production-grade data processing tool"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing data with type: {processing_type}")
        
        # Simulate processing work
        if processing_type == "intensive":
            time.sleep(5)  # Simulate heavy processing
        else:
            time.sleep(1)  # Standard processing
        
        # Simulate different outcomes based on data
        if "error" in data.lower():
            raise ValueError("Simulated processing error")
        
        result = {
            "processed_data": f"Processed: {data}",
            "processing_type": processing_type,
            "processing_time": time.time() - start_time,
            "status": "success",
            "metadata": {
                "processor_version": "2.1.0",
                "timestamp": time.time()
            }
        }
        
        logger.info(f"Data processing completed in {result['processing_time']:.2f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Data processing failed after {processing_time:.2f}s: {e}")
        raise

def production_report_generator(analysis_data: Dict[str, Any], format_type: str = "json") -> str:
    """Production report generation with multiple formats"""
    try:
        logger.info(f"Generating production report in format: {format_type}")
        
        if format_type == "json":
            import json
            return json.dumps(analysis_data, indent=2)
        elif format_type == "xml":
            # Simplified XML generation
            xml_content = "<?xml version='1.0' encoding='UTF-8'?>\n<report>\n"
            for key, value in analysis_data.items():
                xml_content += f"  <{key}>{value}</{key}>\n"
            xml_content += "</report>"
            return xml_content
        else:
            return str(analysis_data)
            
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

class ProductionWorkflowManager:
    """Production-grade workflow management system"""
    
    def __init__(self, config: ProductionConfig):
        """Initialize production workflow manager"""
        self.config = config
        self.workflows: Dict[str, CrewGraph] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        self.security_manager = SecurityManager() if config.enable_security else None
        
        # Thread management
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers // 2)
        
        # Shutdown handling
        self.shutdown_event = threading.Event()
        self._setup_signal_handlers()
        
        # Health monitoring
        self.health_check_thread = None
        if config.enable_monitoring:
            self._start_health_monitoring()
        
        logger.info(f"ProductionWorkflowManager initialized for {config.environment} environment")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        def health_monitor():
            while not self.shutdown_event.is_set():
                try:
                    self._perform_health_check()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
        
        self.health_check_thread = threading.Thread(target=health_monitor, daemon=True)
        self.health_check_thread.start()
        logger.info("Health monitoring started")
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        health_status = {
            "timestamp": time.time(),
            "status": "healthy",
            "checks": {}
        }
        
        # Check memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            health_status["checks"]["memory"] = {
                "status": "ok" if memory_percent < 80 else "warning",
                "usage_percent": memory_percent
            }
        except ImportError:
            health_status["checks"]["memory"] = {"status": "unknown", "reason": "psutil not available"}
        
        # Check active workflows
        active_count = len(self.active_executions)
        health_status["checks"]["workflows"] = {
            "status": "ok" if active_count < self.config.max_concurrent_workflows else "warning",
            "active_count": active_count,
            "max_allowed": self.config.max_concurrent_workflows
        }
        
        # Check Redis connectivity (if used)
        try:
            redis_memory = RedisMemory(
                MemoryConfig(
                    redis_host=self.config.redis_host,
                    redis_port=self.config.redis_port,
                    redis_password=self.config.redis_password
                )
            )
            redis_memory.connect()
            redis_memory.save("health_check", {"timestamp": time.time()}, ttl=60)
            redis_memory.disconnect()
            health_status["checks"]["redis"] = {"status": "ok"}
        except Exception as e:
            health_status["checks"]["redis"] = {"status": "error", "error": str(e)}
        
        # Log health status
        if health_status["status"] == "healthy":
            logger.debug(f"Health check passed: {health_status}")
        else:
            logger.warning(f"Health check issues detected: {health_status}")
        
        # Store metrics
        self.metrics_collector.record_health_check(health_status)
    
    def create_production_workflow(self, workflow_name: str) -> CrewGraph:
        """Create production-ready workflow"""
        logger.info(f"Creating production workflow: {workflow_name}")
        
        # Production memory configuration
        memory_config = MemoryConfig(
            memory_type=MemoryType.REDIS,
            redis_host=self.config.redis_host,
            redis_port=self.config.redis_port,
            redis_password=self.config.redis_password,
            ttl=3600,  # 1 hour TTL
            compression=True,
            encryption_key=os.getenv("CREWGRAPH_ENCRYPTION_KEY")
        )
        
        # Production workflow configuration
        workflow_config = CrewGraphConfig(
            memory_backend=RedisMemory(memory_config),
            enable_planning=True,
            max_concurrent_tasks=5,
            task_timeout=600.0,  # 10 minutes
            enable_logging=True,
            log_level="INFO"
        )
        
        # Create workflow
        workflow = CrewGraph(workflow_name, workflow_config)
        
        # Setup production agents
        self._setup_production_agents(workflow)
        
        # Setup production tasks
        self._setup_production_tasks(workflow)
        
        # Store workflow
        self.workflows[workflow_name] = workflow
        
        logger.info(f"Production workflow '{workflow_name}' created successfully")
        return workflow
    
    def _setup_production_agents(self, workflow: CrewGraph):
        """Setup production-grade agents"""
        
        # Data Processing Agent
        processor_agent = Agent(
            role='Production Data Processor',
            goal='Process data efficiently and reliably in production environment',
            backstory='''You are a production-grade data processor with expertise in 
                        handling large-scale data processing tasks with reliability,
                        error handling, and performance optimization.''',
            tools=[
                Tool(
                    name="process_data",
                    func=production_data_processor,
                    description="Process data with production-grade reliability"
                )
            ],
            verbose=True,
            max_iter=3,
            allow_delegation=False
        )
        
        # Report Generation Agent
        reporter_agent = Agent(
            role='Production Report Generator',
            goal='Generate production-quality reports with multiple format support',
            backstory='''You are a production report generator capable of creating
                        high-quality reports in various formats for enterprise use.''',
            tools=[
                Tool(
                    name="generate_report",
                    func=production_report_generator,
                    description="Generate production reports"
                )
            ],
            verbose=True,
            max_iter=2,
            allow_delegation=False
        )
        
        # Add agents to workflow
        workflow.add_agent(processor_agent, name="data_processor")
        workflow.add_agent(reporter_agent, name="report_generator")
    
    def _setup_production_tasks(self, workflow: CrewGraph):
        """Setup production task pipeline"""
        
        # Data processing task
        processing_task = workflow.add_task(
            name="process_production_data",
            description="Process input data with production-grade reliability",
            agent="data_processor",
            tools=["process_data"]
        )
        
        # Report generation task
        reporting_task = workflow.add_task(
            name="generate_production_report",
            description="Generate production-quality report",
            agent="report_generator",
            tools=["generate_report"],
            dependencies=["process_production_data"]
        )
    
    async def execute_workflow_async(self, 
                                   workflow_name: str,
                                   input_data: Dict[str, Any],
                                   execution_id: str = None) -> Dict[str, Any]:
        """Execute workflow asynchronously with monitoring"""
        
        execution_id = execution_id or f"{workflow_name}_{int(time.time())}"
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow = self.workflows[workflow_name]
        
        # Record execution start
        execution_info = {
            "workflow_name": workflow_name,
            "execution_id": execution_id,
            "start_time": time.time(),
            "status": "running",
            "input_data": input_data
        }
        
        self.active_executions[execution_id] = execution_info
        
        try:
            logger.info(f"Starting async execution: {execution_id}")
            
            # Execute with monitoring
            with self.performance_monitor.track_execution(execution_id):
                result = await workflow.execute_async(input_data)
            
            # Record successful completion
            execution_info.update({
                "status": "completed",
                "end_time": time.time(),
                "result": result
            })
            
            logger.info(f"Execution completed successfully: {execution_id}")
            return result
            
        except Exception as e:
            # Record failure
            execution_info.update({
                "status": "failed",
                "end_time": time.time(),
                "error": str(e)
            })
            
            logger.error(f"Execution failed: {execution_id} - {e}")
            raise
            
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def execute_workflow_batch(self, 
                              workflow_name: str,
                              batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute workflow for batch of inputs"""
        logger.info(f"Starting batch execution for {len(batch_data)} items")
        
        # Submit all executions to thread pool
        futures = []
        for i, input_data in enumerate(batch_data):
            execution_id = f"{workflow_name}_batch_{int(time.time())}_{i}"
            future = self.thread_pool.submit(
                asyncio.run,
                self.execute_workflow_async(workflow_name, input_data, execution_id)
            )
            futures.append(future)
        
        # Collect results
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=600)  # 10 minute timeout
                results.append({"index": i, "status": "success", "result": result})
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append({"index": i, "status": "failed", "error": str(e)})
        
        logger.info(f"Batch execution completed: {len(results)} items processed")
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "active_workflows": len(self.workflows),
            "active_executions": len(self.active_executions),
            "thread_pool_active": self.thread_pool._threads,
            "metrics": self.metrics_collector.get_summary(),
            "performance": self.performance_monitor.get_stats(),
            "timestamp": time.time()
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for active executions to complete (with timeout)
        timeout = 60  # 1 minute timeout
        start_time = time.time()
        
        while self.active_executions and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for {len(self.active_executions)} active executions to complete")
            time.sleep(5)
        
        # Force shutdown if timeout reached
        if self.active_executions:
            logger.warning(f"Force shutdown: {len(self.active_executions)} executions still active")
        
        # Shutdown executors
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Cleanup workflows
        for workflow in self.workflows.values():
            if hasattr(workflow, 'shutdown'):
                workflow.shutdown()
        
        logger.info("Graceful shutdown completed")

async def main():
    """Main production deployment demonstration"""
    logger.info("Starting Production Deployment Example")
    logger.info(f"Deployed by user: Vatsal216 at 2025-07-22 10:19:38")
    
    # Production configuration
    config = ProductionConfig(
        environment="production",
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_password=os.getenv("REDIS_PASSWORD", ""),
        max_workers=int(os.getenv("MAX_WORKERS", "10")),
        max_concurrent_workflows=int(os.getenv("MAX_WORKFLOWS", "50"))
    )
    
    # Create production manager
    manager = ProductionWorkflowManager(config)
    
    try:
        # Create production workflow
        workflow = manager.create_production_workflow("production_pipeline")
        
        print("\n" + "="*60)
        print("Production Workflow Execution")
        print("="*60)
        
        # Single execution
        single_result = await manager.execute_workflow_async(
            "production_pipeline",
            {
                "input_data": "Production data sample for processing",
                "processing_type": "standard",
                "report_format": "json"
            }
        )
        
        print(f"Single execution result: {single_result}")
        
        # Batch execution
        batch_data = [
            {
                "input_data": f"Batch item {i}",
                "processing_type": "intensive" if i % 2 == 0 else "standard",
                "report_format": "json"
            }
            for i in range(5)
        ]
        
        batch_results = manager.execute_workflow_batch("production_pipeline", batch_data)
        print(f"Batch execution results: {len(batch_results)} items processed")
        
        # System metrics
        metrics = manager.get_system_metrics()
        print(f"System metrics: {metrics}")
        
        # Simulate running for a while
        logger.info("Production system running... (Press Ctrl+C to shutdown)")
        await asyncio.sleep(30)  # Run for 30 seconds
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Production system error: {e}")
    finally:
        # Graceful shutdown
        manager.shutdown()
        logger.info("Production deployment example completed")

if __name__ == "__main__":
    asyncio.run(main())