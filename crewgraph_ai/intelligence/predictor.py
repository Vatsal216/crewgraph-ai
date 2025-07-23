"""
Performance and Resource Predictors for CrewGraph AI

AI-driven performance prediction and resource usage forecasting.

Author: Vatsal216
Created: 2025-07-23 06:16:00 UTC
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

from .ml_models import MLModelManager, ModelType, TrainingData
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


@dataclass
class WorkflowFeatures:
    """Features extracted from workflow for prediction"""
    workflow_size: int
    agent_count: int
    task_count: int
    task_complexity: float
    data_size: int
    dependency_depth: int
    parallel_branches: int
    memory_requirements: int
    cpu_requirements: float
    io_operations: int
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model"""
        return np.array([
            self.workflow_size, self.agent_count, self.task_count,
            self.task_complexity, self.data_size, self.dependency_depth,
            self.parallel_branches, self.memory_requirements, 
            self.cpu_requirements, self.io_operations
        ])


@dataclass
class PerformancePrediction:
    """Performance prediction results"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_probability: float
    bottleneck_probability: float
    estimated_cost: float
    confidence_score: float
    recommendations: List[str]
    created_by: str = "Vatsal216"
    created_at: str = "2025-07-23 06:16:00"


@dataclass
class ResourcePrediction:
    """Resource usage prediction results"""
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    disk_usage_mb: float
    network_usage_mb: float
    estimated_duration: float
    resource_score: float
    scaling_recommendations: List[str]
    created_by: str = "Vatsal216" 
    created_at: str = "2025-07-23 06:16:00"
""""
Performance Predictor for CrewGraph AI Intelligence Layer

This module provides ML-based performance prediction capabilities for workflows,
enabling proactive optimization and resource planning.

Author: Vatsal216
Created: 2025-07-23 06:03:54 UTC
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..utils.logging import get_logger
from ..types import WorkflowId, TaskResult

logger = get_logger(__name__)


@dataclass
class WorkflowMetrics:
    """Metrics for workflow performance analysis."""
    workflow_id: WorkflowId
    execution_time: float
    memory_usage: float
    cpu_usage: float
    task_count: int
    success_rate: float
    error_count: int
    timestamp: datetime
    resource_utilization: Dict[str, float]


class PerformancePredictor:
    """
    AI-driven performance prediction for workflows.
    
    Uses machine learning models to predict execution time,
    resource usage, and potential issues.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:16:00 UTC
    """
    
    def __init__(self, model_manager: Optional[MLModelManager] = None):
        """
        Initialize performance predictor.
        
        Args:
            model_manager: ML model manager instance
        """
        self.model_manager = model_manager or MLModelManager()
        self._prediction_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("PerformancePredictor initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:16:00")
    
    def _initialize_models(self):
        """Initialize and train models with synthetic data"""
        try:
            # Create performance prediction model
            perf_data = self.model_manager.generate_synthetic_training_data(
                ModelType.PERFORMANCE_PREDICTOR, num_samples=1000
            )
            self.model_manager.train_model(ModelType.PERFORMANCE_PREDICTOR, perf_data)
            
            # Create bottleneck detection model
            bottleneck_data = self.model_manager.generate_synthetic_training_data(
                ModelType.BOTTLENECK_DETECTOR, num_samples=1000
            )
            self.model_manager.train_model(ModelType.BOTTLENECK_DETECTOR, bottleneck_data)
            
            logger.info("Performance prediction models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def extract_workflow_features(self, workflow_data: Dict[str, Any]) -> WorkflowFeatures:
        """
        Extract features from workflow for prediction.
        
        Args:
            workflow_data: Workflow definition and metadata
            
        Returns:
            Extracted features
        """
        # Default feature extraction - can be enhanced based on actual workflow structure
        agents = workflow_data.get('agents', [])
        tasks = workflow_data.get('tasks', [])
        edges = workflow_data.get('edges', [])
        
        # Calculate workflow characteristics
        agent_count = len(agents)
        task_count = len(tasks)
        
        # Estimate complexity based on task types and dependencies
        task_complexity = sum(
            task.get('complexity', 1.0) for task in tasks
        ) / max(task_count, 1)
        
        # Estimate data size from inputs
        data_size = sum(
            len(str(task.get('input', ''))) for task in tasks
        )
        
        # Calculate dependency depth (longest path)
        dependency_depth = self._calculate_dependency_depth(tasks, edges)
        
        # Count parallel branches
        parallel_branches = self._count_parallel_branches(tasks, edges)
        
        # Estimate resource requirements
        memory_requirements = sum(
            task.get('memory_mb', 128) for task in tasks
        )
        
        cpu_requirements = sum(
            task.get('cpu_cores', 0.5) for task in tasks
        )
        
        io_operations = sum(
            1 for task in tasks 
            if task.get('type', '').lower() in ['file', 'database', 'api', 'network']
        )
        
        return WorkflowFeatures(
            workflow_size=task_count + agent_count,
            agent_count=agent_count,
            task_count=task_count,
            task_complexity=task_complexity,
            data_size=data_size,
            dependency_depth=dependency_depth,
            parallel_branches=parallel_branches,
            memory_requirements=memory_requirements,
            cpu_requirements=cpu_requirements,
            io_operations=io_operations
        )
    
    def predict_performance(self, workflow_data: Dict[str, Any]) -> PerformancePrediction:
        """
        Predict workflow performance.
        
        Args:
            workflow_data: Workflow definition and metadata
            
        Returns:
            Performance prediction results
        """
        with self._lock:
            start_time = time.time()
            
            # Extract features
            features = self.extract_workflow_features(workflow_data)
            feature_array = features.to_array().reshape(1, -1)
            
            # Predict execution time
            try:
                execution_time = self.model_manager.predict(
                    ModelType.PERFORMANCE_PREDICTOR, feature_array
                )[0]
                execution_time = max(execution_time, 1.0)  # Minimum 1 second
            except Exception as e:
                logger.warning(f"Performance prediction failed, using heuristic: {e}")
                execution_time = self._heuristic_execution_time(features)
            
            # Predict bottleneck probability
            try:
                bottleneck_features = np.array([
                    features.task_count,
                    execution_time,
                    0.05,  # Assumed error rate
                    100.0 / max(execution_time, 1.0)  # Throughput estimate
                ]).reshape(1, -1)
                
                _, bottleneck_proba = self.model_manager.predict(
                    ModelType.BOTTLENECK_DETECTOR, bottleneck_features
                )
                bottleneck_probability = bottleneck_proba[0]
            except Exception as e:
                logger.warning(f"Bottleneck prediction failed, using heuristic: {e}")
                bottleneck_probability = self._heuristic_bottleneck_probability(features)
            
            # Estimate resource usage
            memory_usage = self._estimate_memory_usage(features)
            cpu_usage = self._estimate_cpu_usage(features)
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(
                features, execution_time, bottleneck_probability
            )
            
            # Estimate cost (simplified)
            estimated_cost = self._estimate_cost(features, execution_time)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(features)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                features, execution_time, bottleneck_probability, memory_usage, cpu_usage
            )
            
            prediction = PerformancePrediction(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                success_probability=success_probability,
                bottleneck_probability=bottleneck_probability,
                estimated_cost=estimated_cost,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
            # Record prediction
            prediction_time = time.time() - start_time
            self._record_prediction(features, prediction, prediction_time)
            
            metrics.record_metric("performance_predictions_total", 1.0)
            metrics.record_metric("prediction_time_seconds", prediction_time)
            
            logger.info(f"Performance prediction completed in {prediction_time:.3f}s")
            
            return prediction
    
    def _calculate_dependency_depth(self, tasks: List[Dict], edges: List[Dict]) -> int:
        """Calculate the longest dependency path in the workflow"""
        if not tasks or not edges:
            return 1
        
        # Build adjacency list
        graph = {task.get('id', i): [] for i, task in enumerate(tasks)}
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in graph:
                graph[source].append(target)
        
        # Find longest path using DFS
        max_depth = 1
        visited = set()
        
        def dfs(node, depth):
            nonlocal max_depth
            if node in visited:
                return depth
            
            visited.add(node)
            max_depth = max(max_depth, depth)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, depth + 1)
            
            visited.remove(node)
            return depth
        
        for task_id in graph:
            dfs(task_id, 1)
        
        return max_depth
    
    def _count_parallel_branches(self, tasks: List[Dict], edges: List[Dict]) -> int:
        """Count the number of parallel execution branches"""
        if not tasks or not edges:
            return 1
        
        # Simple heuristic: count tasks that don't have dependencies
        dependent_tasks = set()
        for edge in edges:
            dependent_tasks.add(edge.get('target'))
        
        independent_tasks = len(tasks) - len(dependent_tasks)
        return max(independent_tasks, 1)
    
    def _heuristic_execution_time(self, features: WorkflowFeatures) -> float:
        """Heuristic execution time calculation"""
        base_time = features.task_count * 2.0  # 2 seconds per task
        complexity_factor = 1.0 + (features.task_complexity - 1.0) * 0.5
        parallelism_factor = 1.0 / max(features.parallel_branches, 1)
        io_penalty = features.io_operations * 0.5
        
        return base_time * complexity_factor * parallelism_factor + io_penalty
    
    def _heuristic_bottleneck_probability(self, features: WorkflowFeatures) -> float:
        """Heuristic bottleneck probability calculation"""
        # Higher probability with more tasks, deeper dependencies, fewer parallel branches
        task_factor = min(features.task_count / 50.0, 1.0)
        depth_factor = min(features.dependency_depth / 10.0, 1.0)
        parallel_factor = 1.0 / max(features.parallel_branches, 1)
        
        return min((task_factor + depth_factor + parallel_factor) / 3.0, 1.0)
    
    def _estimate_memory_usage(self, features: WorkflowFeatures) -> float:
        """Estimate memory usage in MB"""
        base_memory = features.memory_requirements
        data_memory = features.data_size / 1000.0  # Rough estimate
        agent_memory = features.agent_count * 64.0  # 64MB per agent
        
        return base_memory + data_memory + agent_memory
    
    def _estimate_cpu_usage(self, features: WorkflowFeatures) -> float:
        """Estimate CPU usage percentage"""
        base_cpu = features.cpu_requirements * 10.0  # Convert cores to percentage
        task_cpu = features.task_count * 2.0
        complexity_cpu = features.task_complexity * 10.0
        
        return min(base_cpu + task_cpu + complexity_cpu, 100.0)
    
    def _calculate_success_probability(self, 
                                     features: WorkflowFeatures,
                                     execution_time: float,
                                     bottleneck_probability: float) -> float:
        """Calculate probability of successful execution"""
        # Start with high base probability
        base_probability = 0.95
        
        # Reduce probability based on complexity
        complexity_penalty = features.task_complexity * 0.05
        
        # Reduce probability based on execution time
        time_penalty = min(execution_time / 3600.0, 0.2)  # Max 20% penalty for long runs
        
        # Reduce probability based on bottleneck risk
        bottleneck_penalty = bottleneck_probability * 0.15
        
        success_probability = base_probability - complexity_penalty - time_penalty - bottleneck_penalty
        
        return max(success_probability, 0.1)  # Minimum 10% success probability
    
    def _estimate_cost(self, features: WorkflowFeatures, execution_time: float) -> float:
        """Estimate execution cost in dollars"""
        # Simple cost model based on resource usage and time
        compute_cost = features.cpu_requirements * execution_time * 0.0001  # $0.0001 per core-second
        memory_cost = features.memory_requirements * execution_time * 0.00001  # $0.00001 per MB-second
        io_cost = features.io_operations * 0.001  # $0.001 per IO operation
        
        return compute_cost + memory_cost + io_cost
    
    def _calculate_confidence_score(self, features: WorkflowFeatures) -> float:
        """Calculate confidence in the prediction"""
        # Higher confidence for simpler workflows with more standard patterns
        simplicity_score = 1.0 / (1.0 + features.task_complexity)
        size_score = 1.0 / (1.0 + features.workflow_size / 50.0)
        standard_score = 1.0 if features.io_operations < 5 else 0.7
        
        return (simplicity_score + size_score + standard_score) / 3.0
    
    def _generate_recommendations(self, 
                                features: WorkflowFeatures,
                                execution_time: float,
                                bottleneck_probability: float,
                                memory_usage: float,
                                cpu_usage: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Performance recommendations
        if execution_time > 300:  # 5 minutes
            recommendations.append("Consider breaking down large tasks into smaller ones")
            recommendations.append("Enable parallel execution where possible")
        
        # Resource recommendations
        if memory_usage > 4096:  # 4GB
            recommendations.append("Consider using memory-efficient data structures")
            recommendations.append("Implement data streaming for large datasets")
        
        if cpu_usage > 80:
            recommendations.append("Consider scaling to multiple CPU cores")
            recommendations.append("Optimize compute-intensive operations")
        
        # Bottleneck recommendations
        if bottleneck_probability > 0.7:
            recommendations.append("Review workflow dependencies to reduce bottlenecks")
            recommendations.append("Consider implementing circuit breakers for external services")
        
        # Complexity recommendations
        if features.task_complexity > 5:
            recommendations.append("Simplify complex tasks where possible")
            recommendations.append("Add comprehensive error handling")
        
        # Dependency recommendations
        if features.dependency_depth > 5:
            recommendations.append("Reduce deep dependency chains")
            recommendations.append("Consider event-driven architecture patterns")
        
        return recommendations
    
    def _record_prediction(self, 
                          features: WorkflowFeatures,
                          prediction: PerformancePrediction,
                          prediction_time: float):
        """Record prediction for analysis and model improvement"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "features": features.__dict__,
            "prediction": prediction.__dict__,
            "prediction_time": prediction_time,
            "created_by": "Vatsal216"
        }
        
        self._prediction_history.append(record)
        
        # Keep only last 1000 predictions
        if len(self._prediction_history) > 1000:
            self._prediction_history = self._prediction_history[-1000:]
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction performance"""
        if not self._prediction_history:
            return {"total_predictions": 0}
        
        prediction_times = [p["prediction_time"] for p in self._prediction_history]
        execution_times = [p["prediction"]["execution_time"] for p in self._prediction_history]
        
        return {
            "total_predictions": len(self._prediction_history),
            "avg_prediction_time": np.mean(prediction_times),
            "avg_predicted_execution_time": np.mean(execution_times),
            "min_predicted_execution_time": np.min(execution_times),
            "max_predicted_execution_time": np.max(execution_times),
            "created_by": "Vatsal216",
            "timestamp": datetime.now().isoformat()
        }


class ResourcePredictor:
    """
    AI-driven resource usage prediction for workflows.
    
    Predicts memory, CPU, disk, and network usage patterns.
    
    Created by: Vatsal216
    Date: 2025-07-23 06:16:00 UTC
    """
    
    def __init__(self, model_manager: Optional[MLModelManager] = None):
        """
        Initialize resource predictor.
        
        Args:
            model_manager: ML model manager instance
        """
        self.model_manager = model_manager or MLModelManager()
        self._resource_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Initialize resource prediction model
        self._initialize_models()
        
        logger.info("ResourcePredictor initialized")
        logger.info("User: Vatsal216, Time: 2025-07-23 06:16:00")
    
    def _initialize_models(self):
        """Initialize and train resource prediction models"""
        try:
            # Create resource prediction model
            resource_data = self.model_manager.generate_synthetic_training_data(
                ModelType.RESOURCE_PREDICTOR, num_samples=1000
            )
            self.model_manager.train_model(ModelType.RESOURCE_PREDICTOR, resource_data)
            
            logger.info("Resource prediction models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize resource models: {e}")
    
    def predict_resources(self, workflow_data: Dict[str, Any]) -> ResourcePrediction:
        """
        Predict resource usage for workflow.
        
        Args:
            workflow_data: Workflow definition and metadata
            
        Returns:
            Resource prediction results
        """
        with self._lock:
            start_time = time.time()
            
            # Extract features for resource prediction
            features = self._extract_resource_features(workflow_data)
            feature_array = features.reshape(1, -1)
            
            # Predict resource score
            try:
                resource_score = self.model_manager.predict(
                    ModelType.RESOURCE_PREDICTOR, feature_array
                )[0]
            except Exception as e:
                logger.warning(f"Resource prediction failed, using heuristic: {e}")
                resource_score = self._heuristic_resource_score(workflow_data)
            
            # Estimate specific resources
            peak_memory, avg_memory = self._estimate_memory_usage(workflow_data)
            peak_cpu, avg_cpu = self._estimate_cpu_usage(workflow_data)
            disk_usage = self._estimate_disk_usage(workflow_data)
            network_usage = self._estimate_network_usage(workflow_data)
            estimated_duration = self._estimate_duration(workflow_data)
            
            # Generate scaling recommendations
            scaling_recommendations = self._generate_scaling_recommendations(
                peak_memory, peak_cpu, disk_usage, network_usage, estimated_duration
            )
            
            prediction = ResourcePrediction(
                peak_memory_mb=peak_memory,
                avg_memory_mb=avg_memory,
                peak_cpu_percent=peak_cpu,
                avg_cpu_percent=avg_cpu,
                disk_usage_mb=disk_usage,
                network_usage_mb=network_usage,
                estimated_duration=estimated_duration,
                resource_score=resource_score,
                scaling_recommendations=scaling_recommendations
            )
            
            # Record prediction
            prediction_time = time.time() - start_time
            self._record_resource_prediction(workflow_data, prediction, prediction_time)
            
            metrics.record_metric("resource_predictions_total", 1.0)
            metrics.record_metric("resource_prediction_time_seconds", prediction_time)
            
            logger.info(f"Resource prediction completed in {prediction_time:.3f}s")
            
            return prediction
    
    def _extract_resource_features(self, workflow_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for resource prediction"""
        tasks = workflow_data.get('tasks', [])
        agents = workflow_data.get('agents', [])
        
        task_count = len(tasks)
        concurrent_agents = len(agents)
        
        # Estimate memory usage from task specifications
        memory_usage = sum(task.get('memory_mb', 128) for task in tasks)
        
        # Estimate CPU usage from task specifications
        cpu_usage = sum(task.get('cpu_cores', 0.5) for task in tasks) * 10  # Convert to percentage
        
        return np.array([task_count, concurrent_agents, memory_usage, cpu_usage])
    
    def _heuristic_resource_score(self, workflow_data: Dict[str, Any]) -> float:
        """Heuristic resource score calculation"""
        tasks = workflow_data.get('tasks', [])
        agents = workflow_data.get('agents', [])
        
        # Simple scoring based on workflow complexity
        task_score = len(tasks) * 2.0
        agent_score = len(agents) * 5.0
        complexity_score = sum(task.get('complexity', 1.0) for task in tasks)
        
        return task_score + agent_score + complexity_score
    
    def _estimate_memory_usage(self, workflow_data: Dict[str, Any]) -> Tuple[float, float]:
        """Estimate peak and average memory usage"""
        tasks = workflow_data.get('tasks', [])
        agents = workflow_data.get('agents', [])
        
        # Base memory for agents
        agent_memory = len(agents) * 64.0  # 64MB per agent
        
        # Memory for tasks
        task_memory = sum(task.get('memory_mb', 128) for task in tasks)
        
        # Peak memory (assuming some tasks run concurrently)
        parallel_factor = min(len(tasks), 4)  # Assume max 4 parallel tasks
        peak_memory = agent_memory + (task_memory * parallel_factor / len(tasks))
        
        # Average memory (assuming sequential execution mostly)
        avg_memory = agent_memory + (task_memory / len(tasks)) if tasks else agent_memory
        
        return peak_memory, avg_memory
    
    def _estimate_cpu_usage(self, workflow_data: Dict[str, Any]) -> Tuple[float, float]:
        """Estimate peak and average CPU usage"""
        tasks = workflow_data.get('tasks', [])
        agents = workflow_data.get('agents', [])
        
        # Base CPU for agents
        agent_cpu = len(agents) * 5.0  # 5% per agent
        
        # CPU for tasks
        task_cpu = sum(task.get('cpu_cores', 0.5) for task in tasks) * 15.0  # Convert to percentage
        
        # Peak CPU (assuming some tasks run concurrently)
        parallel_factor = min(len(tasks), 4)
        peak_cpu = min(agent_cpu + (task_cpu * parallel_factor / len(tasks)), 100.0)
        
        # Average CPU
        avg_cpu = min(agent_cpu + (task_cpu / len(tasks)) if tasks else agent_cpu, 80.0)
        
        return peak_cpu, avg_cpu
    
    def _estimate_disk_usage(self, workflow_data: Dict[str, Any]) -> float:
        """Estimate disk usage in MB"""
        tasks = workflow_data.get('tasks', [])
        
        # Estimate based on data processing tasks
        file_tasks = [task for task in tasks if 'file' in task.get('type', '').lower()]
        data_tasks = [task for task in tasks if 'data' in task.get('type', '').lower()]
        
        file_usage = len(file_tasks) * 100.0  # 100MB per file task
        data_usage = len(data_tasks) * 50.0   # 50MB per data task
        base_usage = 200.0  # Base usage
        
        return file_usage + data_usage + base_usage
    
    def _estimate_network_usage(self, workflow_data: Dict[str, Any]) -> float:
        """Estimate network usage in MB"""
        tasks = workflow_data.get('tasks', [])
        
        # Estimate based on API and network tasks
        api_tasks = [task for task in tasks if 'api' in task.get('type', '').lower()]
        network_tasks = [task for task in tasks if 'network' in task.get('type', '').lower()]
        
        api_usage = len(api_tasks) * 10.0      # 10MB per API task
        network_usage = len(network_tasks) * 5.0  # 5MB per network task
        
        return api_usage + network_usage
    
    def _estimate_duration(self, workflow_data: Dict[str, Any]) -> float:
        """Estimate workflow duration in seconds"""
        tasks = workflow_data.get('tasks', [])
        
        if not tasks:
            return 30.0  # Default 30 seconds
        
        # Estimate based on task complexity and type
        total_time = 0.0
        for task in tasks:
            base_time = 10.0  # 10 seconds base
            complexity_multiplier = task.get('complexity', 1.0)
            
            # Adjust for task type
            task_type = task.get('type', '').lower()
            if 'file' in task_type or 'data' in task_type:
                type_multiplier = 2.0
            elif 'api' in task_type or 'network' in task_type:
                type_multiplier = 1.5
            else:
                type_multiplier = 1.0
            
            task_time = base_time * complexity_multiplier * type_multiplier
            total_time += task_time
        
        # Assume some parallelization
        parallel_factor = min(len(tasks), 3) / len(tasks)
        estimated_duration = total_time * parallel_factor
        
        return max(estimated_duration, 10.0)  # Minimum 10 seconds
    
    def _generate_scaling_recommendations(self, 
                                        peak_memory: float,
                                        peak_cpu: float,
                                        disk_usage: float,
                                        network_usage: float,
                                        duration: float) -> List[str]:
        """Generate scaling recommendations based on resource predictions"""
        recommendations = []
        
        # Memory scaling recommendations
        if peak_memory > 8192:  # 8GB
            recommendations.append("Consider upgrading to high-memory instances")
            recommendations.append("Implement memory optimization techniques")
        elif peak_memory > 4096:  # 4GB
            recommendations.append("Monitor memory usage closely")
        
        # CPU scaling recommendations
        if peak_cpu > 90:
            recommendations.append("Scale to multi-core instances")
            recommendations.append("Consider horizontal scaling")
        elif peak_cpu > 70:
            recommendations.append("Monitor CPU usage and consider scaling")
        
        # Disk usage recommendations
        if disk_usage > 10240:  # 10GB
            recommendations.append("Ensure sufficient disk space")
            recommendations.append("Consider SSD storage for better performance")
        
        # Network usage recommendations
        if network_usage > 1024:  # 1GB
            recommendations.append("Optimize network operations")
            recommendations.append("Consider data compression")
        
        # Duration-based recommendations
        if duration > 3600:  # 1 hour
            recommendations.append("Consider workflow optimization for long-running tasks")
            recommendations.append("Implement checkpointing for reliability")
        
        return recommendations
    
    def _record_resource_prediction(self, 
                                  workflow_data: Dict[str, Any],
                                  prediction: ResourcePrediction,
                                  prediction_time: float):
        """Record resource prediction for analysis"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "workflow_data": workflow_data,
            "prediction": prediction.__dict__,
            "prediction_time": prediction_time,
            "created_by": "Vatsal216"
        }
        
        self._resource_history.append(record)
        
        # Keep only last 1000 predictions
        if len(self._resource_history) > 1000:
            self._resource_history = self._resource_history[-1000:]
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get statistics about resource predictions"""
        if not self._resource_history:
            return {"total_predictions": 0}
        
        prediction_times = [p["prediction_time"] for p in self._resource_history]
        peak_memories = [p["prediction"]["peak_memory_mb"] for p in self._resource_history]
        peak_cpus = [p["prediction"]["peak_cpu_percent"] for p in self._resource_history]
        
        return {
            "total_predictions": len(self._resource_history),
            "avg_prediction_time": np.mean(prediction_times),
            "avg_peak_memory_mb": np.mean(peak_memories),
            "avg_peak_cpu_percent": np.mean(peak_cpus),
            "created_by": "Vatsal216",
            "timestamp": datetime.now().isoformat()
        }
    """
    ML-based performance predictor for workflow optimization.
    
    Uses lightweight statistical models and historical data to predict
    workflow performance, execution time, and resource requirements.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the performance predictor.
        
        Args:
            history_size: Maximum number of historical metrics to keep
        """
        self.history_size = history_size
        self.metrics_history: List[WorkflowMetrics] = []
        self.workflow_patterns: Dict[str, List[WorkflowMetrics]] = {}
        
        logger.info(f"PerformancePredictor initialized with history_size={history_size}")
    
    def record_metrics(self, metrics: WorkflowMetrics) -> None:
        """
        Record workflow execution metrics for learning.
        
        Args:
            metrics: Workflow execution metrics to record
        """
        self.metrics_history.append(metrics)
        
        # Maintain history size limit
        if len(self.metrics_history) > self.history_size:
            self.metrics_history.pop(0)
        
        # Group by workflow pattern
        pattern_key = f"{metrics.task_count}_{metrics.workflow_id[:8]}"
        if pattern_key not in self.workflow_patterns:
            self.workflow_patterns[pattern_key] = []
        
        self.workflow_patterns[pattern_key].append(metrics)
        
        logger.debug(f"Recorded metrics for workflow {metrics.workflow_id}")
    
    def predict_execution_time(self, 
                             workflow_id: WorkflowId, 
                             task_count: int,
                             context: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Predict workflow execution time based on historical data.
        
        Args:
            workflow_id: Workflow identifier
            task_count: Number of tasks in the workflow
            context: Additional context for prediction
            
        Returns:
            Tuple of (predicted_time, confidence_score)
        """
        if not self.metrics_history:
            # No historical data, return default estimate
            base_time = task_count * 2.0  # 2 seconds per task baseline
            return base_time, 0.5
        
        # Find similar workflows
        similar_metrics = self._find_similar_workflows(task_count, workflow_id)
        
        if not similar_metrics:
            # Fallback to general statistics
            execution_times = [m.execution_time for m in self.metrics_history[-50:]]
            avg_time = statistics.mean(execution_times)
            scaling_factor = task_count / 5.0  # Assume 5 tasks average
            predicted_time = avg_time * scaling_factor
            confidence = 0.6
        else:
            # Use similar workflow data
            execution_times = [m.execution_time for m in similar_metrics]
            predicted_time = statistics.mean(execution_times)
            
            # Calculate confidence based on data consistency
            if len(execution_times) > 1:
                std_dev = statistics.stdev(execution_times)
                confidence = max(0.1, 1.0 - (std_dev / predicted_time))
            else:
                confidence = 0.7
        
        logger.debug(f"Predicted execution time: {predicted_time:.2f}s (confidence: {confidence:.2f})")
        return predicted_time, confidence
    
    def predict_resource_usage(self, 
                             workflow_id: WorkflowId,
                             task_count: int) -> Dict[str, Tuple[float, float]]:
        """
        Predict resource usage for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            task_count: Number of tasks
            
        Returns:
            Dictionary of resource predictions with confidence scores
        """
        similar_metrics = self._find_similar_workflows(task_count, workflow_id)
        
        if not similar_metrics:
            # Default resource estimates
            return {
                "memory": (task_count * 50.0, 0.5),  # 50MB per task
                "cpu": (min(task_count * 10.0, 80.0), 0.5)  # 10% per task, max 80%
            }
        
        # Calculate resource predictions
        memory_usage = [m.memory_usage for m in similar_metrics]
        cpu_usage = [m.cpu_usage for m in similar_metrics]
        
        predictions = {}
        
        if memory_usage:
            avg_memory = statistics.mean(memory_usage)
            memory_confidence = self._calculate_confidence(memory_usage)
            predictions["memory"] = (avg_memory, memory_confidence)
        
        if cpu_usage:
            avg_cpu = statistics.mean(cpu_usage)
            cpu_confidence = self._calculate_confidence(cpu_usage)
            predictions["cpu"] = (avg_cpu, cpu_confidence)
        
        return predictions
    
    def detect_performance_anomalies(self, 
                                   metrics: WorkflowMetrics) -> List[str]:
        """
        Detect performance anomalies in workflow execution.
        
        Args:
            metrics: Current workflow metrics
            
        Returns:
            List of detected anomaly descriptions
        """
        anomalies = []
        
        if not self.metrics_history:
            return anomalies
        
        # Compare with historical averages
        recent_metrics = self.metrics_history[-20:]  # Last 20 executions
        
        if recent_metrics:
            avg_execution_time = statistics.mean([m.execution_time for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
            avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
            
            # Check for significant deviations (2x threshold)
            if metrics.execution_time > avg_execution_time * 2:
                anomalies.append(f"Execution time {metrics.execution_time:.2f}s is 2x higher than average {avg_execution_time:.2f}s")
            
            if metrics.memory_usage > avg_memory * 2:
                anomalies.append(f"Memory usage {metrics.memory_usage:.1f}MB is 2x higher than average {avg_memory:.1f}MB")
            
            if metrics.cpu_usage > avg_cpu * 1.5:
                anomalies.append(f"CPU usage {metrics.cpu_usage:.1f}% is 1.5x higher than average {avg_cpu:.1f}%")
            
            if metrics.success_rate < 0.8:
                anomalies.append(f"Success rate {metrics.success_rate:.2f} is below acceptable threshold (0.8)")
        
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} performance anomalies for workflow {metrics.workflow_id}")
        
        return anomalies
    
    def get_optimization_recommendations(self, 
                                       workflow_id: WorkflowId) -> List[str]:
        """
        Generate optimization recommendations based on historical data.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if not self.metrics_history:
            return ["Insufficient historical data for recommendations"]
        
        # Analyze recent performance trends
        recent_metrics = [m for m in self.metrics_history[-50:] 
                         if workflow_id in m.workflow_id or m.workflow_id in workflow_id]
        
        if recent_metrics:
            avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
            avg_execution_time = statistics.mean([m.execution_time for m in recent_metrics])
            
            if avg_success_rate < 0.9:
                recommendations.append("Consider implementing retry mechanisms for improved reliability")
            
            if avg_execution_time > 30.0:
                recommendations.append("Workflow execution time is high - consider parallel task execution")
            
            # Check for memory issues
            high_memory_executions = [m for m in recent_metrics if m.memory_usage > 500.0]
            if len(high_memory_executions) > len(recent_metrics) * 0.3:
                recommendations.append("Frequent high memory usage detected - implement memory optimization")
            
            # Check for CPU bottlenecks
            high_cpu_executions = [m for m in recent_metrics if m.cpu_usage > 80.0]
            if len(high_cpu_executions) > len(recent_metrics) * 0.2:
                recommendations.append("CPU usage often high - consider task distribution or resource scaling")
        
        return recommendations
    
    def _find_similar_workflows(self, 
                              task_count: int, 
                              workflow_id: WorkflowId,
                              tolerance: int = 2) -> List[WorkflowMetrics]:
        """Find workflows with similar characteristics."""
        similar = []
        
        for metrics in self.metrics_history:
            # Match by task count (within tolerance)
            if abs(metrics.task_count - task_count) <= tolerance:
                similar.append(metrics)
        
        # If not enough similar workflows, broaden the search
        if len(similar) < 3:
            similar = [m for m in self.metrics_history 
                      if abs(m.task_count - task_count) <= tolerance * 2]
        
        return similar[-10:]  # Return most recent similar workflows
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate confidence score based on data consistency."""
        if len(values) <= 1:
            return 0.5
        
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        # Confidence is higher when standard deviation is lower relative to mean
        if mean_val == 0:
            return 0.5
        
        coefficient_of_variation = std_dev / mean_val
        confidence = max(0.1, 1.0 - coefficient_of_variation)
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance prediction capabilities."""
        return {
            "total_workflows_analyzed": len(self.metrics_history),
            "unique_patterns": len(self.workflow_patterns),
            "average_accuracy": self._estimate_prediction_accuracy(),
            "last_analysis": datetime.now().isoformat(),
            "predictor_version": "1.0.0"
        }
    
    def _estimate_prediction_accuracy(self) -> float:
        """Estimate prediction accuracy based on historical data."""
        if len(self.metrics_history) < 10:
            return 0.6  # Default accuracy for insufficient data
        
        # Simple accuracy estimation based on data consistency
        recent_times = [m.execution_time for m in self.metrics_history[-20:]]
        if len(recent_times) > 5:
            coefficient_of_variation = statistics.stdev(recent_times) / statistics.mean(recent_times)
            accuracy = max(0.5, 1.0 - coefficient_of_variation)
            return min(accuracy, 0.9)
        
        return 0.7
