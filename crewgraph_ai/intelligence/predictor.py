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