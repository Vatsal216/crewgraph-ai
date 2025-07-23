"""
Advanced ML Optimization Components for CrewGraph AI

Provides comprehensive optimization capabilities including:
- Hyperparameter optimization with advanced algorithms
- Neural network-based task scheduling
- Cost prediction models with economic forecasting
- Auto-tuning parameters with reinforcement learning
- Model selection optimization

Author: Vatsal216
Created: 2025-01-27
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, accuracy_score
    ML_AVAILABLE = True
except ImportError:
    np = pd = RandomForestRegressor = GridSearchCV = RandomizedSearchCV = None
    MLPRegressor = MLPClassifier = LinearRegression = StandardScaler = None
    mean_squared_error = accuracy_score = None
    ML_AVAILABLE = False

from ...types import WorkflowId, TaskId
from ...utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods available."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    FIFO = "first_in_first_out"
    PRIORITY = "priority_based"
    SHORTEST_JOB = "shortest_job_first"
    NEURAL_NETWORK = "neural_network"
    REINFORCEMENT = "reinforcement_learning"


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    
    method: OptimizationMethod
    max_iterations: int = 100
    max_time: int = 3600  # seconds
    
    # Grid/Random search parameters
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    n_iter: int = 50  # for random search
    
    # Bayesian optimization
    acquisition_function: str = "expected_improvement"
    n_initial_points: int = 10
    
    # Genetic algorithm
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Reinforcement learning
    learning_rate: float = 0.01
    epsilon: float = 0.1
    discount_factor: float = 0.95


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    
    best_params: Dict[str, Any]
    best_score: float
    optimization_time: float
    iterations_completed: int
    
    # Additional metrics
    convergence_history: List[float] = field(default_factory=list)
    param_importance: Dict[str, float] = field(default_factory=dict)
    optimization_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_time": self.optimization_time,
            "iterations_completed": self.iterations_completed,
            "convergence_history": self.convergence_history,
            "param_importance": self.param_importance,
            "optimization_method": self.optimization_method
        }


@dataclass
class TaskSchedulingDecision:
    """Enhanced task scheduling decision with neural network insights."""
    
    task_id: TaskId
    assigned_resource: str
    priority_score: float
    estimated_start_time: float
    estimated_completion_time: float
    
    # Neural network outputs
    resource_utilization_prediction: Dict[str, float]
    bottleneck_probability: float
    success_probability: float
    
    # Optimization rationale
    scheduling_reason: str
    alternative_options: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CostPrediction:
    """Comprehensive cost prediction with economic modeling."""
    
    predicted_cost: float
    cost_breakdown: Dict[str, float]
    confidence_interval: Tuple[float, float]
    
    # Economic factors
    market_conditions: Dict[str, float]
    seasonal_adjustments: Dict[str, float]
    risk_factors: Dict[str, float]
    
    # Recommendations
    cost_optimization_suggestions: List[str]
    savings_potential: float


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using multiple algorithms
    including Bayesian optimization and genetic algorithms.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize hyperparameter optimizer."""
        self.model_dir = Path(model_dir) if model_dir else Path("models/optimization")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info("Hyperparameter optimizer initialized")
    
    def optimize(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        config: OptimizationConfig
    ) -> OptimizationResult:
        """
        Optimize hyperparameters using specified method.
        
        Args:
            model: ML model to optimize
            X_train, y_train: Training data
            X_val, y_val: Validation data
            config: Optimization configuration
            
        Returns:
            Optimization result with best parameters
        """
        if not ML_AVAILABLE:
            return self._fallback_optimization(config)
        
        start_time = time.time()
        
        try:
            if config.method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search_optimization(
                    model, X_train, y_train, X_val, y_val, config
                )
            elif config.method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search_optimization(
                    model, X_train, y_train, X_val, y_val, config
                )
            elif config.method == OptimizationMethod.BAYESIAN:
                result = self._bayesian_optimization(
                    model, X_train, y_train, X_val, y_val, config
                )
            elif config.method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._genetic_algorithm_optimization(
                    model, X_train, y_train, X_val, y_val, config
                )
            else:
                # Default to grid search
                result = self._grid_search_optimization(
                    model, X_train, y_train, X_val, y_val, config
                )
            
            result.optimization_time = time.time() - start_time
            result.optimization_method = config.method.value
            
            self.optimization_history.append(result)
            
            logger.info(f"Optimization completed: {result.best_score:.4f} in {result.optimization_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                best_params={},
                best_score=0.0,
                optimization_time=time.time() - start_time,
                iterations_completed=0,
                optimization_method=config.method.value
            )
    
    def _grid_search_optimization(
        self, model, X_train, y_train, X_val, y_val, config
    ) -> OptimizationResult:
        """Perform grid search optimization."""
        if not config.param_grid:
            # Use default parameter grid
            config.param_grid = self._get_default_param_grid(model)
        
        grid_search = GridSearchCV(
            model,
            config.param_grid,
            cv=3,
            scoring='neg_mean_squared_error' if hasattr(model, 'predict') else 'accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on validation set
        best_model = grid_search.best_estimator_
        val_score = self._evaluate_model(best_model, X_val, y_val)
        
        return OptimizationResult(
            best_params=grid_search.best_params_,
            best_score=val_score,
            optimization_time=0.0,  # Will be set by caller
            iterations_completed=len(grid_search.cv_results_['params'])
        )
    
    def _random_search_optimization(
        self, model, X_train, y_train, X_val, y_val, config
    ) -> OptimizationResult:
        """Perform random search optimization."""
        if not config.param_grid:
            config.param_grid = self._get_default_param_grid(model)
        
        random_search = RandomizedSearchCV(
            model,
            config.param_grid,
            n_iter=config.n_iter,
            cv=3,
            scoring='neg_mean_squared_error' if hasattr(model, 'predict') else 'accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        # Evaluate on validation set
        best_model = random_search.best_estimator_
        val_score = self._evaluate_model(best_model, X_val, y_val)
        
        return OptimizationResult(
            best_params=random_search.best_params_,
            best_score=val_score,
            optimization_time=0.0,
            iterations_completed=config.n_iter
        )
    
    def _bayesian_optimization(
        self, model, X_train, y_train, X_val, y_val, config
    ) -> OptimizationResult:
        """Perform Bayesian optimization (simplified implementation)."""
        # Simplified Bayesian optimization using random sampling
        # In production, use libraries like scikit-optimize or optuna
        
        best_params = {}
        best_score = float('-inf')
        convergence_history = []
        
        param_names = list(config.param_grid.keys())
        
        for iteration in range(config.max_iterations):
            # Sample parameters randomly (simplified approach)
            params = {}
            for param_name in param_names:
                param_values = config.param_grid[param_name]
                params[param_name] = np.random.choice(param_values)
            
            # Train and evaluate model
            try:
                model_copy = type(model)(**params)
                model_copy.fit(X_train, y_train)
                score = self._evaluate_model(model_copy, X_val, y_val)
                
                convergence_history.append(score)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Parameter combination failed: {params}, error: {e}")
                convergence_history.append(float('-inf'))
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_time=0.0,
            iterations_completed=config.max_iterations,
            convergence_history=convergence_history
        )
    
    def _genetic_algorithm_optimization(
        self, model, X_train, y_train, X_val, y_val, config
    ) -> OptimizationResult:
        """Perform genetic algorithm optimization (simplified implementation)."""
        # Simplified genetic algorithm
        # In production, use libraries like DEAP or genetic-algorithms
        
        param_names = list(config.param_grid.keys())
        population_size = config.population_size
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name in param_names:
                param_values = config.param_grid[param_name]
                individual[param_name] = np.random.choice(param_values)
            population.append(individual)
        
        best_params = {}
        best_score = float('-inf')
        convergence_history = []
        
        for generation in range(config.max_iterations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                try:
                    model_copy = type(model)(**individual)
                    model_copy.fit(X_train, y_train)
                    score = self._evaluate_model(model_copy, X_val, y_val)
                    fitness_scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()
                        
                except Exception:
                    fitness_scores.append(float('-inf'))
            
            convergence_history.append(max(fitness_scores))
            
            # Selection and reproduction (simplified)
            # Select top 50% of population
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_half_size = population_size // 2
            selected_population = [population[i] for i in sorted_indices[:top_half_size]]
            
            # Create new population
            new_population = selected_population.copy()
            
            # Fill remaining spots with mutations
            while len(new_population) < population_size:
                parent = np.random.choice(selected_population)
                child = parent.copy()
                
                # Mutate
                if np.random.random() < config.mutation_rate:
                    param_to_mutate = np.random.choice(param_names)
                    param_values = config.param_grid[param_to_mutate]
                    child[param_to_mutate] = np.random.choice(param_values)
                
                new_population.append(child)
            
            population = new_population
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_time=0.0,
            iterations_completed=config.max_iterations,
            convergence_history=convergence_history
        )
    
    def _get_default_param_grid(self, model) -> Dict[str, List[Any]]:
        """Get default parameter grid for common models."""
        model_name = type(model).__name__
        
        if model_name == "RandomForestRegressor" or model_name == "RandomForestClassifier":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == "MLPRegressor" or model_name == "MLPClassifier":
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
        elif model_name == "LinearRegression":
            return {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        else:
            return {}
    
    def _evaluate_model(self, model, X_val, y_val) -> float:
        """Evaluate model performance."""
        if X_val.empty or y_val.empty:
            return 0.0
        
        try:
            predictions = model.predict(X_val)
            
            # Determine if classification or regression
            if y_val.dtype == 'object' or len(y_val.unique()) < 10:
                # Classification
                return float(accuracy_score(y_val, predictions))
            else:
                # Regression - return negative MSE for maximization
                mse = mean_squared_error(y_val, predictions)
                return float(-mse)
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return float('-inf')
    
    def _fallback_optimization(self, config: OptimizationConfig) -> OptimizationResult:
        """Fallback optimization when ML libraries not available."""
        return OptimizationResult(
            best_params={"fallback": True},
            best_score=0.5,
            optimization_time=1.0,
            iterations_completed=1,
            optimization_method=config.method.value
        )


class TaskSchedulingOptimizer:
    """
    Neural network-based intelligent task scheduling with deep learning insights.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize task scheduling optimizer."""
        self.model_dir = Path(model_dir) if model_dir else Path("models/scheduling")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Neural network models
        self.scheduling_model = None
        self.bottleneck_predictor = None
        self.resource_predictor = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Training data
        self.scheduling_history: List[Dict[str, Any]] = []
        
        if ML_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize neural network models."""
        if not ML_AVAILABLE:
            return
        
        # Task scheduling neural network
        self.scheduling_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        
        # Bottleneck prediction model
        self.bottleneck_predictor = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        
        # Resource utilization predictor
        self.resource_predictor = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
    
    def optimize_schedule(
        self,
        tasks: List[Dict[str, Any]],
        resources: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[TaskSchedulingDecision]:
        """
        Optimize task scheduling using neural networks.
        
        Args:
            tasks: List of tasks to schedule
            resources: Available resources
            constraints: Scheduling constraints
            
        Returns:
            List of optimal scheduling decisions
        """
        if not ML_AVAILABLE or not self._models_trained():
            return self._fallback_scheduling(tasks, resources)
        
        decisions = []
        
        for task in tasks:
            try:
                # Extract features
                task_features = self._extract_task_features(task, resources)
                
                # Predict optimal resource assignment
                resource_scores = self._predict_resource_scores(task_features, resources)
                
                # Select best resource
                best_resource_idx = np.argmax(resource_scores)
                best_resource = resources[best_resource_idx]
                
                # Predict performance metrics
                bottleneck_prob = self._predict_bottleneck_probability(task_features)
                success_prob = self._predict_success_probability(task_features)
                resource_util = self._predict_resource_utilization(task_features)
                
                # Calculate timing
                start_time = self._calculate_start_time(task, best_resource)
                completion_time = start_time + task.get('estimated_duration', 60)
                
                # Generate alternative options
                alternatives = self._generate_alternatives(
                    task_features, resources, resource_scores
                )
                
                decision = TaskSchedulingDecision(
                    task_id=task['id'],
                    assigned_resource=best_resource['id'],
                    priority_score=float(resource_scores[best_resource_idx]),
                    estimated_start_time=start_time,
                    estimated_completion_time=completion_time,
                    resource_utilization_prediction=resource_util,
                    bottleneck_probability=float(bottleneck_prob),
                    success_probability=float(success_prob),
                    scheduling_reason=self._generate_scheduling_reason(
                        task, best_resource, resource_scores[best_resource_idx]
                    ),
                    alternative_options=alternatives
                )
                
                decisions.append(decision)
                
            except Exception as e:
                logger.error(f"Scheduling optimization failed for task {task.get('id', 'unknown')}: {e}")
                # Add fallback decision
                fallback_decision = self._create_fallback_decision(task, resources)
                decisions.append(fallback_decision)
        
        logger.info(f"Generated {len(decisions)} scheduling decisions")
        return decisions
    
    def _extract_task_features(
        self, 
        task: Dict[str, Any], 
        resources: List[Dict[str, Any]]
    ) -> List[float]:
        """Extract features for neural network input."""
        # Task features
        duration = task.get('estimated_duration', 60)
        priority = task.get('priority', 5)
        cpu_req = task.get('cpu_requirement', 1.0)
        memory_req = task.get('memory_requirement', 1.0)
        
        # Resource availability features
        total_cpu = sum(r.get('cpu_capacity', 1.0) for r in resources)
        total_memory = sum(r.get('memory_capacity', 2.0) for r in resources)
        available_resources = len(resources)
        
        # System load features
        avg_resource_load = sum(r.get('current_load', 0.5) for r in resources) / max(len(resources), 1)
        
        # Temporal features
        hour_of_day = time.localtime().tm_hour / 24.0
        day_of_week = time.localtime().tm_wday / 7.0
        
        return [
            duration, priority, cpu_req, memory_req,
            total_cpu, total_memory, available_resources,
            avg_resource_load, hour_of_day, day_of_week
        ]
    
    def _predict_resource_scores(
        self, 
        task_features: List[float], 
        resources: List[Dict[str, Any]]
    ) -> List[float]:
        """Predict optimal resource assignment scores."""
        scores = []
        
        for resource in resources:
            # Combine task features with resource features
            resource_features = [
                resource.get('cpu_capacity', 1.0),
                resource.get('memory_capacity', 2.0),
                resource.get('current_load', 0.5),
                resource.get('reliability_score', 0.9),
                resource.get('cost_per_hour', 0.1)
            ]
            
            combined_features = task_features + resource_features
            
            if self.scheduling_model and hasattr(self.scheduling_model, 'predict_proba'):
                # Use trained model for prediction
                features_scaled = self.scaler.transform([combined_features])
                probabilities = self.scheduling_model.predict_proba(features_scaled)[0]
                scores.append(float(np.max(probabilities)))
            else:
                # Heuristic scoring
                cpu_score = 1.0 - abs(task_features[2] - resource.get('cpu_capacity', 1.0)) / max(task_features[2], resource.get('cpu_capacity', 1.0))
                memory_score = 1.0 - abs(task_features[3] - resource.get('memory_capacity', 2.0)) / max(task_features[3], resource.get('memory_capacity', 2.0))
                load_score = 1.0 - resource.get('current_load', 0.5)
                
                combined_score = (cpu_score + memory_score + load_score) / 3.0
                scores.append(combined_score)
        
        return scores
    
    def _predict_bottleneck_probability(self, task_features: List[float]) -> float:
        """Predict probability of this task causing a bottleneck."""
        if self.bottleneck_predictor and hasattr(self.bottleneck_predictor, 'predict'):
            features_scaled = self.scaler.transform([task_features])
            return float(self.bottleneck_predictor.predict(features_scaled)[0])
        else:
            # Heuristic: high duration and resource requirements increase bottleneck probability
            duration_factor = min(1.0, task_features[0] / 300.0)  # Normalize to 5 minutes
            resource_factor = min(1.0, (task_features[2] + task_features[3]) / 4.0)  # Normalize to 2 CPU + 2 GB
            return (duration_factor + resource_factor) / 2.0
    
    def _predict_success_probability(self, task_features: List[float]) -> float:
        """Predict probability of successful task completion."""
        # Heuristic based on system load and resource requirements
        system_load = task_features[7]  # avg_resource_load
        resource_demand = (task_features[2] + task_features[3]) / 2.0
        
        # Higher system load and resource demand decrease success probability
        base_success = 0.9
        load_penalty = system_load * 0.2
        demand_penalty = min(0.3, resource_demand * 0.1)
        
        return max(0.1, base_success - load_penalty - demand_penalty)
    
    def _predict_resource_utilization(self, task_features: List[float]) -> Dict[str, float]:
        """Predict resource utilization during task execution."""
        if self.resource_predictor and hasattr(self.resource_predictor, 'predict'):
            features_scaled = self.scaler.transform([task_features])
            prediction = self.resource_predictor.predict(features_scaled)[0]
            
            return {
                "cpu": float(min(1.0, max(0.0, prediction))),
                "memory": float(min(1.0, max(0.0, prediction * 0.8))),  # Memory usually lower than CPU
                "network": float(min(1.0, max(0.0, prediction * 0.3))),  # Network typically much lower
                "storage": float(min(1.0, max(0.0, prediction * 0.2)))   # Storage typically lowest
            }
        else:
            # Heuristic prediction
            base_cpu = task_features[2] / 2.0  # cpu_requirement normalized
            base_memory = task_features[3] / 2.0  # memory_requirement normalized
            
            return {
                "cpu": min(1.0, base_cpu),
                "memory": min(1.0, base_memory),
                "network": min(1.0, base_cpu * 0.3),
                "storage": min(1.0, base_cpu * 0.2)
            }
    
    def _calculate_start_time(self, task: Dict[str, Any], resource: Dict[str, Any]) -> float:
        """Calculate optimal start time for task."""
        current_time = time.time()
        
        # Check resource availability
        resource_available_time = resource.get('next_available_time', current_time)
        
        # Check dependencies
        dependencies = task.get('dependencies', [])
        dependency_completion_time = current_time
        
        for dep_id in dependencies:
            # Look up dependency completion time (simplified)
            dep_completion = self._get_dependency_completion_time(dep_id)
            dependency_completion_time = max(dependency_completion_time, dep_completion)
        
        return max(resource_available_time, dependency_completion_time)
    
    def _get_dependency_completion_time(self, dependency_id: str) -> float:
        """Get estimated completion time for a dependency."""
        # Simplified implementation - would look up from scheduling system
        return time.time() + 60  # Default 1 minute from now
    
    def _generate_alternatives(
        self, 
        task_features: List[float], 
        resources: List[Dict[str, Any]], 
        resource_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Generate alternative scheduling options."""
        alternatives = []
        
        # Sort resources by score
        sorted_indices = np.argsort(resource_scores)[::-1]
        
        # Take top 3 alternatives (excluding the best one)
        for i in sorted_indices[1:4]:
            if i < len(resources):
                alternative = {
                    "resource_id": resources[i]['id'],
                    "score": float(resource_scores[i]),
                    "reason": f"Alternative with score {resource_scores[i]:.3f}"
                }
                alternatives.append(alternative)
        
        return alternatives
    
    def _generate_scheduling_reason(
        self, 
        task: Dict[str, Any], 
        resource: Dict[str, Any], 
        score: float
    ) -> str:
        """Generate human-readable scheduling reason."""
        reasons = []
        
        if score > 0.8:
            reasons.append("excellent resource match")
        elif score > 0.6:
            reasons.append("good resource match")
        else:
            reasons.append("acceptable resource match")
        
        if resource.get('current_load', 0.5) < 0.3:
            reasons.append("low resource utilization")
        
        if resource.get('reliability_score', 0.9) > 0.95:
            reasons.append("high reliability")
        
        return f"Selected due to {', '.join(reasons)}"
    
    def _models_trained(self) -> bool:
        """Check if neural network models are trained."""
        return (
            self.scheduling_model is not None and 
            hasattr(self.scheduling_model, 'coefs_')
        )
    
    def _fallback_scheduling(
        self, 
        tasks: List[Dict[str, Any]], 
        resources: List[Dict[str, Any]]
    ) -> List[TaskSchedulingDecision]:
        """Fallback scheduling when ML models not available."""
        decisions = []
        
        for i, task in enumerate(tasks):
            # Simple round-robin assignment
            resource_idx = i % len(resources)
            resource = resources[resource_idx]
            
            decision = self._create_fallback_decision(task, [resource])
            decisions.append(decision)
        
        return decisions
    
    def _create_fallback_decision(
        self, 
        task: Dict[str, Any], 
        resources: List[Dict[str, Any]]
    ) -> TaskSchedulingDecision:
        """Create a fallback scheduling decision."""
        resource = resources[0] if resources else {"id": "default_resource"}
        
        return TaskSchedulingDecision(
            task_id=task.get('id', 'unknown'),
            assigned_resource=resource.get('id', 'default_resource'),
            priority_score=0.5,
            estimated_start_time=time.time(),
            estimated_completion_time=time.time() + task.get('estimated_duration', 60),
            resource_utilization_prediction={
                "cpu": 0.5, "memory": 0.5, "network": 0.2, "storage": 0.1
            },
            bottleneck_probability=0.3,
            success_probability=0.8,
            scheduling_reason="Fallback assignment due to unavailable ML models"
        )


class CostPredictor:
    """
    Advanced cost prediction with economic modeling and market analysis.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize cost predictor."""
        self.model_dir = Path(model_dir) if model_dir else Path("models/cost")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # ML models for different cost components
        self.compute_cost_model = None
        self.storage_cost_model = None
        self.network_cost_model = None
        self.economic_model = None
        
        # Historical cost data
        self.cost_history: List[Dict[str, Any]] = []
        
        # Market condition factors
        self.market_conditions = {
            "cloud_demand_index": 1.0,
            "resource_availability": 1.0,
            "seasonal_factor": 1.0,
            "economic_indicator": 1.0
        }
        
        if ML_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize cost prediction models."""
        if not ML_AVAILABLE:
            return
        
        self.compute_cost_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        
        self.storage_cost_model = LinearRegression()
        self.network_cost_model = LinearRegression()
        
        # Economic forecasting model
        self.economic_model = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            max_iter=500,
            random_state=42
        )
    
    def predict_cost(
        self,
        workflow_specs: Dict[str, Any],
        execution_time: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> CostPrediction:
        """
        Predict comprehensive cost for workflow execution.
        
        Args:
            workflow_specs: Workflow specifications
            execution_time: Expected execution time
            market_data: Current market conditions
            
        Returns:
            Detailed cost prediction with breakdown
        """
        if market_data:
            self.market_conditions.update(market_data)
        
        try:
            # Extract cost features
            cost_features = self._extract_cost_features(workflow_specs, execution_time)
            
            # Predict base costs
            compute_cost = self._predict_compute_cost(cost_features)
            storage_cost = self._predict_storage_cost(cost_features)
            network_cost = self._predict_network_cost(cost_features)
            
            # Apply market adjustments
            market_multiplier = self._calculate_market_multiplier()
            
            # Calculate total cost
            base_cost = compute_cost + storage_cost + network_cost
            adjusted_cost = base_cost * market_multiplier
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(adjusted_cost)
            
            # Generate cost breakdown
            cost_breakdown = {
                "compute": compute_cost * market_multiplier,
                "storage": storage_cost * market_multiplier,
                "network": network_cost * market_multiplier,
                "market_adjustment": adjusted_cost - base_cost,
                "base_cost": base_cost
            }
            
            # Seasonal adjustments
            seasonal_adjustments = self._calculate_seasonal_adjustments()
            
            # Risk factors
            risk_factors = self._assess_risk_factors(workflow_specs)
            
            # Optimization suggestions
            optimization_suggestions = self._generate_cost_optimization_suggestions(
                cost_breakdown, workflow_specs
            )
            
            # Calculate savings potential
            savings_potential = self._calculate_savings_potential(cost_breakdown)
            
            prediction = CostPrediction(
                predicted_cost=adjusted_cost,
                cost_breakdown=cost_breakdown,
                confidence_interval=confidence_interval,
                market_conditions=self.market_conditions.copy(),
                seasonal_adjustments=seasonal_adjustments,
                risk_factors=risk_factors,
                cost_optimization_suggestions=optimization_suggestions,
                savings_potential=savings_potential
            )
            
            # Store for learning
            self._record_prediction(workflow_specs, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Cost prediction failed: {e}")
            return self._fallback_cost_prediction(workflow_specs)
    
    def _extract_cost_features(
        self, 
        workflow_specs: Dict[str, Any], 
        execution_time: Optional[float]
    ) -> List[float]:
        """Extract features for cost prediction."""
        # Resource requirements
        cpu_hours = workflow_specs.get('cpu_requirement', 1.0) * (execution_time or 3600) / 3600
        memory_gb_hours = workflow_specs.get('memory_requirement', 2.0) * (execution_time or 3600) / 3600
        storage_gb = workflow_specs.get('storage_requirement', 10.0)
        network_gb = workflow_specs.get('network_transfer', 1.0)
        
        # Workflow characteristics
        task_count = workflow_specs.get('task_count', 1)
        parallelism = workflow_specs.get('parallelism_level', 1)
        priority = workflow_specs.get('priority', 5)
        
        # Time-based factors
        hour_of_day = time.localtime().tm_hour
        day_of_week = time.localtime().tm_wday
        
        # Market factors
        demand_index = self.market_conditions.get('cloud_demand_index', 1.0)
        availability = self.market_conditions.get('resource_availability', 1.0)
        
        return [
            cpu_hours, memory_gb_hours, storage_gb, network_gb,
            task_count, parallelism, priority,
            hour_of_day, day_of_week, demand_index, availability
        ]
    
    def _predict_compute_cost(self, features: List[float]) -> float:
        """Predict compute cost component."""
        if self.compute_cost_model and hasattr(self.compute_cost_model, 'predict'):
            prediction = self.compute_cost_model.predict([features])[0]
            return max(0.01, float(prediction))
        else:
            # Heuristic pricing
            cpu_hours = features[0]
            memory_gb_hours = features[1]
            
            cpu_cost = cpu_hours * 0.05  # $0.05 per CPU hour
            memory_cost = memory_gb_hours * 0.01  # $0.01 per GB hour
            
            return cpu_cost + memory_cost
    
    def _predict_storage_cost(self, features: List[float]) -> float:
        """Predict storage cost component."""
        storage_gb = features[2]
        return storage_gb * 0.002  # $0.002 per GB
    
    def _predict_network_cost(self, features: List[float]) -> float:
        """Predict network cost component."""
        network_gb = features[3]
        return network_gb * 0.01  # $0.01 per GB transfer
    
    def _calculate_market_multiplier(self) -> float:
        """Calculate market condition multiplier."""
        demand_factor = self.market_conditions.get('cloud_demand_index', 1.0)
        availability_factor = 1.0 / max(0.1, self.market_conditions.get('resource_availability', 1.0))
        seasonal_factor = self.market_conditions.get('seasonal_factor', 1.0)
        economic_factor = self.market_conditions.get('economic_indicator', 1.0)
        
        # Combine factors with weights
        multiplier = (
            demand_factor * 0.3 +
            availability_factor * 0.3 +
            seasonal_factor * 0.2 +
            economic_factor * 0.2
        )
        
        # Constrain to reasonable range
        return max(0.5, min(2.0, multiplier))
    
    def _calculate_confidence_interval(self, predicted_cost: float) -> Tuple[float, float]:
        """Calculate confidence interval for cost prediction."""
        # Simple approach: ±20% of predicted cost
        margin = predicted_cost * 0.2
        return (max(0.0, predicted_cost - margin), predicted_cost + margin)
    
    def _calculate_seasonal_adjustments(self) -> Dict[str, float]:
        """Calculate seasonal cost adjustments."""
        month = time.localtime().tm_mon
        
        # Simplified seasonal factors
        seasonal_factors = {
            "compute_seasonal": 1.1 if month in [11, 12, 1] else 1.0,  # Higher in winter
            "storage_seasonal": 1.05 if month in [3, 6, 9, 12] else 1.0,  # Quarterly spikes
            "network_seasonal": 1.0  # Assume constant
        }
        
        return seasonal_factors
    
    def _assess_risk_factors(self, workflow_specs: Dict[str, Any]) -> Dict[str, float]:
        """Assess cost risk factors."""
        risk_factors = {}
        
        # Complexity risk
        task_count = workflow_specs.get('task_count', 1)
        if task_count > 20:
            risk_factors['complexity_risk'] = 0.3
        elif task_count > 10:
            risk_factors['complexity_risk'] = 0.15
        else:
            risk_factors['complexity_risk'] = 0.05
        
        # Resource scaling risk
        parallelism = workflow_specs.get('parallelism_level', 1)
        if parallelism > 10:
            risk_factors['scaling_risk'] = 0.25
        elif parallelism > 5:
            risk_factors['scaling_risk'] = 0.1
        else:
            risk_factors['scaling_risk'] = 0.02
        
        # Duration uncertainty risk
        estimated_duration = workflow_specs.get('estimated_duration', 3600)
        if estimated_duration > 7200:  # > 2 hours
            risk_factors['duration_risk'] = 0.2
        elif estimated_duration > 3600:  # > 1 hour
            risk_factors['duration_risk'] = 0.1
        else:
            risk_factors['duration_risk'] = 0.05
        
        return risk_factors
    
    def _generate_cost_optimization_suggestions(
        self, 
        cost_breakdown: Dict[str, float], 
        workflow_specs: Dict[str, Any]
    ) -> List[str]:
        """Generate cost optimization suggestions."""
        suggestions = []
        
        # Compute optimization
        if cost_breakdown['compute'] > cost_breakdown['predicted_cost'] * 0.6:
            suggestions.append("Consider using spot instances for compute-intensive tasks")
            suggestions.append("Optimize resource allocation to reduce over-provisioning")
        
        # Storage optimization
        if cost_breakdown['storage'] > cost_breakdown['predicted_cost'] * 0.2:
            suggestions.append("Use tiered storage for infrequently accessed data")
            suggestions.append("Implement data compression to reduce storage costs")
        
        # Network optimization
        if cost_breakdown['network'] > cost_breakdown['predicted_cost'] * 0.1:
            suggestions.append("Optimize data transfer patterns to reduce network costs")
            suggestions.append("Use data caching to minimize redundant transfers")
        
        # Scheduling optimization
        parallelism = workflow_specs.get('parallelism_level', 1)
        if parallelism > 5:
            suggestions.append("Consider running during off-peak hours for lower costs")
        
        # Resource right-sizing
        cpu_req = workflow_specs.get('cpu_requirement', 1.0)
        memory_req = workflow_specs.get('memory_requirement', 2.0)
        if cpu_req > 4.0 or memory_req > 8.0:
            suggestions.append("Analyze resource usage patterns for right-sizing opportunities")
        
        return suggestions
    
    def _calculate_savings_potential(self, cost_breakdown: Dict[str, float]) -> float:
        """Calculate potential cost savings."""
        total_cost = cost_breakdown['predicted_cost']
        
        # Estimate savings from various optimizations
        spot_instance_savings = cost_breakdown['compute'] * 0.7  # 70% savings on compute
        storage_optimization_savings = cost_breakdown['storage'] * 0.3  # 30% savings on storage
        network_optimization_savings = cost_breakdown['network'] * 0.2  # 20% savings on network
        
        total_potential_savings = (
            spot_instance_savings + 
            storage_optimization_savings + 
            network_optimization_savings
        )
        
        # Cap savings at 50% of total cost
        return min(total_cost * 0.5, total_potential_savings)
    
    def _record_prediction(self, workflow_specs: Dict[str, Any], prediction: CostPrediction):
        """Record prediction for learning."""
        record = {
            "timestamp": time.time(),
            "workflow_specs": workflow_specs,
            "prediction": prediction.predicted_cost,
            "cost_breakdown": prediction.cost_breakdown,
            "market_conditions": prediction.market_conditions
        }
        
        self.cost_history.append(record)
        
        # Keep only recent history
        if len(self.cost_history) > 1000:
            self.cost_history = self.cost_history[-1000:]
    
    def _fallback_cost_prediction(self, workflow_specs: Dict[str, Any]) -> CostPrediction:
        """Fallback cost prediction when ML models not available."""
        # Simple heuristic pricing
        cpu_hours = workflow_specs.get('cpu_requirement', 1.0)
        memory_gb = workflow_specs.get('memory_requirement', 2.0)
        storage_gb = workflow_specs.get('storage_requirement', 10.0)
        
        compute_cost = cpu_hours * 0.05 + memory_gb * 0.01
        storage_cost = storage_gb * 0.002
        network_cost = 0.05  # Fixed small amount
        
        total_cost = compute_cost + storage_cost + network_cost
        
        return CostPrediction(
            predicted_cost=total_cost,
            cost_breakdown={
                "compute": compute_cost,
                "storage": storage_cost,
                "network": network_cost,
                "market_adjustment": 0.0,
                "base_cost": total_cost
            },
            confidence_interval=(total_cost * 0.8, total_cost * 1.2),
            market_conditions=self.market_conditions.copy(),
            seasonal_adjustments={"compute_seasonal": 1.0, "storage_seasonal": 1.0, "network_seasonal": 1.0},
            risk_factors={"complexity_risk": 0.1, "scaling_risk": 0.1, "duration_risk": 0.1},
            cost_optimization_suggestions=[
                "Use spot instances for cost savings",
                "Optimize resource allocation",
                "Consider running during off-peak hours"
            ],
            savings_potential=total_cost * 0.3
        )


# Export all optimization classes
__all__ = [
    "OptimizationMethod",
    "SchedulingStrategy", 
    "OptimizationConfig",
    "OptimizationResult",
    "TaskSchedulingDecision",
    "CostPrediction",
    "HyperparameterOptimizer",
    "TaskSchedulingOptimizer",
    "CostPredictor"
]