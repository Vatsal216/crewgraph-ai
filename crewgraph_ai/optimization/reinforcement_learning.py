"""
Reinforcement Learning-based Task Scheduling for CrewGraph AI

Implements Q-learning and other RL algorithms for intelligent task scheduling,
dynamic workflow adaptation, and continuous optimization.

Author: Vatsal216
Created: 2025-07-23 17:20:00 UTC
"""

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    RL_NUMPY_AVAILABLE = True
except ImportError:
    np = None
    RL_NUMPY_AVAILABLE = False

from ..types import WorkflowId, NodeId, TaskStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RLState:
    """Reinforcement learning state representation."""
    
    pending_tasks: List[str]
    running_tasks: List[str]
    completed_tasks: List[str]
    available_resources: Dict[str, float]
    current_time: float
    workflow_progress: float
    resource_utilization: float
    
    def to_vector(self) -> List[float]:
        """Convert state to numerical vector for RL algorithms."""
        return [
            len(self.pending_tasks),
            len(self.running_tasks),
            len(self.completed_tasks),
            self.available_resources.get('cpu', 0.0),
            self.available_resources.get('memory', 0.0),
            self.current_time,
            self.workflow_progress,
            self.resource_utilization
        ]
    
    def get_state_hash(self) -> str:
        """Get hashable state representation."""
        return f"{len(self.pending_tasks)}_{len(self.running_tasks)}_{self.workflow_progress:.2f}_{self.resource_utilization:.2f}"


@dataclass
class RLAction:
    """Reinforcement learning action representation."""
    
    action_type: str  # 'schedule', 'delay', 'parallel', 'resource_adjust'
    task_id: Optional[str] = None
    resource_allocation: Optional[Dict[str, float]] = None
    priority_adjustment: Optional[float] = None
    
    def to_index(self, action_space_size: int) -> int:
        """Convert action to index for Q-table."""
        # Simple mapping for demonstration
        action_map = {
            'schedule': 0,
            'delay': 1,
            'parallel': 2,
            'resource_adjust': 3
        }
        return action_map.get(self.action_type, 0) % action_space_size


@dataclass
class RLReward:
    """Reinforcement learning reward calculation."""
    
    performance_reward: float
    resource_efficiency_reward: float
    completion_reward: float
    penalty: float
    total_reward: float


@dataclass 
class SchedulingDecision:
    """RL-based scheduling decision with rationale."""
    
    recommended_action: RLAction
    confidence_score: float
    expected_reward: float
    alternative_actions: List[RLAction]
    reasoning: str


class QLearningScheduler:
    """
    Q-Learning based task scheduler for intelligent workflow optimization.
    
    Uses Q-learning to learn optimal task scheduling policies based on
    historical performance and resource utilization patterns.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        state_space_size: int = 1000,
        action_space_size: int = 10
    ):
        """Initialize Q-learning scheduler."""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.01
        
        # Q-table and experience storage
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = deque(maxlen=10000)
        
        # State and action spaces
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.learning_episodes = 0
        
        # State-action history for learning
        self.current_episode: List[Tuple[str, int, float]] = []
        
        logger.info("Q-Learning scheduler initialized")
    
    def select_action(
        self,
        state: RLState,
        available_actions: List[RLAction],
        training_mode: bool = True
    ) -> SchedulingDecision:
        """
        Select optimal action using epsilon-greedy Q-learning policy.
        
        Args:
            state: Current RL state
            available_actions: List of possible actions
            training_mode: Whether to use exploration
            
        Returns:
            Scheduling decision with recommended action
        """
        state_hash = state.get_state_hash()
        
        # Get Q-values for all available actions
        action_values = []
        for i, action in enumerate(available_actions):
            action_index = action.to_index(self.action_space_size)
            q_value = self.q_table[state_hash][action_index]
            action_values.append((action, q_value, action_index))
        
        # Epsilon-greedy action selection
        if training_mode and random.random() < self.exploration_rate:
            # Exploration: choose random action
            chosen_action, q_value, action_index = random.choice(action_values)
            reasoning = f"Exploration action (ε={self.exploration_rate:.3f})"
        else:
            # Exploitation: choose best action
            action_values.sort(key=lambda x: x[1], reverse=True)
            chosen_action, q_value, action_index = action_values[0]
            reasoning = f"Exploitation action (Q-value: {q_value:.3f})"
        
        # Calculate confidence based on Q-value distribution
        q_values = [av[1] for av in action_values]
        confidence = self._calculate_confidence(q_values)
        
        # Get alternative actions
        alternatives = [av[0] for av in action_values[1:3]]  # Top 2 alternatives
        
        decision = SchedulingDecision(
            recommended_action=chosen_action,
            confidence_score=confidence,
            expected_reward=q_value,
            alternative_actions=alternatives,
            reasoning=reasoning
        )
        
        # Store for learning
        if training_mode:
            self.current_episode.append((state_hash, action_index, 0.0))  # Reward filled later
        
        return decision
    
    def update_reward(
        self,
        reward: RLReward,
        next_state: Optional[RLState] = None,
        is_terminal: bool = False
    ):
        """
        Update Q-values based on received reward.
        
        Args:
            reward: Reward received for last action
            next_state: Next state (if not terminal)
            is_terminal: Whether this is the final state
        """
        if not self.current_episode:
            return
        
        # Update reward for last state-action pair
        if self.current_episode:
            state_hash, action_index, _ = self.current_episode[-1]
            self.current_episode[-1] = (state_hash, action_index, reward.total_reward)
        
        if is_terminal:
            # Episode finished, perform Q-learning updates
            self._update_q_values()
            self._end_episode()
        elif next_state:
            # Store transition for future update
            next_state_hash = next_state.get_state_hash()
            self.experience_buffer.append({
                'state': state_hash,
                'action': action_index, 
                'reward': reward.total_reward,
                'next_state': next_state_hash,
                'terminal': is_terminal
            })
    
    def _update_q_values(self):
        """Update Q-values using Q-learning algorithm."""
        if len(self.current_episode) < 2:
            return
        
        # Backward update through episode
        for i in range(len(self.current_episode) - 2, -1, -1):
            state_hash, action_index, reward = self.current_episode[i]
            next_state_hash, next_action_index, next_reward = self.current_episode[i + 1]
            
            # Get max Q-value for next state
            next_state_q_values = self.q_table[next_state_hash]
            max_next_q = max(next_state_q_values.values()) if next_state_q_values else 0.0
            
            # Q-learning update
            current_q = self.q_table[state_hash][action_index]
            target_q = reward + self.discount_factor * max_next_q
            
            self.q_table[state_hash][action_index] = (
                current_q + self.learning_rate * (target_q - current_q)
            )
    
    def _end_episode(self):
        """End current episode and update statistics."""
        if self.current_episode:
            episode_reward = sum(step[2] for step in self.current_episode)
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(len(self.current_episode))
        
        self.current_episode.clear()
        self.learning_episodes += 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        logger.debug(f"Episode {self.learning_episodes} completed, reward: {episode_reward:.3f}")
    
    def _calculate_confidence(self, q_values: List[float]) -> float:
        """Calculate confidence score based on Q-value distribution."""
        if len(q_values) < 2:
            return 0.5
        
        q_values = sorted(q_values, reverse=True)
        
        if RL_NUMPY_AVAILABLE:
            # Use standard deviation for confidence
            std_dev = np.std(q_values)
            # Higher std_dev means lower confidence
            confidence = 1.0 / (1.0 + std_dev)
        else:
            # Simple confidence calculation
            max_q = q_values[0]
            second_q = q_values[1] if len(q_values) > 1 else 0
            difference = abs(max_q - second_q)
            confidence = min(1.0, difference / (abs(max_q) + 1e-6))
        
        return confidence
    
    def calculate_reward(
        self,
        previous_state: RLState,
        action: RLAction,
        current_state: RLState,
        execution_metrics: Dict[str, float]
    ) -> RLReward:
        """
        Calculate reward for state-action transition.
        
        Args:
            previous_state: Previous workflow state
            action: Action taken
            current_state: Resulting state
            execution_metrics: Performance metrics
            
        Returns:
            Calculated reward with components
        """
        # Performance reward (faster execution is better)
        execution_time = execution_metrics.get('execution_time', 0.0)
        performance_reward = max(0, 1.0 - (execution_time / 60.0))  # Normalize to 60s
        
        # Resource efficiency reward
        cpu_usage = execution_metrics.get('cpu_usage', 0.0)
        memory_usage = execution_metrics.get('memory_usage', 0.0)
        target_utilization = 0.75  # Target 75% utilization
        
        cpu_efficiency = 1.0 - abs(cpu_usage - target_utilization)
        memory_efficiency = 1.0 - abs(memory_usage - target_utilization)
        resource_efficiency_reward = (cpu_efficiency + memory_efficiency) / 2.0
        
        # Progress reward (completing tasks is good)
        progress_delta = current_state.workflow_progress - previous_state.workflow_progress
        completion_reward = progress_delta * 2.0  # Scale progress reward
        
        # Penalties
        penalty = 0.0
        
        # Penalty for resource waste
        if cpu_usage < 0.3 or memory_usage < 0.3:
            penalty += 0.2  # Under-utilization penalty
        
        if cpu_usage > 0.95 or memory_usage > 0.95:
            penalty += 0.3  # Over-utilization penalty
        
        # Penalty for delays
        if action.action_type == 'delay':
            penalty += 0.1
        
        # Calculate total reward
        total_reward = (
            performance_reward * 0.4 +
            resource_efficiency_reward * 0.3 +
            completion_reward * 0.3 -
            penalty
        )
        
        return RLReward(
            performance_reward=performance_reward,
            resource_efficiency_reward=resource_efficiency_reward,
            completion_reward=completion_reward,
            penalty=penalty,
            total_reward=total_reward
        )
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics and performance metrics."""
        stats = {
            "episodes_completed": self.learning_episodes,
            "current_exploration_rate": self.exploration_rate,
            "q_table_size": len(self.q_table),
            "experience_buffer_size": len(self.experience_buffer),
            "average_episode_reward": 0.0,
            "average_episode_length": 0.0,
            "learning_progress": {
                "recent_rewards": list(self.episode_rewards[-10:]),
                "reward_trend": "improving" if len(self.episode_rewards) > 5 and
                               sum(self.episode_rewards[-5:]) > sum(self.episode_rewards[-10:-5]) else "stable"
            }
        }
        
        if self.episode_rewards:
            stats["average_episode_reward"] = sum(self.episode_rewards) / len(self.episode_rewards)
        
        if self.episode_steps:
            stats["average_episode_length"] = sum(self.episode_steps) / len(self.episode_steps)
        
        return stats
    
    def save_model(self, filepath: str):
        """Save Q-learning model to file."""
        model_data = {
            "q_table": dict(self.q_table),
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "learning_episodes": self.learning_episodes
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            logger.info(f"Q-learning model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath: str):
        """Load Q-learning model from file."""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Restore Q-table
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in model_data.get("q_table", {}).items():
                for action, value in actions.items():
                    self.q_table[state][int(action)] = value
            
            # Restore parameters
            self.learning_rate = model_data.get("learning_rate", self.learning_rate)
            self.discount_factor = model_data.get("discount_factor", self.discount_factor)
            self.exploration_rate = model_data.get("exploration_rate", self.exploration_rate)
            self.episode_rewards = model_data.get("episode_rewards", [])
            self.episode_steps = model_data.get("episode_steps", [])
            self.learning_episodes = model_data.get("learning_episodes", 0)
            
            logger.info(f"Q-learning model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


class MultiObjectiveRLOptimizer:
    """
    Multi-objective reinforcement learning optimizer for complex workflow scenarios.
    
    Optimizes multiple objectives simultaneously: performance, cost, quality, resource efficiency.
    """
    
    def __init__(
        self,
        objectives: List[str] = None,
        objective_weights: Dict[str, float] = None
    ):
        """Initialize multi-objective RL optimizer."""
        self.objectives = objectives or ["performance", "cost", "quality", "resource_efficiency"]
        
        # Default weights for objectives
        default_weights = {obj: 1.0 / len(self.objectives) for obj in self.objectives}
        self.objective_weights = objective_weights or default_weights
        
        # Individual Q-learning schedulers for each objective
        self.schedulers = {
            obj: QLearningScheduler(learning_rate=0.1, exploration_rate=0.2)
            for obj in self.objectives
        }
        
        # Pareto front tracking
        self.pareto_solutions: List[Dict[str, Any]] = []
        
        logger.info(f"Multi-objective RL optimizer initialized with objectives: {self.objectives}")
    
    def select_action(
        self,
        state: RLState,
        available_actions: List[RLAction],
        training_mode: bool = True
    ) -> SchedulingDecision:
        """
        Select action using multi-objective optimization.
        
        Combines decisions from multiple single-objective schedulers.
        """
        # Get decisions from each objective scheduler
        objective_decisions = {}
        for obj, scheduler in self.schedulers.items():
            decision = scheduler.select_action(state, available_actions, training_mode)
            objective_decisions[obj] = decision
        
        # Multi-objective decision fusion
        action_scores = defaultdict(float)
        
        for obj, decision in objective_decisions.items():
            weight = self.objective_weights[obj]
            action = decision.recommended_action
            score = decision.expected_reward * decision.confidence_score
            
            # Create action key for aggregation
            action_key = f"{action.action_type}_{action.task_id}"
            action_scores[action_key] += weight * score
        
        # Select best action based on weighted scores
        best_action_key = max(action_scores.keys(), key=lambda k: action_scores[k])
        best_score = action_scores[best_action_key]
        
        # Find the actual action object
        best_action = None
        best_confidence = 0.0
        for decision in objective_decisions.values():
            action = decision.recommended_action
            action_key = f"{action.action_type}_{action.task_id}"
            if action_key == best_action_key:
                best_action = action
                best_confidence = decision.confidence_score
                break
        
        # Create combined decision
        alternatives = []
        for obj_decision in objective_decisions.values():
            if obj_decision.recommended_action != best_action:
                alternatives.append(obj_decision.recommended_action)
        
        combined_decision = SchedulingDecision(
            recommended_action=best_action,
            confidence_score=best_confidence,
            expected_reward=best_score,
            alternative_actions=alternatives[:2],
            reasoning=f"Multi-objective optimization (weights: {self.objective_weights})"
        )
        
        return combined_decision
    
    def update_reward(
        self,
        objective_rewards: Dict[str, RLReward],
        next_state: Optional[RLState] = None,
        is_terminal: bool = False
    ):
        """Update all objective schedulers with their respective rewards."""
        for obj, reward in objective_rewards.items():
            if obj in self.schedulers:
                self.schedulers[obj].update_reward(reward, next_state, is_terminal)
        
        # Track Pareto solutions
        if is_terminal:
            self._update_pareto_front(objective_rewards)
    
    def _update_pareto_front(self, objective_rewards: Dict[str, RLReward]):
        """Update Pareto front with new solution."""
        current_solution = {
            obj: reward.total_reward for obj, reward in objective_rewards.items()
        }
        
        # Check if current solution is Pareto optimal
        is_pareto_optimal = True
        dominated_solutions = []
        
        for i, existing_solution in enumerate(self.pareto_solutions):
            dominates_existing = all(
                current_solution[obj] >= existing_solution[obj] for obj in self.objectives
            ) and any(
                current_solution[obj] > existing_solution[obj] for obj in self.objectives
            )
            
            dominated_by_existing = all(
                existing_solution[obj] >= current_solution[obj] for obj in self.objectives
            ) and any(
                existing_solution[obj] > current_solution[obj] for obj in self.objectives
            )
            
            if dominates_existing:
                dominated_solutions.append(i)
            elif dominated_by_existing:
                is_pareto_optimal = False
                break
        
        # Update Pareto front
        if is_pareto_optimal:
            # Remove dominated solutions
            for i in reversed(dominated_solutions):
                del self.pareto_solutions[i]
            
            # Add new solution
            self.pareto_solutions.append(current_solution.copy())
            
            # Limit Pareto front size
            if len(self.pareto_solutions) > 20:
                self.pareto_solutions = self.pareto_solutions[-20:]
    
    def get_pareto_solutions(self) -> List[Dict[str, float]]:
        """Get current Pareto optimal solutions."""
        return self.pareto_solutions.copy()
    
    def adapt_objective_weights(self, performance_feedback: Dict[str, float]):
        """Adapt objective weights based on performance feedback."""
        # Simple adaptation: increase weights for poorly performing objectives
        total_feedback = sum(performance_feedback.values())
        
        if total_feedback > 0:
            for obj in self.objectives:
                if obj in performance_feedback:
                    # Lower performance means higher weight needed
                    performance = performance_feedback[obj]
                    normalized_performance = performance / total_feedback
                    
                    # Inverse relationship: lower performance → higher weight
                    adaptation_factor = 1.0 / (normalized_performance + 0.1)
                    self.objective_weights[obj] *= adaptation_factor
            
            # Normalize weights
            total_weight = sum(self.objective_weights.values())
            for obj in self.objectives:
                self.objective_weights[obj] /= total_weight
            
            logger.info(f"Adapted objective weights: {self.objective_weights}")


def create_standard_actions(pending_tasks: List[str], resources: Dict[str, float]) -> List[RLAction]:
    """Create standard set of actions for RL scheduler."""
    actions = []
    
    # Schedule actions for each pending task
    for task_id in pending_tasks[:5]:  # Limit to top 5 tasks
        actions.append(RLAction(
            action_type="schedule",
            task_id=task_id,
            resource_allocation={"cpu": 0.5, "memory": 0.5}
        ))
    
    # Delay action
    actions.append(RLAction(action_type="delay"))
    
    # Parallel execution action
    if len(pending_tasks) > 1:
        actions.append(RLAction(action_type="parallel"))
    
    # Resource adjustment actions
    if resources.get('cpu', 0) > 0.8:
        actions.append(RLAction(
            action_type="resource_adjust",
            resource_allocation={"cpu": 0.6, "memory": 0.8}
        ))
    
    return actions