"""
CrewGraph AI Agent Pool Auto-Scaling
Dynamic agent pool management based on workload and specialization needs

Author: Vatsal216
Created: 2025-07-22 13:17:52 UTC
"""

import time
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
from collections import defaultdict, deque
import threading

from crewai import Agent
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


@dataclass
class AgentPoolConfig:
    """Agent pool configuration"""
    min_agents: int = 2
    max_agents: int = 20
    scale_up_threshold: float = 80.0    # CPU/utilization %
    scale_down_threshold: float = 30.0
    idle_timeout: int = 300             # 5 minutes
    specialization_weight: float = 0.3
    enable_specialization: bool = True


class AgentPoolScaler:
    """
    Dynamic agent pool scaling system.
    
    Features:
    - Agent specialization-based scaling
    - Workload-aware agent provisioning
    - Idle agent management and cleanup
    - Performance-based scaling decisions
    - Agent health monitoring
    
    Created by: Vatsal216
    Date: 2025-07-22 13:17:52 UTC
    """
    
    def __init__(self, 
                 agent_factory: Dict[str, callable],
                 config: Optional[AgentPoolConfig] = None):
        """
        Initialize agent pool scaler.
        
        Args:
            agent_factory: Dictionary of agent type to factory function
            config: Scaling configuration
        """
        self.agent_factory = agent_factory
        self.config = config or AgentPoolConfig()
        
        # Agent pools by type
        self.agent_pools: Dict[str, List[Agent]] = defaultdict(list)
        self.agent_utilization: Dict[str, float] = defaultdict(float)
        self.agent_last_used: Dict[str, float] = defaultdict(time.time)
        
        # Scaling history
        self.scaling_events = deque(maxlen=100)
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        
        logger.info(f"AgentPoolScaler initialized with {len(agent_factory)} agent types")
        logger.info(f"Config: min={config.min_agents}, max={config.max_agents}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 13:17:52")
    
    def start_scaling(self) -> None:
        """Start agent pool scaling"""
        self._running = True
        
        # Initialize minimum agents
        self._ensure_minimum_agents()
        
        # Start monitoring
        self._monitor_thread = threading.Thread(
            target=self._monitor_agent_pools,
            name="AgentPoolScaler-Monitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Agent pool scaling started")
    
    def stop_scaling(self) -> None:
        """Stop agent pool scaling"""
        self._running = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Agent pool scaling stopped")
    
    def _monitor_agent_pools(self) -> None:
        """Monitor agent pools and scale as needed"""
        while self._running:
            try:
                for agent_type in self.agent_factory.keys():
                    self._evaluate_pool_scaling(agent_type)
                
                # Cleanup idle agents
                self._cleanup_idle_agents()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in agent pool monitoring: {e}")
                time.sleep(30)
    
    def _evaluate_pool_scaling(self, agent_type: str) -> None:
        """Evaluate scaling needs for specific agent type"""
        current_pool_size = len(self.agent_pools[agent_type])
        utilization = self.agent_utilization[agent_type]
        
        # Scale up decision
        if (utilization > self.config.scale_up_threshold and 
            current_pool_size < self.config.max_agents):
            self._scale_up_agents(agent_type, 1)
        
        # Scale down decision
        elif (utilization < self.config.scale_down_threshold and 
              current_pool_size > self.config.min_agents):
            self._scale_down_agents(agent_type, 1)
    
    def _scale_up_agents(self, agent_type: str, count: int) -> None:
        """Scale up agents of specific type"""
        try:
            factory = self.agent_factory[agent_type]
            
            for _ in range(count):
                agent = factory()
                self.agent_pools[agent_type].append(agent)
                agent_id = f"{agent_type}_{len(self.agent_pools[agent_type])}"
                self.agent_last_used[agent_id] = time.time()
            
            # Record scaling event
            event = {
                'timestamp': time.time(),
                'agent_type': agent_type,
                'action': 'scale_up',
                'count': count,
                'pool_size': len(self.agent_pools[agent_type])
            }
            self.scaling_events.append(event)
            
            logger.info(f"Scaled UP {agent_type} agents by {count}. "
                       f"Pool size: {len(self.agent_pools[agent_type])}")
            
        except Exception as e:
            logger.error(f"Failed to scale up {agent_type} agents: {e}")
    
    def get_agent(self, agent_type: str, preferred_skills: Optional[List[str]] = None) -> Optional[Agent]:
        """Get available agent from pool"""
        pool = self.agent_pools.get(agent_type, [])
        
        if not pool:
            # Try to create one if none available
            if len(pool) < self.config.max_agents:
                self._scale_up_agents(agent_type, 1)
                pool = self.agent_pools[agent_type]
        
        if pool:
            # Select best matching agent
            selected_agent = self._select_best_agent(pool, preferred_skills)
            
            # Update utilization and usage tracking
            agent_id = f"{agent_type}_{pool.index(selected_agent)}"
            self.agent_last_used[agent_id] = time.time()
            self._update_utilization(agent_type)
            
            return selected_agent
        
        return None
    
    def _select_best_agent(self, pool: List[Agent], skills: Optional[List[str]] = None) -> Agent:
        """Select best agent from pool based on skills and load"""
        if not skills or not self.config.enable_specialization:
            return pool[0]  # Simple selection
        
        # Score agents based on skill match (simplified)
        best_agent = pool[0]
        best_score = 0
        
        for agent in pool:
            score = self._calculate_agent_score(agent, skills)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _calculate_agent_score(self, agent: Agent, required_skills: List[str]) -> float:
        """Calculate agent suitability score"""
        # Simplified scoring - in real implementation, would use agent capabilities
        base_score = 1.0
        
        # Add skill matching logic here
        # For now, return base score
        return base_score
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get agent pool status"""
        status = {
            'total_agents': sum(len(pool) for pool in self.agent_pools.values()),
            'pools': {},
            'scaling_events': list(self.scaling_events)[-10:],
            'created_by': 'Vatsal216',
            'timestamp': '2025-07-22 13:17:52'
        }
        
        for agent_type, pool in self.agent_pools.items():
            status['pools'][agent_type] = {
                'size': len(pool),
                'utilization': self.agent_utilization[agent_type],
                'max_capacity': self.config.max_agents
            }
        
        return status