"""
Workflow execution benchmarks for CrewGraph AI

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import time
import statistics
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from crewgraph_ai.core import GraphOrchestrator, SharedState
from crewgraph_ai.memory import DictMemory, create_memory
from tests.fixtures import create_mock_crew


class WorkflowBenchmark:
    """Benchmark workflow execution performance"""
    
    def __init__(self, memory_type: str = "dict"):
        self.memory_type = memory_type
        self.memory = create_memory(memory_type)
        self.memory.connect()
        
        self.state = SharedState(memory_backend=self.memory)
        self.orchestrator = GraphOrchestrator(state=self.state)
        
        self.results = {}
        
    def cleanup(self):
        """Clean up resources"""
        try:
            self.state.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def benchmark_simple_workflow(self, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark simple linear workflow"""
        execution_times = []
        
        for i in range(iterations):
            # Create fresh mock crew for each iteration
            crew = create_mock_crew(2, 2)
            
            # Register agents and tasks
            agent_ids = []
            for agent in crew["agents"]:
                agent_id = self.orchestrator.register_agent(agent)
                agent_ids.append(agent_id)
            
            task_ids = []
            for task in crew["tasks"]:
                task_id = self.orchestrator.register_task(task)
                task_ids.append(task_id)
            
            # Define simple workflow
            workflow_config = {
                "nodes": (
                    [{"id": aid, "type": "agent"} for aid in agent_ids] +
                    [{"id": tid, "type": "task"} for tid in task_ids]
                ),
                "edges": [
                    {"from": agent_ids[0], "to": task_ids[0]},
                    {"from": task_ids[0], "to": task_ids[1]},
                    {"from": agent_ids[1], "to": task_ids[1]}
                ]
            }
            
            # Benchmark execution
            start_time = time.time()
            self.orchestrator.define_workflow(workflow_config)
            results = self.orchestrator.execute_workflow({"input": f"benchmark_iteration_{i}"})
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
            
            # Clean up for next iteration
            self.state.clear()
        
        return {
            "test_name": "simple_workflow",
            "iterations": iterations,
            "execution_times": execution_times,
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "memory_type": self.memory_type
        }
    
    def benchmark_parallel_workflow(self, iterations: int = 10, num_parallel_tasks: int = 5) -> Dict[str, Any]:
        """Benchmark parallel workflow execution"""
        execution_times = []
        
        for i in range(iterations):
            # Create mock crew with parallel tasks
            crew = create_mock_crew(num_parallel_tasks, num_parallel_tasks)
            
            # Register agents and tasks
            agent_ids = []
            for agent in crew["agents"]:
                agent_id = self.orchestrator.register_agent(agent)
                agent_ids.append(agent_id)
            
            task_ids = []
            for task in crew["tasks"]:
                task_id = self.orchestrator.register_task(task)
                task_ids.append(task_id)
            
            # Define parallel workflow (each agent works on one task)
            workflow_config = {
                "nodes": (
                    [{"id": aid, "type": "agent"} for aid in agent_ids] +
                    [{"id": tid, "type": "task"} for tid in task_ids]
                ),
                "edges": [
                    {"from": agent_ids[j], "to": task_ids[j]} for j in range(len(agent_ids))
                ]
            }
            
            # Benchmark execution
            start_time = time.time()
            self.orchestrator.define_workflow(workflow_config)
            results = self.orchestrator.execute_workflow(
                {"input": f"parallel_benchmark_{i}"}, 
                parallel=True
            )
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
            
            # Clean up for next iteration
            self.state.clear()
        
        return {
            "test_name": "parallel_workflow",
            "iterations": iterations,
            "parallel_tasks": num_parallel_tasks,
            "execution_times": execution_times,
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "memory_type": self.memory_type
        }
    
    def benchmark_complex_workflow(self, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark complex workflow with dependencies"""
        execution_times = []
        
        for i in range(iterations):
            # Create larger mock crew
            crew = create_mock_crew(5, 8)
            
            # Register agents and tasks
            agent_ids = []
            for agent in crew["agents"]:
                agent_id = self.orchestrator.register_agent(agent)
                agent_ids.append(agent_id)
            
            task_ids = []
            for task in crew["tasks"]:
                task_id = self.orchestrator.register_task(task)
                task_ids.append(task_id)
            
            # Define complex workflow with dependencies
            workflow_config = {
                "nodes": (
                    [{"id": aid, "type": "agent"} for aid in agent_ids] +
                    [{"id": tid, "type": "task"} for tid in task_ids]
                ),
                "edges": [
                    # Create complex dependency chain
                    {"from": agent_ids[0], "to": task_ids[0]},
                    {"from": task_ids[0], "to": task_ids[1]},
                    {"from": agent_ids[1], "to": task_ids[1]},
                    {"from": task_ids[1], "to": task_ids[2]},
                    {"from": agent_ids[2], "to": task_ids[2]},
                    {"from": task_ids[2], "to": task_ids[3]},
                    {"from": agent_ids[3], "to": task_ids[3]},
                    # Parallel branches
                    {"from": task_ids[0], "to": task_ids[4]},
                    {"from": task_ids[0], "to": task_ids[5]},
                    {"from": agent_ids[4], "to": task_ids[4]},
                    {"from": agent_ids[4], "to": task_ids[5]},
                    # Convergence
                    {"from": task_ids[4], "to": task_ids[6]},
                    {"from": task_ids[5], "to": task_ids[6]},
                    {"from": task_ids[3], "to": task_ids[7]},
                    {"from": task_ids[6], "to": task_ids[7]}
                ]
            }
            
            # Benchmark execution
            start_time = time.time()
            self.orchestrator.define_workflow(workflow_config)
            results = self.orchestrator.execute_workflow({"input": f"complex_benchmark_{i}"})
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
            
            # Clean up for next iteration
            self.state.clear()
        
        return {
            "test_name": "complex_workflow",
            "iterations": iterations,
            "agents": len(agent_ids),
            "tasks": len(task_ids),
            "execution_times": execution_times,
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "memory_type": self.memory_type
        }
    
    def run_all_benchmarks(self, iterations: int = 10) -> Dict[str, Any]:
        """Run all workflow benchmarks"""
        print("Starting workflow benchmarks...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "memory_type": self.memory_type,
            "iterations": iterations,
            "benchmarks": {}
        }
        
        # Simple workflow benchmark
        print("Running simple workflow benchmark...")
        results["benchmarks"]["simple"] = self.benchmark_simple_workflow(iterations)
        
        # Parallel workflow benchmark
        print("Running parallel workflow benchmark...")
        results["benchmarks"]["parallel"] = self.benchmark_parallel_workflow(iterations, 3)
        
        # Complex workflow benchmark
        print("Running complex workflow benchmark...")
        results["benchmarks"]["complex"] = self.benchmark_complex_workflow(max(1, iterations // 2))
        
        self.results = results
        return results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmarks/reports/workflow_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return
        
        print("\n" + "="*60)
        print("WORKFLOW BENCHMARK SUMMARY")
        print("="*60)
        print(f"Memory Type: {self.results['memory_type']}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Iterations: {self.results['iterations']}")
        print()
        
        for name, result in self.results["benchmarks"].items():
            print(f"{name.upper()} WORKFLOW:")
            print(f"  Average Time: {result['avg_time']:.4f} seconds")
            print(f"  Min Time: {result['min_time']:.4f} seconds")
            print(f"  Max Time: {result['max_time']:.4f} seconds")
            print(f"  Std Dev: {result['std_dev']:.4f} seconds")
            
            if 'parallel_tasks' in result:
                print(f"  Parallel Tasks: {result['parallel_tasks']}")
            if 'agents' in result and 'tasks' in result:
                print(f"  Agents: {result['agents']}, Tasks: {result['tasks']}")
            print()


def run_workflow_benchmarks(memory_type: str = "dict", iterations: int = 10) -> str:
    """Run complete workflow benchmarks and return results file"""
    benchmark = WorkflowBenchmark(memory_type)
    
    try:
        # Run all benchmarks
        results = benchmark.run_all_benchmarks(iterations)
        
        # Print summary
        benchmark.print_summary()
        
        # Save results
        filename = benchmark.save_results()
        
        return filename
        
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    memory_type = sys.argv[1] if len(sys.argv) > 1 else "dict"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Running workflow benchmarks with {memory_type} memory backend...")
    print(f"Iterations: {iterations}")
    
    results_file = run_workflow_benchmarks(memory_type, iterations)
    print(f"\nBenchmark completed. Results saved to: {results_file}")