"""
Memory performance benchmarks for CrewGraph AI

Author: Vatsal216
Created: 2025-07-23 06:14:25 UTC
"""

import time
import statistics
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from crewgraph_ai.memory import create_memory


class MemoryBenchmark:
    """Benchmark memory backend performance"""
    
    def __init__(self, memory_type: str = "dict"):
        self.memory_type = memory_type
        self.memory = create_memory(memory_type)
        self.memory.connect()
        self.results = {}
        
    def cleanup(self):
        """Clean up resources"""
        try:
            self.memory.clear()
            self.memory.disconnect()
        except Exception:
            pass
    
    def benchmark_save_operations(self, iterations: int = 100, value_size: int = 100) -> Dict[str, Any]:
        """Benchmark save operations"""
        test_value = "x" * value_size
        execution_times = []
        
        for i in range(iterations):
            key = f"benchmark_save_{i}"
            
            start_time = time.time()
            self.memory.save(key, test_value)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        return {
            "operation": "save",
            "iterations": iterations,
            "value_size": value_size,
            "execution_times": execution_times,
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "ops_per_second": iterations / sum(execution_times) if sum(execution_times) > 0 else 0
        }
    
    def benchmark_load_operations(self, iterations: int = 100, value_size: int = 100) -> Dict[str, Any]:
        """Benchmark load operations"""
        test_value = "x" * value_size
        
        # Setup data first
        for i in range(iterations):
            key = f"benchmark_load_{i}"
            self.memory.save(key, test_value)
        
        execution_times = []
        
        for i in range(iterations):
            key = f"benchmark_load_{i}"
            
            start_time = time.time()
            result = self.memory.load(key)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        return {
            "operation": "load",
            "iterations": iterations,
            "value_size": value_size,
            "execution_times": execution_times,
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "ops_per_second": iterations / sum(execution_times) if sum(execution_times) > 0 else 0
        }
    
    def benchmark_delete_operations(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark delete operations"""
        # Setup data first
        for i in range(iterations):
            key = f"benchmark_delete_{i}"
            self.memory.save(key, f"value_{i}")
        
        execution_times = []
        
        for i in range(iterations):
            key = f"benchmark_delete_{i}"
            
            start_time = time.time()
            self.memory.delete(key)
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        return {
            "operation": "delete",
            "iterations": iterations,
            "execution_times": execution_times,
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "ops_per_second": iterations / sum(execution_times) if sum(execution_times) > 0 else 0
        }
    
    def benchmark_list_keys_operation(self, num_keys: int = 100) -> Dict[str, Any]:
        """Benchmark list_keys operation"""
        # Setup data first
        for i in range(num_keys):
            key = f"benchmark_list_{i}"
            self.memory.save(key, f"value_{i}")
        
        execution_times = []
        iterations = 10  # Run list_keys multiple times
        
        for i in range(iterations):
            start_time = time.time()
            keys = self.memory.list_keys()
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        return {
            "operation": "list_keys",
            "num_keys": num_keys,
            "iterations": iterations,
            "execution_times": execution_times,
            "avg_time": statistics.mean(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "ops_per_second": iterations / sum(execution_times) if sum(execution_times) > 0 else 0
        }
    
    def run_all_benchmarks(self, iterations: int = 100) -> Dict[str, Any]:
        """Run all memory benchmarks"""
        print(f"Starting memory benchmarks for {self.memory_type} backend...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "memory_type": self.memory_type,
            "iterations": iterations,
            "benchmarks": {}
        }
        
        # Save operations benchmark
        print("Running save operations benchmark...")
        results["benchmarks"]["save"] = self.benchmark_save_operations(iterations)
        
        # Load operations benchmark  
        print("Running load operations benchmark...")
        results["benchmarks"]["load"] = self.benchmark_load_operations(iterations)
        
        # Delete operations benchmark
        print("Running delete operations benchmark...")
        results["benchmarks"]["delete"] = self.benchmark_delete_operations(iterations)
        
        # List keys benchmark
        print("Running list_keys benchmark...")
        results["benchmarks"]["list_keys"] = self.benchmark_list_keys_operation(100)
        
        self.results = results
        return results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmarks/reports/memory_benchmark_{self.memory_type}_{timestamp}.json"
        
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
        print("MEMORY BENCHMARK SUMMARY")
        print("="*60)
        print(f"Memory Type: {self.results['memory_type']}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Iterations: {self.results['iterations']}")
        print()
        
        for name, result in self.results["benchmarks"].items():
            print(f"{name.upper()} OPERATIONS:")
            print(f"  Average Time: {result['avg_time']:.6f} seconds")
            print(f"  Min Time: {result['min_time']:.6f} seconds")
            print(f"  Max Time: {result['max_time']:.6f} seconds")
            print(f"  Ops/Second: {result['ops_per_second']:.2f}")
            if 'value_size' in result:
                print(f"  Value Size: {result['value_size']} bytes")
            print()


def run_memory_benchmarks(memory_type: str = "dict", iterations: int = 100) -> str:
    """Run complete memory benchmarks and return results file"""
    benchmark = MemoryBenchmark(memory_type)
    
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
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"Running memory benchmarks with {memory_type} backend...")
    print(f"Iterations: {iterations}")
    
    results_file = run_memory_benchmarks(memory_type, iterations)
    print(f"\nBenchmark completed. Results saved to: {results_file}")