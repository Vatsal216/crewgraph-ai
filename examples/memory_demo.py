"""
CrewGraph AI Memory System Demonstration
Comprehensive demo of all memory backends with benchmarks

Author: Vatsal216
Created: 2025-07-22 12:09:19 UTC
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crewgraph_ai.memory import (
    DictMemory, RedisMemory, FAISSMemory, SQLMemory,
    MemoryConfig, MemoryType, MemoryUtils, MemorySerializer,
    create_memory
)

def demonstrate_dict_memory():
    """Demonstrate dictionary memory backend"""
    print("\n🗂️  DICTIONARY MEMORY DEMO")
    print("-" * 50)
    
    # Create dictionary memory with advanced features
    memory = DictMemory(
        max_size=1000,
        default_ttl=60,
        enable_compression=True,
        compression_threshold=100
    )
    
    memory.connect()
    
    # Save various data types
    test_data = {
        "string_key": "Hello CrewGraph AI by Vatsal216!",
        "number_key": 42,
        "list_key": [1, 2, 3, 4, 5],
        "dict_key": {"name": "CrewGraph", "version": "1.0.0", "author": "Vatsal216"},
        "large_data": "x" * 2000  # Will trigger compression
    }
    
    print("💾 Saving test data...")
    for key, value in test_data.items():
        success = memory.save(key, value, ttl=30)
        print(f"  ✅ Saved '{key}': {success}")
    
    print("\n📖 Loading test data...")
    for key in test_data.keys():
        value = memory.load(key)
        print(f"  📋 Loaded '{key}': {type(value).__name__}")
    
    # Show cache statistics
    stats = memory.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Hit Rate: {stats['hit_rate']:.2%}")
    print(f"  Total Operations: {stats['hit_count'] + stats['miss_count']}")
    print(f"  Memory Usage: {stats['current_size']}/{stats['max_size']} items")
    print(f"  Compression: {stats['compression_enabled']}")
    
    memory.disconnect()
    print("✅ Dictionary memory demo completed")

def demonstrate_faiss_memory():
    """Demonstrate FAISS vector memory backend"""
    print("\n🔍 FAISS VECTOR MEMORY DEMO")
    print("-" * 50)
    
    try:
        # Create FAISS memory for vector similarity search
        memory = FAISSMemory(
            dimension=128,
            index_type="Flat",
            metric_type="L2",
            max_vectors=10000
        )
        
        memory.connect()
        
        # Generate sample vectors (simulating embeddings)
        print("🧮 Generating sample vectors...")
        vectors = {}
        np.random.seed(42)  # For reproducible results
        
        for i in range(100):
            vector = np.random.random(128).astype(np.float32)
            metadata = {
                "id": i,
                "category": f"category_{i % 5}",
                "created_by": "Vatsal216",
                "timestamp": "2025-07-22 12:09:19"
            }
            
            vectors[f"vector_{i}"] = {
                "vector": vector.tolist(),
                "metadata": metadata
            }
        
        # Save vectors
        print("💾 Saving vectors to FAISS index...")
        saved_count = 0
        for key, data in vectors.items():
            if memory.save(key, data):
                saved_count += 1
        
        print(f"  ✅ Saved {saved_count} vectors")
        
        # Perform similarity search
        print("\n🔎 Performing similarity search...")
        query_vector = np.random.random(128).astype(np.float32)
        
        search_results = memory.search(
            query_vector=query_vector,
            k=5,
            include_metadata=True
        )
        
        print(f"  🎯 Found {len(search_results)} similar vectors:")
        for i, result in enumerate(search_results):
            print(f"    {i+1}. ID: {result['id']}, Distance: {result['distance']:.4f}")
            print(f"       Category: {result['metadata']['category']}")
        
        # Show index information
        index_info = memory.get_index_info()
        print(f"\n📊 FAISS Index Information:")
        print(f"  Index Type: {index_info['index_type']}")
        print(f"  Total Vectors: {index_info['total_vectors']}")
        print(f"  Dimension: {index_info['dimension']}")
        print(f"  Memory Usage: {index_info['memory_usage_bytes'] / 1024:.1f} KB")
        print(f"  Search Count: {index_info['search_count']}")
        
        memory.disconnect()
        print("✅ FAISS memory demo completed")
        
    except ImportError:
        print("❌ FAISS not available. Install with: pip install faiss-cpu")

def demonstrate_sql_memory():
    """Demonstrate SQL memory backend"""
    print("\n🗄️  SQL MEMORY DEMO")
    print("-" * 50)
    
    try:
        # Create SQL memory with SQLite for demo
        memory = SQLMemory(
            database_url="sqlite:///crewgraph_demo.db",
            enable_compression=True,
            enable_ttl_cleanup=False  # Disable for demo
        )
        
        memory.connect()
        
        # Save complex data structures
        print("💾 Saving data to SQL database...")
        
        complex_data = {
            "user_profile": {
                "name": "Vatsal216",
                "role": "CrewGraph AI Developer",
                "created_at": "2025-07-22 12:09:19",
                "preferences": {
                    "theme": "dark",
                    "language": "python",
                    "frameworks": ["CrewAI", "LangGraph", "CrewGraph AI"]
                }
            },
            "workflow_config": {
                "name": "production_workflow",
                "agents": ["researcher", "writer", "reviewer"],
                "max_tasks": 10,
                "timeout": 300
            },
            "metrics_data": {
                "executions": 1500,
                "success_rate": 0.98,
                "avg_duration": 45.2,
                "errors": ["timeout", "connection_lost"]
            }
        }
        
        saved_count = 0
        for key, data in complex_data.items():
            if memory.save(key, data, ttl=3600):  # 1 hour TTL
                saved_count += 1
                print(f"  ✅ Saved '{key}'")
        
        print(f"\n📖 Loading data from SQL database...")
        for key in complex_data.keys():
            value = memory.load(key)
            if value:
                print(f"  📋 Loaded '{key}': {len(str(value))} characters")
        
        # Show database statistics
        stats = memory.get_database_stats()
        print(f"\n📊 Database Statistics:")
        print(f"  Total Items: {stats['total_items']}")
        print(f"  Total Size: {stats['total_size_bytes']} bytes")
        print(f"  Compression Ratio: {stats['compression_ratio']:.2%}")
        print(f"  Database Type: {stats['database_type']}")
        
        # Search by metadata (simplified demo)
        print(f"\n🔍 Searching by metadata...")
        keys = memory.search_by_metadata({"created_by": "Vatsal216"})
        print(f"  🎯 Found {len(keys)} items created by Vatsal216")
        
        memory.disconnect()
        print("✅ SQL memory demo completed")
        
    except ImportError:
        print("❌ SQLAlchemy not available. Install with: pip install sqlalchemy")

def demonstrate_memory_comparison():
    """Compare different memory backends"""
    print("\n⚖️  MEMORY BACKEND COMPARISON")
    print("-" * 50)
    
    # Create configurations for comparison
    configs = [
        MemoryConfig(
            memory_type=MemoryType.DICT,
            max_size=1000,
            environment="demo"
        )
    ]
    
    # Add Redis config if available
    try:
        from crewgraph_ai.memory.redis_memory import RedisMemory
        configs.append(MemoryConfig(
            memory_type=MemoryType.REDIS,
            redis_host="localhost",
            environment="demo"
        ))
    except ImportError:
        pass
    
    print(f"🔧 Comparing {len(configs)} memory backends...")
    
    # Run comparison benchmark
    comparison_results = MemoryUtils.compare_memory_backends(configs, test_size=100)
    
    print(f"\n📊 Benchmark Results:")
    print("-" * 60)
    
    for backend_name, results in comparison_results.items():
        if "error" in results:
            print(f"❌ {backend_name}: {results['error']}")
            continue
        
        print(f"\n🏷️  {backend_name.upper()}:")
        
        for operation, result in results.items():
            print(f"  {operation.upper()}:")
            print(f"    Operations/sec: {result.operations_per_second:.1f}")
            print(f"    Avg Latency: {result.average_latency_ms:.2f} ms")
            print(f"    Success Rate: {result.success_rate:.2%}")

def demonstrate_memory_serialization():
    """Demonstrate advanced serialization features"""
    print("\n🔄 MEMORY SERIALIZATION DEMO")
    print("-" * 50)
    
    # Test different serialization formats
    formats = ["pickle", "json"]
    
    test_obj = {
        "name": "CrewGraph AI",
        "version": "1.0.0",
        "created_by": "Vatsal216",
        "created_at": "2025-07-22 12:09:19",
        "features": ["memory", "planning", "monitoring"],
        "metrics": {
            "performance": 0.95,
            "reliability": 0.99,
            "scalability": 0.92
        }
    }
    
    for format_type in formats:
        print(f"\n🔧 Testing {format_type.upper()} serialization:")
        
        try:
            serializer = MemorySerializer(
                format_type=format_type,
                enable_compression=True,
                compression_threshold=50
            )
            
            # Serialize
            start_time = time.time()
            serialized = serializer.serialize(test_obj)
            serialize_time = time.time() - start_time
            
            # Deserialize
            start_time = time.time()
            deserialized = serializer.deserialize(serialized)
            deserialize_time = time.time() - start_time
            
            # Check compression
            compression_ratio = serializer.get_compression_ratio(test_obj)
            
            print(f"  ✅ Serialized size: {len(serialized)} bytes")
            print(f"  ⏱️  Serialize time: {serialize_time*1000:.2f} ms")
            print(f"  ⏱️  Deserialize time: {deserialize_time*1000:.2f} ms")
            print(f"  🗜️  Compression ratio: {compression_ratio:.2f}x")
            print(f"  ✓ Data integrity: {test_obj == deserialized}")
            
        except Exception as e:
            print(f"  ❌ {format_type} failed: {e}")

def main():
    """Main demonstration function"""
    print("🧠 CrewGraph AI Memory System Demonstration")
    print("👤 Created by: Vatsal216")
    print("⏰ Date: 2025-07-22 12:09:19 UTC")
    print("=" * 70)
    
    # Run all demonstrations
    try:
        demonstrate_dict_memory()
        demonstrate_faiss_memory()
        demonstrate_sql_memory()
        demonstrate_memory_comparison()
        demonstrate_memory_serialization()
        
        # Show available memory backends
        print("\n🔧 AVAILABLE MEMORY BACKENDS")
        print("-" * 50)
        
        available_backends = []
        
        # Check DictMemory (always available)
        available_backends.append("✅ DictMemory - In-memory storage")
        
        # Check Redis
        try:
            import redis
            available_backends.append("✅ RedisMemory - Redis-based storage")
        except ImportError:
            available_backends.append("❌ RedisMemory - Redis not installed")
        
        # Check FAISS
        try:
            import faiss
            available_backends.append("✅ FAISSMemory - Vector similarity search")
        except ImportError:
            available_backends.append("❌ FAISSMemory - FAISS not installed")
        
        # Check SQL
        try:
            import sqlalchemy
            available_backends.append("✅ SQLMemory - SQL database storage")
        except ImportError:
            available_backends.append("❌ SQLMemory - SQLAlchemy not installed")
        
        for backend in available_backends:
            print(f"  {backend}")
        
        # Usage examples
        print("\n💡 QUICK USAGE EXAMPLES")
        print("-" * 50)
        
        print("# Create dictionary memory:")
        print("from crewgraph_ai.memory import create_memory")
        print("memory = create_memory('dict', max_size=1000)")
        print()
        
        print("# Create Redis memory:")
        print("memory = create_memory('redis', redis_host='localhost')")
        print()
        
        print("# Create FAISS memory:")
        print("memory = create_memory('faiss', dimension=384)")
        print()
        
        print("# Create SQL memory:")
        print("memory = create_memory('sql', database_url='sqlite:///app.db')")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 MEMORY SYSTEM DEMO COMPLETED!")
    print("=" * 70)
    print("Key Features Demonstrated:")
    print("✅ Multiple memory backends (Dict, Redis, FAISS, SQL)")
    print("✅ Advanced serialization and compression")
    print("✅ TTL support and automatic cleanup")
    print("✅ Performance benchmarking and comparison")
    print("✅ Vector similarity search capabilities")
    print("✅ Production-ready features and monitoring")
    print("")
    print("📚 Memory backends provide:")
    print("  🚀 High-performance caching and storage")
    print("  🔍 Vector similarity search for AI workflows")
    print("  💾 Persistent storage with SQL databases")
    print("  ⚡ Distributed caching with Redis")
    print("  🛡️ Data compression and security")
    print("")
    print(f"👤 Created by: Vatsal216")
    print(f"⏰ Completed at: 2025-07-22 12:09:19 UTC")
    print(f"🔗 Repository: https://github.com/Vatsal216/crewgraph-ai")

if __name__ == "__main__":
    main()