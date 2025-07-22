"""
CrewGraph AI Memory Backends Test Suite
Comprehensive testing for all memory backend implementations

Author: Vatsal216
Created: 2025-07-22 12:13:00 UTC
"""

import os
import sys
import unittest
import tempfile
import shutil
from typing import Dict, Any, List
import time

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crewgraph_ai.memory import (
    DictMemory, MemoryConfig, MemoryType, 
    MemoryUtils, create_memory
)
from crewgraph_ai.memory.base import BaseMemory, MemoryOperation


class BaseMemoryTest(unittest.TestCase):
    """Base test class for memory backends"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data = {
            "string_key": "Hello CrewGraph AI by Vatsal216!",
            "number_key": 42,
            "list_key": [1, 2, 3, 4, 5],
            "dict_key": {
                "name": "CrewGraph AI",
                "version": "1.0.0",
                "created_by": "Vatsal216",
                "created_at": "2025-07-22 12:13:00"
            },
            "boolean_key": True,
            "float_key": 3.14159
        }
        
        self.memory: BaseMemory = None
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        if self.memory and hasattr(self.memory, '_connected') and self.memory._connected:
            try:
                self.memory.clear()
                self.memory.disconnect()
            except Exception as e:
                print(f"Cleanup warning: {e}")
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_operations(self):
        """Test basic save/load/delete operations"""
        self.assertIsNotNone(self.memory, "Memory backend not initialized")
        
        # Test save operations
        for key, value in self.test_data.items():
            with self.subTest(key=key):
                result = self.memory.save(key, value)
                self.assertTrue(result, f"Failed to save key: {key}")
        
        # Test load operations
        for key, expected_value in self.test_data.items():
            with self.subTest(key=key):
                loaded_value = self.memory.load(key)
                self.assertEqual(loaded_value, expected_value, f"Loaded value mismatch for key: {key}")
        
        # Test exists operations
        for key in self.test_data.keys():
            with self.subTest(key=key):
                exists = self.memory.exists(key)
                self.assertTrue(exists, f"Key should exist: {key}")
        
        # Test delete operations
        for key in self.test_data.keys():
            with self.subTest(key=key):
                result = self.memory.delete(key)
                self.assertTrue(result, f"Failed to delete key: {key}")
                
                # Verify deletion
                exists = self.memory.exists(key)
                self.assertFalse(exists, f"Key should not exist after deletion: {key}")
    
    def test_ttl_functionality(self):
        """Test TTL (Time To Live) functionality"""
        if not hasattr(self.memory, 'save'):
            self.skipTest("Memory backend not available")
        
        key = "ttl_test_key"
        value = "ttl_test_value"
        
        # Save with short TTL
        result = self.memory.save(key, value, ttl=1)  # 1 second TTL
        self.assertTrue(result, "Failed to save with TTL")
        
        # Immediately load (should exist)
        loaded_value = self.memory.load(key)
        self.assertEqual(loaded_value, value, "Value should be available immediately")
        
        # Wait for TTL to expire
        time.sleep(2)
        
        # Try to load after expiration (should be None or not exist)
        expired_value = self.memory.load(key)
        self.assertIsNone(expired_value, "Value should be expired")
    
    def test_batch_operations(self):
        """Test batch save/load operations"""
        if not hasattr(self.memory, 'batch_save'):
            self.skipTest("Batch operations not supported")
        
        # Test batch save
        save_results = self.memory.batch_save(self.test_data)
        self.assertIsInstance(save_results, dict, "Batch save should return dict")
        
        for key, result in save_results.items():
            self.assertTrue(result, f"Batch save failed for key: {key}")
        
        # Test batch load
        keys = list(self.test_data.keys())
        load_results = self.memory.batch_load(keys)
        self.assertIsInstance(load_results, dict, "Batch load should return dict")
        
        for key, expected_value in self.test_data.items():
            loaded_value = load_results.get(key)
            self.assertEqual(loaded_value, expected_value, f"Batch load mismatch for key: {key}")
    
    def test_list_keys(self):
        """Test listing keys functionality"""
        # Save test data
        for key, value in self.test_data.items():
            self.memory.save(key, value)
        
        # List all keys
        keys = self.memory.list_keys()
        self.assertIsInstance(keys, list, "list_keys should return a list")
        
        # Check that all saved keys are present
        for key in self.test_data.keys():
            self.assertIn(key, keys, f"Key should be in list: {key}")
    
    def test_clear_functionality(self):
        """Test clear all data functionality"""
        # Save test data
        for key, value in self.test_data.items():
            self.memory.save(key, value)
        
        # Verify data exists
        keys = self.memory.list_keys()
        self.assertGreater(len(keys), 0, "Should have data before clear")
        
        # Clear all data
        result = self.memory.clear()
        self.assertTrue(result, "Clear operation should succeed")
        
        # Verify data is cleared
        keys_after_clear = self.memory.list_keys()
        self.assertEqual(len(keys_after_clear), 0, "Should have no data after clear")
    
    def test_health_check(self):
        """Test health check functionality"""
        health = self.memory.get_health()
        self.assertIsInstance(health, dict, "Health check should return dict")
        self.assertIn('status', health, "Health should include status")
        self.assertIn('backend_type', health, "Health should include backend_type")
        
        # Status should be healthy for working backend
        self.assertEqual(health['status'], 'healthy', "Backend should be healthy")
    
    def test_statistics(self):
        """Test statistics functionality"""
        # Save some data first
        for key, value in list(self.test_data.items())[:3]:  # Save first 3 items
            self.memory.save(key, value)
            self.memory.load(key)  # Load to generate stats
        
        stats = self.memory.get_stats()
        self.assertIsInstance(stats, object, "Stats should return an object")
        
        # Check basic stats properties
        self.assertTrue(hasattr(stats, 'total_operations'), "Should have total_operations")
        self.assertTrue(hasattr(stats, 'successful_operations'), "Should have successful_operations")
        self.assertGreater(stats.total_operations, 0, "Should have recorded operations")


class TestDictMemory(BaseMemoryTest):
    """Test DictMemory backend"""
    
    def setUp(self):
        super().setUp()
        self.memory = DictMemory(
            max_size=1000,
            enable_compression=True,
            compression_threshold=100
        )
        self.memory.connect()
    
    def test_dict_specific_features(self):
        """Test DictMemory specific features"""
        # Test LRU eviction
        small_memory = DictMemory(max_size=2)
        small_memory.connect()
        
        # Fill beyond capacity
        small_memory.save("key1", "value1")
        small_memory.save("key2", "value2")
        small_memory.save("key3", "value3")  # Should evict key1
        
        # Check eviction
        self.assertFalse(small_memory.exists("key1"), "key1 should be evicted")
        self.assertTrue(small_memory.exists("key2"), "key2 should exist")
        self.assertTrue(small_memory.exists("key3"), "key3 should exist")
        
        small_memory.disconnect()
    
    def test_compression_functionality(self):
        """Test compression functionality"""
        large_data = "x" * 2000  # Large enough to trigger compression
        key = "compression_test"
        
        result = self.memory.save(key, large_data)
        self.assertTrue(result, "Should save large data")
        
        loaded_data = self.memory.load(key)
        self.assertEqual(loaded_data, large_data, "Compressed data should load correctly")
    
    def test_cache_statistics(self):
        """Test cache-specific statistics"""
        # Generate some cache activity
        for i in range(10):
            self.memory.save(f"key_{i}", f"value_{i}")
            self.memory.load(f"key_{i}")
        
        # Try to load non-existent key (miss)
        self.memory.load("non_existent_key")
        
        stats = self.memory.get_cache_stats()
        self.assertIsInstance(stats, dict, "Cache stats should be dict")
        self.assertIn('hit_count', stats, "Should have hit_count")
        self.assertIn('miss_count', stats, "Should have miss_count")
        self.assertIn('hit_rate', stats, "Should have hit_rate")
        
        self.assertGreater(stats['hit_count'], 0, "Should have cache hits")
        self.assertGreater(stats['miss_count'], 0, "Should have cache misses")


class TestMemoryFactory(unittest.TestCase):
    """Test memory factory functions"""
    
    def test_create_memory_dict(self):
        """Test creating dict memory via factory"""
        memory = create_memory("dict", max_size=100)
        self.assertIsInstance(memory, DictMemory, "Should create DictMemory")
        self.assertTrue(memory._connected, "Should be connected")
        memory.disconnect()
    
    def test_create_memory_with_config(self):
        """Test creating memory with configuration"""
        config = MemoryConfig(
            memory_type=MemoryType.DICT,
            max_size=500,
            compression=True
        )
        
        memory = MemoryUtils.create_memory_backend(config)
        self.assertIsInstance(memory, DictMemory, "Should create DictMemory from config")
        memory.connect()
        memory.disconnect()
    
    def test_invalid_memory_type(self):
        """Test creating memory with invalid type"""
        with self.assertRaises(ValueError):
            create_memory("invalid_type")


class TestMemoryBenchmark(unittest.TestCase):
    """Test memory benchmarking functionality"""
    
    def test_benchmark_dict_memory(self):
        """Test benchmarking DictMemory"""
        memory = DictMemory(max_size=1000)
        memory.connect()
        
        try:
            results = MemoryUtils.benchmark_memory_backend(
                memory, 
                test_data_size=50,  # Small size for quick test
                value_size=100
            )
            
            self.assertIsInstance(results, dict, "Benchmark should return dict")
            self.assertIn('save', results, "Should have save benchmark")
            self.assertIn('load', results, "Should have load benchmark")
            self.assertIn('delete', results, "Should have delete benchmark")
            
            # Check save benchmark
            save_result = results['save']
            self.assertGreater(save_result.operations_per_second, 0, "Should have positive ops/sec")
            self.assertGreaterEqual(save_result.success_rate, 0.9, "Should have high success rate")
            
        finally:
            memory.disconnect()


def run_specific_backend_test(backend_type: str) -> bool:
    """Run tests for a specific backend type"""
    print(f"\n🧪 Testing {backend_type.upper()} Memory Backend")
    print(f"👤 User: Vatsal216")
    print(f"⏰ Time: 2025-07-22 12:13:00 UTC")
    print("-" * 50)
    
    try:
        if backend_type == "dict":
            suite = unittest.TestLoader().loadTestsFromTestCase(TestDictMemory)
        elif backend_type == "redis":
            # Import and test Redis if available
            try:
                from crewgraph_ai.memory.redis_memory import RedisMemory
                # Create Redis-specific test class dynamically
                class TestRedisMemory(BaseMemoryTest):
                    def setUp(self):
                        super().setUp()
                        self.memory = RedisMemory({
                            'redis_host': 'localhost',
                            'redis_port': 6379,
                            'redis_database': 15  # Use test database
                        })
                        try:
                            self.memory.connect()
                        except Exception as e:
                            self.skipTest(f"Redis not available: {e}")
                
                suite = unittest.TestLoader().loadTestsFromTestCase(TestRedisMemory)
            except ImportError:
                print("❌ Redis backend not available")
                return False
                
        elif backend_type == "faiss":
            # Import and test FAISS if available
            try:
                from crewgraph_ai.memory.faiss_memory import FAISSMemory
                import numpy as np
                
                class TestFAISSMemory(BaseMemoryTest):
                    def setUp(self):
                        super().setUp()
                        # Override test data for vectors
                        self.test_data = {}
                        for i in range(5):
                            vector = np.random.random(128).astype(np.float32)
                            self.test_data[f"vector_{i}"] = {
                                "vector": vector.tolist(),
                                "metadata": {"id": i, "created_by": "Vatsal216"}
                            }
                        
                        self.memory = FAISSMemory(
                            dimension=128,
                            index_type="Flat",
                            max_vectors=1000
                        )
                        try:
                            self.memory.connect()
                        except Exception as e:
                            self.skipTest(f"FAISS not available: {e}")
                    
                    def test_vector_search(self):
                        """Test vector similarity search"""
                        # Save some vectors
                        for key, data in self.test_data.items():
                            self.memory.save(key, data)
                        
                        # Perform search
                        query_vector = np.random.random(128).astype(np.float32)
                        results = self.memory.search(query_vector, k=3)
                        
                        self.assertIsInstance(results, list, "Search should return list")
                        self.assertLessEqual(len(results), 3, "Should return at most k results")
                
                suite = unittest.TestLoader().loadTestsFromTestCase(TestFAISSMemory)
            except ImportError:
                print("❌ FAISS backend not available")
                return False
                
        elif backend_type == "sql":
            # Import and test SQL if available
            try:
                from crewgraph_ai.memory.sql_memory import SQLMemory
                
                class TestSQLMemory(BaseMemoryTest):
                    def setUp(self):
                        super().setUp()
                        # Use temporary SQLite database
                        db_path = os.path.join(self.temp_dir, "test.db")
                        self.memory = SQLMemory(
                            database_url=f"sqlite:///{db_path}",
                            enable_ttl_cleanup=False  # Disable for tests
                        )
                        try:
                            self.memory.connect()
                        except Exception as e:
                            self.skipTest(f"SQL backend not available: {e}")
                
                suite = unittest.TestLoader().loadTestsFromTestCase(TestSQLMemory)
            except ImportError:
                print("❌ SQL backend not available")
                return False
        else:
            print(f"❌ Unknown backend type: {backend_type}")
            return False
        
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        success = result.wasSuccessful()
        print(f"\n{'✅' if success else '❌'} {backend_type.upper()} tests {'PASSED' if success else 'FAILED'}")
        
        if not success:
            print(f"Failures: {len(result.failures)}")
            print(f"Errors: {len(result.errors)}")
        
        return success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner"""
    print("🧠 CrewGraph AI Memory Backends Test Suite")
    print("👤 Created by: Vatsal216")
    print("⏰ Date: 2025-07-22 12:13:00 UTC")
    print("🔗 Repository: https://github.com/Vatsal216/crewgraph-ai")
    print("=" * 70)
    
    # Test configuration
    backends_to_test = ["dict", "redis", "faiss", "sql"]
    results = {}
    
    # Run factory and utility tests first
    print("\n🔧 TESTING MEMORY UTILITIES")
    print("=" * 40)
    
    utility_suite = unittest.TestSuite()
    utility_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMemoryFactory))
    utility_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMemoryBenchmark))
    
    utility_runner = unittest.TextTestRunner(verbosity=1)
    utility_result = utility_runner.run(utility_suite)
    
    print(f"{'✅' if utility_result.wasSuccessful() else '❌'} Utility tests {'PASSED' if utility_result.wasSuccessful() else 'FAILED'}")
    
    # Test each backend
    print(f"\n🧪 TESTING MEMORY BACKENDS")
    print("=" * 40)
    
    for backend in backends_to_test:
        results[backend] = run_specific_backend_test(backend)
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 40)
    
    utility_status = "✅ PASSED" if utility_result.wasSuccessful() else "❌ FAILED"
    print(f"Utilities: {utility_status}")
    
    for backend, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{backend.upper()}: {status}")
    
    # Overall results
    passed_backends = sum(results.values())
    total_backends = len([r for r in results.values() if r is not False])  # Exclude unavailable
    utility_passed = utility_result.wasSuccessful()
    
    print(f"\nOverall Results:")
    print(f"  Utility Tests: {'✅' if utility_passed else '❌'}")
    print(f"  Backend Tests: {passed_backends}/{total_backends} passed")
    
    if utility_passed and passed_backends == total_backends and total_backends > 0:
        print("\n🎉 All available memory backends are working correctly!")
    else:
        print("\n⚠️ Some tests failed or backends are unavailable")
    
    print(f"\n📝 Notes:")
    print(f"  - Install 'redis' package for Redis backend testing")
    print(f"  - Install 'faiss-cpu' or 'faiss-gpu' for FAISS backend testing")
    print(f"  - Install 'sqlalchemy' for SQL backend testing")
    print(f"  - All backends are optional except DictMemory")
    
    print(f"\n👤 Testing completed by: Vatsal216")
    print(f"⏰ Completed at: 2025-07-22 12:13:00 UTC")
    
    # Return exit code
    return 0 if (utility_passed and passed_backends > 0) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)