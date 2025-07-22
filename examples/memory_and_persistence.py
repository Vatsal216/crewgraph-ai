"""
CrewGraph AI - Memory and Persistence Example
Demonstrates advanced memory backends, state persistence, and workflow resume
"""

import time
import asyncio
from crewai import Agent, Tool
from crewgraph_ai import CrewGraph, CrewGraphConfig
from crewgraph_ai.memory import DictMemory, RedisMemory, FAISSMemory
from crewgraph_ai.core.state import SharedState, StateManager
from crewgraph_ai.utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def research_task(topic: str, depth: str = "basic") -> dict:
    """Simulate research task with memory usage"""
    logger.info(f"Researching topic: {topic} with depth: {depth}")
    
    # Simulate research work
    time.sleep(2)
    
    return {
        "topic": topic,
        "findings": [
            f"Key insight 1 about {topic}",
            f"Key insight 2 about {topic}",
            f"Key insight 3 about {topic}"
        ],
        "sources": [
            f"https://example.com/{topic}-1",
            f"https://example.com/{topic}-2"
        ],
        "confidence": 0.85,
        "timestamp": time.time()
    }

def summarize_research(research_data: dict) -> str:
    """Summarize research findings"""
    logger.info("Summarizing research findings")
    
    topic = research_data.get("topic", "Unknown")
    findings = research_data.get("findings", [])
    confidence = research_data.get("confidence", 0.0)
    
    summary = f"""
# Research Summary: {topic}

## Key Findings:
{chr(10).join(f"- {finding}" for finding in findings)}

## Confidence Level: {confidence:.2%}
## Sources: {len(research_data.get('sources', []))} references
"""
    
    return summary

class MemoryPersistenceExample:
    """Demonstrate memory and persistence features"""
    
    def __init__(self):
        """Initialize memory example"""
        self.dict_memory = DictMemory()
        
        # Redis memory (requires Redis server)
        try:
            self.redis_memory = RedisMemory()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_memory = None
        
        # FAISS memory for vector storage
        try:
            self.faiss_memory = FAISSMemory(dimension=384)
        except Exception as e:
            logger.warning(f"FAISS not available: {e}")
            self.faiss_memory = None
    
    def demonstrate_dict_memory(self):
        """Demonstrate in-memory storage"""
        logger.info("Demonstrating DictMemory capabilities")
        
        config = CrewGraphConfig(memory_backend=self.dict_memory)
        workflow = CrewGraph("memory_demo", config)
        
        # Create agent with memory
        researcher = Agent(
            role='Memory-Enabled Researcher',
            goal='Conduct research and store findings in memory',
            backstory='Researcher who learns from previous investigations',
            tools=[Tool(name="research", func=research_task, description="Research topics")],
            verbose=True
        )
        
        workflow.add_agent(researcher, name="researcher")
        
        # Add task that uses memory
        research_task_wrapper = workflow.add_task(
            name="research_with_memory",
            description="Research topic and store in memory",
            agent="researcher",
            tools=["research"]
        )
        
        # Execute workflow multiple times to show memory persistence
        topics = ["artificial_intelligence", "machine_learning", "deep_learning"]
        
        for topic in topics:
            logger.info(f"Researching: {topic}")
            
            result = workflow.execute({
                "topic": topic,
                "depth": "detailed"
            })
            
            # Store results in memory
            memory_key = f"research:{topic}"
            workflow.get_state().set(memory_key, result)
            
            logger.info(f"Stored research for {topic} in memory")
        
        # Retrieve and display memory contents
        logger.info("Memory contents:")
        for topic in topics:
            memory_key = f"research:{topic}"
            stored_result = workflow.get_state().get(memory_key)
            logger.info(f"{topic}: {stored_result is not None}")
        
        # Demonstrate memory statistics
        if hasattr(self.dict_memory, 'get_memory_usage'):
            stats = self.dict_memory.get_memory_usage()
            logger.info(f"Memory usage stats: {stats}")
        
        return workflow
    
    def demonstrate_redis_memory(self):
        """Demonstrate Redis-based memory"""
        if not self.redis_memory:
            logger.warning("Redis memory not available, skipping demonstration")
            return None
        
        logger.info("Demonstrating Redis memory capabilities")
        
        config = CrewGraphConfig(memory_backend=self.redis_memory)
        workflow = CrewGraph("redis_demo", config)
        
        # Test Redis memory operations
        test_data = {
            "workflow_id": "redis_test_001",
            "execution_time": time.time(),
            "status": "running",
            "results": {"step1": "completed", "step2": "in_progress"}
        }
        
        # Store data with TTL
        self.redis_memory.save("workflow:state", test_data, ttl=300)  # 5 minutes TTL
        
        # Retrieve data
        retrieved_data = self.redis_memory.load("workflow:state")
        logger.info(f"Retrieved from Redis: {retrieved_data}")
        
        # Demonstrate persistence across workflow restarts
        workflow.save_state("redis_workflow_state.json")
        logger.info("Workflow state saved to Redis")
        
        return workflow
    
    def demonstrate_faiss_memory(self):
        """Demonstrate FAISS vector memory"""
        if not self.faiss_memory:
            logger.warning("FAISS memory not available, skipping demonstration")
            return None
        
        logger.info("Demonstrating FAISS vector memory capabilities")
        
        # Store vectors (embeddings) for semantic search
        research_topics = [
            "artificial intelligence applications",
            "machine learning algorithms", 
            "natural language processing",
            "computer vision techniques",
            "robotics and automation"
        ]
        
        # Simulate embeddings (in real use, these would come from an embedding model)
        import numpy as np
        
        for i, topic in enumerate(research_topics):
            # Create mock embedding
            embedding = np.random.rand(384).astype(np.float32)
            
            # Store in FAISS memory
            vector_id = f"topic_{i}"
            self.faiss_memory.save(vector_id, {
                "embedding": embedding,
                "text": topic,
                "metadata": {"category": "research", "index": i}
            })
        
        # Demonstrate similarity search
        query_embedding = np.random.rand(384).astype(np.float32)
        # In real implementation, you would search for similar vectors
        logger.info("FAISS memory populated with research topic embeddings")
        
        return self.faiss_memory
    
    def demonstrate_state_persistence(self):
        """Demonstrate workflow state persistence and resume"""
        logger.info("Demonstrating state persistence and workflow resume")
        
        # Create workflow with state management
        config = CrewGraphConfig(memory_backend=self.dict_memory)
        workflow = CrewGraph("persistence_demo", config)
        
        # Create agents
        researcher = Agent(
            role='Persistent Researcher',
            goal='Conduct multi-step research with state persistence',
            backstory='Researcher who can resume work after interruptions',
            tools=[Tool(name="research", func=research_task, description="Research topics")],
            verbose=True
        )
        
        summarizer = Agent(
            role='Research Summarizer',
            goal='Summarize research findings',
            backstory='Expert at creating concise summaries',
            tools=[Tool(name="summarize", func=summarize_research, description="Summarize research")],
            verbose=True
        )
        
        workflow.add_agent(researcher, name="researcher")
        workflow.add_agent(summarizer, name="summarizer")
        
        # Add tasks with dependencies
        research_tasks = []
        topics = ["quantum_computing", "blockchain_technology", "edge_computing"]
        
        for topic in topics:
            task = workflow.add_task(
                name=f"research_{topic}",
                description=f"Research {topic.replace('_', ' ')}",
                agent="researcher",
                tools=["research"]
            )
            research_tasks.append(f"research_{topic}")
        
        # Summary task depends on all research tasks
        summary_task = workflow.add_task(
            name="create_summary",
            description="Create comprehensive summary of all research",
            agent="summarizer",
            tools=["summarize"],
            dependencies=research_tasks
        )
        
        # Execute workflow with state tracking
        initial_state = {
            "research_depth": "comprehensive",
            "output_format": "detailed_report"
        }
        
        # Simulate workflow execution with interruption
        try:
            logger.info("Starting workflow execution...")
            
            # Execute first part
            result = workflow.execute(initial_state)
            
            # Save state during execution
            state_file = "workflow_checkpoint.json"
            workflow.save_state(state_file)
            logger.info(f"Workflow state saved to {state_file}")
            
            # Simulate workflow resume (in real scenario, this would be a new process)
            logger.info("Simulating workflow resume...")
            
            # Create new workflow instance
            resumed_workflow = CrewGraph("resumed_workflow", config)
            
            # Load previous state
            resumed_workflow.load_state(state_file)
            logger.info("Workflow state loaded successfully")
            
            # Continue execution
            final_result = resumed_workflow.execute(initial_state)
            
            logger.info("Workflow resumed and completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return None
    
    def demonstrate_cross_workflow_memory(self):
        """Demonstrate memory sharing across different workflows"""
        logger.info("Demonstrating cross-workflow memory sharing")
        
        # Shared memory backend
        shared_memory = self.dict_memory
        
        # Workflow 1: Data Collection
        config1 = CrewGraphConfig(memory_backend=shared_memory)
        collection_workflow = CrewGraph("data_collection", config1)
        
        collector = Agent(
            role='Data Collector',
            goal='Collect and store data for other workflows',
            backstory='Specialist in data gathering and storage',
            tools=[Tool(name="research", func=research_task, description="Collect data")],
            verbose=True
        )
        
        collection_workflow.add_agent(collector, name="collector")
        collection_workflow.add_task(
            name="collect_data",
            description="Collect research data",
            agent="collector"
        )
        
        # Execute data collection
        collection_result = collection_workflow.execute({
            "topic": "shared_research_topic",
            "depth": "comprehensive"
        })
        
        # Store in shared memory
        shared_memory.save("shared:research_data", collection_result)
        
        # Workflow 2: Data Analysis
        config2 = CrewGraphConfig(memory_backend=shared_memory)
        analysis_workflow = CrewGraph("data_analysis", config2)
        
        analyzer = Agent(
            role='Data Analyzer',
            goal='Analyze data from shared memory',
            backstory='Expert at analyzing pre-collected data',
            tools=[Tool(name="summarize", func=summarize_research, description="Analyze data")],
            verbose=True
        )
        
        analysis_workflow.add_agent(analyzer, name="analyzer")
        analysis_workflow.add_task(
            name="analyze_data",
            description="Analyze shared research data",
            agent="analyzer"
        )
        
        # Execute analysis using shared data
        shared_data = shared_memory.load("shared:research_data")
        
        analysis_result = analysis_workflow.execute({
            "shared_data": shared_data,
            "analysis_type": "comprehensive"
        })
        
        logger.info("Cross-workflow memory sharing completed successfully")
        return {
            "collection_result": collection_result,
            "analysis_result": analysis_result,
            "shared_data": shared_data
        }

async def main():
    """Main execution function"""
    logger.info("Starting Memory and Persistence Examples")
    logger.info(f"Executed by user: Vatsal216 at 2025-07-22 10:19:38")
    
    example = MemoryPersistenceExample()
    
    print("\n" + "="*60)
    print("1. DictMemory Demonstration")
    print("="*60)
    dict_workflow = example.demonstrate_dict_memory()
    
    print("\n" + "="*60)
    print("2. Redis Memory Demonstration")
    print("="*60)
    redis_workflow = example.demonstrate_redis_memory()
    
    print("\n" + "="*60)
    print("3. FAISS Vector Memory Demonstration")
    print("="*60)
    faiss_memory = example.demonstrate_faiss_memory()
    
    print("\n" + "="*60)
    print("4. State Persistence and Resume")
    print("="*60)
    persistence_result = example.demonstrate_state_persistence()
    
    print("\n" + "="*60)
    print("5. Cross-Workflow Memory Sharing")
    print("="*60)
    cross_workflow_result = example.demonstrate_cross_workflow_memory()
    
    logger.info("All memory and persistence examples completed!")
    
    return {
        "dict_memory": dict_workflow is not None,
        "redis_memory": redis_workflow is not None,
        "faiss_memory": faiss_memory is not None,
        "persistence": persistence_result is not None,
        "cross_workflow": cross_workflow_result is not None
    }

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nExample execution summary: {result}")