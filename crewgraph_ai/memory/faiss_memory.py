"""
FAISS Memory Backend for CrewGraph AI
Vector similarity search and storage using Facebook AI Similarity Search

Author: Vatsal216
Created: 2025-07-22 12:05:13 UTC
"""

import time
import json
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import threading
import uuid

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .base import BaseMemory, MemoryOperation
from ..utils.logging import get_logger
from ..utils.exceptions import MemoryError

logger = get_logger(__name__)


@dataclass
class VectorItem:
    """Vector item with metadata"""
    vector_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    created_at: float
    ttl: Optional[int] = None
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if item has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class FAISSMemory(BaseMemory):
    """
    FAISS-based vector memory backend for CrewGraph AI.
    
    Provides high-performance vector similarity search capabilities with:
    - Multiple index types (Flat, IVF, HNSW, etc.)
    - GPU acceleration support
    - Metadata storage alongside vectors
    - TTL support for vector expiration
    - Batch operations for efficiency
    - Similarity search with configurable metrics
    
    Perfect for:
    - Semantic search in AI workflows
    - Document embeddings storage
    - Feature vector caching
    - Recommendation systems
    - Clustering and classification tasks
    
    Created by: Vatsal216
    Date: 2025-07-22 12:05:13 UTC
    """
    
    def __init__(self, 
                 dimension: int = 384,
                 index_type: str = "Flat",
                 metric_type: str = "L2",
                 use_gpu: bool = False,
                 nlist: int = 100,
                 max_vectors: int = 1000000,
                 enable_metadata: bool = True,
                 enable_ttl: bool = True):
        """
        Initialize FAISS memory backend.
        
        Args:
            dimension: Vector dimension
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW', 'LSH')
            metric_type: Distance metric ('L2', 'IP' for inner product)
            use_gpu: Use GPU acceleration if available
            nlist: Number of clusters for IVF index
            max_vectors: Maximum number of vectors to store
            enable_metadata: Enable metadata storage
            enable_ttl: Enable TTL support
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu or faiss-gpu")
        
        super().__init__()
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.use_gpu = use_gpu
        self.nlist = nlist
        self.max_vectors = max_vectors
        self.enable_metadata = enable_metadata
        self.enable_ttl = enable_ttl
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.gpu_resources = None
        
        # Vector storage and metadata
        self._vectors: Dict[str, VectorItem] = {}
        self._id_to_index: Dict[str, int] = {}  # Maps vector_id to FAISS index position
        self._index_to_id: Dict[int, str] = {}  # Maps FAISS index position to vector_id
        self._next_index = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._search_count = 0
        self._total_search_time = 0.0
        
        logger.info(f"FAISSMemory initialized: dim={dimension}, type={index_type}")
        logger.info(f"User: Vatsal216, Time: 2025-07-22 12:05:13")
    
    def connect(self) -> None:
        """Initialize FAISS index"""
        try:
            with self._lock:
                self._create_index()
                self._connected = True
                
            logger.info(f"FAISS index created successfully: {self.index_type}")
            logger.info(f"Connected by user: Vatsal216 at 2025-07-22 12:05:13")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise MemoryError(
                "FAISS index creation failed",
                backend="FAISS",
                operation="connect",
                original_error=e
            )
    
    def disconnect(self) -> None:
        """Cleanup FAISS resources"""
        with self._lock:
            if self.gpu_resources and self.use_gpu:
                # Cleanup GPU resources
                pass
            
            self.index = None
            self._connected = False
            
        logger.info("FAISS disconnected successfully")
    
    def _create_index(self) -> None:
        """Create FAISS index based on configuration"""
        if self.metric_type == "L2":
            metric = faiss.METRIC_L2
        elif self.metric_type == "IP":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        
        # Create index based on type
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
        
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric)
        
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, metric)
        
        elif self.index_type == "LSH":
            self.index = faiss.IndexLSH(self.dimension, 2048)
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # GPU acceleration if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                logger.info("FAISS GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"Failed to enable GPU acceleration: {e}")
                self.use_gpu = False
        
        # Train index if needed (for IVF)
        if self.index_type == "IVF":
            # For IVF, we'll train when we have enough vectors
            self._index_trained = False
        else:
            self._index_trained = True
        
        logger.info(f"FAISS index created: {self.index_type}, dimension: {self.dimension}")
    
    def save(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Save vector to FAISS index"""
        def _save():
            # Extract or generate vector from value
            if isinstance(value, dict) and 'vector' in value:
                vector = np.array(value['vector'], dtype=np.float32)
                metadata = value.get('metadata', {})
            elif isinstance(value, (list, tuple, np.ndarray)):
                vector = np.array(value, dtype=np.float32)
                metadata = {}
            else:
                raise ValueError("Value must contain 'vector' field or be a vector array")
            
            # Validate vector dimension
            if vector.shape[0] != self.dimension:
                raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match index dimension {self.dimension}")
            
            vector = vector.reshape(1, -1)  # FAISS expects 2D arrays
            
            with self._lock:
                # Train index if needed and we have enough vectors
                if not self._index_trained and self._next_index >= self.nlist:
                    self._train_index()
                
                # Check capacity
                if self._next_index >= self.max_vectors:
                    self._evict_oldest()
                
                # Add vector to index
                self.index.add(vector)
                
                # Store metadata
                vector_item = VectorItem(
                    vector_id=key,
                    vector=vector.flatten(),
                    metadata=metadata,
                    created_at=time.time(),
                    ttl=ttl
                )
                
                self._vectors[key] = vector_item
                self._id_to_index[key] = self._next_index
                self._index_to_id[self._next_index] = key
                self._next_index += 1
                
            logger.debug(f"Saved vector '{key}' to FAISS index")
            return True
        
        return self._execute_with_metrics(MemoryOperation.SAVE, _save)
    
    def load(self, key: str) -> Any:
        """Load vector from FAISS index"""
        def _load():
            with self._lock:
                if key not in self._vectors:
                    logger.debug(f"Vector '{key}' not found in FAISS")
                    return None
                
                vector_item = self._vectors[key]
                
                # Check TTL
                if self.enable_ttl and vector_item.is_expired():
                    self._remove_vector(key)
                    logger.debug(f"Vector '{key}' expired and removed")
                    return None
                
                # Update access count
                vector_item.access_count += 1
                
                # Return vector with metadata
                result = {
                    'vector': vector_item.vector.tolist(),
                    'metadata': vector_item.metadata,
                    'created_at': vector_item.created_at,
                    'access_count': vector_item.access_count
                }
                
                logger.debug(f"Loaded vector '{key}' from FAISS")
                return result
        
        return self._execute_with_metrics(MemoryOperation.LOAD, _load)
    
    def delete(self, key: str) -> bool:
        """Delete vector from FAISS index"""
        def _delete():
            with self._lock:
                if key not in self._vectors:
                    return False
                
                self._remove_vector(key)
                logger.debug(f"Deleted vector '{key}' from FAISS")
                return True
        
        return self._execute_with_metrics(MemoryOperation.DELETE, _delete)
    
    def exists(self, key: str) -> bool:
        """Check if vector exists in FAISS index"""
        def _exists():
            with self._lock:
                if key not in self._vectors:
                    return False
                
                # Check TTL
                if self.enable_ttl and self._vectors[key].is_expired():
                    self._remove_vector(key)
                    return False
                
                return True
        
        return self._execute_with_metrics(MemoryOperation.EXISTS, _exists)
    
    def clear(self) -> bool:
        """Clear all vectors from FAISS index"""
        def _clear():
            with self._lock:
                # Reset index
                self.index.reset()
                
                # Clear all data structures
                self._vectors.clear()
                self._id_to_index.clear()
                self._index_to_id.clear()
                self._next_index = 0
                self._index_trained = True if self.index_type != "IVF" else False
                
            logger.info("FAISS index cleared")
            return True
        
        return self._execute_with_metrics(MemoryOperation.CLEAR, _clear)
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all vector keys"""
        def _list_keys():
            with self._lock:
                # Clean up expired vectors first
                self._cleanup_expired()
                
                keys = list(self._vectors.keys())
                
                # Apply pattern filter if provided
                if pattern:
                    import fnmatch
                    keys = [key for key in keys if fnmatch.fnmatch(key, pattern)]
                
                return sorted(keys)
        
        return self._execute_with_metrics(MemoryOperation.LIST_KEYS, _list_keys)
    
    def get_size(self) -> int:
        """Get total size of stored vectors in bytes"""
        def _get_size():
            with self._lock:
                total_size = 0
                
                # Vector data size
                total_size += self._next_index * self.dimension * 4  # float32 = 4 bytes
                
                # Metadata size (rough estimate)
                for vector_item in self._vectors.values():
                    try:
                        metadata_size = len(pickle.dumps(vector_item.metadata))
                        total_size += metadata_size
                    except Exception:
                        total_size += 1024  # Rough estimate
                
                return total_size
        
        return self._execute_with_metrics(MemoryOperation.GET_SIZE, _get_size)
    
    def search(self, 
               query_vector: Union[List[float], np.ndarray], 
               k: int = 10,
               include_metadata: bool = True,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            include_metadata: Include metadata in results
            filter_metadata: Filter results by metadata
            
        Returns:
            List of similar vectors with distances and metadata
        """
        def _search():
            start_time = time.time()
            
            # Prepare query vector
            query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            
            if query.shape[1] != self.dimension:
                raise ValueError(f"Query vector dimension {query.shape[1]} doesn't match index dimension {self.dimension}")
            
            with self._lock:
                # Clean up expired vectors
                self._cleanup_expired()
                
                if self.index.ntotal == 0:
                    return []
                
                # Perform search
                distances, indices = self.index.search(query, min(k, self.index.ntotal))
                
                # Prepare results
                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx == -1:  # FAISS returns -1 for invalid indices
                        continue
                    
                    # Get vector ID
                    vector_id = self._index_to_id.get(idx)
                    if not vector_id or vector_id not in self._vectors:
                        continue
                    
                    vector_item = self._vectors[vector_id]
                    
                    # Apply metadata filter if provided
                    if filter_metadata:
                        match = True
                        for key, value in filter_metadata.items():
                            if key not in vector_item.metadata or vector_item.metadata[key] != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    result = {
                        'id': vector_id,
                        'distance': float(distance),
                        'score': 1.0 / (1.0 + distance) if distance >= 0 else 1.0  # Convert distance to similarity score
                    }
                    
                    if include_metadata:
                        result['metadata'] = vector_item.metadata
                        result['vector'] = vector_item.vector.tolist()
                    
                    results.append(result)
                
                # Update search statistics
                search_time = time.time() - start_time
                self._search_count += 1
                self._total_search_time += search_time
                
                # Sort by distance (ascending)
                results.sort(key=lambda x: x['distance'])
                
                logger.debug(f"FAISS search completed: {len(results)} results in {search_time:.3f}s")
                return results
        
        return self._execute_with_metrics(MemoryOperation.SEARCH, _search)
    
    def _train_index(self) -> None:
        """Train IVF index with current vectors"""
        if self.index_type != "IVF" or self._index_trained:
            return
        
        # Collect training vectors
        training_vectors = []
        for vector_item in self._vectors.values():
            training_vectors.append(vector_item.vector)
        
        if len(training_vectors) >= self.nlist:
            training_data = np.vstack(training_vectors)
            self.index.train(training_data)
            self._index_trained = True
            logger.info(f"IVF index trained with {len(training_vectors)} vectors")
    
    def _remove_vector(self, key: str) -> None:
        """Remove vector from all data structures"""
        if key in self._vectors:
            # Note: FAISS doesn't support individual vector removal
            # In production, you'd need to rebuild the index periodically
            # For now, we just remove from our tracking structures
            
            if key in self._id_to_index:
                idx = self._id_to_index[key]
                del self._id_to_index[key]
                if idx in self._index_to_id:
                    del self._index_to_id[idx]
            
            del self._vectors[key]
    
    def _evict_oldest(self) -> None:
        """Evict oldest vector to make space"""
        if not self._vectors:
            return
        
        # Find oldest vector
        oldest_key = min(self._vectors.keys(), key=lambda k: self._vectors[k].created_at)
        self._remove_vector(oldest_key)
        logger.debug(f"Evicted oldest vector: {oldest_key}")
    
    def _cleanup_expired(self) -> None:
        """Clean up expired vectors"""
        if not self.enable_ttl:
            return
        
        expired_keys = []
        for key, vector_item in self._vectors.items():
            if vector_item.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_vector(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired vectors")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get FAISS index information"""
        with self._lock:
            info = {
                "index_type": self.index_type,
                "dimension": self.dimension,
                "metric_type": self.metric_type,
                "total_vectors": self.index.ntotal if self.index else 0,
                "max_vectors": self.max_vectors,
                "is_trained": getattr(self, '_index_trained', True),
                "use_gpu": self.use_gpu,
                "memory_usage_bytes": self.get_size(),
                "search_count": self._search_count,
                "average_search_time": self._total_search_time / self._search_count if self._search_count > 0 else 0,
                "created_by": "Vatsal216",
                "created_at": "2025-07-22 12:05:13"
            }
            
            if self.index:
                info["faiss_description"] = self.index.__class__.__name__
            
            return info
    
    def rebuild_index(self) -> bool:
        """Rebuild FAISS index (useful for cleanup after deletions)"""
        try:
            with self._lock:
                # Collect all current vectors
                vectors = []
                vector_items = []
                
                for key, vector_item in self._vectors.items():
                    if not (self.enable_ttl and vector_item.is_expired()):
                        vectors.append(vector_item.vector)
                        vector_items.append((key, vector_item))
                
                if not vectors:
                    self.clear()
                    return True
                
                # Recreate index
                self._create_index()
                
                # Re-add all vectors
                vector_matrix = np.vstack(vectors)
                self.index.add(vector_matrix)
                
                # Rebuild mappings
                self._vectors.clear()
                self._id_to_index.clear()
                self._index_to_id.clear()
                
                for i, (key, vector_item) in enumerate(vector_items):
                    self._vectors[key] = vector_item
                    self._id_to_index[key] = i
                    self._index_to_id[i] = key
                
                self._next_index = len(vector_items)
                
            logger.info(f"FAISS index rebuilt with {len(vector_items)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
            return False
    
    def save_index(self, filepath: str) -> bool:
        """Save FAISS index to file"""
        try:
            with self._lock:
                if self.use_gpu:
                    # Move to CPU for saving
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, filepath)
                else:
                    faiss.write_index(self.index, filepath)
                
                # Save metadata separately
                metadata_file = f"{filepath}.metadata"
                with open(metadata_file, 'wb') as f:
                    pickle.dump({
                        'vectors': self._vectors,
                        'id_to_index': self._id_to_index,
                        'index_to_id': self._index_to_id,
                        'next_index': self._next_index,
                        'config': {
                            'dimension': self.dimension,
                            'index_type': self.index_type,
                            'metric_type': self.metric_type
                        }
                    }, f)
                
            logger.info(f"FAISS index saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """Load FAISS index from file"""
        try:
            with self._lock:
                # Load index
                self.index = faiss.read_index(filepath)
                
                # Move to GPU if requested
                if self.use_gpu and faiss.get_num_gpus() > 0:
                    self.gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                
                # Load metadata
                metadata_file = f"{filepath}.metadata"
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self._vectors = data['vectors']
                    self._id_to_index = data['id_to_index']
                    self._index_to_id = data['index_to_id']
                    self._next_index = data['next_index']
                
                self._index_trained = True
                self._connected = True
                
            logger.info(f"FAISS index loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False