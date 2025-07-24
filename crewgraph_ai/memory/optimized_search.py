"""High-performance message searching with indexing and optimization."""

import time
import threading
import json
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchIndex:
    """Search index for fast message lookup."""
    word_to_messages: Dict[str, Set[str]] = None  # word -> message_ids
    message_metadata: Dict[str, Dict[str, Any]] = None  # message_id -> metadata
    conversation_index: Dict[str, List[str]] = None  # conversation_id -> message_ids
    last_updated: float = 0
    
    def __post_init__(self):
        if self.word_to_messages is None:
            self.word_to_messages = defaultdict(set)
        if self.message_metadata is None:
            self.message_metadata = {}
        if self.conversation_index is None:
            self.conversation_index = defaultdict(list)

class OptimizedMessageSearch:
    """High-performance message search with indexing."""
    
    def __init__(self, memory_backend):
        self.memory = memory_backend
        self.search_index = SearchIndex()
        self._index_lock = threading.RLock()
        self._index_dirty = True
        self._background_indexer = None
        self._stop_indexing = False
        
        # Start background indexer
        self._start_background_indexer()
    
    def search_messages_optimized(self, query: str, message_type: Optional[type] = None, 
                                limit: int = 10, conversation_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Optimized message search with indexing.
        
        Returns:
            List of message metadata with conversation context
        """
        start_time = time.time()
        
        # Ensure index is up to date
        self._ensure_index_current()
        
        # Parse search query
        search_terms = self._parse_query(query)
        
        # Fast index-based search
        candidate_message_ids = self._search_index(search_terms, conversation_ids)
        
        # Rank and filter results
        ranked_results = self._rank_and_filter_results(
            candidate_message_ids, query, message_type, limit
        )
        
        # Load full message content for top results
        full_results = self._load_full_messages(ranked_results[:limit])
        
        search_time = time.time() - start_time
        logger.info(f"Optimized search completed in {search_time:.3f}s, found {len(full_results)} results")
        
        return full_results
    
    def _parse_query(self, query: str) -> List[str]:
        """Parse search query into terms."""
        # Simple tokenization - can be enhanced with NLP
        import re
        terms = re.findall(r'\b\w+\b', query.lower())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [term for term in terms if term not in stop_words and len(term) > 2]
    
    def _search_index(self, search_terms: List[str], conversation_ids: Optional[List[str]] = None) -> Set[str]:
        """Search index for matching message IDs."""
        with self._index_lock:
            if not search_terms:
                return set()
            
            # Find messages containing ALL search terms (AND logic)
            matching_messages = None
            
            for term in search_terms:
                term_matches = self.search_index.word_to_messages.get(term, set())
                
                if matching_messages is None:
                    matching_messages = term_matches.copy()
                else:
                    matching_messages = matching_messages.intersection(term_matches)
                
                # Early termination if no matches
                if not matching_messages:
                    break
            
            # Filter by conversation IDs if specified
            if conversation_ids and matching_messages:
                conversation_messages = set()
                for conv_id in conversation_ids:
                    conversation_messages.update(self.search_index.conversation_index.get(conv_id, []))
                matching_messages = matching_messages.intersection(conversation_messages)
            
            return matching_messages or set()
    
    def _rank_and_filter_results(self, message_ids: Set[str], original_query: str, 
                               message_type: Optional[type], limit: int) -> List[Tuple[str, float]]:
        """Rank search results by relevance."""
        scored_results = []
        query_lower = original_query.lower()
        
        with self._index_lock:
            for msg_id in message_ids:
                metadata = self.search_index.message_metadata.get(msg_id, {})
                
                # Filter by message type
                if message_type:
                    msg_type = metadata.get('message_type')
                    if msg_type != message_type.__name__:
                        continue
                
                # Calculate relevance score
                content = metadata.get('content', '').lower()
                score = self._calculate_relevance_score(content, query_lower)
                
                scored_results.append((msg_id, score))
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:limit * 2]  # Get extra for final filtering
    
    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """Calculate relevance score for content."""
        if not content or not query:
            return 0.0
        
        score = 0.0
        
        # Exact phrase match (highest score)
        if query in content:
            score += 10.0
        
        # Word frequency scoring
        query_words = query.split()
        content_words = content.split()
        
        for word in query_words:
            word_count = content_words.count(word)
            if word_count > 0:
                # TF-IDF-like scoring
                tf = word_count / len(content_words)
                score += tf * 5.0
        
        # Length penalty for very long content
        if len(content) > 1000:
            score *= 0.9
        
        return score
    
    def _load_full_messages(self, ranked_results: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Load full message content for top results."""
        full_messages = []
        
        # Group message IDs by conversation for efficient loading
        conv_to_messages = defaultdict(list)
        
        with self._index_lock:
            for msg_id, score in ranked_results:
                metadata = self.search_index.message_metadata.get(msg_id, {})
                conv_id = metadata.get('conversation_id')
                if conv_id:
                    conv_to_messages[conv_id].append((msg_id, score, metadata))
        
        # Load conversations in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_conv = {
                executor.submit(self._load_conversation_messages, conv_id, msg_list): conv_id
                for conv_id, msg_list in conv_to_messages.items()
            }
            
            for future in future_to_conv:
                try:
                    conv_messages = future.result(timeout=5.0)
                    full_messages.extend(conv_messages)
                except Exception as e:
                    logger.error(f"Failed to load conversation messages: {e}")
        
        # Sort by relevance score
        full_messages.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return full_messages
    
    def _load_conversation_messages(self, conv_id: str, msg_list: List[Tuple[str, float, Dict]]) -> List[Dict[str, Any]]:
        """Load specific messages from a conversation."""
        try:
            # Load conversation once
            messages = self.memory.load_conversation(conv_id)
            conv_messages = []
            
            # Create message lookup by index
            message_lookup = {f"{conv_id}_{i}": msg for i, msg in enumerate(messages)}
            
            for msg_id, score, metadata in msg_list:
                if msg_id in message_lookup:
                    message = message_lookup[msg_id]
                    conv_messages.append({
                        'message_id': msg_id,
                        'conversation_id': conv_id,
                        'content': message.content,
                        'message_type': message.__class__.__name__,
                        'additional_kwargs': getattr(message, 'additional_kwargs', {}),
                        'relevance_score': score,
                        'timestamp': metadata.get('timestamp', time.time())
                    })
            
            return conv_messages
        
        except Exception as e:
            logger.error(f"Failed to load conversation {conv_id}: {e}")
            return []
    
    def _ensure_index_current(self):
        """Ensure search index is up to date."""
        if self._index_dirty or (time.time() - self.search_index.last_updated) > 300:  # 5 minutes
            self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the search index."""
        logger.info("Rebuilding search index...")
        start_time = time.time()
        
        with self._index_lock:
            # Clear existing index
            self.search_index = SearchIndex()
            
            try:
                # Get all conversation keys efficiently
                all_keys = self.memory.list_keys()
                conversation_keys = [k for k in all_keys if k.startswith("conversation:")]
                
                # Process conversations in batches
                batch_size = 10
                for i in range(0, len(conversation_keys), batch_size):
                    batch = conversation_keys[i:i + batch_size]
                    self._process_conversation_batch(batch)
                
                self.search_index.last_updated = time.time()
                self._index_dirty = False
                
                build_time = time.time() - start_time
                logger.info(f"Search index rebuilt in {build_time:.3f}s, indexed {len(conversation_keys)} conversations")
                
            except Exception as e:
                logger.error(f"Failed to rebuild search index: {e}")
    
    def _process_conversation_batch(self, conversation_keys: List[str]):
        """Process a batch of conversations for indexing."""
        for key in conversation_keys:
            try:
                conv_id = key.split(":", 1)[1]
                messages = self.memory.load_conversation(conv_id)
                
                for i, message in enumerate(messages):
                    msg_id = f"{conv_id}_{i}"
                    
                    # Index message content
                    content = message.content.lower()
                    words = content.split()
                    
                    for word in words:
                        # Clean word
                        word = ''.join(c for c in word if c.isalnum())
                        if len(word) > 2:  # Skip very short words
                            self.search_index.word_to_messages[word].add(msg_id)
                    
                    # Store message metadata
                    self.search_index.message_metadata[msg_id] = {
                        'conversation_id': conv_id,
                        'message_type': message.__class__.__name__,
                        'content': message.content,
                        'timestamp': getattr(message, 'additional_kwargs', {}).get('timestamp', time.time()),
                        'content_length': len(message.content)
                    }
                    
                    # Update conversation index
                    self.search_index.conversation_index[conv_id].append(msg_id)
            
            except Exception as e:
                logger.error(f"Failed to index conversation {key}: {e}")
    
    def _start_background_indexer(self):
        """Start background thread for index maintenance."""
        def indexer_worker():
            while not self._stop_indexing:
                try:
                    if self._index_dirty:
                        self._rebuild_index()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Background indexer error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self._background_indexer = threading.Thread(target=indexer_worker, daemon=True)
        self._background_indexer.start()
    
    def invalidate_index(self, conversation_id: Optional[str] = None):
        """Mark index as dirty for rebuild."""
        if conversation_id:
            # Selective invalidation for specific conversation
            with self._index_lock:
                conv_msg_ids = self.search_index.conversation_index.get(conversation_id, [])
                for msg_id in conv_msg_ids:
                    # Remove from word index
                    for word_set in self.search_index.word_to_messages.values():
                        word_set.discard(msg_id)
                    # Remove metadata
                    self.search_index.message_metadata.pop(msg_id, None)
                # Remove conversation index
                self.search_index.conversation_index.pop(conversation_id, None)
        else:
            # Full invalidation
            self._index_dirty = True
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        with self._index_lock:
            return {
                'indexed_conversations': len(self.search_index.conversation_index),
                'indexed_messages': len(self.search_index.message_metadata),
                'unique_words': len(self.search_index.word_to_messages),
                'last_updated': self.search_index.last_updated,
                'index_age_seconds': time.time() - self.search_index.last_updated
            }
    
    def cleanup(self):
        """Cleanup resources."""
        self._stop_indexing = True
        if self._background_indexer and self._background_indexer.is_alive():
            self._background_indexer.join(timeout=5.0)

# Global optimized search instance
_optimized_search = None

def get_optimized_search(memory_backend) -> OptimizedMessageSearch:
    """Get or create optimized search instance."""
    global _optimized_search
    if _optimized_search is None:
        _optimized_search = OptimizedMessageSearch(memory_backend)
    return _optimized_search