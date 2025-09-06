"""
Production FAISS vector database for oceanographic data with hybrid search capabilities.
Supports multi-modal embeddings, distributed search, and advanced optimization.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path
import json
import pickle
import hashlib
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# FAISS and ML libraries
try:
    import faiss
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
except ImportError as e:
    print(f"Missing FAISS dependencies: {e}")
    print("Install with: pip install faiss-cpu scikit-learn torch")
    raise

from ...core.config import get_settings
from ..embeddings import EmbeddingMetadata

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with metadata."""
    id: str
    score: float
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    source_data: Optional[Dict] = None


@dataclass 
class SearchQuery:
    """Search query configuration."""
    query_embedding: np.ndarray
    k: int = 10
    filter_criteria: Optional[Dict] = None
    search_type: str = 'similarity'  # 'similarity', 'hybrid', 'filtered'
    rerank: bool = True
    include_embeddings: bool = False


@dataclass
class IndexStats:
    """Vector index statistics."""
    total_vectors: int
    dimensions: int
    index_type: str
    memory_usage_mb: float
    last_updated: datetime
    search_metrics: Dict[str, float]


class QueryRouter:
    """Intelligent query routing based on complexity and patterns."""
    
    def __init__(self):
        self.query_patterns = {
            'simple_similarity': {'complexity': 'low', 'method': 'flat_search'},
            'spatial_filtered': {'complexity': 'medium', 'method': 'ivf_search'},
            'complex_hybrid': {'complexity': 'high', 'method': 'hnsw_search'},
            'batch_queries': {'complexity': 'high', 'method': 'parallel_search'}
        }
        
    def route_query(self, query: SearchQuery) -> str:
        """Determine optimal search method for query."""
        
        # Simple similarity search
        if not query.filter_criteria and query.k <= 50:
            return 'flat_search'
        
        # Filtered search with moderate complexity
        if query.filter_criteria and len(query.filter_criteria) <= 3:
            return 'ivf_search' 
        
        # Complex queries
        if query.filter_criteria and len(query.filter_criteria) > 3:
            return 'hnsw_search'
        
        # Large result sets
        if query.k > 100:
            return 'parallel_search'
            
        return 'ivf_search'  # Default


class CacheManager:
    """Advanced caching for search results and index data."""
    
    def __init__(self, max_cache_size: int = 10000):
        self.max_cache_size = max_cache_size
        self.query_cache = {}
        self.embedding_cache = {}
        self.access_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.RLock()
        
    def _generate_cache_key(self, query_embedding: np.ndarray, k: int, 
                          filter_criteria: Optional[Dict] = None) -> str:
        """Generate cache key for query."""
        # Hash query parameters
        query_hash = hashlib.md5(query_embedding.tobytes()).hexdigest()[:16]
        filter_hash = hashlib.md5(str(sorted((filter_criteria or {}).items())).encode()).hexdigest()[:8]
        return f"{query_hash}_{k}_{filter_hash}"
    
    def get_cached_result(self, query: SearchQuery) -> Optional[List[SearchResult]]:
        """Retrieve cached search results."""
        with self._lock:
            cache_key = self._generate_cache_key(query.query_embedding, query.k, query.filter_criteria)
            
            if cache_key in self.query_cache:
                self.access_times[cache_key] = time.time()
                self.cache_hits += 1
                return self.query_cache[cache_key]
            
            self.cache_misses += 1
            return None
    
    def cache_result(self, query: SearchQuery, results: List[SearchResult]):
        """Cache search results."""
        with self._lock:
            cache_key = self._generate_cache_key(query.query_embedding, query.k, query.filter_criteria)
            
            # Evict old entries if cache is full
            if len(self.query_cache) >= self.max_cache_size:
                self._evict_oldest()
            
            self.query_cache[cache_key] = results
            self.access_times[cache_key] = time.time()
    
    def _evict_oldest(self):
        """Evict least recently used cache entries."""
        if not self.access_times:
            return
            
        # Remove 10% of oldest entries
        num_to_evict = max(1, len(self.access_times) // 10)
        oldest_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])[:num_to_evict]
        
        for key in oldest_keys:
            self.query_cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            'hit_rate': hit_rate,
            'total_hits': self.cache_hits,
            'total_misses': self.cache_misses,
            'cached_queries': len(self.query_cache),
            'cache_size': len(self.query_cache)
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self.query_cache.clear()
            self.embedding_cache.clear()
            self.access_times.clear()


class FAISSVectorStore:
    """Production-grade FAISS vector store with advanced optimization."""
    
    def __init__(self, dimensions: int, index_type: str = 'IVF', config: Optional[Dict] = None):
        self.dimensions = dimensions
        self.index_type = index_type
        self.config = config or {}
        self.settings = get_settings()
        
        # Initialize FAISS index
        self.index = None
        self.id_map = {}  # Maps FAISS indices to our IDs
        self.metadata_store = {}  # Stores metadata for each vector
        self.next_index = 0
        
        # Advanced features
        self.query_router = QueryRouter()
        self.cache_manager = CacheManager(max_cache_size=self.config.get('cache_size', 10000))
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'search_times': [],
            'index_builds': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"FAISSVectorStore initialized: {dimensions}D, type: {index_type}")
    
    def _create_index(self, nlist: int = 100, m: int = 8) -> faiss.Index:
        """Create optimized FAISS index based on configuration."""
        
        if self.index_type == 'Flat':
            # Exact search index
            index = faiss.IndexFlatIP(self.dimensions)  # Inner Product for cosine similarity
            logger.info("Created Flat index for exact search")
            
        elif self.index_type == 'IVF':
            # Inverted File index for balanced speed/accuracy
            quantizer = faiss.IndexFlatIP(self.dimensions)
            index = faiss.IndexIVFFlat(quantizer, self.dimensions, nlist)
            logger.info(f"Created IVF index with {nlist} clusters")
            
        elif self.index_type == 'HNSW':
            # Hierarchical Navigable Small World for fast approximate search
            index = faiss.IndexHNSWFlat(self.dimensions, m)
            index.hnsw.efConstruction = 200  # Build-time parameter
            index.hnsw.efSearch = 50  # Search-time parameter
            logger.info(f"Created HNSW index with M={m}")
            
        elif self.index_type == 'PQ':
            # Product Quantization for memory efficiency
            m_pq = min(m, self.dimensions // 4)  # Ensure valid PQ parameters
            index = faiss.IndexPQ(self.dimensions, m_pq, 8)  # 8 bits per sub-quantizer
            logger.info(f"Created PQ index with M={m_pq}")
            
        elif self.index_type == 'IVF_PQ':
            # Combined IVF + PQ for large-scale deployment
            quantizer = faiss.IndexFlatIP(self.dimensions)
            m_pq = min(m, self.dimensions // 4)
            index = faiss.IndexIVFPQ(quantizer, self.dimensions, nlist, m_pq, 8)
            logger.info(f"Created IVF-PQ index with {nlist} clusters, M={m_pq}")
            
        else:
            # Default to IVF
            quantizer = faiss.IndexFlatIP(self.dimensions)
            index = faiss.IndexIVFFlat(quantizer, self.dimensions, nlist)
            logger.info(f"Created default IVF index with {nlist} clusters")
        
        return index
    
    async def initialize_index(self, expected_size: int = 100000):
        """Initialize FAISS index with optimal parameters."""
        
        # Calculate optimal parameters based on expected size
        if expected_size < 10000:
            nlist = min(100, expected_size // 10)
            self.index_type = 'Flat' if expected_size < 1000 else 'IVF'
        elif expected_size < 100000:
            nlist = min(1000, expected_size // 100)
            self.index_type = 'IVF'
        else:
            nlist = min(4000, expected_size // 1000)
            self.index_type = 'IVF_PQ'  # Memory efficient for large scale
        
        # Create index
        self.index = self._create_index(nlist=max(1, nlist))
        
        logger.info(f"FAISS index initialized for {expected_size} expected vectors")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str], 
                   metadata: Optional[List[Dict]] = None) -> bool:
        """Add vectors to the index with metadata."""
        
        if vectors.shape[1] != self.dimensions:
            raise ValueError(f"Vector dimensions {vectors.shape[1]} don't match index {self.dimensions}")
        
        with self._lock:
            try:
                # Normalize vectors for cosine similarity
                normalized_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
                
                # Train index if needed (for IVF-based indices)
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    if len(normalized_vectors) >= 100:  # Need sufficient training data
                        logger.info("Training FAISS index...")
                        self.index.train(normalized_vectors)
                        logger.info("FAISS index trained")
                
                # Add vectors to index
                start_idx = self.next_index
                self.index.add(normalized_vectors)
                
                # Update mappings
                for i, vector_id in enumerate(ids):
                    faiss_idx = start_idx + i
                    self.id_map[faiss_idx] = vector_id
                    
                    if metadata:
                        self.metadata_store[vector_id] = metadata[i]
                
                self.next_index += len(vectors)
                logger.info(f"Added {len(vectors)} vectors to index (total: {self.index.ntotal})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add vectors: {e}")
                return False
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Advanced search with caching, routing, and optimization."""
        
        # Check cache first
        cached_result = self.cache_manager.get_cached_result(query)
        if cached_result:
            logger.debug("Returning cached search result")
            return cached_result
        
        start_time = time.time()
        
        try:
            # Route query to optimal search method
            search_method = self.query_router.route_query(query)
            
            # Perform search based on routing decision
            if search_method == 'flat_search':
                results = await self._flat_search(query)
            elif search_method == 'ivf_search':
                results = await self._ivf_search(query)
            elif search_method == 'hnsw_search':
                results = await self._hnsw_search(query)
            elif search_method == 'parallel_search':
                results = await self._parallel_search(query)
            else:
                results = await self._default_search(query)
            
            # Apply post-processing
            if query.filter_criteria:
                results = self._apply_filters(results, query.filter_criteria)
            
            if query.rerank:
                results = await self._rerank_results(results, query)
            
            # Cache results
            self.cache_manager.cache_result(query, results)
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            logger.debug(f"Search completed: {len(results)} results in {search_time:.3f}s")
            return results[:query.k]  # Ensure we return exactly k results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _default_search(self, query: SearchQuery) -> List[SearchResult]:
        """Default FAISS search implementation."""
        
        # Normalize query vector
        query_vector = query.query_embedding.reshape(1, -1)
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Search with extra candidates for filtering
        search_k = min(query.k * 3, self.index.ntotal) if query.filter_criteria else query.k
        
        with self._lock:
            distances, indices = self.index.search(query_vector, search_k)
        
        # Convert to SearchResult objects
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
                
            vector_id = self.id_map.get(idx)
            if not vector_id:
                continue
            
            metadata = self.metadata_store.get(vector_id, {})
            
            # Convert distance to similarity score (for inner product)
            similarity_score = float(dist)  # Already similarity for normalized vectors
            
            result = SearchResult(
                id=vector_id,
                score=similarity_score,
                metadata=metadata,
                embedding=None  # Don't include embeddings by default
            )
            results.append(result)
        
        return results
    
    async def _flat_search(self, query: SearchQuery) -> List[SearchResult]:
        """Optimized flat search for small datasets."""
        return await self._default_search(query)
    
    async def _ivf_search(self, query: SearchQuery) -> List[SearchResult]:
        """IVF search with probe optimization."""
        # Increase nprobe for higher recall
        if hasattr(self.index, 'nprobe'):
            original_nprobe = self.index.nprobe
            self.index.nprobe = min(10, original_nprobe * 2)
            
            try:
                results = await self._default_search(query)
            finally:
                self.index.nprobe = original_nprobe
            
            return results
        
        return await self._default_search(query)
    
    async def _hnsw_search(self, query: SearchQuery) -> List[SearchResult]:
        """HNSW search with dynamic efSearch."""
        if hasattr(self.index, 'hnsw'):
            # Increase efSearch for complex queries
            original_ef = self.index.hnsw.efSearch
            self.index.hnsw.efSearch = min(500, max(50, query.k * 3))
            
            try:
                results = await self._default_search(query)
            finally:
                self.index.hnsw.efSearch = original_ef
            
            return results
        
        return await self._default_search(query)
    
    async def _parallel_search(self, query: SearchQuery) -> List[SearchResult]:
        """Parallel search for large result sets."""
        # For now, use default search (can be extended for true parallelization)
        return await self._default_search(query)
    
    def _apply_filters(self, results: List[SearchResult], filter_criteria: Dict) -> List[SearchResult]:
        """Apply metadata-based filtering."""
        filtered_results = []
        
        for result in results:
            metadata = result.metadata or {}
            match = True
            
            for key, expected_value in filter_criteria.items():
                if key not in metadata:
                    match = False
                    break
                
                actual_value = metadata[key]
                
                # Handle different filter types
                if isinstance(expected_value, dict):
                    # Range filter: {"min": 10, "max": 20}
                    if "min" in expected_value and actual_value < expected_value["min"]:
                        match = False
                        break
                    if "max" in expected_value and actual_value > expected_value["max"]:
                        match = False
                        break
                elif isinstance(expected_value, list):
                    # Include filter: ["value1", "value2"]
                    if actual_value not in expected_value:
                        match = False
                        break
                else:
                    # Exact match
                    if actual_value != expected_value:
                        match = False
                        break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    async def _rerank_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Advanced reranking based on multiple criteria."""
        
        # For now, sort by similarity score (can be extended with cross-encoders)
        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Could add:
        # - Cross-encoder reranking
        # - Diversity-based reranking  
        # - Temporal relevance boost
        # - Geographic proximity boost
        
        return reranked
    
    def _update_search_stats(self, search_time: float):
        """Update search performance statistics."""
        self.search_stats['total_searches'] += 1
        
        # Update average search time
        total = self.search_stats['total_searches']
        prev_avg = self.search_stats['avg_search_time']
        new_avg = (prev_avg * (total - 1) + search_time) / total
        self.search_stats['avg_search_time'] = new_avg
        
        # Keep recent search times
        self.search_stats['search_times'].append(search_time)
        if len(self.search_stats['search_times']) > 1000:
            self.search_stats['search_times'] = self.search_stats['search_times'][-1000:]
    
    def get_index_stats(self) -> IndexStats:
        """Get comprehensive index statistics."""
        
        # Calculate memory usage (approximate)
        memory_usage = 0.0
        if self.index:
            # Rough estimation based on index type and size
            base_memory = self.index.ntotal * self.dimensions * 4 / (1024 * 1024)  # 4 bytes per float, convert to MB
            
            if self.index_type == 'PQ' or self.index_type == 'IVF_PQ':
                memory_usage = base_memory * 0.25  # PQ compression
            else:
                memory_usage = base_memory
            
            memory_usage += len(self.metadata_store) * 0.001  # Rough metadata overhead
        
        # Search performance metrics
        search_metrics = {
            'avg_search_time': self.search_stats['avg_search_time'],
            'total_searches': self.search_stats['total_searches'],
            'p95_search_time': np.percentile(self.search_stats['search_times'], 95) if self.search_stats['search_times'] else 0.0,
            'cache_hit_rate': self.cache_manager.get_cache_stats()['hit_rate']
        }
        
        return IndexStats(
            total_vectors=self.index.ntotal if self.index else 0,
            dimensions=self.dimensions,
            index_type=self.index_type,
            memory_usage_mb=memory_usage,
            last_updated=datetime.now(),
            search_metrics=search_metrics
        )
    
    def save_index(self, filepath: Path):
        """Save FAISS index and metadata to disk."""
        try:
            with self._lock:
                # Save FAISS index
                faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
                
                # Save metadata
                metadata = {
                    'id_map': self.id_map,
                    'metadata_store': self.metadata_store,
                    'next_index': self.next_index,
                    'dimensions': self.dimensions,
                    'index_type': self.index_type,
                    'config': self.config,
                    'stats': self.search_stats
                }
                
                with open(filepath.with_suffix('.meta'), 'wb') as f:
                    pickle.dump(metadata, f)
                
                logger.info(f"Index saved to {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, filepath: Path) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            with self._lock:
                # Load FAISS index
                self.index = faiss.read_index(str(filepath.with_suffix('.faiss')))
                
                # Load metadata
                with open(filepath.with_suffix('.meta'), 'rb') as f:
                    metadata = pickle.load(f)
                
                self.id_map = metadata['id_map']
                self.metadata_store = metadata['metadata_store']
                self.next_index = metadata['next_index']
                self.dimensions = metadata['dimensions']
                self.index_type = metadata['index_type']
                self.config = metadata.get('config', {})
                self.search_stats = metadata.get('stats', self.search_stats)
                
                logger.info(f"Index loaded from {filepath}: {self.index.ntotal} vectors")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    async def optimize_index(self):
        """Optimize index for better performance."""
        if not self.index or self.index.ntotal == 0:
            return
        
        logger.info("Optimizing FAISS index...")
        
        with self._lock:
            try:
                # For IVF indices, optimize clustering
                if hasattr(self.index, 'nprobe'):
                    # Automatically adjust nprobe based on index size
                    optimal_nprobe = min(20, max(1, self.index.nlist // 10))
                    self.index.nprobe = optimal_nprobe
                    logger.info(f"Set nprobe to {optimal_nprobe}")
                
                # Clear old cache after optimization
                self.cache_manager.clear_cache()
                
                logger.info("Index optimization completed")
                
            except Exception as e:
                logger.error(f"Index optimization failed: {e}")
    
    async def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        self.cache_manager.clear_cache()
        logger.info("FAISSVectorStore closed")