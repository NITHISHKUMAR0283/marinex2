"""
Unit Tests for FAISS Vector Database
Tests vector storage, retrieval, and hybrid search capabilities
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from floatchat.ai.vector_database.faiss_vector_store import (
    OceanographicVectorStore,
    HybridSearchEngine,
    QueryCache,
    VectorMetadata,
    SearchResult
)

class TestOceanographicVectorStore:
    """Test the main vector store functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create vector store for testing."""
        return OceanographicVectorStore(
            index_path=str(temp_dir / "test_index"),
            dimension=512
        )
    
    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors for testing."""
        vectors = []
        metadata = []
        
        for i in range(100):
            vector = np.random.randn(512).astype(np.float32)
            metadata = VectorMetadata(
                measurement_id=f"test_{i}",
                wmo_id=f"290274{i % 10}",
                latitude=np.random.uniform(-30, 30),
                longitude=np.random.uniform(40, 100),
                depth=np.random.uniform(0, 2000),
                parameters=['temperature', 'salinity'] if i % 2 == 0 else ['oxygen'],
                timestamp=f"2023-{(i % 12) + 1:02d}-01T00:00:00Z",
                content_hash=f"hash_{i}"
            )
            
            vectors.append(vector)
            metdata.append(metadata)
        
        return vectors, metdata
    
    @pytest.mark.asyncio
    async def test_initialization(self, vector_store):
        """Test vector store initialization."""
        await vector_store.initialize()
        
        assert vector_store.dimension == 512
        assert vector_store.index is not None
        assert vector_store.metadata_store == {}
        assert isinstance(vector_store.search_engine, HybridSearchEngine)
        assert isinstance(vector_store.cache, QueryCache)
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, vector_store, sample_vectors):
        """Test adding vectors to the store."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        
        # Add vectors in batches
        batch_size = 10
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            await vector_store.add_vectors(batch_vectors, batch_metadata)
        
        # Verify vectors were added
        assert vector_store.index.ntotal == len(vectors)
        assert len(vector_store.metadata_store) == len(vectors)
    
    @pytest.mark.asyncio  
    async def test_similarity_search(self, vector_store, sample_vectors):
        """Test similarity search functionality."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        # Search with a query vector
        query_vector = vectors[0]  # Use first vector as query
        results = await vector_store.similarity_search(
            query_vector=query_vector,
            k=5
        )
        
        assert len(results) == 5
        assert all(isinstance(result, SearchResult) for result in results)
        assert all(0 <= result.similarity_score <= 1 for result in results)
        
        # First result should be exact match (similarity = 1.0)
        assert results[0].similarity_score == pytest.approx(1.0, abs=1e-6)
        assert results[0].metadata.measurement_id == "test_0"
    
    @pytest.mark.asyncio
    async def test_filtered_search(self, vector_store, sample_vectors):
        """Test search with metadata filters."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        query_vector = np.random.randn(512).astype(np.float32)
        
        # Filter by WMO ID
        results = await vector_store.filtered_search(
            query_vector=query_vector,
            filters={'wmo_id': '2902740'},
            k=10
        )
        
        # All results should match the filter
        assert all(result.metadata.wmo_id == '2902740' for result in results)
        assert len(results) <= 10
    
    @pytest.mark.asyncio
    async def test_spatial_search(self, vector_store, sample_vectors):
        """Test spatial proximity search."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        query_vector = np.random.randn(512).astype(np.float32)
        
        # Search near specific coordinates
        results = await vector_store.spatial_search(
            query_vector=query_vector,
            center_lat=15.0,
            center_lon=70.0,
            radius_km=500,
            k=10
        )
        
        # Verify results are within spatial bounds
        for result in results:
            # Calculate approximate distance (simplified)
            lat_diff = abs(result.metadata.latitude - 15.0)
            lon_diff = abs(result.metadata.longitude - 70.0)
            # Rough distance check (not exact but good for testing)
            assert lat_diff < 10 and lon_diff < 10  # Should be reasonably close
    
    @pytest.mark.asyncio
    async def test_temporal_search(self, vector_store, sample_vectors):
        """Test temporal range search."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        query_vector = np.random.randn(512).astype(np.float32)
        
        # Search within time range
        results = await vector_store.temporal_search(
            query_vector=query_vector,
            start_time="2023-06-01T00:00:00Z",
            end_time="2023-08-31T23:59:59Z",
            k=20
        )
        
        # Verify results are within temporal bounds
        for result in results:
            timestamp = result.metadata.timestamp
            assert "2023-06" in timestamp or "2023-07" in timestamp or "2023-08" in timestamp
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, vector_store, sample_vectors):
        """Test hybrid search combining multiple criteria."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        query_vector = np.random.randn(512).astype(np.float32)
        
        search_criteria = {
            'semantic_weight': 0.6,
            'spatial_weight': 0.3,
            'temporal_weight': 0.1,
            'center_lat': 10.0,
            'center_lon': 75.0,
            'radius_km': 1000,
            'start_time': "2023-01-01T00:00:00Z",
            'end_time': "2023-12-31T23:59:59Z",
            'parameters': ['temperature']
        }
        
        results = await vector_store.hybrid_search(
            query_vector=query_vector,
            criteria=search_criteria,
            k=15
        )
        
        assert len(results) <= 15
        assert all(isinstance(result, SearchResult) for result in results)
        # Results should be sorted by combined score
        scores = [result.similarity_score for result in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_batch_search(self, vector_store, sample_vectors):
        """Test batch search functionality."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        # Multiple query vectors
        query_vectors = [
            np.random.randn(512).astype(np.float32),
            np.random.randn(512).astype(np.float32),
            np.random.randn(512).astype(np.float32)
        ]
        
        results = await vector_store.batch_search(
            query_vectors=query_vectors,
            k=5
        )
        
        assert len(results) == 3  # One result set per query
        assert all(len(result_set) == 5 for result_set in results)
    
    @pytest.mark.asyncio
    async def test_save_load_index(self, vector_store, sample_vectors, temp_dir):
        """Test saving and loading index."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        # Save index
        await vector_store.save_index()
        
        # Verify files were created
        assert (Path(vector_store.index_path) / "index.faiss").exists()
        assert (Path(vector_store.index_path) / "metadata.json").exists()
        
        # Create new vector store and load
        new_vector_store = OceanographicVectorStore(
            index_path=str(temp_dir / "test_index"),
            dimension=512
        )
        await new_vector_store.load_index()
        
        # Verify loaded correctly
        assert new_vector_store.index.ntotal == len(vectors)
        assert len(new_vector_store.metadata_store) == len(vectors)
    
    @pytest.mark.asyncio
    async def test_remove_vectors(self, vector_store, sample_vectors):
        """Test vector removal functionality."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        original_count = vector_store.index.ntotal
        
        # Remove specific vectors by ID
        ids_to_remove = ["test_0", "test_1", "test_2"]
        removed_count = await vector_store.remove_vectors(ids_to_remove)
        
        assert removed_count == 3
        assert vector_store.index.ntotal == original_count - 3
        
        # Verify metadata was also removed
        for vector_id in ids_to_remove:
            assert vector_id not in vector_store.metadata_store
    
    @pytest.mark.asyncio
    async def test_update_vectors(self, vector_store, sample_vectors):
        """Test vector update functionality."""
        vectors, metadata = sample_vectors
        await vector_store.initialize()
        await vector_store.add_vectors(vectors, metadata)
        
        # Update a vector
        new_vector = np.random.randn(512).astype(np.float32)
        new_metadata = VectorMetadata(
            measurement_id="test_0",
            wmo_id="updated_wmo",
            latitude=999.0,  # Distinctive value
            longitude=999.0,
            depth=0.0,
            parameters=['updated'],
            timestamp="2024-01-01T00:00:00Z",
            content_hash="updated_hash"
        )
        
        success = await vector_store.update_vector("test_0", new_vector, new_metadata)
        assert success
        
        # Verify update
        updated_metadata = vector_store.metadata_store["test_0"]
        assert updated_metadata.wmo_id == "updated_wmo"
        assert updated_metadata.latitude == 999.0

class TestHybridSearchEngine:
    """Test hybrid search engine functionality."""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine for testing."""
        return HybridSearchEngine()
    
    def test_combine_scores(self, search_engine):
        """Test score combination logic."""
        semantic_scores = [0.9, 0.8, 0.7]
        spatial_scores = [0.6, 0.9, 0.5]
        temporal_scores = [0.7, 0.6, 0.8]
        
        weights = {
            'semantic_weight': 0.5,
            'spatial_weight': 0.3,
            'temporal_weight': 0.2
        }
        
        combined = search_engine.combine_scores(
            semantic_scores, spatial_scores, temporal_scores, weights
        )
        
        assert len(combined) == 3
        assert all(0 <= score <= 1 for score in combined)
        
        # Manually calculate first score to verify
        expected_first = (0.9 * 0.5) + (0.6 * 0.3) + (0.7 * 0.2)
        assert combined[0] == pytest.approx(expected_first, abs=1e-6)
    
    def test_spatial_distance_scoring(self, search_engine):
        """Test spatial distance to similarity conversion."""
        distances = [0, 100, 500, 1000, 5000]  # km
        max_distance = 2000  # km
        
        scores = search_engine.distance_to_similarity(distances, max_distance)
        
        assert len(scores) == len(distances)
        assert scores[0] == 1.0  # Zero distance = perfect similarity
        assert scores[-1] == 0.0  # Beyond max distance = zero similarity
        assert all(0 <= score <= 1 for score in scores)
        assert scores == sorted(scores, reverse=True)  # Should be descending
    
    def test_temporal_distance_scoring(self, search_engine):
        """Test temporal distance to similarity conversion."""
        # Time differences in hours
        time_diffs = [0, 1, 24, 168, 720]  # 0h, 1h, 1d, 1w, 1m
        max_time_diff = 720  # hours
        
        scores = search_engine.temporal_distance_to_similarity(time_diffs, max_time_diff)
        
        assert len(scores) == len(time_diffs)
        assert scores[0] == 1.0  # Zero time diff = perfect similarity
        assert scores[-1] == 0.0  # Max time diff = zero similarity
        assert all(0 <= score <= 1 for score in scores)

class TestQueryCache:
    """Test query caching functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create cache for testing."""
        return QueryCache(max_size=100, ttl_seconds=60)
    
    def test_cache_operations(self, cache):
        """Test basic cache operations."""
        query_vector = np.random.randn(512)
        query_params = {'k': 5, 'filters': {'wmo_id': '123'}}
        
        # Cache miss initially
        result = cache.get(query_vector, query_params)
        assert result is None
        
        # Store result
        search_results = [Mock(), Mock(), Mock()]
        cache.put(query_vector, query_params, search_results)
        
        # Cache hit
        cached_result = cache.get(query_vector, query_params)
        assert cached_result == search_results
    
    def test_cache_key_generation(self, cache):
        """Test cache key generation consistency."""
        query_vector = np.random.randn(512)
        params1 = {'k': 5, 'wmo_id': '123'}
        params2 = {'wmo_id': '123', 'k': 5}  # Same params, different order
        
        key1 = cache._generate_key(query_vector, params1)
        key2 = cache._generate_key(query_vector, params2)
        
        assert key1 == key2  # Should be same regardless of order
    
    def test_cache_size_limit(self, cache):
        """Test cache size limiting."""
        # Fill cache beyond max size
        for i in range(150):  # More than max_size=100
            query_vector = np.random.randn(512)
            params = {'test_param': i}
            cache.put(query_vector, params, [f"result_{i}"])
        
        # Cache should be limited to max size
        assert len(cache.cache) <= cache.max_size
    
    def test_cache_ttl(self, cache):
        """Test time-to-live functionality."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000
            
            query_vector = np.random.randn(512)
            params = {'test': 'value'}
            cache.put(query_vector, params, ['result'])
            
            # Still valid within TTL
            mock_time.return_value = 1030  # 30 seconds later
            result = cache.get(query_vector, params)
            assert result == ['result']
            
            # Expired after TTL
            mock_time.return_value = 1100  # 100 seconds later (> 60 TTL)
            result = cache.get(query_vector, params)
            assert result is None

class TestVectorMetadata:
    """Test vector metadata functionality."""
    
    def test_metadata_creation(self):
        """Test metadata object creation."""
        metadata = VectorMetadata(
            measurement_id="test_123",
            wmo_id="2902746",
            latitude=15.5,
            longitude=68.2,
            depth=250.0,
            parameters=['temperature', 'salinity'],
            timestamp="2023-04-15T12:30:00Z",
            content_hash="abc123"
        )
        
        assert metadata.measurement_id == "test_123"
        assert metadata.wmo_id == "2902746"
        assert metadata.latitude == 15.5
        assert metadata.longitude == 68.2
        assert metadata.depth == 250.0
        assert metadata.parameters == ['temperature', 'salinity']
        assert metadata.timestamp == "2023-04-15T12:30:00Z"
        assert metadata.content_hash == "abc123"
    
    def test_metadata_serialization(self):
        """Test metadata serialization to/from dict."""
        metadata = VectorMetadata(
            measurement_id="test_123",
            wmo_id="2902746",
            latitude=15.5,
            longitude=68.2,
            depth=250.0,
            parameters=['temperature'],
            timestamp="2023-04-15T12:30:00Z",
            content_hash="abc123"
        )
        
        # Serialize to dict
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data['measurement_id'] == "test_123"
        assert data['parameters'] == ['temperature']
        
        # Deserialize from dict
        restored = VectorMetadata.from_dict(data)
        assert restored.measurement_id == metadata.measurement_id
        assert restored.parameters == metadata.parameters
        assert restored.latitude == metadata.latitude

class TestSearchResult:
    """Test search result functionality."""
    
    def test_search_result_creation(self):
        """Test search result object creation."""
        metadata = VectorMetadata(
            measurement_id="test_123",
            wmo_id="2902746", 
            latitude=15.5,
            longitude=68.2,
            depth=250.0,
            parameters=['temperature'],
            timestamp="2023-04-15T12:30:00Z",
            content_hash="abc123"
        )
        
        result = SearchResult(
            metadata=metadata,
            similarity_score=0.95,
            vector_id=42,
            distance=0.05
        )
        
        assert result.metadata == metadata
        assert result.similarity_score == 0.95
        assert result.vector_id == 42
        assert result.distance == 0.05
    
    def test_search_result_comparison(self):
        """Test search result comparison for sorting."""
        metadata1 = VectorMetadata("test_1", "wmo1", 0, 0, 0, [], "2023-01-01T00:00:00Z", "hash1")
        metadata2 = VectorMetadata("test_2", "wmo2", 0, 0, 0, [], "2023-01-01T00:00:00Z", "hash2")
        
        result1 = SearchResult(metadata1, 0.9, 1, 0.1)
        result2 = SearchResult(metadata2, 0.8, 2, 0.2)
        
        # Higher similarity should be "greater"
        assert result1 > result2
        assert not result2 > result1
        
        # Test sorting
        results = [result2, result1]
        sorted_results = sorted(results, reverse=True)
        assert sorted_results[0] == result1
        assert sorted_results[1] == result2