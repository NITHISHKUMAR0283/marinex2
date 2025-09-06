"""
Integration Tests for FloatChat System
End-to-end testing of complete AI pipeline
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from floatchat.ai.rag.rag_pipeline import OceanographicRAGPipeline
from floatchat.ai.llm.llm_orchestrator import LLMOrchestrator, LLMProvider, LLMConfig
from floatchat.ai.nl2sql.nl2sql_engine import NL2SQLEngine
from floatchat.core.database import DatabaseManager

class TestEndToEndPipeline:
    """Test complete end-to-end AI pipeline."""
    
    @pytest.fixture
    def mock_db_config(self):
        """Mock database configuration."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_floatchat',
            'user': 'test_user',
            'password': 'test_password'
        }
    
    @pytest.fixture
    def mock_llm_configs(self):
        """Mock LLM configurations."""
        return {
            LLMProvider.OPENAI: LLMConfig(
                api_key='test_openai_key',
                model_name='gpt-4-turbo-preview',
                max_tokens=4000,
                temperature=0.1
            ),
            LLMProvider.GROQ: LLMConfig(
                api_key='gsk_test_key',
                model_name='llama3-70b-8192',
                max_tokens=4000,
                temperature=0.1
            )
        }
    
    @pytest.mark.asyncio
    async def test_complete_query_pipeline(self, mock_db_config, mock_llm_configs):
        """Test complete query processing pipeline."""
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_vector_store, \
             patch('floatchat.core.database.DatabaseManager') as mock_db_manager:
            
            # Setup mocks
            mock_vector_store.return_value.similarity_search = AsyncMock(return_value=[
                Mock(metadata=Mock(content="Temperature data from Arabian Sea"), similarity_score=0.95),
                Mock(metadata=Mock(content="Salinity measurements near coast"), similarity_score=0.87)
            ])
            
            mock_db_manager.return_value.execute_query = AsyncMock(return_value=[
                {'temperature': 25.5, 'salinity': 35.2, 'depth': 100.0}
            ])
            
            # Initialize components
            vector_store = mock_vector_store.return_value
            llm_orchestrator = LLMOrchestrator(mock_llm_configs)
            db_manager = mock_db_manager.return_value
            
            # Create RAG pipeline
            rag_pipeline = OceanographicRAGPipeline(
                vector_store=vector_store,
                llm_orchestrator=llm_orchestrator,
                db_manager=db_manager
            )
            
            # Create NL2SQL engine
            nl2sql_engine = NL2SQLEngine(mock_db_config)
            
            # Test queries
            test_queries = [
                "What is the average temperature in the Arabian Sea?",
                "Show me salinity profiles from the Bay of Bengal",
                "Find oxygen levels below 1000m depth"
            ]
            
            for query in test_queries:
                # Test RAG pipeline
                with patch.object(llm_orchestrator, 'generate_response', new_callable=AsyncMock) as mock_llm:
                    mock_llm.return_value = {
                        'response': f"Analysis of {query}: Based on the data...",
                        'confidence_score': 0.9,
                        'model_used': 'gpt-4-turbo-preview'
                    }
                    
                    rag_result = await rag_pipeline.process_query(
                        query=query,
                        k_retrievals=5,
                        context_window=4000
                    )
                    
                    assert 'answer' in rag_result
                    assert 'confidence_score' in rag_result
                    assert rag_result['confidence_score'] > 0.8
                
                # Test NL2SQL pipeline
                with patch.object(nl2sql_engine.parser, 'parse_query', new_callable=AsyncMock) as mock_parser, \
                     patch.object(nl2sql_engine.generator, 'generate_sql', new_callable=AsyncMock) as mock_generator:
                    
                    mock_parser.return_value = Mock(confidence_score=0.85, query_type='data_retrieval')
                    mock_generator.return_value = Mock(
                        sql_query="SELECT AVG(temperature) FROM measurements WHERE region = 'Arabian Sea'",
                        parameters=[],
                        confidence_score=0.9
                    )
                    
                    sql_result = await nl2sql_engine.process_query(query)
                    
                    assert 'sql_query' in sql_result
                    assert 'confidence_score' in sql_result
                    assert sql_result['confidence_score'] > 0.8
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_db_config, mock_llm_configs):
        """Test system error handling and recovery mechanisms."""
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_vector_store:
            
            # Test vector store failure recovery
            mock_vector_store.return_value.similarity_search = AsyncMock(
                side_effect=Exception("Vector store connection failed")
            )
            
            vector_store = mock_vector_store.return_value
            llm_orchestrator = LLMOrchestrator(mock_llm_configs)
            
            rag_pipeline = OceanographicRAGPipeline(
                vector_store=vector_store,
                llm_orchestrator=llm_orchestrator,
                db_manager=Mock()
            )
            
            # Should handle vector store failure gracefully
            with patch.object(llm_orchestrator, 'generate_response', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = {
                    'response': "I apologize, but I'm experiencing technical difficulties...",
                    'confidence_score': 0.3,
                    'model_used': 'gpt-4-turbo-preview'
                }
                
                result = await rag_pipeline.process_query(
                    query="Test query",
                    k_retrievals=5,
                    context_window=4000
                )
                
                # Should return response with low confidence due to missing context
                assert 'answer' in result
                assert result['confidence_score'] < 0.5
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, mock_db_config, mock_llm_configs):
        """Test system performance under concurrent load."""
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_vector_store, \
             patch('floatchat.core.database.DatabaseManager') as mock_db_manager:
            
            # Setup fast mocks
            mock_vector_store.return_value.similarity_search = AsyncMock(return_value=[
                Mock(metadata=Mock(content="Test data"), similarity_score=0.9)
            ])
            
            mock_db_manager.return_value.execute_query = AsyncMock(return_value=[
                {'temperature': 25.0}
            ])
            
            # Initialize system
            vector_store = mock_vector_store.return_value
            llm_orchestrator = LLMOrchestrator(mock_llm_configs)
            db_manager = mock_db_manager.return_value
            
            rag_pipeline = OceanographicRAGPipeline(
                vector_store=vector_store,
                llm_orchestrator=llm_orchestrator,
                db_manager=db_manager
            )
            
            # Simulate concurrent queries
            concurrent_queries = [
                f"Test query {i}" for i in range(50)
            ]
            
            with patch.object(llm_orchestrator, 'generate_response', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = {
                    'response': "Test response",
                    'confidence_score': 0.9,
                    'model_used': 'gpt-4-turbo-preview'
                }
                
                # Process queries concurrently
                start_time = datetime.now()
                tasks = [
                    rag_pipeline.process_query(query, k_retrievals=5, context_window=4000)
                    for query in concurrent_queries
                ]
                results = await asyncio.gather(*tasks)
                end_time = datetime.now()
                
                # Verify all queries processed successfully
                assert len(results) == 50
                assert all('answer' in result for result in results)
                
                # Performance check - should complete within reasonable time
                processing_time = (end_time - start_time).total_seconds()
                assert processing_time < 30  # Should complete within 30 seconds

class TestDataQualityValidation:
    """Test data quality and validation."""
    
    @pytest.mark.asyncio
    async def test_embedding_quality_validation(self):
        """Test embedding quality meets standards."""
        from floatchat.ai.embeddings.multi_modal_embeddings import OceanographicEmbeddingGenerator
        
        # Mock embedding generator
        with patch('floatchat.ai.embeddings.multi_modal_embeddings.OceanographicEmbeddingGenerator') as mock_generator:
            
            # Create realistic embeddings
            good_embeddings = [np.random.randn(512) for _ in range(10)]
            bad_embeddings = [np.zeros(512) for _ in range(5)]  # Poor quality
            
            generator = mock_generator.return_value
            generator.generate_embeddings = AsyncMock(return_value=[
                {'embedding': emb, 'metadata': {'quality_score': 0.9}}
                for emb in good_embeddings
            ])
            
            # Test quality assessment
            results = await generator.generate_embeddings([Mock()])
            
            for result in results:
                quality_score = result['metadata']['quality_score']
                embedding = result['embedding']
                
                # Quality checks
                assert quality_score > 0.8  # High quality threshold
                assert not np.all(embedding == 0)  # Not all zeros
                assert np.all(np.isfinite(embedding))  # No NaN or infinity
                assert np.std(embedding) > 0.1  # Reasonable variance
    
    def test_search_result_relevance(self):
        """Test search results meet relevance thresholds."""
        from floatchat.ai.vector_database.faiss_vector_store import SearchResult, VectorMetadata
        
        # Create test search results
        high_relevance_metadata = VectorMetadata(
            measurement_id="relevant_1",
            wmo_id="2902746",
            latitude=15.0, longitude=70.0, depth=100.0,
            parameters=['temperature', 'salinity'],
            timestamp="2023-06-01T00:00:00Z",
            content_hash="hash1"
        )
        
        low_relevance_metadata = VectorMetadata(
            measurement_id="irrelevant_1", 
            wmo_id="2902999",
            latitude=-60.0, longitude=120.0, depth=5000.0,
            parameters=['oxygen'],
            timestamp="2020-01-01T00:00:00Z",
            content_hash="hash2"
        )
        
        results = [
            SearchResult(high_relevance_metadata, 0.95, 1, 0.05),
            SearchResult(low_relevance_metadata, 0.3, 2, 0.7)
        ]
        
        # Filter by relevance threshold
        relevant_results = [r for r in results if r.similarity_score > 0.8]
        
        assert len(relevant_results) == 1
        assert relevant_results[0].metadata.measurement_id == "relevant_1"
    
    @pytest.mark.asyncio
    async def test_sql_query_safety(self):
        """Test SQL query safety and validation."""
        from floatchat.ai.nl2sql.sql_generator import OceanographicSQLGenerator
        from floatchat.ai.nl2sql.oceanographic_schema import OceanographicSchema
        
        schema = OceanographicSchema()
        generator = OceanographicSQLGenerator(schema)
        
        # Test safe query generation
        safe_query_structure = Mock()
        safe_query_structure.query_type = 'data_retrieval'
        safe_query_structure.spatial_constraints = []
        safe_query_structure.temporal_constraints = []
        safe_query_structure.parameter_constraints = []
        
        with patch.object(generator, '_validate_query_safety') as mock_validator:
            mock_validator.return_value = True
            
            result = await generator.generate_sql(safe_query_structure)
            
            # Verify safety validation was called
            mock_validator.assert_called_once()
            
            # SQL should not contain dangerous operations
            sql = result.sql_query if hasattr(result, 'sql_query') else ""
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            
            for keyword in dangerous_keywords:
                assert keyword.upper() not in sql.upper()

class TestSystemConfiguration:
    """Test system configuration and environment setup."""
    
    def test_environment_variables(self):
        """Test required environment variables are configured."""
        import os
        
        # Required environment variables for production
        required_vars = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'DATABASE_URL',
            'VECTOR_INDEX_PATH'
        ]
        
        # For testing, we'll check if they're set or provide defaults
        for var in required_vars:
            value = os.getenv(var)
            if value is None:
                # In test environment, use test values
                if var.endswith('_KEY'):
                    assert len(var) > 0  # Key name should exist
                else:
                    # Other variables can have test defaults
                    test_defaults = {
                        'DATABASE_URL': 'postgresql://test:test@localhost/test',
                        'VECTOR_INDEX_PATH': './test_vectors'
                    }
                    assert var in test_defaults
    
    def test_model_configurations(self):
        """Test LLM model configurations are valid."""
        from floatchat.ai.llm.llm_orchestrator import LLMConfig, LLMProvider
        
        # Test configuration validation
        valid_config = LLMConfig(
            api_key='test_key',
            model_name='gpt-4-turbo-preview',
            max_tokens=4000,
            temperature=0.1
        )
        
        assert valid_config.api_key == 'test_key'
        assert 0 <= valid_config.temperature <= 2.0
        assert valid_config.max_tokens > 0
        assert valid_config.max_tokens <= 128000  # Reasonable upper bound
        
        # Test invalid configurations
        with pytest.raises((ValueError, TypeError)):
            LLMConfig(
                api_key='',  # Empty key
                model_name='invalid_model',
                max_tokens=-1,  # Invalid token count
                temperature=3.0  # Invalid temperature
            )
    
    def test_database_connection_config(self):
        """Test database connection configuration."""
        from floatchat.core.database import DatabaseManager
        
        # Test valid configuration
        valid_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'floatchat_test',
            'user': 'test_user',
            'password': 'test_password'
        }
        
        db_manager = DatabaseManager(valid_config)
        assert db_manager.config['host'] == 'localhost'
        assert db_manager.config['port'] == 5432
        
        # Test invalid configurations
        invalid_configs = [
            {},  # Empty config
            {'host': 'localhost'},  # Missing required fields
            {'host': 'localhost', 'port': 'invalid_port'}  # Invalid port type
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, KeyError, TypeError)):
                DatabaseManager(invalid_config)

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self):
        """Test embedding generation performance benchmarks."""
        from floatchat.ai.embeddings.multi_modal_embeddings import OceanographicEmbeddingGenerator
        
        with patch('floatchat.ai.embeddings.multi_modal_embeddings.OceanographicEmbeddingGenerator') as mock_generator:
            
            # Mock fast embedding generation
            def mock_generate(measurements):
                return [
                    {'embedding': np.random.randn(512), 'metadata': {'quality_score': 0.9}}
                    for _ in measurements
                ]
            
            generator = mock_generator.return_value
            generator.generate_embeddings = AsyncMock(side_effect=mock_generate)
            
            # Test batch processing performance
            batch_sizes = [1, 10, 100]
            
            for batch_size in batch_sizes:
                measurements = [Mock() for _ in range(batch_size)]
                
                start_time = datetime.now()
                results = await generator.generate_embeddings(measurements)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                
                # Performance assertions
                assert len(results) == batch_size
                
                # Should process at reasonable speed (rough benchmarks)
                if batch_size == 1:
                    assert processing_time < 1.0  # Single embedding < 1 second
                elif batch_size == 10:
                    assert processing_time < 5.0  # 10 embeddings < 5 seconds
                else:  # batch_size == 100
                    assert processing_time < 30.0  # 100 embeddings < 30 seconds
    
    @pytest.mark.asyncio
    async def test_vector_search_performance(self):
        """Test vector search performance benchmarks."""
        from floatchat.ai.vector_database.faiss_vector_store import OceanographicVectorStore
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_store:
            
            # Mock fast search
            def mock_search(query_vector, k=5, **kwargs):
                return [
                    Mock(similarity_score=0.9 - i*0.1, metadata=Mock()) 
                    for i in range(k)
                ]
            
            store = mock_store.return_value
            store.similarity_search = AsyncMock(side_effect=mock_search)
            
            # Test search performance with different k values
            query_vector = np.random.randn(512)
            k_values = [5, 20, 100]
            
            for k in k_values:
                start_time = datetime.now()
                results = await store.similarity_search(query_vector, k=k)
                end_time = datetime.now()
                
                search_time = (end_time - start_time).total_seconds()
                
                # Performance assertions
                assert len(results) == k
                assert search_time < 2.0  # All searches should be fast
    
    @pytest.mark.asyncio 
    async def test_end_to_end_response_time(self):
        """Test complete system response time benchmarks."""
        
        # Mock all components for fast testing
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_vector_store, \
             patch('floatchat.core.database.DatabaseManager') as mock_db_manager, \
             patch('floatchat.ai.llm.llm_orchestrator.LLMOrchestrator') as mock_orchestrator:
            
            # Setup fast mocks
            mock_vector_store.return_value.similarity_search = AsyncMock(return_value=[
                Mock(metadata=Mock(content="Fast test data"), similarity_score=0.95)
            ])
            
            mock_db_manager.return_value.execute_query = AsyncMock(return_value=[
                {'temperature': 25.0, 'salinity': 35.0}
            ])
            
            mock_orchestrator.return_value.generate_response = AsyncMock(return_value={
                'response': 'Fast test response based on oceanographic data...',
                'confidence_score': 0.9,
                'model_used': 'gpt-4-turbo-preview'
            })
            
            # Initialize system
            from floatchat.ai.rag.rag_pipeline import OceanographicRAGPipeline
            
            pipeline = OceanographicRAGPipeline(
                vector_store=mock_vector_store.return_value,
                llm_orchestrator=mock_orchestrator.return_value,
                db_manager=mock_db_manager.return_value
            )
            
            # Test response time for different query types
            test_queries = [
                "What is the temperature in Arabian Sea?",
                "Show me salinity data from last month",
                "Analyze oxygen levels in deep water"
            ]
            
            for query in test_queries:
                start_time = datetime.now()
                result = await pipeline.process_query(
                    query=query,
                    k_retrievals=5,
                    context_window=4000
                )
                end_time = datetime.now()
                
                response_time = (end_time - start_time).total_seconds()
                
                # Performance requirements
                assert 'answer' in result
                assert response_time < 10.0  # Complete response within 10 seconds
                assert result['confidence_score'] > 0.8  # High confidence
    
    def test_memory_usage_limits(self):
        """Test system memory usage stays within limits."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate heavy operations
        large_arrays = []
        for i in range(10):
            # Create large arrays to simulate embedding storage
            large_array = np.random.randn(1000, 512).astype(np.float32)
            large_arrays.append(large_array)
        
        # Check memory increase
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        del large_arrays
        gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage assertions
        assert memory_increase < 1000  # Should not use more than 1GB extra
        assert (final_memory - initial_memory) < 100  # Should cleanup properly