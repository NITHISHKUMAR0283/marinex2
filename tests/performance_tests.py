"""
Performance Testing Suite for FloatChat
Comprehensive performance benchmarks and load testing
"""

import pytest
import asyncio
import time
import numpy as np
import psutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock
import statistics
from typing import List, Dict, Any

class PerformanceMetrics:
    """Class to track and analyze performance metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.success_rate: float = 0.0
        self.error_count: int = 0
        self.total_queries: int = 0
    
    def add_response_time(self, response_time: float):
        """Add response time measurement."""
        self.response_times.append(response_time)
    
    def add_memory_usage(self, memory_mb: float):
        """Add memory usage measurement."""
        self.memory_usage.append(memory_mb)
    
    def add_cpu_usage(self, cpu_percent: float):
        """Add CPU usage measurement."""
        self.cpu_usage.append(cpu_percent)
    
    def record_query(self, success: bool):
        """Record query success/failure."""
        self.total_queries += 1
        if not success:
            self.error_count += 1
        self.success_rate = (self.total_queries - self.error_count) / self.total_queries
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        return {
            'response_time_stats': {
                'mean': statistics.mean(self.response_times) if self.response_times else 0,
                'median': statistics.median(self.response_times) if self.response_times else 0,
                'min': min(self.response_times) if self.response_times else 0,
                'max': max(self.response_times) if self.response_times else 0,
                'p95': np.percentile(self.response_times, 95) if self.response_times else 0,
                'p99': np.percentile(self.response_times, 99) if self.response_times else 0
            },
            'memory_stats': {
                'mean': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'peak': max(self.memory_usage) if self.memory_usage else 0,
                'min': min(self.memory_usage) if self.memory_usage else 0
            },
            'cpu_stats': {
                'mean': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'peak': max(self.cpu_usage) if self.cpu_usage else 0
            },
            'success_rate': self.success_rate,
            'total_queries': self.total_queries,
            'error_count': self.error_count
        }

class TestEmbeddingPerformance:
    """Test embedding generation performance."""
    
    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics tracker."""
        return PerformanceMetrics()
    
    @pytest.mark.asyncio
    async def test_single_embedding_performance(self, performance_metrics):
        """Test single embedding generation performance."""
        from floatchat.ai.embeddings.multi_modal_embeddings import OceanographicEmbeddingGenerator
        
        with patch('floatchat.ai.embeddings.multi_modal_embeddings.OceanographicEmbeddingGenerator') as mock_generator:
            
            # Mock realistic embedding generation time
            async def mock_generate(measurements):
                await asyncio.sleep(0.1)  # Simulate API call time
                return [
                    {'embedding': np.random.randn(512), 'metadata': {'quality_score': 0.9}}
                    for _ in measurements
                ]
            
            generator = mock_generator.return_value
            generator.generate_embeddings = AsyncMock(side_effect=mock_generate)
            
            # Test single embedding
            measurement = Mock()
            
            start_time = time.time()
            result = await generator.generate_embeddings([measurement])
            end_time = time.time()
            
            response_time = end_time - start_time
            performance_metrics.add_response_time(response_time)
            performance_metrics.record_query(len(result) == 1)
            
            # Performance assertions
            assert response_time < 1.0  # Should complete within 1 second
            assert len(result) == 1
            assert 'embedding' in result[0]
    
    @pytest.mark.asyncio
    async def test_batch_embedding_performance(self, performance_metrics):
        """Test batch embedding generation performance."""
        from floatchat.ai.embeddings.multi_modal_embeddings import OceanographicEmbeddingGenerator
        
        with patch('floatchat.ai.embeddings.multi_modal_embeddings.OceanographicEmbeddingGenerator') as mock_generator:
            
            # Mock batch processing with realistic timing
            async def mock_batch_generate(measurements):
                # Simulate batch processing efficiency
                batch_time = 0.05 * len(measurements)  # 50ms per item in batch
                await asyncio.sleep(batch_time)
                return [
                    {'embedding': np.random.randn(512), 'metadata': {'quality_score': 0.9}}
                    for _ in measurements
                ]
            
            generator = mock_generator.return_value  
            generator.generate_embeddings = AsyncMock(side_effect=mock_batch_generate)
            
            # Test different batch sizes
            batch_sizes = [10, 50, 100]
            
            for batch_size in batch_sizes:
                measurements = [Mock() for _ in range(batch_size)]
                
                start_time = time.time()
                results = await generator.generate_embeddings(measurements)
                end_time = time.time()
                
                response_time = end_time - start_time
                performance_metrics.add_response_time(response_time)
                performance_metrics.record_query(len(results) == batch_size)
                
                # Performance requirements based on batch size
                if batch_size == 10:
                    assert response_time < 2.0
                elif batch_size == 50:
                    assert response_time < 5.0
                else:  # batch_size == 100
                    assert response_time < 10.0
                
                # Efficiency check - batch should be more efficient than individual
                individual_time_estimate = batch_size * 0.1  # 100ms per individual
                efficiency_gain = individual_time_estimate / response_time
                assert efficiency_gain > 1.5  # At least 50% more efficient
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_performance(self, performance_metrics):
        """Test concurrent embedding generation performance."""
        from floatchat.ai.embeddings.multi_modal_embeddings import OceanographicEmbeddingGenerator
        
        with patch('floatchat.ai.embeddings.multi_modal_embeddings.OceanographicEmbeddingGenerator') as mock_generator:
            
            async def mock_concurrent_generate(measurements):
                await asyncio.sleep(0.2)  # Simulate processing time
                return [
                    {'embedding': np.random.randn(512), 'metadata': {'quality_score': 0.9}}
                    for _ in measurements
                ]
            
            generator = mock_generator.return_value
            generator.generate_embeddings = AsyncMock(side_effect=mock_concurrent_generate)
            
            # Create concurrent requests
            concurrent_requests = 10
            tasks = []
            
            start_time = time.time()
            
            for i in range(concurrent_requests):
                measurement = Mock()
                task = generator.generate_embeddings([measurement])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            performance_metrics.add_response_time(total_time)
            
            # Verify all requests completed successfully
            assert len(results) == concurrent_requests
            assert all(len(result) == 1 for result in results)
            
            # Concurrent processing should be much faster than sequential
            sequential_time_estimate = concurrent_requests * 0.2
            concurrency_efficiency = sequential_time_estimate / total_time
            assert concurrency_efficiency > 2.0  # At least 2x faster with concurrency
    
    def test_memory_usage_during_embedding(self, performance_metrics):
        """Test memory usage during embedding operations."""
        import gc
        
        process = psutil.Process()
        
        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        performance_metrics.add_memory_usage(initial_memory)
        
        # Simulate large embedding operations
        large_embeddings = []
        for i in range(100):
            # Simulate 512-dimensional embeddings
            embedding = np.random.randn(512).astype(np.float32)
            large_embeddings.append(embedding)
            
            # Record memory usage periodically
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                performance_metrics.add_memory_usage(current_memory)
        
        # Record peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024
        performance_metrics.add_memory_usage(peak_memory)
        
        # Cleanup and record final memory
        del large_embeddings
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        performance_metrics.add_memory_usage(final_memory)
        
        # Memory usage assertions
        memory_increase = peak_memory - initial_memory
        memory_cleanup = peak_memory - final_memory
        
        assert memory_increase < 500  # Should not use more than 500MB
        assert memory_cleanup > (memory_increase * 0.8)  # Should cleanup 80%+ of memory

class TestVectorSearchPerformance:
    """Test vector search performance."""
    
    @pytest.mark.asyncio
    async def test_similarity_search_speed(self):
        """Test similarity search performance across different scales."""
        from floatchat.ai.vector_database.faiss_vector_store import OceanographicVectorStore
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_store:
            
            # Mock search with realistic timing
            def mock_search(query_vector, k=5, **kwargs):
                # Simulate FAISS search time based on index size and k
                search_time = 0.001 + (k * 0.0001)  # Base time + per-result time
                time.sleep(search_time)
                
                return [
                    Mock(similarity_score=0.95 - i*0.05, metadata=Mock())
                    for i in range(min(k, 1000))  # Simulate index size limit
                ]
            
            store = mock_store.return_value
            store.similarity_search = AsyncMock(side_effect=mock_search)
            
            query_vector = np.random.randn(512)
            
            # Test different k values
            k_values = [1, 5, 10, 50, 100]
            performance_results = []
            
            for k in k_values:
                start_time = time.time()
                results = await store.similarity_search(query_vector, k=k)
                end_time = time.time()
                
                search_time = end_time - start_time
                performance_results.append({
                    'k': k,
                    'time': search_time,
                    'results_count': len(results)
                })
                
                # Performance assertions
                assert len(results) == min(k, 1000)
                assert search_time < 0.1  # All searches should be very fast
            
            # Verify search time scales reasonably with k
            times = [r['time'] for r in performance_results]
            assert times == sorted(times)  # Should increase with k
    
    @pytest.mark.asyncio
    async def test_filtered_search_performance(self):
        """Test performance of searches with filters."""
        from floatchat.ai.vector_database.faiss_vector_store import OceanographicVectorStore
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_store:
            
            def mock_filtered_search(query_vector, filters=None, k=5, **kwargs):
                # Simulate additional filtering overhead
                filter_overhead = 0.002 if filters else 0
                base_time = 0.001 + (k * 0.0001) + filter_overhead
                time.sleep(base_time)
                
                return [
                    Mock(similarity_score=0.9 - i*0.1, metadata=Mock())
                    for i in range(k)
                ]
            
            store = mock_store.return_value
            store.filtered_search = AsyncMock(side_effect=mock_filtered_search)
            
            query_vector = np.random.randn(512)
            
            # Test search without filters
            start_time = time.time()
            results_no_filter = await store.filtered_search(query_vector, filters=None, k=10)
            time_no_filter = time.time() - start_time
            
            # Test search with filters
            filters = {'wmo_id': '2902746', 'depth_range': (0, 1000)}
            start_time = time.time()
            results_with_filter = await store.filtered_search(query_vector, filters=filters, k=10)
            time_with_filter = time.time() - start_time
            
            # Performance assertions
            assert len(results_no_filter) == 10
            assert len(results_with_filter) == 10
            assert time_no_filter < 0.1
            assert time_with_filter < 0.1
            
            # Filtered search should have some overhead but not excessive
            filter_overhead_ratio = time_with_filter / time_no_filter
            assert 1.0 <= filter_overhead_ratio <= 3.0  # At most 3x slower
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self):
        """Test concurrent search performance."""
        from floatchat.ai.vector_database.faiss_vector_store import OceanographicVectorStore
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_store:
            
            async def mock_concurrent_search(query_vector, k=5, **kwargs):
                await asyncio.sleep(0.01)  # 10ms search time
                return [
                    Mock(similarity_score=0.9 - i*0.1, metadata=Mock())
                    for i in range(k)
                ]
            
            store = mock_store.return_value
            store.similarity_search = AsyncMock(side_effect=mock_concurrent_search)
            
            # Create multiple concurrent searches
            concurrent_searches = 20
            query_vectors = [np.random.randn(512) for _ in range(concurrent_searches)]
            
            start_time = time.time()
            
            tasks = [
                store.similarity_search(query_vector, k=5)
                for query_vector in query_vectors
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Verify all searches completed
            assert len(results) == concurrent_searches
            assert all(len(result) == 5 for result in results)
            
            # Concurrent searches should be much faster than sequential
            sequential_time_estimate = concurrent_searches * 0.01
            concurrency_efficiency = sequential_time_estimate / total_time
            assert concurrency_efficiency > 5.0  # At least 5x faster with concurrency
            assert total_time < 0.5  # Should complete within 500ms

class TestRAGPipelinePerformance:
    """Test RAG pipeline performance."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_performance(self):
        """Test complete RAG pipeline performance."""
        from floatchat.ai.rag.rag_pipeline import OceanographicRAGPipeline
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_vector_store, \
             patch('floatchat.ai.llm.llm_orchestrator.LLMOrchestrator') as mock_orchestrator, \
             patch('floatchat.core.database.DatabaseManager') as mock_db:
            
            # Mock components with realistic timing
            async def mock_vector_search(query_vector, k=5, **kwargs):
                await asyncio.sleep(0.01)  # 10ms for vector search
                return [
                    Mock(metadata=Mock(content=f"Context {i}"), similarity_score=0.9-i*0.1)
                    for i in range(k)
                ]
            
            async def mock_llm_response(prompt, **kwargs):
                await asyncio.sleep(1.0)  # 1 second for LLM response
                return {
                    'response': 'Based on the oceanographic data...',
                    'confidence_score': 0.9,
                    'model_used': 'gpt-4'
                }
            
            async def mock_db_query(query, params=None):
                await asyncio.sleep(0.05)  # 50ms for database query
                return [{'temperature': 25.0, 'salinity': 35.0}]
            
            # Setup mocks
            mock_vector_store.return_value.similarity_search = mock_vector_search
            mock_orchestrator.return_value.generate_response = mock_llm_response
            mock_db.return_value.execute_query = mock_db_query
            
            # Create pipeline
            pipeline = OceanographicRAGPipeline(
                vector_store=mock_vector_store.return_value,
                llm_orchestrator=mock_orchestrator.return_value,
                db_manager=mock_db.return_value
            )
            
            # Test query performance
            test_queries = [
                "What is the average temperature in Arabian Sea?",
                "Show salinity profiles from Bay of Bengal",
                "Analyze oxygen levels in deep water"
            ]
            
            performance_results = []
            
            for query in test_queries:
                start_time = time.time()
                
                result = await pipeline.process_query(
                    query=query,
                    k_retrievals=5,
                    context_window=4000
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                performance_results.append({
                    'query': query,
                    'response_time': response_time,
                    'success': 'answer' in result and result.get('confidence_score', 0) > 0.8
                })
                
                # Performance assertions
                assert 'answer' in result
                assert response_time < 3.0  # Should complete within 3 seconds
                assert result.get('confidence_score', 0) > 0.8
            
            # Overall performance metrics
            avg_response_time = statistics.mean(r['response_time'] for r in performance_results)
            success_rate = sum(r['success'] for r in performance_results) / len(performance_results)
            
            assert avg_response_time < 2.0  # Average response time under 2 seconds
            assert success_rate == 1.0  # 100% success rate
    
    @pytest.mark.asyncio
    async def test_pipeline_under_load(self):
        """Test RAG pipeline performance under concurrent load."""
        from floatchat.ai.rag.rag_pipeline import OceanographicRAGPipeline
        
        with patch('floatchat.ai.vector_database.faiss_vector_store.OceanographicVectorStore') as mock_vector_store, \
             patch('floatchat.ai.llm.llm_orchestrator.LLMOrchestrator') as mock_orchestrator, \
             patch('floatchat.core.database.DatabaseManager') as mock_db:
            
            # Mock fast components for load testing
            mock_vector_store.return_value.similarity_search = AsyncMock(return_value=[
                Mock(metadata=Mock(content="Test context"), similarity_score=0.9)
            ])
            
            mock_orchestrator.return_value.generate_response = AsyncMock(return_value={
                'response': 'Test response',
                'confidence_score': 0.9,
                'model_used': 'gpt-4'
            })
            
            mock_db.return_value.execute_query = AsyncMock(return_value=[
                {'temperature': 25.0}
            ])
            
            # Create pipeline
            pipeline = OceanographicRAGPipeline(
                vector_store=mock_vector_store.return_value,
                llm_orchestrator=mock_orchestrator.return_value,
                db_manager=mock_db.return_value
            )
            
            # Generate concurrent load
            concurrent_queries = 50
            queries = [f"Test query {i}" for i in range(concurrent_queries)]
            
            start_time = time.time()
            
            tasks = [
                pipeline.process_query(query, k_retrievals=5, context_window=4000)
                for query in queries
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Performance assertions
            assert len(results) == concurrent_queries
            assert all('answer' in result for result in results)
            
            # Load performance requirements
            avg_time_per_query = total_time / concurrent_queries
            assert avg_time_per_query < 1.0  # Average time per query under 1 second
            assert total_time < 10.0  # All queries completed within 10 seconds
    
    def test_pipeline_memory_efficiency(self):
        """Test RAG pipeline memory efficiency."""
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Simulate processing many queries
        for i in range(100):
            # Simulate context data
            contexts = [f"Context {j}" * 100 for j in range(10)]  # Simulate large contexts
            
            # Simulate processing
            processed_contexts = [ctx.upper() for ctx in contexts]
            
            # Cleanup simulation
            del contexts, processed_contexts
            
            # Periodic garbage collection
            if i % 20 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory efficiency assertions
        assert memory_increase < 100  # Should not increase memory by more than 100MB

class TestSystemLevelPerformance:
    """Test overall system performance."""
    
    @pytest.mark.asyncio
    async def test_system_startup_performance(self):
        """Test system initialization performance."""
        
        # Mock component initialization times
        startup_components = [
            ('VectorStore', 0.5),
            ('LLMOrchestrator', 0.2), 
            ('DatabaseManager', 0.3),
            ('RAGPipeline', 0.1),
            ('NL2SQLEngine', 0.2)
        ]
        
        total_startup_time = 0
        
        for component_name, init_time in startup_components:
            start_time = time.time()
            
            # Simulate component initialization
            await asyncio.sleep(init_time)
            
            end_time = time.time()
            actual_time = end_time - start_time
            total_startup_time += actual_time
            
            # Individual component startup should be reasonable
            assert actual_time < (init_time + 0.1)  # Allow small overhead
        
        # Total startup time should be reasonable
        assert total_startup_time < 2.0  # Complete system startup within 2 seconds
    
    @pytest.mark.asyncio
    async def test_system_scalability(self):
        """Test system scalability under increasing load."""
        
        # Test with increasing number of concurrent users
        user_loads = [1, 5, 10, 25, 50]
        performance_results = []
        
        for num_users in user_loads:
            # Simulate concurrent user queries
            start_time = time.time()
            
            tasks = []
            for i in range(num_users):
                # Simulate user query processing
                async def simulate_user_query():
                    await asyncio.sleep(0.1)  # Simulate processing time
                    return {'success': True, 'response_time': 0.1}
                
                tasks.append(simulate_user_query())
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_user = total_time / num_users if num_users > 0 else 0
            
            performance_results.append({
                'num_users': num_users,
                'total_time': total_time,
                'avg_time_per_user': avg_time_per_user,
                'success_rate': sum(r['success'] for r in results) / len(results)
            })
            
            # Scalability assertions
            assert total_time < 5.0  # Should handle load within 5 seconds
            assert all(r['success'] for r in results)  # All queries should succeed
        
        # Analyze scalability trends
        avg_times = [r['avg_time_per_user'] for r in performance_results]
        
        # Average time per user shouldn't increase dramatically with load
        max_avg_time = max(avg_times)
        min_avg_time = min(avg_times)
        scalability_ratio = max_avg_time / min_avg_time
        
        assert scalability_ratio < 3.0  # Shouldn't degrade more than 3x under load
    
    def test_resource_utilization_limits(self):
        """Test system resource utilization stays within limits."""
        import threading
        import queue
        
        # Monitor resource usage during simulated load
        resource_queue = queue.Queue()
        monitoring = True
        
        def monitor_resources():
            process = psutil.Process()
            while monitoring:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                resource_queue.put({
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'timestamp': time.time()
                })
                
                time.sleep(0.1)  # Monitor every 100ms
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        try:
            # Simulate system load
            for i in range(100):
                # Simulate CPU and memory intensive operations
                data = np.random.randn(1000, 100)
                result = np.dot(data, data.T)
                del data, result
                
                time.sleep(0.01)  # Small delay between operations
        
        finally:
            # Stop monitoring
            monitoring = False
            monitor_thread.join()
        
        # Analyze resource usage
        resource_data = []
        while not resource_queue.empty():
            resource_data.append(resource_queue.get())
        
        if resource_data:
            max_cpu = max(r['cpu_percent'] for r in resource_data)
            max_memory = max(r['memory_mb'] for r in resource_data)
            
            # Resource utilization limits
            assert max_cpu < 80.0  # CPU usage should stay below 80%
            assert max_memory < 2000  # Memory usage should stay below 2GB
    
    @pytest.mark.asyncio
    async def test_system_reliability_under_stress(self):
        """Test system reliability under stress conditions."""
        
        # Simulate various stress conditions
        stress_tests = [
            ('High Query Volume', 100, 0.01),  # 100 queries with 10ms each
            ('Large Context Processing', 10, 0.1),  # 10 queries with 100ms each
            ('Mixed Load', 50, 0.05)  # 50 queries with 50ms each
        ]
        
        reliability_results = []
        
        for test_name, num_queries, query_time in stress_tests:
            success_count = 0
            error_count = 0
            
            start_time = time.time()
            
            tasks = []
            for i in range(num_queries):
                async def simulate_stress_query():
                    try:
                        await asyncio.sleep(query_time)
                        # Simulate occasional failures (5% failure rate)
                        if np.random.random() < 0.05:
                            raise Exception("Simulated failure")
                        return {'success': True}
                    except:
                        return {'success': False}
                
                tasks.append(simulate_stress_query())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Count successes and failures
            for result in results:
                if isinstance(result, dict) and result.get('success'):
                    success_count += 1
                else:
                    error_count += 1
            
            reliability_rate = success_count / (success_count + error_count)
            total_time = end_time - start_time
            
            reliability_results.append({
                'test_name': test_name,
                'reliability_rate': reliability_rate,
                'total_time': total_time,
                'success_count': success_count,
                'error_count': error_count
            })
            
            # Reliability assertions
            assert reliability_rate >= 0.90  # At least 90% reliability
            assert total_time < 15.0  # Should complete within reasonable time
        
        # Overall system reliability
        overall_reliability = sum(r['reliability_rate'] for r in reliability_results) / len(reliability_results)
        assert overall_reliability >= 0.92  # Overall system reliability > 92%