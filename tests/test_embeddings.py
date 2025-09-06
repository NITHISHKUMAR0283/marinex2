"""
Unit Tests for Multi-Modal Embeddings System
Tests embedding generation, fusion, and quality assessment
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from floatchat.ai.embeddings.multi_modal_embeddings import (
    OceanographicEmbeddingGenerator,
    TextEmbeddingProcessor,
    SpatialEmbeddingProcessor, 
    TemporalEmbeddingProcessor,
    ParametricEmbeddingProcessor,
    NeuralFusionLayer,
    EmbeddingQualityAssessor
)

class TestOceanographicEmbeddingGenerator:
    """Test the main embedding generator."""
    
    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator for testing."""
        return OceanographicEmbeddingGenerator(
            text_model="text-embedding-3-small",
            fusion_dimension=512
        )
    
    @pytest.fixture
    def sample_measurement(self):
        """Sample oceanographic measurement."""
        return {
            'wmo_id': '2902746',
            'profile_id': '20230415_001',
            'latitude': 15.5,
            'longitude': 68.2,
            'depth': 250.0,
            'measurement_date': '2023-04-15T12:30:00Z',
            'temperature': 18.5,
            'salinity': 35.2,
            'pressure': 25.0,
            'oxygen': 180.5,
            'description': 'Temperature and salinity profile in Arabian Sea'
        }
    
    def test_initialization(self, embedding_generator):
        """Test proper initialization of embedding generator."""
        assert embedding_generator.text_model == "text-embedding-3-small"
        assert embedding_generator.fusion_dimension == 512
        assert isinstance(embedding_generator.text_processor, TextEmbeddingProcessor)
        assert isinstance(embedding_generator.spatial_processor, SpatialEmbeddingProcessor)
        assert isinstance(embedding_generator.temporal_processor, TemporalEmbeddingProcessor)
        assert isinstance(embedding_generator.parametric_processor, ParametricEmbeddingProcessor)
        assert isinstance(embedding_generator.fusion_layer, NeuralFusionLayer)
        assert isinstance(embedding_generator.quality_assessor, EmbeddingQualityAssessor)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, embedding_generator, sample_measurement):
        """Test end-to-end embedding generation."""
        # Mock the individual processors
        with patch.object(embedding_generator.text_processor, 'process', new_callable=AsyncMock) as mock_text, \
             patch.object(embedding_generator.spatial_processor, 'process', new_callable=AsyncMock) as mock_spatial, \
             patch.object(embedding_generator.temporal_processor, 'process', new_callable=AsyncMock) as mock_temporal, \
             patch.object(embedding_generator.parametric_processor, 'process', new_callable=AsyncMock) as mock_parametric:
            
            # Setup mock returns
            mock_text.return_value = np.random.randn(512)
            mock_spatial.return_value = np.random.randn(128)
            mock_temporal.return_value = np.random.randn(64)
            mock_parametric.return_value = np.random.randn(256)
            
            result = await embedding_generator.generate_embeddings([sample_measurement])
            
            assert len(result) == 1
            assert 'embedding' in result[0]
            assert 'metadata' in result[0]
            assert len(result[0]['embedding']) == 512  # fusion_dimension
            
            # Verify all processors were called
            mock_text.assert_called_once()
            mock_spatial.assert_called_once()
            mock_temporal.assert_called_once()
            mock_parametric.assert_called_once()
    
    def test_invalid_input(self, embedding_generator):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            asyncio.run(embedding_generator.generate_embeddings([]))
        
        with pytest.raises((TypeError, KeyError)):
            asyncio.run(embedding_generator.generate_embeddings([{"invalid": "data"}]))

class TestTextEmbeddingProcessor:
    """Test text embedding processing."""
    
    @pytest.fixture
    def text_processor(self):
        """Create text processor for testing."""
        return TextEmbeddingProcessor(model_name="text-embedding-3-small")
    
    @pytest.mark.asyncio
    async def test_text_processing(self, text_processor, sample_measurement):
        """Test text embedding generation."""
        with patch('openai.embeddings.create') as mock_openai:
            # Mock OpenAI response
            mock_openai.return_value = Mock()
            mock_openai.return_value.data = [Mock()]
            mock_openai.return_value.data[0].embedding = [0.1] * 1536
            
            result = await text_processor.process(sample_measurement)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == 1536
            mock_openai.assert_called_once()
    
    def test_text_construction(self, text_processor, sample_measurement):
        """Test oceanographic text construction."""
        text = text_processor._construct_oceanographic_text(sample_measurement)
        
        assert "Arabian Sea" in text
        assert "2902746" in text
        assert "18.5°C" in text
        assert "35.2 PSU" in text
        assert "250.0m" in text

class TestSpatialEmbeddingProcessor:
    """Test spatial embedding processing."""
    
    @pytest.fixture
    def spatial_processor(self):
        """Create spatial processor for testing."""
        return SpatialEmbeddingProcessor()
    
    def test_spatial_encoding(self, spatial_processor, sample_measurement):
        """Test spatial coordinate encoding."""
        result = spatial_processor.process(sample_measurement)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 128  # Expected spatial embedding dimension
        
        # Test coordinate normalization
        lat_norm = sample_measurement['latitude'] / 90.0
        lon_norm = sample_measurement['longitude'] / 180.0
        depth_norm = sample_measurement['depth'] / 6000.0
        
        # Verify normalization is reasonable
        assert -1 <= lat_norm <= 1
        assert -1 <= lon_norm <= 1
        assert 0 <= depth_norm <= 1
    
    def test_ocean_region_detection(self, spatial_processor, sample_measurement):
        """Test ocean region detection."""
        region = spatial_processor._detect_ocean_region(
            sample_measurement['latitude'],
            sample_measurement['longitude']
        )
        
        assert region in ["arabian_sea", "bay_of_bengal", "southern_indian_ocean", "equatorial_indian_ocean"]
        # For coordinates (15.5, 68.2), should be Arabian Sea
        assert region == "arabian_sea"

class TestTemporalEmbeddingProcessor:
    """Test temporal embedding processing."""
    
    @pytest.fixture
    def temporal_processor(self):
        """Create temporal processor for testing."""
        return TemporalEmbeddingProcessor()
    
    def test_temporal_encoding(self, temporal_processor, sample_measurement):
        """Test temporal encoding."""
        result = temporal_processor.process(sample_measurement)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 64  # Expected temporal embedding dimension
        
        # Verify encoding captures different time aspects
        assert np.any(result != 0)  # Should not be all zeros
    
    def test_seasonal_patterns(self, temporal_processor):
        """Test seasonal pattern detection."""
        # Test different seasons
        dates = [
            datetime(2023, 3, 15),  # Spring
            datetime(2023, 6, 15),  # Summer 
            datetime(2023, 9, 15),  # Monsoon
            datetime(2023, 12, 15)  # Winter
        ]
        
        embeddings = []
        for date in dates:
            measurement = {
                'measurement_date': date.isoformat() + 'Z'
            }
            embedding = temporal_processor.process(measurement)
            embeddings.append(embedding)
        
        # Seasonal embeddings should be different
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not np.allclose(embeddings[i], embeddings[j])

class TestParametricEmbeddingProcessor:
    """Test parametric embedding processing."""
    
    @pytest.fixture  
    def parametric_processor(self):
        """Create parametric processor for testing."""
        return ParametricEmbeddingProcessor()
    
    def test_parametric_encoding(self, parametric_processor, sample_measurement):
        """Test parametric value encoding."""
        result = parametric_processor.process(sample_measurement)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 256  # Expected parametric embedding dimension
        
        # Should encode all available parameters
        assert np.any(result != 0)
    
    def test_parameter_normalization(self, parametric_processor):
        """Test parameter value normalization."""
        # Test with extreme values
        test_cases = [
            {'temperature': -2.0},  # Minimum
            {'temperature': 35.0},  # Maximum
            {'salinity': 30.0},     # Low salinity
            {'salinity': 40.0},     # High salinity
            {'oxygen': 0.0},        # Anoxic
            {'oxygen': 400.0}       # Supersaturated
        ]
        
        for case in test_cases:
            result = parametric_processor.process(case)
            assert np.all(np.isfinite(result))  # No NaN or infinity
            assert np.any(result != 0)  # Should produce non-zero encoding
    
    def test_missing_parameters(self, parametric_processor):
        """Test handling of missing parameters."""
        incomplete_measurement = {
            'temperature': 20.0
            # Missing other parameters
        }
        
        result = parametric_processor.process(incomplete_measurement)
        assert isinstance(result, np.ndarray)
        assert len(result) == 256
        assert np.all(np.isfinite(result))

class TestNeuralFusionLayer:
    """Test neural fusion layer."""
    
    @pytest.fixture
    def fusion_layer(self):
        """Create fusion layer for testing."""
        return NeuralFusionLayer(
            text_dim=1536,
            spatial_dim=128, 
            temporal_dim=64,
            parametric_dim=256,
            fusion_dim=512
        )
    
    def test_fusion_initialization(self, fusion_layer):
        """Test fusion layer initialization."""
        assert fusion_layer.text_dim == 1536
        assert fusion_layer.spatial_dim == 128
        assert fusion_layer.temporal_dim == 64
        assert fusion_layer.parametric_dim == 256
        assert fusion_layer.fusion_dim == 512
        
        # Check neural network components exist
        assert hasattr(fusion_layer, 'attention_weights')
        assert hasattr(fusion_layer, 'fusion_network')
    
    def test_fusion_process(self, fusion_layer):
        """Test fusion of multiple embeddings."""
        # Create sample embeddings
        text_embedding = np.random.randn(1536)
        spatial_embedding = np.random.randn(128)
        temporal_embedding = np.random.randn(64)
        parametric_embedding = np.random.randn(256)
        
        result = fusion_layer.fuse(
            text_embedding,
            spatial_embedding, 
            temporal_embedding,
            parametric_embedding
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 512
        assert np.all(np.isfinite(result))
    
    def test_attention_mechanism(self, fusion_layer):
        """Test attention mechanism in fusion."""
        # Create embeddings with different magnitudes
        text_embedding = np.random.randn(1536) * 0.1  # Small magnitude
        spatial_embedding = np.random.randn(128) * 1.0  # Large magnitude
        temporal_embedding = np.random.randn(64) * 0.5
        parametric_embedding = np.random.randn(256) * 2.0  # Largest magnitude
        
        result1 = fusion_layer.fuse(text_embedding, spatial_embedding, temporal_embedding, parametric_embedding)
        
        # Swap magnitudes
        text_embedding_large = np.random.randn(1536) * 2.0
        spatial_embedding_small = np.random.randn(128) * 0.1
        
        result2 = fusion_layer.fuse(text_embedding_large, spatial_embedding_small, temporal_embedding, parametric_embedding)
        
        # Results should be different due to attention mechanism
        assert not np.allclose(result1, result2)

class TestEmbeddingQualityAssessor:
    """Test embedding quality assessment."""
    
    @pytest.fixture
    def quality_assessor(self):
        """Create quality assessor for testing."""
        return EmbeddingQualityAssessor()
    
    def test_quality_assessment(self, quality_assessor):
        """Test embedding quality scoring."""
        # Good quality embedding (normal distribution)
        good_embedding = np.random.randn(512)
        good_score = quality_assessor.assess_quality(good_embedding)
        
        # Poor quality embedding (all zeros)
        poor_embedding = np.zeros(512)
        poor_score = quality_assessor.assess_quality(poor_embedding)
        
        # Bad quality embedding (all same value)
        bad_embedding = np.ones(512) * 5.0
        bad_score = quality_assessor.assess_quality(bad_embedding)
        
        assert 0 <= good_score <= 1
        assert 0 <= poor_score <= 1
        assert 0 <= bad_score <= 1
        
        # Good embedding should score higher than poor/bad
        assert good_score > poor_score
        assert good_score > bad_score
    
    def test_diversity_score(self, quality_assessor):
        """Test embedding diversity scoring."""
        # High diversity embeddings
        diverse_embeddings = [
            np.random.randn(512),
            np.random.randn(512) * 2,
            np.random.randn(512) * 0.5
        ]
        
        # Low diversity embeddings (similar)
        base = np.random.randn(512)
        similar_embeddings = [
            base,
            base + np.random.randn(512) * 0.01,  # Very similar
            base + np.random.randn(512) * 0.02   # Very similar
        ]
        
        diverse_score = quality_assessor.assess_diversity(diverse_embeddings)
        similar_score = quality_assessor.assess_diversity(similar_embeddings)
        
        assert diverse_score > similar_score
    
    def test_coherence_score(self, quality_assessor):
        """Test embedding coherence scoring."""
        # Coherent embeddings (related oceanographic data)
        coherent_data = [
            {'description': 'Temperature profile in Arabian Sea', 'temperature': 25.0},
            {'description': 'Salinity measurement in Arabian Sea', 'salinity': 35.5},
            {'description': 'Oxygen levels in Arabian Sea', 'oxygen': 180.0}
        ]
        
        # Incoherent embeddings (unrelated data)
        incoherent_data = [
            {'description': 'Temperature in Arctic Ocean', 'temperature': -1.5},
            {'description': 'Deep ocean trench measurement', 'depth': 5000.0},
            {'description': 'Surface water in tropical Pacific', 'temperature': 30.0}
        ]
        
        # For testing, we'll create mock embeddings
        coherent_embeddings = [np.random.randn(512) + np.array([1.0] * 512) for _ in coherent_data]
        incoherent_embeddings = [np.random.randn(512) * (i+1) for i in range(len(incoherent_data))]
        
        coherent_score = quality_assessor.assess_coherence(coherent_embeddings, coherent_data)
        incoherent_score = quality_assessor.assess_coherence(incoherent_embeddings, incoherent_data)
        
        # Coherent embeddings should score higher
        assert coherent_score >= incoherent_score