"""
Multi-modal embedding generation for oceanographic data.
"""

from .multi_modal_embeddings import (
    MultiModalEmbeddingGenerator,
    MultiModalData,
    EmbeddingMetadata,
    SpatialEncoder,
    TemporalEncoder,
    ParameterEncoder,
    EmbeddingFusion
)

__all__ = [
    'MultiModalEmbeddingGenerator',
    'MultiModalData', 
    'EmbeddingMetadata',
    'SpatialEncoder',
    'TemporalEncoder',
    'ParameterEncoder',
    'EmbeddingFusion'
]