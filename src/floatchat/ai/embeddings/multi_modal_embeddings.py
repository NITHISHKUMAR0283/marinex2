"""
Multi-modal embedding generation system for oceanographic data.
Combines textual, spatial, temporal, and parametric information for RAG system.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, date
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
import hashlib

# ML and embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    print(f"Missing ML dependencies: {e}")
    print("Install with: pip install sentence-transformers faiss-cpu scikit-learn torch")
    raise

from ...core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """Metadata for generated embeddings."""
    source_type: str  # 'float', 'profile', 'measurement', 'region'
    source_id: str
    embedding_version: str
    generation_timestamp: datetime
    model_name: str
    dimensions: int
    quality_score: float
    features_used: List[str]


@dataclass 
class MultiModalData:
    """Container for multi-modal data inputs."""
    # Text data
    text_description: Optional[str] = None
    location_name: Optional[str] = None
    parameter_descriptions: Optional[List[str]] = None
    
    # Spatial data  
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    region_code: Optional[str] = None
    
    # Temporal data
    measurement_date: Optional[datetime] = None
    season: Optional[str] = None
    year: Optional[int] = None
    day_of_year: Optional[int] = None
    
    # Parametric data
    temperature: Optional[float] = None
    salinity: Optional[float] = None
    pressure: Optional[float] = None
    oxygen: Optional[float] = None
    
    # Contextual data
    float_type: Optional[str] = None
    dac_name: Optional[str] = None
    project_name: Optional[str] = None
    data_quality: Optional[str] = None


class SpatialEncoder(nn.Module):
    """Neural network encoder for spatial information."""
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # Geographic coordinate encoding
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, 64),  # lat, lon, depth
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Regional encoding (learned embeddings for ocean regions)
        self.region_embeddings = nn.Embedding(50, 32)  # 50 ocean regions
        
        # Final spatial representation
        self.spatial_fusion = nn.Sequential(
            nn.Linear(64, output_dim),  # 32 + 32 = 64
            nn.Tanh()
        )
        
    def forward(self, coordinates: torch.Tensor, region_ids: torch.Tensor) -> torch.Tensor:
        """Encode spatial information."""
        coord_features = self.coord_encoder(coordinates)
        region_features = self.region_embeddings(region_ids)
        
        combined = torch.cat([coord_features, region_features], dim=-1)
        return self.spatial_fusion(combined)


class TemporalEncoder(nn.Module):
    """Neural network encoder for temporal patterns."""
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # Cyclical time encoding (sin/cos for seasonality)
        self.time_encoder = nn.Sequential(
            nn.Linear(8, 32),  # year, month, day, hour + 4 cyclical features
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Seasonal pattern encoder
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(4, 32),  # seasonal indicators
            nn.ReLU(),
        )
        
        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
        
    def encode_cyclical_time(self, dt: datetime) -> np.ndarray:
        """Encode datetime with cyclical features."""
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        
        # Cyclical encoding
        year_cycle = dt.year / 2024.0  # Normalized year
        day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        return np.array([
            year_cycle, dt.month / 12.0, dt.day / 31.0, hour / 24.0,
            day_sin, day_cos, hour_sin, hour_cos
        ])
    
    def forward(self, time_features: torch.Tensor, seasonal_features: torch.Tensor) -> torch.Tensor:
        """Encode temporal information."""
        time_encoded = self.time_encoder(time_features)
        seasonal_encoded = self.seasonal_encoder(seasonal_features)
        
        combined = torch.cat([time_encoded, seasonal_encoded], dim=-1)
        return self.temporal_fusion(combined)


class ParameterEncoder(nn.Module):
    """Neural network encoder for oceanographic parameters."""
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # Parameter relationship encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(8, 64),  # T, S, P, O2, + derived features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Quality and metadata encoder
        self.quality_encoder = nn.Sequential(
            nn.Linear(4, 16),  # Quality flags and confidence
            nn.ReLU(),
        )
        
        # Parameter fusion
        self.param_fusion = nn.Sequential(
            nn.Linear(48, output_dim),  # 32 + 16 = 48
            nn.Tanh()
        )
        
    def compute_derived_features(self, temp: float, sal: float, pres: float) -> np.ndarray:
        """Compute derived oceanographic features."""
        # Density approximation (simplified)
        density = 1025 + 0.7 * sal - 0.2 * temp + 0.0005 * pres
        
        # Buoyancy frequency approximation
        buoyancy = np.sqrt(max(0, (temp - 4.0) * 0.1))
        
        # Water mass indicators
        temp_sal_ratio = temp / max(sal, 1.0)
        stability_index = (temp - 4.0) / max(sal - 34.0, 0.1)
        
        return np.array([density, buoyancy, temp_sal_ratio, stability_index])
    
    def forward(self, param_features: torch.Tensor, quality_features: torch.Tensor) -> torch.Tensor:
        """Encode parameter information."""
        param_encoded = self.param_encoder(param_features)
        quality_encoded = self.quality_encoder(quality_features)
        
        combined = torch.cat([param_encoded, quality_encoded], dim=-1)
        return self.param_fusion(combined)


class EmbeddingFusion(nn.Module):
    """Neural network for fusing multi-modal embeddings."""
    
    def __init__(self, text_dim: int = 384, spatial_dim: int = 128, 
                 temporal_dim: int = 128, param_dim: int = 128, output_dim: int = 768):
        super().__init__()
        
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        # Attention-based fusion
        self.attention_weights = nn.Parameter(torch.randn(4, 1))  # 4 modalities
        
        # Modal projections to common dimension
        self.text_proj = nn.Linear(text_dim, output_dim // 4)
        self.spatial_proj = nn.Linear(spatial_dim, output_dim // 4)
        self.temporal_proj = nn.Linear(temporal_dim, output_dim // 4)
        self.param_proj = nn.Linear(param_dim, output_dim // 4)
        
        # Final fusion layers
        self.fusion_net = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, text_emb: torch.Tensor, spatial_emb: torch.Tensor,
                temporal_emb: torch.Tensor, param_emb: torch.Tensor) -> torch.Tensor:
        """Fuse multi-modal embeddings with learned attention."""
        
        # Project to common dimension
        text_proj = self.text_proj(text_emb)
        spatial_proj = self.spatial_proj(spatial_emb)
        temporal_proj = self.temporal_proj(temporal_emb)
        param_proj = self.param_proj(param_emb)
        
        # Concatenate projected embeddings
        fused = torch.cat([text_proj, spatial_proj, temporal_proj, param_proj], dim=-1)
        
        # Apply fusion network
        return self.fusion_net(fused)


class MultiModalEmbeddingGenerator:
    """Production multi-modal embedding generation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.settings = get_settings()
        
        # Initialize text encoder
        self.text_model_name = self.config.get('text_model', 'all-MiniLM-L6-v2')
        self.text_encoder = None
        
        # Initialize neural encoders
        self.spatial_encoder = SpatialEncoder(output_dim=128)
        self.temporal_encoder = TemporalEncoder(output_dim=128)
        self.parameter_encoder = ParameterEncoder(output_dim=128)
        self.fusion_network = EmbeddingFusion(
            text_dim=384,  # MiniLM dimension
            output_dim=768
        )
        
        # Scalers for normalization
        self.spatial_scaler = StandardScaler()
        self.temporal_scaler = StandardScaler()
        self.parameter_scaler = StandardScaler()
        
        # Ocean region mapping
        self.region_mapping = self._load_region_mapping()
        
        # Performance tracking
        self.generation_stats = {
            'total_generated': 0,
            'avg_generation_time': 0.0,
            'quality_scores': []
        }
        
        logger.info("MultiModalEmbeddingGenerator initialized")
    
    async def initialize(self):
        """Initialize heavy models asynchronously."""
        try:
            logger.info("Loading sentence transformer model...")
            self.text_encoder = SentenceTransformer(self.text_model_name)
            logger.info(f"Text encoder loaded: {self.text_model_name}")
            
            # Load pre-trained neural encoders if available
            await self._load_pretrained_encoders()
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
            raise
    
    def _load_region_mapping(self) -> Dict[str, int]:
        """Load ocean region to ID mapping."""
        # Indian Ocean regions mapping
        return {
            'central_arabian_sea': 0,
            'northern_arabian_sea': 1,
            'western_arabian_sea': 2,
            'somali_coast': 3,
            'central_bay_of_bengal': 4,
            'northern_bay_of_bengal': 5,
            'southern_bay_of_bengal': 6,
            'eastern_bay_of_bengal': 7,
            'central_indian_basin': 8,
            'tropical_indian_ocean': 9,
            'equatorial_indian_ocean': 10,
            'southwest_indian_ocean': 11,
            'south_central_indian_ocean': 12,
            'southeast_indian_ocean': 13,
            'andaman_sea': 14,
            'lakshadweep_sea': 15,
            'ceylon_basin': 16,
            'maldives_region': 17,
            'unknown_region': 18
        }
    
    async def _load_pretrained_encoders(self):
        """Load pre-trained neural encoders if available."""
        model_dir = Path(self.settings.model_cache_dir) if hasattr(self.settings, 'model_cache_dir') else Path("models")
        model_dir.mkdir(exist_ok=True)
        
        encoder_paths = {
            'spatial': model_dir / 'spatial_encoder.pth',
            'temporal': model_dir / 'temporal_encoder.pth', 
            'parameter': model_dir / 'parameter_encoder.pth',
            'fusion': model_dir / 'fusion_network.pth'
        }
        
        # Load if models exist
        for name, path in encoder_paths.items():
            if path.exists():
                try:
                    if name == 'spatial':
                        self.spatial_encoder.load_state_dict(torch.load(path, map_location='cpu'))
                    elif name == 'temporal':
                        self.temporal_encoder.load_state_dict(torch.load(path, map_location='cpu'))
                    elif name == 'parameter':
                        self.parameter_encoder.load_state_dict(torch.load(path, map_location='cpu'))
                    elif name == 'fusion':
                        self.fusion_network.load_state_dict(torch.load(path, map_location='cpu'))
                    logger.info(f"Loaded pre-trained {name} encoder")
                except Exception as e:
                    logger.warning(f"Failed to load {name} encoder: {e}")
    
    def _encode_text(self, data: MultiModalData) -> np.ndarray:
        """Encode textual information."""
        text_parts = []
        
        if data.text_description:
            text_parts.append(data.text_description)
        
        if data.location_name:
            text_parts.append(f"Location: {data.location_name}")
        
        if data.region_code:
            text_parts.append(f"Ocean region: {data.region_code}")
        
        if data.float_type:
            text_parts.append(f"Float type: {data.float_type}")
        
        if data.project_name:
            text_parts.append(f"Project: {data.project_name}")
        
        if data.parameter_descriptions:
            text_parts.extend(data.parameter_descriptions)
        
        # Create comprehensive text representation
        full_text = ". ".join(text_parts) if text_parts else "Oceanographic measurement data"
        
        # Generate embedding
        text_embedding = self.text_encoder.encode(full_text, convert_to_numpy=True)
        return text_embedding
    
    def _encode_spatial(self, data: MultiModalData) -> np.ndarray:
        """Encode spatial information."""
        # Prepare coordinate data
        lat = data.latitude if data.latitude is not None else 0.0
        lon = data.longitude if data.longitude is not None else 0.0
        depth = data.depth if data.depth is not None else 0.0
        
        # Normalize coordinates
        lat_norm = lat / 90.0  # [-1, 1]
        lon_norm = lon / 180.0  # [-1, 1] 
        depth_norm = min(depth / 6000.0, 1.0)  # [0, 1] for depths up to 6000m
        
        coordinates = torch.tensor([[lat_norm, lon_norm, depth_norm]], dtype=torch.float32)
        
        # Region encoding
        region_key = data.region_code.lower().replace(' ', '_') if data.region_code else 'unknown_region'
        region_id = self.region_mapping.get(region_key, 18)  # 18 = unknown
        region_tensor = torch.tensor([region_id], dtype=torch.long)
        
        # Generate spatial embedding
        with torch.no_grad():
            spatial_embedding = self.spatial_encoder(coordinates, region_tensor)
        
        return spatial_embedding.numpy().flatten()
    
    def _encode_temporal(self, data: MultiModalData) -> np.ndarray:
        """Encode temporal information."""
        if data.measurement_date:
            # Use provided date
            dt = data.measurement_date if isinstance(data.measurement_date, datetime) else datetime.combine(data.measurement_date, datetime.min.time())
        else:
            # Use current date as fallback
            dt = datetime.now()
        
        # Encode cyclical time features
        time_features = self.temporal_encoder.encode_cyclical_time(dt)
        
        # Seasonal features
        month = dt.month
        seasonal_features = np.array([
            1.0 if month in [12, 1, 2] else 0.0,  # Winter
            1.0 if month in [3, 4, 5] else 0.0,   # Spring
            1.0 if month in [6, 7, 8] else 0.0,   # Summer (Monsoon)
            1.0 if month in [9, 10, 11] else 0.0  # Autumn (Post-monsoon)
        ])
        
        time_tensor = torch.tensor(time_features, dtype=torch.float32).unsqueeze(0)
        seasonal_tensor = torch.tensor(seasonal_features, dtype=torch.float32).unsqueeze(0)
        
        # Generate temporal embedding
        with torch.no_grad():
            temporal_embedding = self.temporal_encoder(time_tensor, seasonal_tensor)
        
        return temporal_embedding.numpy().flatten()
    
    def _encode_parameters(self, data: MultiModalData) -> np.ndarray:
        """Encode oceanographic parameters."""
        # Extract parameters with defaults
        temp = data.temperature if data.temperature is not None else 15.0
        sal = data.salinity if data.salinity is not None else 34.7
        pres = data.pressure if data.pressure is not None else 1000.0
        oxy = data.oxygen if data.oxygen is not None else 200.0
        
        # Compute derived features
        derived = self.parameter_encoder.compute_derived_features(temp, sal, pres)
        
        # Combine all parameter features
        param_features = np.array([temp, sal, pres, oxy] + derived.tolist())
        
        # Quality features
        quality_score = 1.0 if data.data_quality == '1' else (0.8 if data.data_quality == '2' else 0.5)
        completeness = sum([
            1.0 if data.temperature is not None else 0.0,
            1.0 if data.salinity is not None else 0.0,
            1.0 if data.pressure is not None else 0.0,
            1.0 if data.oxygen is not None else 0.0
        ]) / 4.0
        
        quality_features = np.array([quality_score, completeness, 1.0, 1.0])  # Padding
        
        param_tensor = torch.tensor(param_features, dtype=torch.float32).unsqueeze(0)
        quality_tensor = torch.tensor(quality_features, dtype=torch.float32).unsqueeze(0)
        
        # Generate parameter embedding
        with torch.no_grad():
            param_embedding = self.parameter_encoder(param_tensor, quality_tensor)
        
        return param_embedding.numpy().flatten()
    
    async def generate_embedding(self, data: MultiModalData, 
                               metadata: Optional[Dict] = None) -> Tuple[np.ndarray, EmbeddingMetadata]:
        """Generate multi-modal embedding for oceanographic data."""
        start_time = datetime.now()
        
        try:
            # Generate individual modality embeddings
            text_emb = self._encode_text(data)
            spatial_emb = self._encode_spatial(data)
            temporal_emb = self._encode_temporal(data)
            param_emb = self._encode_parameters(data)
            
            # Convert to tensors for fusion
            text_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0)
            spatial_tensor = torch.tensor(spatial_emb, dtype=torch.float32).unsqueeze(0)
            temporal_tensor = torch.tensor(temporal_emb, dtype=torch.float32).unsqueeze(0)
            param_tensor = torch.tensor(param_emb, dtype=torch.float32).unsqueeze(0)
            
            # Fuse embeddings
            with torch.no_grad():
                fused_embedding = self.fusion_network(
                    text_tensor, spatial_tensor, temporal_tensor, param_tensor
                )
            
            final_embedding = fused_embedding.numpy().flatten()
            
            # Calculate quality score based on data completeness
            quality_score = self._calculate_quality_score(data, [text_emb, spatial_emb, temporal_emb, param_emb])
            
            # Create metadata
            embedding_metadata = EmbeddingMetadata(
                source_type=metadata.get('source_type', 'unknown') if metadata else 'unknown',
                source_id=metadata.get('source_id', 'unknown') if metadata else 'unknown',
                embedding_version="v1.0.0",
                generation_timestamp=datetime.now(),
                model_name=f"{self.text_model_name}+custom_fusion",
                dimensions=len(final_embedding),
                quality_score=quality_score,
                features_used=['text', 'spatial', 'temporal', 'parameters']
            )
            
            # Update stats
            generation_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(generation_time, quality_score)
            
            logger.debug(f"Generated embedding: {len(final_embedding)}D, quality: {quality_score:.3f}")
            return final_embedding, embedding_metadata
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _calculate_quality_score(self, data: MultiModalData, embeddings: List[np.ndarray]) -> float:
        """Calculate embedding quality score based on data completeness and embedding properties."""
        
        # Data completeness score
        total_fields = 0
        filled_fields = 0
        
        for field_name, field_value in asdict(data).items():
            total_fields += 1
            if field_value is not None:
                filled_fields += 1
        
        completeness_score = filled_fields / total_fields
        
        # Embedding quality metrics
        embedding_quality = 0.0
        for emb in embeddings:
            # Check for NaN or infinite values
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                embedding_quality -= 0.2
            else:
                # Measure embedding diversity (avoid all-zeros or constant values)
                variance = np.var(emb)
                if variance > 1e-6:
                    embedding_quality += 0.25
        
        # Combined quality score
        quality = 0.6 * completeness_score + 0.4 * max(0.0, embedding_quality)
        return min(1.0, max(0.0, quality))
    
    def _update_stats(self, generation_time: float, quality_score: float):
        """Update generation statistics."""
        self.generation_stats['total_generated'] += 1
        
        # Update average generation time
        prev_avg = self.generation_stats['avg_generation_time']
        total = self.generation_stats['total_generated']
        new_avg = (prev_avg * (total - 1) + generation_time) / total
        self.generation_stats['avg_generation_time'] = new_avg
        
        # Track quality scores
        self.generation_stats['quality_scores'].append(quality_score)
        
        # Keep only recent quality scores (last 1000)
        if len(self.generation_stats['quality_scores']) > 1000:
            self.generation_stats['quality_scores'] = self.generation_stats['quality_scores'][-1000:]
    
    async def batch_generate_embeddings(self, data_list: List[Tuple[MultiModalData, Dict]], 
                                      batch_size: int = 32) -> List[Tuple[np.ndarray, EmbeddingMetadata]]:
        """Generate embeddings in batches for efficiency."""
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            batch_results = []
            
            # Process batch concurrently
            tasks = [self.generate_embedding(data, metadata) for data, metadata in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_embeddings:
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding failed: {result}")
                    continue
                batch_results.append(result)
            
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(data_list)-1)//batch_size + 1}")
        
        return results
    
    def get_stats(self) -> Dict:
        """Get generation statistics."""
        stats = self.generation_stats.copy()
        if stats['quality_scores']:
            stats['avg_quality'] = np.mean(stats['quality_scores'])
            stats['min_quality'] = np.min(stats['quality_scores'])
            stats['max_quality'] = np.max(stats['quality_scores'])
        return stats
    
    async def save_model_state(self, save_dir: Path):
        """Save trained neural encoder states."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            torch.save(self.spatial_encoder.state_dict(), save_dir / 'spatial_encoder.pth')
            torch.save(self.temporal_encoder.state_dict(), save_dir / 'temporal_encoder.pth')
            torch.save(self.parameter_encoder.state_dict(), save_dir / 'parameter_encoder.pth')
            torch.save(self.fusion_network.state_dict(), save_dir / 'fusion_network.pth')
            
            # Save generation statistics
            with open(save_dir / 'generation_stats.json', 'w') as f:
                json.dump(self.get_stats(), f, indent=2)
                
            logger.info(f"Model state saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")
            raise