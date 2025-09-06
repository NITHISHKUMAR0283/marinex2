"""
Simplified LLM Orchestrator using only Groq (free models).
Optimized for Smart India Hackathon 2025 with zero API cost.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import time

from .groq_client import GroqClient, LLMResponse

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    api_key: str
    model_name: str = "llama-3.1-70b-versatile"
    max_tokens: int = 8192
    temperature: float = 0.2

@dataclass
class OrchestratedResponse:
    """Complete orchestrated response with metadata."""
    query: str
    answer: str
    model_used: str
    total_time: float
    confidence_score: float
    metadata: Dict[str, Any]

class LLMOrchestrator:
    """Simplified LLM orchestrator using only Groq free models."""
    
    def __init__(self, llm_configs: Dict[LLMProvider, LLMConfig]):
        """Initialize orchestrator with Groq configuration."""
        self.llm_configs = llm_configs
        
        # Initialize Groq client
        groq_config = llm_configs.get(LLMProvider.GROQ)
        if not groq_config:
            raise ValueError("Groq configuration is required")
        
        self.groq_client = GroqClient({
            'api_key': groq_config.api_key,
            'model': groq_config.model_name,
            'max_tokens': groq_config.max_tokens,
            'temperature': groq_config.temperature
        })
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }
        
        logger.info(f"LLM Orchestrator initialized with Groq model: {groq_config.model_name}")
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Groq."""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Use Groq client to generate response
            response = await self.groq_client.generate_response(
                prompt=prompt,
                context=context,
                **kwargs
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Update statistics
            self.stats['successful_requests'] += 1
            self.stats['total_response_time'] += response_time
            self.stats['avg_response_time'] = self.stats['total_response_time'] / self.stats['total_requests']
            
            # Return standardized response
            return {
                'response': response.content,
                'model_used': response.model,
                'confidence_score': response.confidence_score,
                'response_time': response_time,
                'tokens_used': response.tokens_used,
                'metadata': {
                    'provider': 'groq',
                    'reasoning_steps': response.reasoning_steps,
                    'citations': response.citations,
                    **response.metadata
                }
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            self.stats['failed_requests'] += 1
            
            logger.error(f"Failed to generate response with Groq: {str(e)}")
            
            # Return error response
            return {
                'response': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                'model_used': 'groq-error',
                'confidence_score': 0.0,
                'response_time': response_time,
                'tokens_used': 0,
                'metadata': {
                    'provider': 'groq',
                    'error': str(e),
                    'reasoning_steps': [],
                    'citations': []
                }
            }
    
    async def process_oceanographic_query(
        self,
        query: str,
        retrieved_contexts: Optional[List[Dict]] = None,
        **kwargs
    ) -> OrchestratedResponse:
        """Process oceanographic query with context."""
        start_time = time.time()
        
        # Prepare context from retrieved documents
        context = ""
        if retrieved_contexts:
            context_parts = []
            for i, ctx in enumerate(retrieved_contexts[:5], 1):  # Limit to top 5
                content = ctx.get('content', '')
                if content:
                    context_parts.append(f"Context {i}: {content}")
            context = "\n\n".join(context_parts)
        
        # Generate response
        response = await self.generate_response(
            prompt=query,
            context=context,
            **kwargs
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return OrchestratedResponse(
            query=query,
            answer=response['response'],
            model_used=response['model_used'],
            total_time=total_time,
            confidence_score=response['confidence_score'],
            metadata=response['metadata']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self.stats,
            'success_rate': (self.stats['successful_requests'] / max(self.stats['total_requests'], 1)) * 100,
            'groq_model': self.llm_configs[LLMProvider.GROQ].model_name,
            'provider': 'groq_only'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM services."""
        try:
            # Test Groq with a simple query
            test_response = await self.generate_response(
                prompt="Test query: What is oceanography?",
                context=None
            )
            
            groq_healthy = test_response['confidence_score'] > 0.0
            
            return {
                'groq': {
                    'status': 'healthy' if groq_healthy else 'unhealthy',
                    'model': self.llm_configs[LLMProvider.GROQ].model_name,
                    'response_time': test_response['response_time']
                },
                'overall_status': 'healthy' if groq_healthy else 'unhealthy',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'groq': {
                    'status': 'unhealthy',
                    'error': str(e)
                },
                'overall_status': 'unhealthy',
                'timestamp': datetime.now().isoformat()
            }

# For backward compatibility
QueryComplexityAnalyzer = None  # Not needed for single provider