"""
Groq client for fast inference with open-source LLMs (Llama, Mixtral).
Optimized for oceanographic queries with high-speed processing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from datetime import datetime
import json
import time
from dataclasses import dataclass
import backoff

try:
    import openai  # Groq uses OpenAI-compatible API
    from openai import AsyncOpenAI
except ImportError as e:
    print(f"OpenAI library required for Groq: {e}")
    print("Install with: pip install openai")
    raise

from ...core.config import get_settings
from ..rag import RAGContext
from .openai_client import LLMResponse, OceanographicPromptEngineering

logger = logging.getLogger(__name__)


class GroqClient:
    """Production Groq client for fast LLM inference with oceanographic optimization."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.settings = get_settings()
        
        # Initialize Groq client (OpenAI-compatible)
        api_key = self.config.get('api_key', os.getenv('GROQ_API_KEY', ''))
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=self.config.get('timeout', 30.0),  # Groq is fast
            max_retries=0  # We handle retries manually
        )
        
        # Model configuration optimized for different use cases
        self.model_configs = {
            'fast': {
                'model': 'llama-3.1-8b-instant',
                'max_tokens': 8000,
                'temperature': 0.2,
                'use_case': 'Quick responses, simple queries'
            },
            'balanced': {
                'model': 'llama-3.3-70b-versatile', 
                'max_tokens': 32768,
                'temperature': 0.3,
                'use_case': 'Complex analysis, reasoning'
            },
            'guard': {
                'model': 'meta-llama/llama-guard-4-12b',
                'max_tokens': 1024,
                'temperature': 0.1,
                'use_case': 'Safety and content filtering'
            }
        }
        
        # Default configuration
        self.default_model_tier = self.config.get('default_tier', 'balanced')
        default_config = self.model_configs[self.default_model_tier]
        self.default_model = default_config['model']
        self.max_tokens = default_config['max_tokens']
        self.temperature = default_config['temperature']
        
        # Advanced features
        self.prompt_engineer = OceanographicPromptEngineering()
        
        # Performance tracking
        self.usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'avg_response_time': 0.0,
            'error_count': 0,
            'cache_hits': 0,
            'model_usage': {model: 0 for model in self.model_configs.keys()}
        }
        
        # Response cache
        self.response_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 1800)  # 30 minutes
        
        logger.info(f"Groq client initialized with default model: {self.default_model}")
    
    def _cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for response caching."""
        import hashlib
        cache_data = f"{prompt}_{model}_{sorted(kwargs.items())}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cached response is still valid."""
        if not cache_entry:
            return False
        return (time.time() - cache_entry['timestamp']) < self.cache_ttl
    
    def _select_optimal_model(self, query: str, context: Optional[RAGContext] = None) -> str:
        """Select optimal Groq model based on query characteristics."""
        query_lower = query.lower()
        
        # Use fast model for simple queries
        simple_indicators = ['show', 'list', 'get', 'find', 'what is']
        if any(indicator in query_lower for indicator in simple_indicators) and len(query.split()) < 10:
            return self.model_configs['fast']['model']
        
        # Use balanced model for complex analysis
        complex_indicators = ['analyze', 'compare', 'trend', 'correlation', 'statistics', 'pattern']
        if any(indicator in query_lower for indicator in complex_indicators):
            return self.model_configs['balanced']['model']
        
        # Use guard model for content filtering (if needed)
        if self.config.get('enable_safety_check', False):
            safety_check_needed = any(word in query_lower for word in ['sensitive', 'personal', 'private'])
            if safety_check_needed:
                return self.model_configs['guard']['model']
        
        # Default to balanced model
        return self.model_configs['balanced']['model']
    
    def _build_llama_optimized_prompt(self, query: str, context: Optional[RAGContext] = None, 
                                    user_type: str = 'researcher') -> List[Dict]:
        """Build Llama-optimized prompt structure."""
        
        # Llama models work better with specific formatting
        system_prompt = self.prompt_engineer.get_system_prompt(user_type)
        
        # Enhanced system prompt for Llama
        llama_system_additions = """
RESPONSE GUIDELINES FOR LLAMA:
- Provide clear, structured responses with numbered points when appropriate
- Use specific scientific terminology accurately
- Include uncertainty quantification when discussing data
- Format numerical data clearly with appropriate units
- Conclude with actionable insights or recommendations

LLAMA REASONING APPROACH:
- Think step-by-step through complex oceanographic problems
- Explain your reasoning process clearly
- Connect different oceanographic concepts logically
- Validate conclusions against known oceanographic principles
"""
        
        enhanced_system = system_prompt + "\n\n" + llama_system_additions
        
        messages = [{"role": "system", "content": enhanced_system}]
        
        # Add context if available
        if context and context.retrieved_documents:
            context_prompt = self._format_context_for_llama(context)
            messages.append({
                "role": "user", 
                "content": f"<oceanographic_context>\n{context_prompt}\n</oceanographic_context>"
            })
            messages.append({
                "role": "assistant",
                "content": "I've reviewed the oceanographic context. I'm ready to analyze your query using this data."
            })
        
        # Add main query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _format_context_for_llama(self, context: RAGContext) -> str:
        """Format RAG context optimized for Llama processing."""
        
        context_parts = [
            f"QUERY: {context.query}",
            f"RETRIEVED_DOCUMENTS: {len(context.retrieved_documents)}",
            f"CONFIDENCE: {context.confidence_score:.3f}"
        ]
        
        # Add top relevant documents
        if context.retrieved_documents:
            context_parts.append("\nRELEVANT_DATA:")
            for i, doc in enumerate(context.retrieved_documents[:3]):
                metadata = doc.metadata or {}
                context_parts.append(f"Document_{i+1}:")
                context_parts.append(f"  - Relevance: {doc.score:.3f}")
                context_parts.append(f"  - Source: {doc.id}")
                context_parts.append(f"  - Location: {metadata.get('latitude', 'N/A')}, {metadata.get('longitude', 'N/A')}")
                context_parts.append(f"  - Date: {metadata.get('measurement_date', 'N/A')}")
                context_parts.append(f"  - Parameters: {metadata.get('parameters', 'N/A')}")
        
        # Add SQL query if available
        if context.sql_query:
            context_parts.extend([
                "\nSQL_QUERY_EXECUTED:",
                f"```sql\n{context.sql_query}\n```"
            ])
        
        # Add database results summary
        if context.database_results:
            context_parts.extend([
                f"\nDATABASE_RESULTS: {len(context.database_results)} records returned",
                "Use this quantitative data for accurate analysis."
            ])
        
        return "\n".join(context_parts)
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        max_tries=2,  # Groq is usually fast, so fewer retries
        factor=1.5
    )
    async def _make_completion_request(self, messages: List[Dict], **kwargs) -> Any:
        """Make completion request with automatic retry."""
        return await self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )
    
    async def generate_response(
        self,
        query: str,
        context: Optional[RAGContext] = None,
        user_type: str = 'researcher',
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate response using Groq's fast inference."""
        
        start_time = time.time()
        
        # Select optimal model
        if not model:
            model = self._select_optimal_model(query, context)
        
        # Get model configuration
        model_config = None
        for tier, config in self.model_configs.items():
            if config['model'] == model:
                model_config = config
                break
        
        if not model_config:
            model_config = self.model_configs['balanced']
            model = model_config['model']
        
        try:
            # Check cache first
            cache_key = self._cache_key(query, model, user_type=user_type, **kwargs)
            if cache_key in self.response_cache:
                cached_entry = self.response_cache[cache_key]
                if self._is_cache_valid(cached_entry):
                    self.usage_stats['cache_hits'] += 1
                    logger.debug("Returning cached Groq response")
                    return cached_entry['response']
            
            # Build Llama-optimized messages
            messages = self._build_llama_optimized_prompt(query, context, user_type)
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'messages': messages,
                'max_tokens': min(kwargs.get('max_tokens', model_config['max_tokens']), model_config['max_tokens']),
                'temperature': kwargs.get('temperature', model_config['temperature']),
                'stream': stream,
                'top_p': kwargs.get('top_p', 0.9),
                'stop': kwargs.get('stop', None)
            }
            
            if stream:
                return self._stream_response(request_params, start_time)
            else:
                return await self._complete_response(request_params, start_time, cache_key, model_config)
                
        except Exception as e:
            self.usage_stats['error_count'] += 1
            logger.error(f"Groq request failed: {e}")
            
            # Return error response
            return LLMResponse(
                content=f"I encountered an error processing your oceanographic query with Groq: {str(e)}. Please try again.",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                finish_reason="error"
            )
    
    async def _complete_response(self, request_params: Dict, start_time: float, 
                               cache_key: str, model_config: Dict) -> LLMResponse:
        """Handle non-streaming response with Groq optimization."""
        
        # Make the request
        response = await self._make_completion_request(**request_params)
        
        response_time = time.time() - start_time
        
        # Extract response data
        choice = response.choices[0]
        content = choice.message.content
        finish_reason = choice.finish_reason
        
        # Extract token usage
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        # Calculate confidence score (Llama-specific heuristics)
        confidence_score = self._calculate_llama_confidence_score(
            content, finish_reason, response_time, request_params['model']
        )
        
        # Extract citations
        citations = self._extract_citations(content)
        
        llm_response = LLMResponse(
            content=content,
            model=request_params['model'],
            tokens_used=tokens_used,
            response_time=response_time,
            confidence_score=confidence_score,
            finish_reason=finish_reason,
            citations=citations
        )
        
        # Cache the response
        self.response_cache[cache_key] = {
            'response': llm_response,
            'timestamp': time.time()
        }
        
        # Update statistics
        self._update_stats(tokens_used, response_time, request_params['model'])
        
        return llm_response
    
    async def _stream_response(self, request_params: Dict, start_time: float) -> AsyncGenerator[str, None]:
        """Handle streaming response from Groq."""
        
        try:
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            yield f"\n\n[Groq Error: {str(e)}]"
    
    def _calculate_llama_confidence_score(self, content: str, finish_reason: str, 
                                        response_time: float, model: str) -> float:
        """Calculate confidence score optimized for Llama models."""
        
        base_score = 0.6
        
        # Adjust based on finish reason
        if finish_reason == 'stop':
            base_score += 0.3
        elif finish_reason == 'length':
            base_score += 0.15  # Potentially truncated but complete
        else:
            base_score -= 0.2
        
        # Llama-specific quality indicators
        quality_indicators = [
            'based on the data', 'analysis shows', 'statistical', 'correlation',
            'temperature', 'salinity', 'oceanographic', 'argo float', 'measurement',
            'however', 'therefore', 'furthermore', 'in conclusion'
        ]
        
        indicator_count = sum(1 for indicator in quality_indicators if indicator.lower() in content.lower())
        base_score += min(0.2, indicator_count * 0.02)
        
        # Response structure bonus (Llama tends to be well-structured)
        if any(marker in content for marker in ['1.', '2.', '•', '-', 'First', 'Second']):
            base_score += 0.05
        
        # Speed bonus (Groq is fast)
        if response_time < 2.0:
            base_score += 0.05
        elif response_time > 10.0:
            base_score -= 0.05
        
        # Model-specific adjustments
        if '70b' in model:  # Larger models generally more capable
            base_score += 0.05
        elif '8b' in model:  # Smaller models, slightly lower confidence
            base_score -= 0.02
        
        return max(0.0, min(1.0, base_score))
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from Llama response content."""
        import re
        
        citation_patterns = [
            r'\[([^\]]+)\]',
            r'\(([^)]+\d{4}[^)]*)\)',
            r'(?:Source|Reference|Data from):\s*([^\n]+)',
            r'ARGO\s+float\s+(\w+)',
            r'(?:doi|DOI):\s*([^\s]+)',
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))
    
    def _update_stats(self, tokens_used: int, response_time: float, model: str):
        """Update usage statistics."""
        self.usage_stats['total_requests'] += 1
        self.usage_stats['total_tokens'] += tokens_used
        
        # Update average response time
        total_requests = self.usage_stats['total_requests']
        prev_avg = self.usage_stats['avg_response_time']
        new_avg = (prev_avg * (total_requests - 1) + response_time) / total_requests
        self.usage_stats['avg_response_time'] = new_avg
        
        # Update model usage
        for tier, config in self.model_configs.items():
            if config['model'] == model:
                self.usage_stats['model_usage'][tier] += 1
                break
    
    async def generate_with_structured_output(self, query: str, 
                                            output_schema: Dict,
                                            context: Optional[RAGContext] = None) -> Dict[str, Any]:
        """Generate structured output using Llama's reasoning capabilities."""
        
        # Create structured prompt
        schema_description = json.dumps(output_schema, indent=2)
        structured_query = f"""
Please analyze the following oceanographic query and provide a response in the exact JSON format specified below.

Query: {query}

Required JSON Schema:
{schema_description}

Important: Your response must be valid JSON that matches this schema exactly. Include all required fields and use appropriate data types.

Response:
"""
        
        try:
            response = await self.generate_response(
                structured_query,
                context=context,
                model=self.model_configs['balanced']['model'],  # Use larger model for structured output
                temperature=0.1  # Lower temperature for more consistent structure
            )
            
            # Try to parse JSON from response
            content = response.content.strip()
            
            # Extract JSON if wrapped in markdown
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            elif '```' in content:
                json_start = content.find('```') + 3
                json_end = content.rfind('```')
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            
            # Parse JSON
            try:
                structured_data = json.loads(content)
                return structured_data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse structured output as JSON: {e}")
                return {
                    "error": "Failed to parse structured output",
                    "raw_response": response.content,
                    "parsing_error": str(e)
                }
                
        except Exception as e:
            logger.error(f"Structured output generation failed: {e}")
            return {
                "error": "Structured output generation failed",
                "exception": str(e)
            }
    
    def get_usage_stats(self) -> Dict:
        """Get client usage statistics."""
        stats = self.usage_stats.copy()
        
        # Add model information
        stats['available_models'] = list(self.model_configs.keys())
        stats['default_model'] = self.default_model
        
        return stats
    
    async def health_check(self) -> bool:
        """Check if Groq API is accessible."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_configs['fast']['model'],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache.clear()
        logger.info("Groq response cache cleared")
    
    async def close(self):
        """Clean up resources."""
        await self.client.close()
        self.clear_cache()
        logger.info("Groq client closed")