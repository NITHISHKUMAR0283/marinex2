"""
Production OpenAI client for oceanographic queries with advanced retry logic,
streaming support, and domain-specific prompt engineering.
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
    import openai
    from openai import AsyncOpenAI
except ImportError as e:
    print(f"OpenAI library not found: {e}")
    print("Install with: pip install openai")
    raise

from ...core.config import get_settings
from ..rag import RAGContext

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    content: str
    model: str
    tokens_used: int
    response_time: float
    confidence_score: float
    finish_reason: str
    tool_calls: Optional[List[Dict]] = None
    citations: Optional[List[str]] = None


@dataclass
class OceanographicPromptTemplate:
    """Domain-specific prompt templates for oceanographic queries."""
    system_prompt: str
    user_template: str
    context_integration: str
    citation_format: str
    constraints: List[str]


class OceanographicPromptEngineering:
    """Advanced prompt engineering for oceanographic domain."""
    
    def __init__(self):
        self.system_prompts = {
            'researcher': """You are OceanGPT, an expert oceanographer and marine scientist with deep knowledge of ARGO float data, ocean dynamics, and marine ecosystems. You provide scientifically accurate, well-cited responses to oceanographic queries.

EXPERTISE DOMAINS:
- Physical oceanography (temperature, salinity, density, currents)
- Marine biogeochemistry (oxygen, nutrients, pH, chlorophyll)
- Ocean circulation and water mass analysis
- Climate change impacts on ocean systems
- ARGO float technology and data interpretation

RESPONSE GUIDELINES:
- Always provide scientifically accurate information with uncertainty quantification
- Cite data sources and reference relevant scientific literature
- Explain complex concepts at appropriate technical levels
- Include data quality considerations and limitations
- Suggest follow-up analyses when relevant
- Use proper oceanographic terminology and units""",

            'educator': """You are OceanEd, an oceanographic educator who makes complex marine science concepts accessible and engaging. You excel at explaining ocean processes, ARGO data, and marine research to students and the general public.

TEACHING APPROACH:
- Start with fundamental concepts and build complexity gradually
- Use analogies and real-world examples to illustrate abstract concepts
- Provide interactive learning opportunities and suggest hands-on activities
- Connect oceanographic phenomena to everyday life and current events
- Encourage scientific curiosity and critical thinking
- Adapt explanations to appropriate grade/knowledge levels

RESPONSE STYLE:
- Clear, engaging language with defined technical terms
- Step-by-step explanations of complex processes
- Visual learning suggestions (charts, diagrams, animations)
- Questions to check understanding and promote deeper thinking
- Connections to broader environmental and climate topics""",

            'policy_maker': """You are OceanPolicy, a science advisor specializing in translating oceanographic research into policy-relevant insights. You excel at communicating scientific findings for decision-making contexts.

POLICY FOCUS AREAS:
- Climate change impacts and ocean warming trends
- Marine ecosystem health and biodiversity conservation
- Coastal vulnerability and sea level rise
- Ocean acidification and marine resource management
- International ocean observation and data sharing
- Sustainable blue economy development

COMMUNICATION STYLE:
- Executive summary format with key findings highlighted
- Clear implications for policy and decision-making
- Uncertainty and risk assessment with confidence levels
- Economic and social impact considerations
- Actionable recommendations with implementation timelines
- Alignment with international frameworks (SDGs, Paris Agreement)"""
        }
        
        self.context_templates = {
            'data_analysis': """Based on the following ARGO oceanographic data context:

DATA SOURCES: {sources}
SPATIAL COVERAGE: {spatial_info}
TEMPORAL RANGE: {temporal_info}
PARAMETERS: {parameters}
DATA QUALITY: {quality_info}

Please analyze and respond to the user's query with scientific rigor.""",

            'comparative_analysis': """For comparative oceanographic analysis:

DATASET A: {dataset_a_info}
DATASET B: {dataset_b_info}
COMPARISON METRICS: {comparison_criteria}
STATISTICAL METHODS: {statistical_approach}

Provide a comprehensive comparison addressing the query.""",

            'trend_analysis': """For oceanographic trend analysis:

HISTORICAL DATA RANGE: {time_range}
TREND DETECTION METHOD: {trend_method}
CONFIDENCE INTERVALS: {confidence_info}
SEASONAL ADJUSTMENTS: {seasonal_info}

Analyze trends and provide scientifically sound interpretation."""
        }
        
        self.constraint_templates = {
            'scientific_accuracy': "Ensure all oceanographic facts are scientifically accurate and cite uncertainty levels",
            'data_limitations': "Acknowledge data limitations and spatial/temporal coverage gaps",
            'unit_consistency': "Use consistent oceanographic units (°C, PSU, dbar, mg/L, etc.)",
            'quality_control': "Reference data quality flags and validation procedures",
            'ethical_guidelines': "Follow scientific ethics and avoid overinterpretation of data"
        }
    
    def get_system_prompt(self, user_type: str = 'researcher') -> str:
        """Get appropriate system prompt for user type."""
        return self.system_prompts.get(user_type, self.system_prompts['researcher'])
    
    def format_context_prompt(self, template_type: str, **kwargs) -> str:
        """Format context-specific prompt template."""
        template = self.context_templates.get(template_type, self.context_templates['data_analysis'])
        return template.format(**kwargs)
    
    def apply_constraints(self, prompt: str, constraint_types: List[str]) -> str:
        """Apply domain-specific constraints to prompt."""
        constraints = [self.constraint_templates[ct] for ct in constraint_types if ct in self.constraint_templates]
        if constraints:
            constraint_text = "\n".join([f"- {constraint}" for constraint in constraints])
            prompt += f"\n\nIMPORTANT CONSTRAINTS:\n{constraint_text}"
        return prompt


class OpenAIClient:
    """Production OpenAI client with advanced features for oceanographic applications."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.settings = get_settings()
        
        # Initialize OpenAI client
        api_key = self.config.get('api_key') or getattr(self.settings, 'openai_api_key', None)
        if not api_key:
            raise ValueError("OpenAI API key not provided in config or settings")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=self.config.get('timeout', 60.0),
            max_retries=0  # We handle retries manually
        )
        
        # Model configuration
        self.default_model = self.config.get('default_model', 'gpt-4-turbo-preview')
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.temperature = self.config.get('temperature', 0.3)  # Lower for scientific accuracy
        
        # Advanced features
        self.prompt_engineer = OceanographicPromptEngineering()
        
        # Performance tracking
        self.usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'avg_response_time': 0.0,
            'error_count': 0,
            'cache_hits': 0
        }
        
        # Response cache for similar queries
        self.response_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        logger.info(f"OpenAI client initialized with model: {self.default_model}")
    
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
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError),
        max_tries=3,
        factor=2
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
        """Generate response to oceanographic query with context integration."""
        
        start_time = time.time()
        model = model or self.default_model
        
        try:
            # Check cache first
            cache_key = self._cache_key(query, model, user_type=user_type, **kwargs)
            if cache_key in self.response_cache:
                cached_entry = self.response_cache[cache_key]
                if self._is_cache_valid(cached_entry):
                    self.usage_stats['cache_hits'] += 1
                    logger.debug("Returning cached response")
                    return cached_entry['response']
            
            # Build prompt with domain expertise
            system_prompt = self.prompt_engineer.get_system_prompt(user_type)
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add context if available
            if context and context.retrieved_documents:
                context_prompt = self._format_context_from_rag(context)
                messages.append({"role": "system", "content": context_prompt})
            
            # Add user query
            messages.append({"role": "user", "content": query})
            
            # Apply oceanographic constraints
            constraints = ['scientific_accuracy', 'data_limitations', 'unit_consistency']
            final_system_content = self.prompt_engineer.apply_constraints(
                messages[0]["content"], constraints
            )
            messages[0]["content"] = final_system_content
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'messages': messages,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'stream': stream,
                **kwargs
            }
            
            if stream:
                return self._stream_response(request_params, start_time)
            else:
                return await self._complete_response(request_params, start_time, cache_key)
                
        except Exception as e:
            self.usage_stats['error_count'] += 1
            logger.error(f"OpenAI request failed: {e}")
            
            # Return error response
            return LLMResponse(
                content=f"I encountered an error processing your oceanographic query: {str(e)}. Please try rephrasing your question or contact support if the issue persists.",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                finish_reason="error"
            )
    
    async def _complete_response(self, request_params: Dict, start_time: float, cache_key: str) -> LLMResponse:
        """Handle non-streaming response."""
        
        # Make the request with retry logic
        response = await self._make_completion_request(**request_params)
        
        response_time = time.time() - start_time
        
        # Extract response data
        choice = response.choices[0]
        content = choice.message.content
        finish_reason = choice.finish_reason
        
        # Extract token usage
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        # Calculate confidence score (heuristic based on response characteristics)
        confidence_score = self._calculate_confidence_score(content, finish_reason, response_time)
        
        # Extract citations if present
        citations = self._extract_citations(content)
        
        # Extract tool calls if present
        tool_calls = []
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            tool_calls = [
                {
                    'id': call.id,
                    'function': call.function.name,
                    'arguments': json.loads(call.function.arguments)
                }
                for call in choice.message.tool_calls
            ]
        
        llm_response = LLMResponse(
            content=content,
            model=request_params['model'],
            tokens_used=tokens_used,
            response_time=response_time,
            confidence_score=confidence_score,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            citations=citations
        )
        
        # Cache the response
        self.response_cache[cache_key] = {
            'response': llm_response,
            'timestamp': time.time()
        }
        
        # Update statistics
        self._update_stats(tokens_used, response_time)
        
        return llm_response
    
    async def _stream_response(self, request_params: Dict, start_time: float) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        
        try:
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\n[Error: {str(e)}]"
    
    def _format_context_from_rag(self, context: RAGContext) -> str:
        """Format RAG context for LLM prompt."""
        
        context_parts = [
            "RELEVANT OCEANOGRAPHIC DATA CONTEXT:",
            f"Query: {context.query}",
            f"Retrieved Documents: {len(context.retrieved_documents)}",
            f"Context Confidence: {context.confidence_score:.3f}"
        ]
        
        # Add information about retrieved documents
        for i, doc in enumerate(context.retrieved_documents[:5]):  # Top 5 most relevant
            metadata = doc.metadata or {}
            context_parts.extend([
                f"\nDocument {i+1} (Relevance: {doc.score:.3f}):",
                f"ID: {doc.id}",
                f"Type: {metadata.get('source_type', 'unknown')}",
                f"Location: {metadata.get('latitude', 'N/A')}, {metadata.get('longitude', 'N/A')}",
                f"Date: {metadata.get('measurement_date', 'N/A')}",
                f"Parameters: {metadata.get('parameters', 'N/A')}"
            ])
        
        # Add SQL query if generated
        if context.sql_query:
            context_parts.extend([
                f"\nGenerated SQL Query:",
                f"```sql\n{context.sql_query}\n```"
            ])
        
        # Add database results if available
        if context.database_results:
            context_parts.extend([
                f"\nDatabase Query Results: {len(context.database_results)} records",
                "Use this data to provide accurate, data-driven responses."
            ])
        
        return "\n".join(context_parts)
    
    def _calculate_confidence_score(self, content: str, finish_reason: str, response_time: float) -> float:
        """Calculate confidence score based on response characteristics."""
        
        base_score = 0.5
        
        # Adjust based on finish reason
        if finish_reason == 'stop':
            base_score += 0.3
        elif finish_reason == 'length':
            base_score += 0.1  # Potentially truncated
        else:
            base_score -= 0.2  # Error or other issues
        
        # Adjust based on content quality indicators
        if 'uncertainty' in content.lower() or 'approximately' in content.lower():
            base_score += 0.1  # Good uncertainty quantification
        
        if any(term in content.lower() for term in ['citation', 'reference', 'source', 'data from']):
            base_score += 0.1  # Good referencing
        
        if len(content) > 100 and len(content) < 2000:
            base_score += 0.1  # Appropriate length
        
        # Adjust based on response time (faster might indicate cached or simpler response)
        if response_time < 2.0:
            base_score += 0.05
        elif response_time > 10.0:
            base_score -= 0.05
        
        return max(0.0, min(1.0, base_score))
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from response content."""
        import re
        
        # Look for common citation patterns
        citation_patterns = [
            r'\[([^\]]+)\]',  # [Citation]
            r'\(([^)]+\d{4}[^)]*)\)',  # (Author et al., 2023)
            r'doi:[^\s]+',  # DOI references
            r'https?://[^\s]+',  # URLs
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def _update_stats(self, tokens_used: int, response_time: float):
        """Update usage statistics."""
        self.usage_stats['total_requests'] += 1
        self.usage_stats['total_tokens'] += tokens_used
        
        # Update average response time
        total_requests = self.usage_stats['total_requests']
        prev_avg = self.usage_stats['avg_response_time']
        new_avg = (prev_avg * (total_requests - 1) + response_time) / total_requests
        self.usage_stats['avg_response_time'] = new_avg
    
    async def generate_with_tools(
        self,
        query: str,
        tools: List[Dict],
        context: Optional[RAGContext] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """Generate response with tool calling capabilities (MCP integration)."""
        
        model = model or 'gpt-4-turbo-preview'  # Tool calling requires newer models
        
        try:
            # Build messages with tool context
            messages = [
                {
                    "role": "system",
                    "content": self.prompt_engineer.get_system_prompt('researcher') + 
                              "\n\nYou have access to specialized oceanographic tools. Use them appropriately to answer queries."
                }
            ]
            
            if context:
                context_prompt = self._format_context_from_rag(context)
                messages.append({"role": "system", "content": context_prompt})
            
            messages.append({"role": "user", "content": query})
            
            # Make request with tools
            response = await self._make_completion_request(
                messages=messages,
                model=model,
                tools=tools,
                tool_choice="auto",
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Process tool calls if present
            choice = response.choices[0]
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                # Tool calls present - this would be handled by MCP orchestrator
                tool_calls = [
                    {
                        'id': call.id,
                        'function': call.function.name,
                        'arguments': json.loads(call.function.arguments)
                    }
                    for call in choice.message.tool_calls
                ]
                
                return LLMResponse(
                    content=choice.message.content or "",
                    model=model,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                    response_time=0.0,  # Would be calculated by orchestrator
                    confidence_score=0.8,  # Higher confidence with tools
                    finish_reason=choice.finish_reason,
                    tool_calls=tool_calls
                )
            else:
                # Regular response without tool calls
                return LLMResponse(
                    content=choice.message.content,
                    model=model,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                    response_time=0.0,
                    confidence_score=0.7,
                    finish_reason=choice.finish_reason
                )
                
        except Exception as e:
            logger.error(f"Tool-enabled request failed: {e}")
            return LLMResponse(
                content=f"I encountered an error while processing your request with oceanographic tools: {str(e)}",
                model=model,
                tokens_used=0,
                response_time=0.0,
                confidence_score=0.0,
                finish_reason="error"
            )
    
    def get_usage_stats(self) -> Dict:
        """Get client usage statistics."""
        return self.usage_stats.copy()
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    async def close(self):
        """Clean up resources."""
        await self.client.close()
        self.clear_cache()
        logger.info("OpenAI client closed")