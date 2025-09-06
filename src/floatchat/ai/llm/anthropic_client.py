"""
Production Anthropic Claude client for oceanographic queries with advanced capabilities.
Supports Claude-3 models with streaming, tool calling, and domain-specific optimization.
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
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError as e:
    print(f"Anthropic library not found: {e}")
    print("Install with: pip install anthropic")
    raise

from ...core.config import get_settings
from ..rag import RAGContext
from .openai_client import LLMResponse, OceanographicPromptEngineering

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Production Anthropic Claude client optimized for oceanographic applications."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.settings = get_settings()
        
        # Initialize Anthropic client
        api_key = self.config.get('api_key') or getattr(self.settings, 'anthropic_api_key', None)
        if not api_key:
            raise ValueError("Anthropic API key not provided in config or settings")
        
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=self.config.get('timeout', 60.0),
            max_retries=0  # We handle retries manually
        )
        
        # Model configuration optimized for Claude
        self.default_model = self.config.get('default_model', 'claude-3-5-sonnet-20241022')
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.temperature = self.config.get('temperature', 0.2)  # Lower for scientific accuracy
        
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
        
        logger.info(f"Anthropic client initialized with model: {self.default_model}")
    
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
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError),
        max_tries=3,
        factor=2
    )
    async def _make_message_request(self, messages: List[Dict], system: str = "", **kwargs) -> Any:
        """Make message request with automatic retry."""
        return await self.client.messages.create(
            system=system,
            messages=messages,
            **kwargs
        )
    
    def _format_messages_for_anthropic(self, messages: List[Dict]) -> tuple[str, List[Dict]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system_content = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                if system_content:
                    system_content += "\n\n" + msg["content"]
                else:
                    system_content = msg["content"]
            else:
                user_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system_content, user_messages
    
    async def generate_response(
        self,
        query: str,
        context: Optional[RAGContext] = None,
        user_type: str = 'researcher',
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate response to oceanographic query with Claude's advanced reasoning."""
        
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
            
            # Build Claude-optimized prompt
            system_prompt = self._build_claude_system_prompt(user_type, context)
            
            # Build user messages
            user_messages = []
            
            # Add context information if available
            if context and context.retrieved_documents:
                context_prompt = self._format_context_for_claude(context)
                user_messages.append({
                    "role": "user", 
                    "content": f"<oceanographic_context>\n{context_prompt}\n</oceanographic_context>\n\nBased on this oceanographic data context, please answer the following query:"
                })
            
            # Add main query
            user_messages.append({"role": "user", "content": query})
            
            # Prepare request parameters
            request_params = {
                'model': model,
                'system': system_prompt,
                'messages': user_messages,
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
            logger.error(f"Anthropic request failed: {e}")
            
            # Return error response
            return LLMResponse(
                content=f"I encountered an error processing your oceanographic query: {str(e)}. Please try rephrasing your question or contact support if the issue persists.",
                model=model,
                tokens_used=0,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                finish_reason="error"
            )
    
    def _build_claude_system_prompt(self, user_type: str, context: Optional[RAGContext] = None) -> str:
        """Build Claude-optimized system prompt with domain expertise."""
        
        base_prompt = self.prompt_engineer.get_system_prompt(user_type)
        
        # Add Claude-specific enhancements
        claude_enhancements = """
CLAUDE-SPECIFIC CAPABILITIES:
- You excel at complex reasoning and analysis of multidimensional oceanographic data
- You can identify patterns, anomalies, and relationships in marine datasets
- You provide step-by-step analysis of complex oceanographic phenomena
- You excel at uncertainty quantification and statistical interpretation
- You can suggest appropriate visualization approaches for different data types

RESPONSE FORMAT:
- Use structured thinking with clear reasoning steps
- Provide confidence levels for key findings
- Include relevant caveats and limitations
- Suggest follow-up analyses when appropriate
- Use precise oceanographic terminology

TOOL USAGE:
When tools are available, use them strategically to:
1. Query specific oceanographic datasets
2. Generate appropriate visualizations
3. Perform statistical analyses
4. Export data in requested formats
"""
        
        enhanced_prompt = base_prompt + "\n\n" + claude_enhancements
        
        # Add context-specific guidance
        if context and context.retrieved_documents:
            context_guidance = f"""
CONTEXT ANALYSIS:
You have access to {len(context.retrieved_documents)} relevant oceanographic documents.
Context confidence score: {context.confidence_score:.3f}
Focus your response on the most relevant and high-quality data sources.
"""
            enhanced_prompt += "\n\n" + context_guidance
        
        return enhanced_prompt
    
    def _format_context_for_claude(self, context: RAGContext) -> str:
        """Format RAG context optimized for Claude's processing style."""
        
        context_sections = []
        
        # Query information
        context_sections.append(f"<query>{context.query}</query>")
        
        # Context metadata
        context_sections.append(f"""<context_metadata>
Retrieved documents: {len(context.retrieved_documents)}
Context confidence: {context.confidence_score:.3f}
SQL query: {context.sql_query if context.sql_query else 'None generated'}
</context_metadata>""")
        
        # Document information (structured for Claude)
        if context.retrieved_documents:
            doc_info = []
            for i, doc in enumerate(context.retrieved_documents[:5]):
                metadata = doc.metadata or {}
                doc_info.append(f"""<document id="{i+1}" relevance="{doc.score:.3f}">
Source ID: {doc.id}
Type: {metadata.get('source_type', 'unknown')}
Location: {metadata.get('latitude', 'N/A')}, {metadata.get('longitude', 'N/A')}
Date: {metadata.get('measurement_date', 'N/A')}
Parameters: {metadata.get('parameters', 'N/A')}
Quality: {metadata.get('data_quality', 'N/A')}
</document>""")
            
            context_sections.append("<retrieved_documents>\n" + "\n\n".join(doc_info) + "\n</retrieved_documents>")
        
        # Database results if available
        if context.database_results:
            context_sections.append(f"""<database_results>
Query returned: {len(context.database_results)} records
Use this quantitative data to provide accurate, evidence-based responses.
</database_results>""")
        
        return "\n\n".join(context_sections)
    
    async def _complete_response(self, request_params: Dict, start_time: float, cache_key: str) -> LLMResponse:
        """Handle non-streaming response with Claude-specific processing."""
        
        # Make the request with retry logic
        response = await self._make_message_request(**request_params)
        
        response_time = time.time() - start_time
        
        # Extract response data (Claude format)
        content = response.content[0].text if response.content else ""
        finish_reason = response.stop_reason or "stop"
        
        # Extract token usage
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        total_tokens = input_tokens + output_tokens
        
        # Calculate confidence score (Claude-optimized)
        confidence_score = self._calculate_claude_confidence_score(content, finish_reason, response_time, response)
        
        # Extract citations and structured information
        citations = self._extract_citations(content)
        
        # Check for tool use (if supported)
        tool_calls = []
        if hasattr(response, 'tool_use') and response.tool_use:
            # Process Claude tool use format
            for tool_use in response.tool_use:
                tool_calls.append({
                    'id': tool_use.id,
                    'function': tool_use.name,
                    'arguments': tool_use.input
                })
        
        llm_response = LLMResponse(
            content=content,
            model=request_params['model'],
            tokens_used=total_tokens,
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
        self._update_stats(total_tokens, response_time)
        
        return llm_response
    
    async def _stream_response(self, request_params: Dict, start_time: float) -> AsyncGenerator[str, None]:
        """Handle streaming response with Claude."""
        
        try:
            stream = await self.client.messages.create(**request_params)
            
            async for event in stream:
                if event.type == 'content_block_delta' and event.delta.type == 'text_delta':
                    yield event.delta.text
                    
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            yield f"\n\n[Error: {str(e)}]"
    
    def _calculate_claude_confidence_score(self, content: str, finish_reason: str, 
                                         response_time: float, response: Any) -> float:
        """Calculate confidence score optimized for Claude responses."""
        
        base_score = 0.6  # Claude generally provides higher quality responses
        
        # Adjust based on finish reason
        if finish_reason == 'end_turn':
            base_score += 0.3
        elif finish_reason == 'max_tokens':
            base_score += 0.1  # Potentially truncated
        else:
            base_score -= 0.1
        
        # Claude-specific quality indicators
        quality_indicators = [
            'analysis shows', 'data indicates', 'evidence suggests',
            'confidence interval', 'uncertainty', 'statistical significance',
            'based on the data', 'oceanographic patterns', 'scientific literature'
        ]
        
        indicator_count = sum(1 for indicator in quality_indicators if indicator in content.lower())
        base_score += min(0.15, indicator_count * 0.03)
        
        # Check for structured reasoning
        if any(marker in content for marker in ['1.', '2.', 'First,', 'Second,', 'Therefore,']):
            base_score += 0.05
        
        # Check for appropriate length and detail
        if 200 <= len(content) <= 3000:
            base_score += 0.1
        
        # Response time considerations (Claude can be slower but more thoughtful)
        if 2.0 <= response_time <= 8.0:
            base_score += 0.05
        
        return max(0.0, min(1.0, base_score))
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from Claude response content."""
        import re
        
        # Claude often provides more detailed citations
        citation_patterns = [
            r'\[([^\]]+)\]',  # [Citation]
            r'\(([^)]+\d{4}[^)]*)\)',  # (Author et al., 2023)
            r'(?:doi|DOI):\s*([^\s]+)',  # DOI references
            r'https?://[^\s\)]+',  # URLs
            r'(?:Source|Reference|Citation):\s*([^\n]+)',  # Explicit source labels
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
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
        """Generate response with tool calling capabilities for Claude."""
        
        model = model or self.default_model
        
        try:
            # Build system prompt with tool context
            system_prompt = self._build_claude_system_prompt('researcher', context)
            system_prompt += f"""

AVAILABLE TOOLS:
You have access to the following oceanographic analysis tools:
{json.dumps(tools, indent=2)}

Use these tools strategically to provide accurate, data-driven responses to oceanographic queries.
Always explain your tool usage and interpret the results in your response.
"""
            
            # Build messages
            messages = []
            
            if context:
                context_prompt = self._format_context_for_claude(context)
                messages.append({
                    "role": "user",
                    "content": f"<oceanographic_context>\n{context_prompt}\n</oceanographic_context>\n\nQuery: {query}"
                })
            else:
                messages.append({"role": "user", "content": query})
            
            # Make request with tools (Claude format)
            response = await self._make_message_request(
                system=system_prompt,
                messages=messages,
                model=model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=tools if tools else None
            )
            
            # Process response
            content = response.content[0].text if response.content else ""
            tokens_used = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
            
            # Extract tool calls if present
            tool_calls = []
            if hasattr(response, 'tool_use') and response.tool_use:
                for tool_use in response.tool_use:
                    tool_calls.append({
                        'id': getattr(tool_use, 'id', 'unknown'),
                        'function': tool_use.name,
                        'arguments': tool_use.input
                    })
            
            return LLMResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                response_time=0.0,  # Would be calculated by orchestrator
                confidence_score=0.85,  # Higher confidence with Claude + tools
                finish_reason=response.stop_reason or "stop",
                tool_calls=tool_calls
            )
            
        except Exception as e:
            logger.error(f"Claude tool-enabled request failed: {e}")
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
        """Check if Anthropic API is accessible."""
        try:
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    async def close(self):
        """Clean up resources."""
        await self.client.close()
        self.clear_cache()
        logger.info("Anthropic client closed")