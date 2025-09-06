"""
LLM Orchestrator for multi-provider access with intelligent routing, 
fallback mechanisms, and advanced oceanographic query processing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
import random

from .groq_client import GroqClient
from .mcp_integration import MCPOrchestrator, ToolExecutionResult
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    model: str
    max_tokens: int
    temperature: float
    fallback_provider: Optional[LLMProvider] = LLMProvider.GROQ
    use_tools: bool = True
    stream_response: bool = False


@dataclass
class QueryAnalysis:
    """Analysis of query complexity and routing requirements."""
    complexity_level: str  # simple, moderate, complex, expert
    requires_tools: bool
    requires_reasoning: bool
    estimated_tokens: int
    recommended_provider: LLMProvider
    confidence: float


@dataclass
class OrchestratedResponse:
    """Complete orchestrated response with metadata."""
    query: str
    answer: str
    llm_response: LLMResponse
    tool_results: List[ToolExecutionResult]
    provider_used: str
    total_time: float
    reasoning_steps: List[str]
    confidence_score: float
    citations: List[str]
    suggestions: List[str]


class QueryComplexityAnalyzer:
    """Analyzes query complexity for optimal LLM routing."""
    
    def __init__(self):
        self.complexity_indicators = {
            'simple': [
                'what is', 'define', 'explain briefly', 'show me', 'list',
                'temperature', 'salinity', 'depth', 'location'
            ],
            'moderate': [
                'compare', 'analyze', 'trend', 'pattern', 'relationship',
                'between', 'over time', 'seasonal', 'correlation'
            ],
            'complex': [
                'optimize', 'predict', 'model', 'simulate', 'algorithm',
                'machine learning', 'statistical significance', 'hypothesis'
            ],
            'expert': [
                'thermohaline circulation', 'water mass analysis', 'mixing',
                'biogeochemical cycles', 'climate change impacts', 'ocean acidification'
            ]
        }
        
        self.tool_indicators = [
            'show', 'plot', 'visualize', 'map', 'chart', 'graph', 'export',
            'download', 'save', 'data', 'statistics', 'analysis'
        ]
        
        self.reasoning_indicators = [
            'why', 'how', 'explain', 'reason', 'cause', 'effect',
            'relationship', 'mechanism', 'process', 'implications'
        ]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query complexity and requirements."""
        query_lower = query.lower()
        
        # Determine complexity level
        complexity_scores = {}
        for level, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            complexity_scores[level] = score
        
        complexity_level = max(complexity_scores, key=complexity_scores.get)
        if max(complexity_scores.values()) == 0:
            complexity_level = 'simple'
        
        # Check if tools are needed
        requires_tools = any(indicator in query_lower for indicator in self.tool_indicators)
        
        # Check if reasoning is needed
        requires_reasoning = any(indicator in query_lower for indicator in self.reasoning_indicators)
        
        # Estimate token requirements
        estimated_tokens = len(query.split()) * 4  # Rough estimation
        if complexity_level in ['complex', 'expert']:
            estimated_tokens *= 2
        if requires_reasoning:
            estimated_tokens *= 1.5
        
        # Recommend provider based on analysis
        if complexity_level in ['complex', 'expert'] or requires_reasoning:
            recommended_provider = LLMProvider.ANTHROPIC  # Claude for complex reasoning
            confidence = 0.8
        elif requires_tools:
            recommended_provider = LLMProvider.OPENAI  # GPT for tool usage
            confidence = 0.7
        else:
            recommended_provider = LLMProvider.AUTO  # Either works
            confidence = 0.6
        
        return QueryAnalysis(
            complexity_level=complexity_level,
            requires_tools=requires_tools,
            requires_reasoning=requires_reasoning,
            estimated_tokens=estimated_tokens,
            recommended_provider=recommended_provider,
            confidence=confidence
        )


class IntelligentRouter:
    """Intelligent routing for LLM requests based on query characteristics."""
    
    def __init__(self):
        self.analyzer = QueryComplexityAnalyzer()
        
        # Provider capabilities and strengths
        self.provider_strengths = {
            LLMProvider.OPENAI: {
                'tool_usage': 0.9,
                'code_generation': 0.9,
                'structured_output': 0.8,
                'reasoning': 0.7,
                'scientific_accuracy': 0.7
            },
            LLMProvider.ANTHROPIC: {
                'reasoning': 0.9,
                'scientific_accuracy': 0.9,
                'complex_analysis': 0.9,
                'long_context': 0.8,
                'tool_usage': 0.7
            }
        }
        
        # Current provider status (health, rate limits, etc.)
        self.provider_status = {
            LLMProvider.OPENAI: {'available': True, 'load': 0.3, 'errors': 0},
            LLMProvider.ANTHROPIC: {'available': True, 'load': 0.4, 'errors': 0}
        }
    
    def route_request(self, query: str, config: Optional[LLMConfig] = None) -> LLMProvider:
        """Route request to optimal provider."""
        
        # If provider explicitly specified, use it
        if config and config.provider != LLMProvider.AUTO:
            return config.provider
        
        # Analyze query characteristics
        analysis = self.analyzer.analyze_query(query)
        
        # If analysis strongly recommends a provider, use it
        if analysis.confidence > 0.75 and analysis.recommended_provider != LLMProvider.AUTO:
            provider = analysis.recommended_provider
        else:
            # Use capabilities-based routing
            scores = {}
            for provider, capabilities in self.provider_strengths.items():
                score = 0.0
                
                if analysis.requires_tools:
                    score += capabilities['tool_usage'] * 0.4
                if analysis.requires_reasoning:
                    score += capabilities['reasoning'] * 0.4
                if analysis.complexity_level in ['complex', 'expert']:
                    score += capabilities['complex_analysis'] * 0.3
                    score += capabilities['scientific_accuracy'] * 0.3
                
                # Factor in current provider status
                status = self.provider_status[provider]
                if not status['available']:
                    score = 0.0
                else:
                    score *= (1.0 - status['load'] * 0.2)  # Reduce score for high load
                    score *= (1.0 - status['errors'] * 0.1)  # Reduce score for errors
                
                scores[provider] = score
            
            provider = max(scores, key=scores.get)
        
        logger.debug(f"Routed query to {provider.value} (complexity: {analysis.complexity_level})")
        return provider
    
    def update_provider_status(self, provider: LLMProvider, success: bool, response_time: float):
        """Update provider status based on request outcomes."""
        status = self.provider_status[provider]
        
        if success:
            status['errors'] = max(0, status['errors'] - 1)
            # Update load based on response time
            if response_time < 2.0:
                status['load'] = max(0.0, status['load'] - 0.1)
            elif response_time > 10.0:
                status['load'] = min(1.0, status['load'] + 0.1)
        else:
            status['errors'] += 1
            if status['errors'] > 3:
                status['available'] = False  # Mark as unavailable after multiple errors


class LLMOrchestrator:
    """Main orchestrator for multi-provider LLM access with advanced features."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.settings = get_settings()
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        self.mcp_orchestrator = MCPOrchestrator()
        self.rag_pipeline = None  # Will be injected
        
        # Advanced routing and analysis
        self.router = IntelligentRouter()
        
        # Performance tracking
        self.orchestration_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'provider_usage': {
                'openai': 0,
                'anthropic': 0
            },
            'fallback_usage': 0
        }
        
        logger.info("LLM Orchestrator initialized")
    
    async def initialize(self):
        """Initialize all LLM clients and dependencies."""
        try:
            # Initialize OpenAI client
            openai_config = self.config.get('openai', {})
            if openai_config.get('enabled', True):
                self.openai_client = OpenAIClient(openai_config)
                logger.info("OpenAI client initialized")
            
            # Initialize Anthropic client
            anthropic_config = self.config.get('anthropic', {})
            if anthropic_config.get('enabled', True):
                self.anthropic_client = AnthropicClient(anthropic_config)
                logger.info("Anthropic client initialized")
            
            # Test connectivity
            await self._health_check_all_providers()
            
        except Exception as e:
            logger.error(f"LLM Orchestrator initialization failed: {e}")
            raise
    
    def set_rag_pipeline(self, rag_pipeline: 'RAGPipeline'):
        """Inject RAG pipeline for context-aware responses."""
        self.rag_pipeline = rag_pipeline
        logger.info("RAG pipeline integrated with LLM orchestrator")
    
    async def process_query(self, 
                           query: str, 
                           config: Optional[LLMConfig] = None,
                           user_type: str = 'researcher',
                           context: Optional[RAGContext] = None) -> OrchestratedResponse:
        """Process query with full orchestration including RAG, tools, and multi-provider routing."""
        
        start_time = asyncio.get_event_loop().time()
        self.orchestration_stats['total_requests'] += 1
        
        try:
            # Step 1: Get RAG context if not provided
            if not context and self.rag_pipeline:
                rag_response = await self.rag_pipeline.process_query(query)
                context = rag_response.context
            
            # Step 2: Determine optimal provider
            config = config or LLMConfig(
                provider=LLMProvider.AUTO,
                model="auto",
                max_tokens=4000,
                temperature=0.3,
                use_tools=True
            )
            
            provider = self.router.route_request(query, config)
            client = self._get_client(provider)
            
            if not client:
                raise ValueError(f"Provider {provider.value} not available")
            
            reasoning_steps = []
            tool_results = []
            
            # Step 3: Execute with tools if needed
            if config.use_tools:
                reasoning_steps.append("Analyzing query for tool requirements")
                
                # Get available tools
                available_tools = self.mcp_orchestrator.get_available_tools()
                
                # Generate response with tool calling
                llm_response = await client.generate_with_tools(
                    query=query,
                    tools=available_tools,
                    context=context,
                    model=self._get_model_for_provider(provider, config)
                )
                
                # Execute tool calls if present
                if llm_response.tool_calls:
                    reasoning_steps.append(f"Executing {len(llm_response.tool_calls)} tool calls")
                    tool_results = await self.mcp_orchestrator.execute_tool_sequence(
                        llm_response.tool_calls, context
                    )
                    
                    # Generate final response incorporating tool results
                    tool_context = self.mcp_orchestrator.format_tool_results_for_llm(tool_results)
                    
                    enhanced_query = f"""
Original Query: {query}

Tool Execution Results:
{tool_context}

Please provide a comprehensive response incorporating the tool results above. 
Focus on scientific accuracy and provide clear explanations of the findings.
"""
                    
                    reasoning_steps.append("Synthesizing final response with tool results")
                    final_response = await client.generate_response(
                        query=enhanced_query,
                        context=context,
                        user_type=user_type,
                        stream=config.stream_response
                    )
                    
                    # Combine responses
                    combined_content = final_response.content
                    llm_response.content = combined_content
                    llm_response.tool_calls = llm_response.tool_calls  # Preserve original tool calls
                    
            else:
                # Step 4: Generate response without tools
                reasoning_steps.append("Generating direct response without tools")
                llm_response = await client.generate_response(
                    query=query,
                    context=context,
                    user_type=user_type,
                    stream=config.stream_response
                )
            
            # Step 5: Generate follow-up suggestions
            suggestions = await self._generate_suggestions(query, llm_response, context)
            
            # Step 6: Extract citations
            citations = llm_response.citations or []
            if context and context.retrieved_documents:
                # Add RAG citations
                for doc in context.retrieved_documents[:3]:
                    if doc.metadata:
                        citation = f"ARGO Float {doc.metadata.get('platform_number', 'Unknown')} - {doc.metadata.get('measurement_date', 'Unknown date')}"
                        if citation not in citations:
                            citations.append(citation)
            
            total_time = asyncio.get_event_loop().time() - start_time
            
            # Step 7: Build orchestrated response
            orchestrated_response = OrchestratedResponse(
                query=query,
                answer=llm_response.content,
                llm_response=llm_response,
                tool_results=tool_results,
                provider_used=provider.value,
                total_time=total_time,
                reasoning_steps=reasoning_steps,
                confidence_score=self._calculate_overall_confidence(llm_response, tool_results, context),
                citations=citations,
                suggestions=suggestions
            )
            
            # Update statistics
            self.orchestration_stats['successful_requests'] += 1
            self.orchestration_stats['provider_usage'][provider.value] += 1
            self._update_average_response_time(total_time)
            
            # Update router status
            self.router.update_provider_status(provider, True, total_time)
            
            return orchestrated_response
            
        except Exception as e:
            logger.error(f"Query orchestration failed: {e}")
            
            # Try fallback if available
            if config and config.fallback_provider:
                try:
                    return await self._execute_fallback(query, config, user_type, context, str(e))
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            # Return error response
            self.orchestration_stats['failed_requests'] += 1
            error_response = LLMResponse(
                content=f"I encountered an error processing your oceanographic query: {str(e)}. Please try rephrasing your question or contact support.",
                model="error",
                tokens_used=0,
                response_time=asyncio.get_event_loop().time() - start_time,
                confidence_score=0.0,
                finish_reason="error"
            )
            
            return OrchestratedResponse(
                query=query,
                answer=error_response.content,
                llm_response=error_response,
                tool_results=[],
                provider_used="error",
                total_time=asyncio.get_event_loop().time() - start_time,
                reasoning_steps=["Error occurred during processing"],
                confidence_score=0.0,
                citations=[],
                suggestions=["Please try rephrasing your query", "Check system status"]
            )
    
    async def stream_response(self, 
                            query: str, 
                            config: Optional[LLMConfig] = None,
                            user_type: str = 'researcher',
                            context: Optional[RAGContext] = None) -> AsyncGenerator[str, None]:
        """Stream response with real-time processing."""
        
        config = config or LLMConfig(
            provider=LLMProvider.AUTO,
            model="auto",
            max_tokens=4000,
            temperature=0.3,
            stream_response=True
        )
        
        provider = self.router.route_request(query, config)
        client = self._get_client(provider)
        
        if not client:
            yield f"Error: Provider {provider.value} not available"
            return
        
        try:
            # Get context if not provided
            if not context and self.rag_pipeline:
                rag_response = await self.rag_pipeline.process_query(query)
                context = rag_response.context
            
            # Stream response
            async for chunk in client.generate_response(
                query=query,
                context=context,
                user_type=user_type,
                stream=True
            ):
                yield chunk
                
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"
    
    async def _execute_fallback(self, query: str, config: LLMConfig, user_type: str, 
                              context: Optional[RAGContext], original_error: str) -> OrchestratedResponse:
        """Execute fallback provider."""
        
        self.orchestration_stats['fallback_usage'] += 1
        logger.warning(f"Using fallback provider due to error: {original_error}")
        
        fallback_client = self._get_client(config.fallback_provider)
        if not fallback_client:
            raise ValueError(f"Fallback provider {config.fallback_provider.value} not available")
        
        # Simple fallback without tools
        llm_response = await fallback_client.generate_response(
            query=query,
            context=context,
            user_type=user_type
        )
        
        return OrchestratedResponse(
            query=query,
            answer=f"[Fallback Response] {llm_response.content}",
            llm_response=llm_response,
            tool_results=[],
            provider_used=f"{config.fallback_provider.value}_fallback",
            total_time=llm_response.response_time,
            reasoning_steps=["Executed fallback provider due to primary failure"],
            confidence_score=max(0.3, llm_response.confidence_score - 0.2),  # Reduced confidence for fallback
            citations=llm_response.citations or [],
            suggestions=["Primary system experienced issues", "Response provided via fallback system"]
        )
    
    def _get_client(self, provider: LLMProvider) -> Union[OpenAIClient, AnthropicClient, None]:
        """Get client for specified provider."""
        if provider == LLMProvider.OPENAI:
            return self.openai_client
        elif provider == LLMProvider.ANTHROPIC:
            return self.anthropic_client
        else:
            # Auto-select based on availability
            if self.anthropic_client and self.router.provider_status[LLMProvider.ANTHROPIC]['available']:
                return self.anthropic_client
            elif self.openai_client and self.router.provider_status[LLMProvider.OPENAI]['available']:
                return self.openai_client
            return None
    
    def _get_model_for_provider(self, provider: LLMProvider, config: LLMConfig) -> str:
        """Get appropriate model for provider."""
        if config.model != "auto":
            return config.model
        
        # Default models for each provider
        if provider == LLMProvider.OPENAI:
            return "gpt-4-turbo-preview"
        elif provider == LLMProvider.ANTHROPIC:
            return "claude-3-5-sonnet-20241022"
        else:
            return "gpt-4-turbo-preview"  # Default fallback
    
    def _calculate_overall_confidence(self, llm_response: LLMResponse, 
                                    tool_results: List[ToolExecutionResult],
                                    context: Optional[RAGContext]) -> float:
        """Calculate overall confidence score."""
        base_confidence = llm_response.confidence_score
        
        # Boost confidence if tools executed successfully
        if tool_results:
            successful_tools = sum(1 for result in tool_results if result.success)
            tool_success_rate = successful_tools / len(tool_results)
            base_confidence += tool_success_rate * 0.2
        
        # Boost confidence if high-quality context available
        if context and context.confidence_score > 0.8:
            base_confidence += 0.1
        
        # Reduce confidence for error conditions
        if tool_results and any(not result.success for result in tool_results):
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    async def _generate_suggestions(self, query: str, llm_response: LLMResponse, 
                                  context: Optional[RAGContext]) -> List[str]:
        """Generate intelligent follow-up suggestions."""
        
        suggestions = []
        
        # Analysis-based suggestions
        if "temperature" in query.lower():
            suggestions.append("Explore salinity patterns in the same region")
            suggestions.append("Compare with historical temperature data")
        
        if "region" in query.lower() or "area" in query.lower():
            suggestions.append("Analyze seasonal variations")
            suggestions.append("Compare with adjacent ocean regions")
        
        # Context-based suggestions
        if context and context.retrieved_documents:
            if len(context.retrieved_documents) > 5:
                suggestions.append("Refine your query for more specific results")
            else:
                suggestions.append("Expand your search to nearby regions")
        
        # Tool-based suggestions
        if llm_response.tool_calls:
            suggestions.append("Export the analysis results")
            suggestions.append("Generate visualizations for this data")
        else:
            suggestions.append("Request detailed data analysis")
            suggestions.append("Generate maps and charts")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    async def _health_check_all_providers(self):
        """Check health of all LLM providers."""
        
        if self.openai_client:
            try:
                openai_healthy = await self.openai_client.health_check()
                self.router.provider_status[LLMProvider.OPENAI]['available'] = openai_healthy
                logger.info(f"OpenAI health check: {'✓' if openai_healthy else '✗'}")
            except Exception as e:
                logger.error(f"OpenAI health check failed: {e}")
                self.router.provider_status[LLMProvider.OPENAI]['available'] = False
        
        if self.anthropic_client:
            try:
                anthropic_healthy = await self.anthropic_client.health_check()
                self.router.provider_status[LLMProvider.ANTHROPIC]['available'] = anthropic_healthy
                logger.info(f"Anthropic health check: {'✓' if anthropic_healthy else '✗'}")
            except Exception as e:
                logger.error(f"Anthropic health check failed: {e}")
                self.router.provider_status[LLMProvider.ANTHROPIC]['available'] = False
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time statistics."""
        total_requests = self.orchestration_stats['successful_requests']
        if total_requests == 1:
            self.orchestration_stats['avg_response_time'] = response_time
        else:
            prev_avg = self.orchestration_stats['avg_response_time']
            new_avg = (prev_avg * (total_requests - 1) + response_time) / total_requests
            self.orchestration_stats['avg_response_time'] = new_avg
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics."""
        stats = self.orchestration_stats.copy()
        
        # Add provider statistics
        if self.openai_client:
            stats['openai_stats'] = self.openai_client.get_usage_stats()
        if self.anthropic_client:
            stats['anthropic_stats'] = self.anthropic_client.get_usage_stats()
        
        # Add MCP statistics
        stats['mcp_stats'] = self.mcp_orchestrator.get_execution_statistics()
        
        # Add routing statistics
        stats['provider_status'] = self.router.provider_status.copy()
        
        return stats
    
    async def close(self):
        """Clean up orchestrator resources."""
        if self.openai_client:
            await self.openai_client.close()
        if self.anthropic_client:
            await self.anthropic_client.close()
        await self.mcp_orchestrator.close()
        
        logger.info("LLM Orchestrator closed")