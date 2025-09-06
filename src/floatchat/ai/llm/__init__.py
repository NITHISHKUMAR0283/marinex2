"""
LLM integration module for FloatChat.
Provides multi-provider access to OpenAI, Anthropic with MCP and advanced orchestration.
"""

from .openai_client import OpenAIClient, LLMResponse, OceanographicPromptEngineering
from .anthropic_client import AnthropicClient
from .mcp_integration import (
    MCPOrchestrator, 
    ToolDefinition, 
    ToolParameter, 
    ToolParameterType, 
    ToolExecutionResult,
    OceanographicToolRegistry
)
from .llm_orchestrator import (
    LLMOrchestrator, 
    LLMProvider, 
    LLMConfig, 
    OrchestratedResponse,
    QueryComplexityAnalyzer
)

__all__ = [
    'OpenAIClient',
    'AnthropicClient', 
    'LLMResponse',
    'OceanographicPromptEngineering',
    'MCPOrchestrator',
    'ToolDefinition',
    'ToolParameter',
    'ToolParameterType',
    'ToolExecutionResult',
    'OceanographicToolRegistry',
    'LLMOrchestrator',
    'LLMProvider',
    'LLMConfig',
    'OrchestratedResponse',
    'QueryComplexityAnalyzer'
]