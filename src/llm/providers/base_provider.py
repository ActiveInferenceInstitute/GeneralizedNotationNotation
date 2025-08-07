#!/usr/bin/env python3
"""
Base LLM Provider

Abstract base class defining the interface for all LLM providers.
This ensures consistent behavior across different provider implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    PERPLEXITY = "perplexity"
    OLLAMA = "ollama"

@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    model_used: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LLMMessage:
    """Standardized message format for LLM conversations."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    name: Optional[str] = None

@dataclass
class LLMConfig:
    """Configuration parameters for LLM requests."""
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stream: bool = False
    timeout: Optional[float] = None

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods to ensure consistent behavior.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.client = None
        self._is_initialized = False
        
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type enum."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Return list of available models for this provider."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: LLMConfig) -> bool:
        """
        Validate configuration parameters for this provider.
        
        Args:
            config: LLM configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Returns:
            Standardized LLM response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Yields:
            Chunks of the response content
        """
        pass
    
    def construct_system_prompt(self, domain_context: str = "") -> str:
        """
        Construct a system prompt with domain-specific context.
        
        Args:
            domain_context: Additional context about the domain
            
        Returns:
            Formatted system prompt
        """
        base_prompt = (
            "You are an expert in Active Inference and GNN (Generalized Notation Notation) specifications. "
            "You provide accurate, comprehensive, and scientifically rigorous responses about Active Inference "
            "concepts, model structure, and practical implications."
        )
        
        if domain_context:
            return f"{base_prompt}\n\nAdditional Context: {domain_context}"
        
        return base_prompt
    
    def format_gnn_analysis_prompt(
        self, 
        gnn_content: str, 
        analysis_type: str = "general"
    ) -> List[LLMMessage]:
        """
        Format a prompt for GNN analysis.
        
        Args:
            gnn_content: The GNN file content to analyze
            analysis_type: Type of analysis ('summary', 'structure', 'questions', 'general')
            
        Returns:
            List of formatted messages
        """
        system_prompt = self.construct_system_prompt(
            "Focus on GNN model specifications, Active Inference frameworks, "
            "and scientific accuracy in your analysis."
        )
        
        task_prompts = {
            "summary": (
                "Provide a concise summary of this GNN model, highlighting:\n"
                "- Model purpose and domain\n"
                "- Key components and structure\n"
                "- Notable features or complexity\n"
                "Keep the summary under 500 words."
            ),
            "structure": (
                "Analyze this GNN model structure in detail, providing:\n"
                "1. Model purpose and application domain\n"
                "2. State variables and their dimensions\n"
                "3. Observation variables and modalities\n"
                "4. Control factors and actions\n"
                "5. Matrix relationships (A, B, C, D)\n"
                "6. Temporal dynamics and time horizon\n"
                "7. Notable features, complexity, or design patterns"
            ),
            "questions": (
                "Generate 5-7 insightful questions about this GNN model that would help understand:\n"
                "- Model behavior and dynamics\n"
                "- Implementation challenges\n"
                "- Potential applications and use cases\n"
                "- Active Inference principles involved\n"
                "- Scientific validity and assumptions\n"
                "Format as a numbered list."
            ),
            "general": (
                "Analyze this GNN model comprehensively, providing insights into "
                "its structure, purpose, and Active Inference implementation."
            )
        }
        
        task_prompt = task_prompts.get(analysis_type, task_prompts["general"])
        
        user_content = f"{task_prompt}\n\nGNN Model Content:\n{gnn_content}"
        
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content)
        ]
    
    def is_initialized(self) -> bool:
        """Check if the provider is properly initialized."""
        return self._is_initialized
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider_type": self.provider_type.value,
            "default_model": self.default_model,
            "available_models": self.available_models,
            "is_initialized": self.is_initialized()
        } 