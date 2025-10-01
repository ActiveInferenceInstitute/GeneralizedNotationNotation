#!/usr/bin/env python3
"""
LLM Processor - Unified Multi-Provider Interface

This module provides a unified interface for working with multiple LLM providers
(OpenAI, OpenRouter, Perplexity) within the GNN pipeline. It handles provider
selection, fallback mechanisms, and provides high-level methods for GNN analysis.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from enum import Enum
import logging

from .providers.base_provider import (
    BaseLLMProvider,
    ProviderType,
    LLMResponse,
    LLMMessage,
    LLMConfig,
)
from .providers import (
    get_openai_provider_class,
    get_openrouter_provider_class,
    get_perplexity_provider_class,
    get_ollama_provider_class,
)

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of GNN analysis that can be performed."""
    SUMMARY = "summary"
    STRUCTURE = "structure" 
    QUESTIONS = "questions"
    ENHANCEMENT = "enhancement"
    VALIDATION = "validation"
    COMPARISON = "comparison"
    SEARCH_ENHANCED = "search_enhanced"

def load_api_keys_from_env() -> Dict[str, str]:
    """
    Load API keys from environment variables.
    
    Returns:
        Dictionary mapping provider names to API keys
    """
    # Try to load .env file if it exists
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        
        # Look for .env file in the llm directory
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.debug(f"Loaded .env file from {env_file}")
        else:
            logger.debug("No .env file found in llm directory")
    except ImportError:
        logger.debug("python-dotenv not available, skipping .env file loading")
    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")
    
    api_keys = {}
    
    # OpenAI
    if openai_key := os.getenv('OPENAI_API_KEY'):
        api_keys['openai'] = openai_key
    
    # OpenRouter
    if openrouter_key := os.getenv('OPENROUTER_API_KEY'):
        api_keys['openrouter'] = openrouter_key
    
    # Perplexity
    if perplexity_key := os.getenv('PERPLEXITY_API_KEY'):
        api_keys['perplexity'] = perplexity_key
    
    # Ollama runs locally and doesn't need an API key; detect via env toggle
    if os.getenv('OLLAMA_DISABLED', '0') not in ('1', 'true', 'True'):
        api_keys['ollama'] = 'local'
    
    logger.info(f"Loaded API keys for providers: {list(api_keys.keys())}")
    if 'ollama' in api_keys and all(k not in api_keys for k in ('openai','openrouter','perplexity')):
        logger.info("No cloud API keys detected. Defaulting to local Ollama for LLM operations.")
    return api_keys

def get_default_provider_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get default provider configurations from environment variables.
    
    Returns:
        Dictionary with provider-specific configurations
    """
    configs = {
        'openai': {
            'organization': os.getenv('OPENAI_ORG_ID'),
            'base_url': os.getenv('OPENAI_BASE_URL')
        },
        'openrouter': {
            'site_url': os.getenv('OPENROUTER_SITE_URL', 'http://localhost'),
            'site_name': os.getenv('OPENROUTER_SITE_NAME', 'GNN Pipeline')
        },
        'perplexity': {
            # Perplexity-specific configs can be added here
        },
        'ollama': {
            # Fast small default; override with OLLAMA_MODEL
            'default_model': os.getenv('OLLAMA_MODEL', os.getenv('OLLAMA_TEST_MODEL', 'smollm2:135m-instruct-q4_K_S')),
            # Enforce low latency
            'default_max_tokens': int(os.getenv('OLLAMA_MAX_TOKENS', '256')),
            'timeout': float(os.getenv('OLLAMA_TIMEOUT', '60')),
            'base_url': os.getenv('OLLAMA_HOST')
        },
    }
    
    # Remove None values
    for provider, config in configs.items():
        configs[provider] = {k: v for k, v in config.items() if v is not None}
    
    return configs

def get_preferred_providers_from_env() -> List[ProviderType]:
    """
    Get preferred provider order from environment variables.
    
    Returns:
        List of providers in order of preference
    """
    default_provider = os.getenv('DEFAULT_PROVIDER', 'ollama').lower()
    
    # Map string names to provider types
    provider_map = {
        'openai': ProviderType.OPENAI,
        'openrouter': ProviderType.OPENROUTER,
        'perplexity': ProviderType.PERPLEXITY,
        'ollama': ProviderType.OLLAMA
    }
    
    # Create preferred order with default first
    preferred = []
    if default_provider in provider_map:
        preferred.append(provider_map[default_provider])
    
    # Add remaining providers
    for provider_type in ProviderType:
        if provider_type not in preferred:
            preferred.append(provider_type)
    
    return preferred

class LLMProcessor:
    """
    Main LLM processor that coordinates multiple providers for GNN analysis.
    
    Provides a unified interface for accessing different LLM providers with
    automatic fallback, load balancing, and provider-specific optimizations.
    """
    
    def __init__(
        self,
        preferred_providers: Optional[List[ProviderType]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        provider_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the LLM processor.
        
        Args:
            preferred_providers: List of providers in order of preference
            api_keys: Dictionary mapping provider names to API keys
            provider_configs: Provider-specific configuration options
        """
        self.providers: Dict[ProviderType, BaseLLMProvider] = {}
        self.preferred_providers = preferred_providers or [
            ProviderType.OPENAI,
            ProviderType.OPENROUTER,
            ProviderType.PERPLEXITY,
            ProviderType.OLLAMA,
        ]
        self.api_keys = api_keys or {}
        self.provider_configs = provider_configs or {}
        self._initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize all available providers.
        
        Returns:
            True if at least one provider initialized successfully
        """
        initialized_count = 0
        
        for provider_type in self.preferred_providers:
            try:
                provider = self._create_provider(provider_type)
                if provider and provider.initialize():
                    self.providers[provider_type] = provider
                    initialized_count += 1
                    logger.info(f"Initialized {provider_type.value} provider")
                else:
                    logger.warning(f"Failed to initialize {provider_type.value} provider")
                    
            except Exception as e:
                logger.error(f"Error initializing {provider_type.value}: {e}")
        
        self._initialized = initialized_count > 0
        
        if self._initialized:
            logger.info(f"LLM Processor initialized with {initialized_count} providers")
        else:
            logger.warning("No LLM providers could be initialized")
            
        return self._initialized
    
    def _create_provider(self, provider_type: ProviderType) -> Optional[BaseLLMProvider]:
        """
        Create a provider instance based on type.
        
        Args:
            provider_type: The type of provider to create
            
        Returns:
            Provider instance or None if creation failed
        """
        api_key = self.api_keys.get(provider_type.value)
        config = self.provider_configs.get(provider_type.value, {})
        
        if provider_type == ProviderType.OPENAI:
            return get_openai_provider_class()(api_key=api_key, **config)
        elif provider_type == ProviderType.OPENROUTER:
            return get_openrouter_provider_class()(api_key=api_key, **config)
        elif provider_type == ProviderType.PERPLEXITY:
            return get_perplexity_provider_class()(api_key=api_key, **config)
        elif provider_type == ProviderType.OLLAMA:
            return get_ollama_provider_class()(**config)
        else:
            logger.error(f"Unknown provider type: {provider_type}")
            return None
    
    def get_available_providers(self) -> List[ProviderType]:
        """Get list of available (initialized) providers."""
        return list(self.providers.keys())
    
    def get_provider(self, provider_type: ProviderType) -> Optional[BaseLLMProvider]:
        """
        Get a specific provider instance.
        
        Args:
            provider_type: Type of provider to retrieve
            
        Returns:
            Provider instance or None if not available
        """
        return self.providers.get(provider_type)
    
    def get_best_provider_for_task(self, analysis_type: AnalysisType) -> Optional[BaseLLMProvider]:
        """
        Select the best provider for a specific analysis type.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            Best provider for the task or None if no providers available
        """
        if not self.providers:
            return None
        
        # Provider preferences based on task type
        task_preferences = {
            AnalysisType.SEARCH_ENHANCED: [ProviderType.PERPLEXITY, ProviderType.OPENROUTER, ProviderType.OPENAI],
            AnalysisType.STRUCTURE: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.PERPLEXITY],
            AnalysisType.SUMMARY: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.PERPLEXITY],
            AnalysisType.QUESTIONS: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.PERPLEXITY],
            AnalysisType.ENHANCEMENT: [ProviderType.OPENROUTER, ProviderType.OPENAI, ProviderType.PERPLEXITY],
            AnalysisType.VALIDATION: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.PERPLEXITY],
            AnalysisType.COMPARISON: [ProviderType.OPENROUTER, ProviderType.OPENAI, ProviderType.PERPLEXITY]
        }
        
        preferred_order = task_preferences.get(analysis_type, self.preferred_providers)
        
        for provider_type in preferred_order:
            if provider_type in self.providers:
                return self.providers[provider_type]
        
        # Fallback to any available provider
        return next(iter(self.providers.values()))
    
    async def analyze_gnn(
        self,
        gnn_content: str,
        analysis_type: AnalysisType = AnalysisType.SUMMARY,
        provider_type: Optional[ProviderType] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Analyze GNN content using the appropriate provider.
        
        Args:
            gnn_content: The GNN file content to analyze
            analysis_type: Type of analysis to perform
            provider_type: Specific provider to use (optional)
            config: LLM configuration (optional)
            
        Returns:
            Analysis results
        """
        if not self._initialized:
            raise RuntimeError("LLM Processor not initialized")
        
        # Select provider
        if provider_type and provider_type in self.providers:
            provider = self.providers[provider_type]
        else:
            provider = self.get_best_provider_for_task(analysis_type)
        
        if not provider:
            raise RuntimeError("No suitable provider available")
        
        # Handle search-enhanced analysis for Perplexity
        if (analysis_type == AnalysisType.SEARCH_ENHANCED and 
            isinstance(provider, PerplexityProvider)):
            query = "Analyze this GNN model and provide insights based on current Active Inference research"
            return await provider.search_and_analyze(query, gnn_content)
        
        # Format messages based on analysis type
        messages = provider.format_gnn_analysis_prompt(gnn_content, analysis_type.value)
        
        try:
            return await provider.generate_response(messages, config)
        except Exception as e:
            logger.error(f"Analysis failed with {provider.provider_type.value}: {e}")
            # Try fallback provider
            return await self._try_fallback_analysis(gnn_content, analysis_type, provider_type, config)
    
    async def _try_fallback_analysis(
        self,
        gnn_content: str,
        analysis_type: AnalysisType,
        excluded_provider: Optional[ProviderType],
        config: Optional[LLMConfig]
    ) -> LLMResponse:
        """Try analysis with fallback providers."""
        available_providers = [
            p for p_type, p in self.providers.items() 
            if p_type != excluded_provider
        ]
        
        for provider in available_providers:
            try:
                messages = provider.format_gnn_analysis_prompt(gnn_content, analysis_type.value)
                return await provider.generate_response(messages, config)
            except Exception as e:
                logger.warning(f"Fallback failed with {provider.provider_type.value}: {e}")
                continue
        
        raise RuntimeError("All providers failed for analysis")
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        provider_type: Optional[ProviderType] = None,
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response.
        
        Args:
            messages: Conversation messages
            provider_type: Specific provider to use (optional)
            config: LLM configuration (optional)
            
        Yields:
            Response chunks
        """
        if not self._initialized:
            raise RuntimeError("LLM Processor not initialized")
        
        # Select provider
        if provider_type and provider_type in self.providers:
            provider = self.providers[provider_type]
        else:
            provider = next(iter(self.providers.values()))
        
        if not provider:
            raise RuntimeError("No providers available")
        
        async for chunk in provider.generate_stream(messages, config):
            yield chunk
    
    async def compare_providers(
        self,
        gnn_content: str,
        analysis_type: AnalysisType = AnalysisType.SUMMARY,
        config: Optional[LLMConfig] = None
    ) -> Dict[str, LLMResponse]:
        """
        Compare responses from multiple providers for the same GNN content.
        
        Args:
            gnn_content: The GNN file content to analyze
            analysis_type: Type of analysis to perform
            config: LLM configuration (optional)
            
        Returns:
            Dictionary mapping provider names to their responses
        """
        results = {}
        
        tasks = []
        for provider_type, provider in self.providers.items():
            task = self._analyze_with_provider(provider, gnn_content, analysis_type, config)
            tasks.append((provider_type.value, task))
        
        # Execute all analyses in parallel
        for provider_name, task in tasks:
            try:
                response = await task
                results[provider_name] = response
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                results[provider_name] = None
        
        return results
    
    async def _analyze_with_provider(
        self,
        provider: BaseLLMProvider,
        gnn_content: str,
        analysis_type: AnalysisType,
        config: Optional[LLMConfig]
    ) -> LLMResponse:
        """Helper method for provider-specific analysis."""
        if (analysis_type == AnalysisType.SEARCH_ENHANCED and 
            isinstance(provider, PerplexityProvider)):
            query = "Analyze this GNN model and provide insights based on current Active Inference research"
            return await provider.search_and_analyze(query, gnn_content)
        
        messages = provider.format_gnn_analysis_prompt(gnn_content, analysis_type.value)
        return await provider.generate_response(messages, config)
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available providers.
        
        Returns:
            Dictionary with provider information
        """
        return {
            provider_type.value: provider.get_provider_info()
            for provider_type, provider in self.providers.items()
        }
    
    async def close(self):
        """Close all provider connections."""
        for provider in self.providers.values():
            try:
                await provider.close()
            except Exception as e:
                logger.error(f"Error closing provider: {e}")
        
        self.providers.clear()
        self._initialized = False
        logger.info("LLM Processor closed")

    async def get_response(
        self,
        messages: List[LLMMessage],
        provider_type: Optional[ProviderType] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Get a response from an LLM provider.
        
        Args:
            messages: List of conversation messages
            provider_type: Specific provider to use (optional)
            model_name: Model to use (optional)
            max_tokens: Maximum tokens in response (optional)
            temperature: Temperature for generation (optional)
            config: Full LLM configuration (optional)
            
        Returns:
            LLM response
        """
        if not self._initialized:
            raise RuntimeError("LLM Processor not initialized")
        
        # Select provider
        if provider_type and provider_type in self.providers:
            provider = self.providers[provider_type]
        else:
            provider = next(iter(self.providers.values()))
        
        if not provider:
            raise RuntimeError("No providers available")
        
        # Create or update config
        if config is None:
            config = LLMConfig()
        
        if model_name:
            config.model = model_name
        if max_tokens:
            config.max_tokens = max_tokens
        if temperature is not None:
            config.temperature = temperature
        
        try:
            return await provider.generate_response(messages, config)
        except Exception as e:
            logger.error(f"Response generation failed with {provider.provider_type.value}: {e}")
            # Try fallback providers
            for fallback_provider in self.providers.values():
                if fallback_provider != provider:
                    try:
                        logger.info(f"Trying fallback provider: {fallback_provider.provider_type.value}")
                        return await fallback_provider.generate_response(messages, config)
                    except Exception as fallback_e:
                        logger.warning(f"Fallback provider {fallback_provider.provider_type.value} also failed: {fallback_e}")
                        continue
            
            # If all providers fail, raise the original exception
            raise e

# Global processor instance for easy access
_global_processor: Optional[LLMProcessor] = None

def get_processor() -> Optional[LLMProcessor]:
    """Get the global LLM processor instance."""
    return _global_processor

async def initialize_global_processor(
    preferred_providers: Optional[List[ProviderType]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    provider_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> LLMProcessor:
    """
    Initialize the global LLM processor instance.
    
    Args:
        preferred_providers: List of providers in order of preference
        api_keys: Dictionary mapping provider names to API keys
        provider_configs: Provider-specific configuration options
        
    Returns:
        Initialized LLM processor
    """
    global _global_processor
    
    # Use environment-based defaults if not provided
    if preferred_providers is None:
        preferred_providers = get_preferred_providers_from_env()
    
    if api_keys is None:
        api_keys = load_api_keys_from_env()
    
    if provider_configs is None:
        provider_configs = get_default_provider_configs()
    
    _global_processor = LLMProcessor(
        preferred_providers=preferred_providers,
        api_keys=api_keys,
        provider_configs=provider_configs
    )
    
    await _global_processor.initialize()
    return _global_processor

async def create_processor_from_env() -> LLMProcessor:
    """
    Create and initialize an LLM processor using environment variables.
    
    Returns:
        Initialized LLM processor configured from environment
    """
    return await initialize_global_processor()

async def close_global_processor():
    """Close the global LLM processor instance."""
    global _global_processor
    
    if _global_processor:
        await _global_processor.close()
        _global_processor = None 

class GNNLLMProcessor:
    """
    Specialized LLM processor for GNN model analysis and enhancement.
    
    This class provides GNN-specific analysis capabilities including
    model interpretation, validation, enhancement suggestions, and
    natural language explanations.
    """
    
    def __init__(self, base_processor: Optional[LLMProcessor] = None):
        """
        Initialize the GNN LLM processor.
        
        Args:
            base_processor: Base LLM processor to use for operations
        """
        self.base_processor = base_processor or LLMProcessor()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the GNN LLM processor.
        
        Returns:
            True if initialization successful
        """
        try:
            self.initialized = await self.base_processor.initialize()
            return self.initialized
        except Exception as e:
            logger.error(f"Failed to initialize GNN LLM processor: {e}")
            return False
    
    async def analyze_gnn_model(self, gnn_content: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """
        Analyze a GNN model using LLM capabilities.
        
        Args:
            gnn_content: GNN model content as string
            analysis_type: Type of analysis to perform
        
        Returns:
            Dictionary with analysis results
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "GNN LLM processor not initialized"
            }
        
        try:
            # Map analysis type to AnalysisType enum
            analysis_enum = AnalysisType.SUMMARY
            if analysis_type == "structure":
                analysis_enum = AnalysisType.STRUCTURE
            elif analysis_type == "enhancement":
                analysis_enum = AnalysisType.ENHANCEMENT
            elif analysis_type == "validation":
                analysis_enum = AnalysisType.VALIDATION
            elif analysis_type == "comparison":
                analysis_enum = AnalysisType.COMPARISON
            
            response = await self.base_processor.analyze_gnn(gnn_content, analysis_enum)
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "response": response.content,
                "provider": response.provider.value if response.provider else "unknown",
                "model": response.model,
                "usage": response.usage
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def generate_model_explanation(self, gnn_content: str) -> Dict[str, Any]:
        """
        Generate a natural language explanation of a GNN model.
        
        Args:
            gnn_content: GNN model content as string
        
        Returns:
            Dictionary with explanation results
        """
        return await self.analyze_gnn_model(gnn_content, "summary")
    
    async def suggest_enhancements(self, gnn_content: str) -> Dict[str, Any]:
        """
        Suggest enhancements for a GNN model.
        
        Args:
            gnn_content: GNN model content as string
        
        Returns:
            Dictionary with enhancement suggestions
        """
        return await self.analyze_gnn_model(gnn_content, "enhancement")
    
    async def validate_model(self, gnn_content: str) -> Dict[str, Any]:
        """
        Validate a GNN model using LLM analysis.
        
        Args:
            gnn_content: GNN model content as string
        
        Returns:
            Dictionary with validation results
        """
        return await self.analyze_gnn_model(gnn_content, "validation")
    
    async def compare_models(self, gnn_content_1: str, gnn_content_2: str) -> Dict[str, Any]:
        """
        Compare two GNN models.
        
        Args:
            gnn_content_1: First GNN model content
            gnn_content_2: Second GNN model content
        
        Returns:
            Dictionary with comparison results
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "GNN LLM processor not initialized"
            }
        
        try:
            # Create a combined analysis request
            combined_content = f"Model 1:\n{gnn_content_1}\n\nModel 2:\n{gnn_content_2}"
            response = await self.base_processor.analyze_gnn(combined_content, AnalysisType.COMPARISON)
            
            return {
                "success": True,
                "analysis_type": "comparison",
                "response": response.content,
                "provider": response.provider.value if response.provider else "unknown",
                "model": response.model,
                "usage": response.usage
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def close(self):
        """Close the GNN LLM processor."""
        if self.base_processor:
            await self.base_processor.close()
        self.initialized = False


def create_gnn_llm_processor() -> GNNLLMProcessor:
    """
    Create a GNN LLM processor with default configuration.
    
    Returns:
        GNNLLMProcessor instance
    """
    return GNNLLMProcessor()


async def analyze_gnn_with_llm(gnn_content: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """
    Convenience function to analyze GNN content with LLM.
    
    Args:
        gnn_content: GNN model content as string
        analysis_type: Type of analysis to perform
    
    Returns:
        Dictionary with analysis results
    """
    processor = create_gnn_llm_processor()
    
    try:
        await processor.initialize()
        result = await processor.analyze_gnn_model(gnn_content, analysis_type)
        return result
    finally:
        await processor.close()


async def explain_gnn_model(gnn_content: str) -> Dict[str, Any]:
    """
    Convenience function to explain a GNN model.
    
    Args:
        gnn_content: GNN model content as string
    
    Returns:
        Dictionary with explanation results
    """
    return await analyze_gnn_with_llm(gnn_content, "summary") 

def analyze_gnn_model(gnn_content: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """
    Analyze a GNN model using LLM capabilities.
    
    Args:
        gnn_content: GNN model content as string
        analysis_type: Type of analysis to perform
    
    Returns:
        Dictionary with analysis results
    """
    import asyncio
    
    try:
        # Create a new event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        result = loop.run_until_complete(analyze_gnn_with_llm(gnn_content, analysis_type))
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        } 

def generate_explanation(gnn_content: str) -> dict:
    """
    Generate a natural language explanation for a GNN model.
    
    Args:
        gnn_content: GNN model content as string
    
    Returns:
        Dictionary with explanation results
    """
    import asyncio
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        result = loop.run_until_complete(explain_gnn_model(gnn_content))
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__} 

def enhance_model(gnn_content: str) -> dict:
    """
    Suggest enhancements for a GNN model using LLM.
    
    Args:
        gnn_content: GNN model content as string
    
    Returns:
        Dictionary with enhancement suggestions
    """
    import asyncio
    try:
        processor = GNNLLMProcessor()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(processor.initialize())
        result = loop.run_until_complete(processor.suggest_enhancements(gnn_content))
        loop.run_until_complete(processor.close())
        return result
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__} 