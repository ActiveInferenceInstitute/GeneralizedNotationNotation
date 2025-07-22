#!/usr/bin/env python3
"""
Perplexity LLM Provider

Implementation of the Perplexity API for AI-powered search and reasoning.
Perplexity provides real-time web search capabilities combined with LLM reasoning.
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError as e:
    AIOHTTP_AVAILABLE = False
    AIOHTTP_IMPORT_ERROR = str(e)
    # Let aiohttp remain undefined - we'll check AIOHTTP_AVAILABLE before using it

from .base_provider import (
    BaseLLMProvider, 
    ProviderType, 
    LLMResponse, 
    LLMMessage, 
    LLMConfig
)

logger = logging.getLogger(__name__)

class PerplexityProvider(BaseLLMProvider):
    """Perplexity implementation of the LLM provider interface."""
    
    # Available Perplexity models
    AVAILABLE_MODELS = [
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-online", 
        "llama-3.1-sonar-huge-128k-online",
        "llama-3.1-sonar-small-128k-chat",
        "llama-3.1-sonar-large-128k-chat",
        "llama-3.1-8b-instruct",
        "llama-3.1-70b-instruct"
    ]
    
    DEFAULT_MODEL = "llama-3.1-sonar-large-128k-online"
    BASE_URL = "https://api.perplexity.ai"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Perplexity provider.
        
        Args:
            api_key: Perplexity API key (if None, will try environment variables)
            **kwargs: Additional Perplexity-specific configuration
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.session = None
        
    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.PERPLEXITY
    
    @property
    def default_model(self) -> str:
        """Return the default Perplexity model."""
        return self.DEFAULT_MODEL
    
    @property
    def available_models(self) -> List[str]:
        """Return list of available Perplexity models."""
        return self.AVAILABLE_MODELS.copy()
    
    def initialize(self) -> bool:
        """
        Initialize the Perplexity client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.error(f"Cannot initialize Perplexity provider: aiohttp is not available ({AIOHTTP_IMPORT_ERROR})")
            logger.error("Install aiohttp with: pip install aiohttp>=3.9.0")
            return False
            
        if not self.api_key:
            logger.error("Perplexity API key not provided")
            return False
            
        try:
            # Initialize aiohttp session with proper headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            if not AIOHTTP_AVAILABLE:
                raise ImportError(f"aiohttp is required but not available: {AIOHTTP_IMPORT_ERROR}")
            
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout
            )
            
            self._is_initialized = True
            logger.info("Perplexity provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Perplexity client: {e}")
            return False
    
    def validate_config(self, config: LLMConfig) -> bool:
        """
        Validate Perplexity-specific configuration.
        
        Args:
            config: LLM configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if config.model and config.model not in self.available_models:
            logger.warning(f"Model {config.model} not in available models list")
            
        # Validate parameter ranges
        if config.temperature is not None and not (0.0 <= config.temperature <= 2.0):
            logger.error("Temperature must be between 0.0 and 2.0")
            return False
            
        if config.top_p is not None and not (0.0 <= config.top_p <= 1.0):
            logger.error("top_p must be between 0.0 and 1.0")
            return False
            
        if config.max_tokens is not None and config.max_tokens <= 0:
            logger.error("max_tokens must be positive")
            return False
            
        return True
    
    async def generate_response(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generate a response using Perplexity API.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Returns:
            Standardized LLM response
        """
        if not self.is_initialized():
            raise RuntimeError("Perplexity provider not initialized")
        
        # Use default config if none provided
        if config is None:
            config = LLMConfig()
        
        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        
        # Convert messages to Perplexity format (follows OpenAI convention)
        perplexity_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        
        # Prepare request payload
        payload = {
            "model": config.model or self.default_model,
            "messages": perplexity_messages,
            "stream": False
        }
        
        # Add optional parameters
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        
        try:
            async with self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                choice = data["choices"][0]
                usage_dict = None
                
                if "usage" in data:
                    usage_dict = {
                        "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                        "completion_tokens": data["usage"].get("completion_tokens", 0),
                        "total_tokens": data["usage"].get("total_tokens", 0)
                    }
                
                return LLMResponse(
                    content=choice["message"]["content"] or "",
                    model_used=data.get("model", payload["model"]),
                    provider=self.provider_type.value,
                    usage=usage_dict,
                    finish_reason=choice.get("finish_reason"),
                    metadata={
                        "response_id": data.get("id"),
                        "created": data.get("created"),
                        "citations": data.get("citations", []),
                        "web_results": data.get("web_results", [])
                    }
                )
                
        except Exception as e:
            # Handle both aiohttp.ClientError and other exceptions
            if AIOHTTP_AVAILABLE and hasattr(e, '__module__') and 'aiohttp' in e.__module__:
                logger.error(f"Perplexity API call failed: {e}")
            else:
                logger.error(f"Perplexity API call failed: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using Perplexity API.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Yields:
            Chunks of the response content
        """
        if not self.is_initialized():
            raise RuntimeError("Perplexity provider not initialized")
        
        # Use default config if none provided
        if config is None:
            config = LLMConfig(stream=True)
        else:
            config.stream = True
        
        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        
        # Convert messages to Perplexity format
        perplexity_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        
        # Prepare request payload
        payload = {
            "model": config.model or self.default_model,
            "messages": perplexity_messages,
            "stream": True
        }
        
        # Add optional parameters
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        
        try:
            async with self.session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    
                    if line_text.startswith('data: '):
                        data_str = line_text[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_str)
                            if (chunk_data.get("choices") and 
                                chunk_data["choices"][0].get("delta") and
                                chunk_data["choices"][0]["delta"].get("content")):
                                yield chunk_data["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            # Skip malformed chunks
                            continue
                            
        except Exception as e:
            logger.error(f"Perplexity streaming API call failed: {e}")
            raise
    
    def is_online_model(self, model_name: str) -> bool:
        """
        Check if a model has online search capabilities.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model has online search, False otherwise
        """
        return "online" in model_name.lower()
    
    def construct_search_prompt(self, query: str, gnn_context: str = "") -> List[LLMMessage]:
        """
        Construct a search-optimized prompt for Perplexity.
        
        Args:
            query: The search query
            gnn_context: Optional GNN context for domain-specific search
            
        Returns:
            List of formatted messages optimized for search
        """
        system_prompt = self.construct_system_prompt(
            "You have access to real-time web search. Use this capability to provide "
            "current information about Active Inference, GNN specifications, and related "
            "research. Always cite your sources when using web-retrieved information."
        )
        
        search_context = ""
        if gnn_context:
            search_context = f"\n\nContext: I'm working with GNN (Generalized Notation Notation) files and Active Inference models. Here's relevant context:\n{gnn_context}"
        
        user_content = f"{query}{search_context}"
        
        return [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content)
        ]
    
    async def search_and_analyze(
        self, 
        query: str, 
        gnn_content: str = "",
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Perform a search-enhanced analysis using Perplexity's online capabilities.
        
        Args:
            query: The search query or analysis request
            gnn_content: Optional GNN content to analyze
            model: Optional model to use (defaults to online model)
            
        Returns:
            Search-enhanced LLM response with citations
        """
        # Use online model by default for search capabilities
        search_model = model or "llama-3.1-sonar-large-128k-online"
        
        config = LLMConfig(model=search_model)
        messages = self.construct_search_prompt(query, gnn_content)
        
        return await self.generate_response(messages, config)
    
    def analyze(self, content: str, task: str) -> str:
        """Perform analysis on GNN content."""
        prompt = f"Analyze this GNN model for {task}: {content}"
        
        # Handle async call properly
        try:
            # Try to run in existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop in a separate thread
                import concurrent.futures
                import threading
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(
                            self.generate_response([{"role": "user", "content": prompt}])
                        )
                        return result.content if hasattr(result, 'content') else str(result)
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result(timeout=30)
            else:
                # Run directly in current loop
                result = loop.run_until_complete(
                    self.generate_response([{"role": "user", "content": prompt}])
                )
                return result.content if hasattr(result, 'content') else str(result)
                
        except RuntimeError:
            # No event loop, create new one
            result = asyncio.run(
                self.generate_response([{"role": "user", "content": prompt}])
            )
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            logger.error(f"Perplexity analysis failed: {e}")
            return f"Analysis failed: {e}"
    
    async def close(self):
        """Close the Perplexity client session."""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self._is_initialized = False
        logger.info("Perplexity provider connection closed") 