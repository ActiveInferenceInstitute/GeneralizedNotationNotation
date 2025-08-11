#!/usr/bin/env python3
from __future__ import annotations
"""
OpenRouter LLM Provider

Implementation of the OpenRouter API for accessing multiple LLM models
through a unified interface. OpenRouter provides access to models from
OpenAI, Anthropic, Google, Meta, and other providers.
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

class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter implementation of the LLM provider interface."""
    
    # Popular models available through OpenRouter
    AVAILABLE_MODELS = [
        # OpenAI models
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-3.5-turbo",
        
        # Anthropic models
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-opus",
        
        # Google models
        "google/gemini-pro",
        "google/gemini-pro-vision",
        
        # Meta models
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        
        # Moonshot AI models
        "moonshotai/kimi-k2:free",
        
        # Other providers
        "mistralai/mistral-7b-instruct",
        "cohere/command-r-plus",
        "perplexity/llama-3.1-sonar-large-128k-online",
    ]
    
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (if None, will try environment variables)
            **kwargs: Additional OpenRouter-specific configuration
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.site_url = kwargs.get('site_url') or os.getenv('OPENROUTER_SITE_URL', 'http://localhost')
        self.site_name = kwargs.get('site_name') or os.getenv('OPENROUTER_SITE_NAME', 'GNN Pipeline')
        # Use environment variable for model selection if available
        self.preferred_model = os.getenv('OPENROUTER_MODEL', self.DEFAULT_MODEL)
        self.session = None
        
    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.OPENROUTER
    
    @property
    def default_model(self) -> str:
        """Return the default OpenRouter model."""
        return self.preferred_model
    
    @property
    def available_models(self) -> List[str]:
        """Return list of available OpenRouter models."""
        return self.AVAILABLE_MODELS.copy()
    
    def initialize(self) -> bool:
        """
        Initialize the OpenRouter client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not AIOHTTP_AVAILABLE:
            logger.error(f"Cannot initialize OpenRouter provider: aiohttp is not available ({AIOHTTP_IMPORT_ERROR})")
            logger.error("Install aiohttp with: pip install aiohttp>=3.9.0")
            return False
            
        if not self.api_key:
            logger.error("OpenRouter API key not provided")
            return False
            
        try:
            # Initialize aiohttp session with proper headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name
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
            logger.info("OpenRouter provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            return False
    
    def validate_config(self, config: LLMConfig) -> bool:
        """
        Validate OpenRouter-specific configuration.
        
        Args:
            config: LLM configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if config.model and config.model not in self.available_models:
            logger.warning(f"Model {config.model} not in available models list")
            
        # Validate parameter ranges (OpenRouter generally follows OpenAI conventions)
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
        Generate a response using OpenRouter API.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Returns:
            Standardized LLM response
        """
        if not self.is_initialized():
            raise RuntimeError("OpenRouter provider not initialized")
        
        # Use default config if none provided
        if config is None:
            config = LLMConfig()
        
        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        
        # Convert messages to OpenRouter format (follows OpenAI convention)
        openrouter_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
        
        # Prepare request payload
        payload = {
            "model": config.model or self.default_model,
            "messages": openrouter_messages,
            "stream": False
        }
        
        # Add optional parameters
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            payload["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            payload["presence_penalty"] = config.presence_penalty
        
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
                        "provider_info": data.get("provider", {}),
                        "native_finish_reason": choice.get("native_finish_reason")
                    }
                )
                
        except Exception as e:
            # Handle both aiohttp.ClientError and other exceptions  
            if AIOHTTP_AVAILABLE and hasattr(e, '__module__') and 'aiohttp' in e.__module__:
                logger.error(f"OpenRouter API call failed: {e}")
            else:
                logger.error(f"OpenRouter API call failed: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using OpenRouter API.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Yields:
            Chunks of the response content
        """
        if not self.is_initialized():
            raise RuntimeError("OpenRouter provider not initialized")
        
        # Use default config if none provided
        if config is None:
            config = LLMConfig(stream=True)
        else:
            config.stream = True
        
        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        
        # Convert messages to OpenRouter format
        openrouter_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
        
        # Prepare request payload
        payload = {
            "model": config.model or self.default_model,
            "messages": openrouter_messages,
            "stream": True
        }
        
        # Add optional parameters
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            payload["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            payload["presence_penalty"] = config.presence_penalty
        
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
            logger.error(f"OpenRouter streaming API call failed: {e}")
            raise
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Fetch the list of available models from OpenRouter.
        
        Returns:
            List of model information dictionaries
        """
        if not self.is_initialized():
            raise RuntimeError("OpenRouter provider not initialized")
        
        try:
            async with self.session.get(f"{self.BASE_URL}/models") as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("data", [])
                
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
            raise
    
    async def get_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific generation.
        
        Args:
            generation_id: The ID of the generation to query
            
        Returns:
            Generation information including usage and cost
        """
        if not self.is_initialized():
            raise RuntimeError("OpenRouter provider not initialized")
        
        try:
            async with self.session.get(
                f"{self.BASE_URL}/generation?id={generation_id}"
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"Failed to fetch generation info: {e}")
            raise
    
    async def close(self):
        """Close the OpenRouter client session."""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self._is_initialized = False
        logger.info("OpenRouter provider connection closed") 

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
            logger.error(f"OpenRouter analysis failed: {e}")
            return f"Analysis failed: {e}" 