#!/usr/bin/env python3
from __future__ import annotations
"""
OpenAI LLM Provider

Implementation of the OpenAI API for LLM operations within the GNN pipeline.
Supports both GPT-4 and GPT-3.5 models with full feature compatibility.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging

from .base_provider import (
    BaseLLMProvider, 
    ProviderType, 
    LLMResponse, 
    LLMMessage, 
    LLMConfig
)

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI implementation of the LLM provider interface."""
    
    # Available OpenAI models for GNN analysis
    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, will try environment variables)
            **kwargs: Additional OpenAI-specific configuration
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.organization = kwargs.get('organization') or os.getenv('OPENAI_ORG_ID')
        self.base_url = kwargs.get('base_url')
        
    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.OPENAI
    
    @property
    def default_model(self) -> str:
        """Return the default OpenAI model."""
        return self.DEFAULT_MODEL
    
    @property
    def available_models(self) -> List[str]:
        """Return list of available OpenAI models."""
        return self.AVAILABLE_MODELS.copy()
    
    def initialize(self) -> bool:
        """
        Initialize the OpenAI client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.api_key:
            logger.debug("OpenAI API key not provided - OpenAI provider will not be available")
            return False
            
        try:
            import openai
            
            client_kwargs = {
                'api_key': self.api_key
            }
            
            if self.organization:
                client_kwargs['organization'] = self.organization
                
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            
            self.client = openai.AsyncOpenAI(**client_kwargs)
            self._is_initialized = True
            
            logger.info("OpenAI provider initialized successfully")
            return True
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return False
        except Exception as e:
            logger.debug(f"OpenAI provider initialization issue (will use other providers if available): {e}")
            return False
    
    def validate_config(self, config: LLMConfig) -> bool:
        """
        Validate OpenAI-specific configuration.
        
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
        Generate a response using OpenAI API.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Returns:
            Standardized LLM response
        """
        if not self.is_initialized():
            raise RuntimeError("OpenAI provider not initialized")
        
        # Use default config if none provided
        if config is None:
            config = LLMConfig()
        
        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        
        # Convert messages to OpenAI format
        openai_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
        
        # Prepare request parameters
        request_params = {
            "model": config.model or self.default_model,
            "messages": openai_messages,
            "stream": False
        }
        
        # Add optional parameters
        if config.max_tokens is not None:
            request_params["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            request_params["temperature"] = config.temperature
        if config.top_p is not None:
            request_params["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            request_params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            request_params["presence_penalty"] = config.presence_penalty
        
        try:
            response = await self.client.chat.completions.create(**request_params)
            
            choice = response.choices[0]
            usage_dict = None
            
            if hasattr(response, 'usage') and response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return LLMResponse(
                content=choice.message.content or "",
                model_used=response.model,
                provider=self.provider_type.value,
                usage=usage_dict,
                finish_reason=choice.finish_reason,
                metadata={
                    "response_id": response.id,
                    "created": response.created,
                    "system_fingerprint": getattr(response, 'system_fingerprint', None)
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using OpenAI API.
        
        Args:
            messages: List of messages for the conversation
            config: Optional configuration parameters
            
        Yields:
            Chunks of the response content
        """
        if not self.is_initialized():
            raise RuntimeError("OpenAI provider not initialized")
        
        # Use default config if none provided
        if config is None:
            config = LLMConfig(stream=True)
        else:
            config.stream = True
        
        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")
        
        # Convert messages to OpenAI format
        openai_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
        
        # Prepare request parameters
        request_params = {
            "model": config.model or self.default_model,
            "messages": openai_messages,
            "stream": True
        }
        
        # Add optional parameters
        if config.max_tokens is not None:
            request_params["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            request_params["temperature"] = config.temperature
        if config.top_p is not None:
            request_params["top_p"] = config.top_p
        if config.frequency_penalty is not None:
            request_params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty is not None:
            request_params["presence_penalty"] = config.presence_penalty
        
        try:
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming API call failed: {e}")
            raise
    
    async def get_embeddings(
        self, 
        texts: List[str], 
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        Generate embeddings for text using OpenAI's embedding models.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        if not self.is_initialized():
            raise RuntimeError("OpenAI provider not initialized")
        
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=model
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI embeddings API call failed: {e}")
            raise
    
    async def close(self):
        """Close the OpenAI client connection."""
        if self.client and hasattr(self.client, 'close'):
            await self.client.close()
        
        self._is_initialized = False
        logger.info("OpenAI provider connection closed") 

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
            logger.error(f"OpenAI analysis failed: {e}")
            return f"Analysis failed: {e}" 