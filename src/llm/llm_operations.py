#!/usr/bin/env python3
"""
LLM Operations Module for GNN Pipeline

This module provides Large Language Model operations for analyzing
and processing GNN files, including summarization, analysis, and enhancement.

Updated to leverage the new multi-provider LLM system while maintaining
backward compatibility with existing code.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
import logging

# Import the new LLM system directly to avoid circular imports
from .llm_processor import (
    LLMProcessor, 
    AnalysisType, 
    ProviderType,
    load_api_keys_from_env,
    get_default_provider_configs,
    initialize_global_processor,
    get_processor as get_global_processor
)
from .providers.base_provider import LLMResponse

logger = logging.getLogger(__name__)

# Legacy configuration for backward compatibility
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 8000

class LLMOperations:
    """
    Main class for LLM operations on GNN content.
    
    This class now uses the multi-provider LLM system internally but maintains
    the same interface for backward compatibility.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_legacy: bool = False):
        """
        Initialize LLM operations.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from env)
            use_legacy: If True, use legacy OpenAI-only implementation
        """
        self.use_legacy = use_legacy
        self.processor: Optional[LLMProcessor] = None
        self._initialized = False
        
        if use_legacy:
            # Legacy initialization
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.client = None
            
            if self.api_key:
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=self.api_key)
                    logger.info("Initialized legacy OpenAI client")
                except ImportError:
                    logger.warning("OpenAI library not available")
            else:
                logger.warning("No OpenAI API key provided")
        else:
            # New multi-provider initialization
            try:
                # Try to use global processor first
                self.processor = get_global_processor()
                if self.processor:
                    self._initialized = True
                    logger.info("Using global LLM processor")
                else:
                    # Create new processor
                    api_keys = load_api_keys_from_env()
                    if api_key:
                        api_keys['openai'] = api_key
                    
                    self.processor = LLMProcessor(
                        api_keys=api_keys,
                        provider_configs=get_default_provider_configs()
                    )
                    logger.info("Created new LLM processor")
            except Exception as e:
                logger.error(f"Failed to initialize multi-provider system: {e}")
                # Fall back to legacy mode
                self.use_legacy = True
                self.__init__(api_key, use_legacy=True)
    
    async def _ensure_initialized(self) -> bool:
        """Ensure the processor is initialized."""
        if self.use_legacy:
            return self.client is not None
        
        if not self._initialized and self.processor:
            self._initialized = await self.processor.initialize()
        
        return self._initialized
    
    def construct_prompt(self, content_parts: List[str], task_description: str) -> str:
        """
        Construct a well-formatted prompt for LLM processing.
        
        Args:
            content_parts: List of content pieces to include
            task_description: Description of the task to perform
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are an expert in Active Inference and GNN (Generalized Notation Notation) specifications.",
            "",
            f"Task: {task_description}",
            "",
            "Content to analyze:"
        ]
        
        for i, content in enumerate(content_parts, 1):
            prompt_parts.extend([
                f"--- Content Part {i} ---",
                content,
                ""
            ])
        
        prompt_parts.extend([
            "Please provide a comprehensive and accurate response based on the content above.",
            "Focus on Active Inference concepts, model structure, and practical implications."
        ])
        
        return "\n".join(prompt_parts)
    
    def get_llm_response(self, prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """
        Get response from LLM for given prompt.
        
        Args:
            prompt: Input prompt for the LLM
            model: Model to use for generation
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response or error message
        """
        if self.use_legacy:
            return self._get_legacy_response(prompt, model, max_tokens)
        
        # Use async method correctly: if running inside an event loop, create a task and await
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside existing event loop; return coroutine for caller to await
            return self._get_async_response(prompt, model, max_tokens)
        else:
            try:
                return asyncio.run(self._get_async_response(prompt, model, max_tokens))
            except Exception as e:
                logger.error(f"Async LLM call failed: {e}")
                return f"Error: LLM call failed - {str(e)}"
    
    async def _get_async_response(self, prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Async version of get_llm_response."""
        if not await self._ensure_initialized():
            return "Error: LLM processor not initialized"
        
        try:
            # Use the new processor system
            response = await self.processor.get_response(
                messages=[{"role": "user", "content": prompt}],
                model_name=model,
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Multi-provider LLM call failed: {e}")
            return f"Error: LLM call failed - {str(e)}"
    
    def _get_legacy_response(self, prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Legacy OpenAI-only response method."""
        if not self.client:
            return "Error: LLM client not initialized (check API key)"
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in Active Inference and GNN specifications."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Legacy LLM API call failed: {e}")
            return f"Error: Legacy LLM API call failed - {str(e)}"
    
    def summarize_gnn(self, gnn_content: str, max_length: int = 500) -> str:
        """
        Generate a summary of GNN content.
        
        Args:
            gnn_content: The GNN file content to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        if not self.use_legacy and self.processor:
            # Use the new analysis system
            try:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Return coroutine for the caller to await
                    return self._async_summarize_gnn(gnn_content, max_length)
                else:
                    return asyncio.run(self._async_summarize_gnn(gnn_content, max_length))
            except Exception as e:
                logger.error(f"Async summarization failed: {e}")
                # Fall back to legacy method
        
        # Legacy method
        task_desc = f"Provide a concise summary (max {max_length} words) of this GNN model, highlighting key components and purpose."
        prompt = self.construct_prompt([gnn_content], task_desc)
        
        return self.get_llm_response(prompt, max_tokens=max_length*2)
    
    async def _async_summarize_gnn(self, gnn_content: str, max_length: int = 500) -> str:
        """Async version using new analysis system."""
        if not await self._ensure_initialized():
            raise Exception("Processor not initialized")
        
        response = await self.processor.analyze_gnn(
            gnn_content=gnn_content,
            analysis_type=AnalysisType.SUMMARY
        )
        
        return response.content
    
    def analyze_gnn_structure(self, gnn_content: str) -> str:
        """
        Analyze the structure and components of a GNN model.
        
        Args:
            gnn_content: The GNN file content to analyze
            
        Returns:
            Structured analysis
        """
        if not self.use_legacy and self.processor:
            # Use the new analysis system
            try:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    return self._async_analyze_structure(gnn_content)
                else:
                    return asyncio.run(self._async_analyze_structure(gnn_content))
            except Exception as e:
                logger.error(f"Async structure analysis failed: {e}")
                # Fall back to legacy method
        
        # Legacy method
        task_desc = """Analyze this GNN model structure and provide:
        1. Model purpose and domain
        2. Key state variables and observations
        3. Control factors and actions
        4. Matrix relationships (A, B, C, D)
        5. Notable features or complexity"""
        
        prompt = self.construct_prompt([gnn_content], task_desc)
        
        return self.get_llm_response(prompt, max_tokens=2000)
    
    async def _async_analyze_structure(self, gnn_content: str) -> str:
        """Async version using new analysis system."""
        if not await self._ensure_initialized():
            raise Exception("Processor not initialized")
        
        response = await self.processor.analyze_gnn(
            gnn_content=gnn_content,
            analysis_type=AnalysisType.STRUCTURE
        )
        
        return response.content
    
    def generate_questions(self, gnn_content: str, num_questions: int = 5) -> List[str]:
        """
        Generate relevant questions about a GNN model.
        
        Args:
            gnn_content: The GNN file content
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        if not self.use_legacy and self.processor:
            # Use the new analysis system
            try:
                return asyncio.run(self._async_generate_questions(gnn_content, num_questions))
            except Exception as e:
                logger.error(f"Async question generation failed: {e}")
                # Fall back to legacy method
        
        # Legacy method
        task_desc = f"""Generate {num_questions} insightful questions about this GNN model that would help understand:
        - Model behavior and dynamics
        - Implementation challenges
        - Potential applications
        - Active Inference principles involved
        
        Format as a numbered list."""
        
        prompt = self.construct_prompt([gnn_content], task_desc)
        response = self.get_llm_response(prompt, max_tokens=1000)
        
        # Extract questions from response
        return self._extract_questions_from_response(response, num_questions)
    
    async def _async_generate_questions(self, gnn_content: str, num_questions: int = 5) -> List[str]:
        """Async version using new analysis system."""
        if not await self._ensure_initialized():
            raise Exception("Processor not initialized")
        
        response = await self.processor.analyze_gnn(
            gnn_content=gnn_content,
            analysis_type=AnalysisType.QUESTIONS,
            additional_context={"num_questions": num_questions}
        )
        
        return self._extract_questions_from_response(response.content, num_questions)
    
    def _extract_questions_from_response(self, response: str, num_questions: int) -> List[str]:
        """Extract questions from LLM response."""
        lines = response.split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                question = line.split('.', 1)[-1].strip()
                if question:
                    questions.append(question)
        
        return questions[:num_questions]
    
    # New methods leveraging the multi-provider system
    def enhance_gnn(self, gnn_content: str) -> str:
        """
        Generate enhancement suggestions for a GNN model.
        
        Args:
            gnn_content: The GNN file content to enhance
            
        Returns:
            Enhancement suggestions
        """
        if not self.use_legacy and self.processor:
            try:
                return asyncio.run(self._async_enhance_gnn(gnn_content))
            except Exception as e:
                logger.error(f"Async enhancement failed: {e}")
        
        # Fallback to basic analysis
        task_desc = "Suggest improvements and enhancements for this GNN model, focusing on Active Inference best practices."
        prompt = self.construct_prompt([gnn_content], task_desc)
        return self.get_llm_response(prompt, max_tokens=2000)
    
    async def _async_enhance_gnn(self, gnn_content: str) -> str:
        """Async version using new analysis system."""
        if not await self._ensure_initialized():
            raise Exception("Processor not initialized")
        
        response = await self.processor.analyze_gnn(
            gnn_content=gnn_content,
            analysis_type=AnalysisType.ENHANCEMENT
        )
        
        return response.content
    
    def validate_gnn(self, gnn_content: str) -> str:
        """
        Validate a GNN model for correctness and completeness.
        
        Args:
            gnn_content: The GNN file content to validate
            
        Returns:
            Validation results
        """
        if not self.use_legacy and self.processor:
            try:
                return asyncio.run(self._async_validate_gnn(gnn_content))
            except Exception as e:
                logger.error(f"Async validation failed: {e}")
        
        # Fallback to basic analysis
        task_desc = "Validate this GNN model for correctness, completeness, and adherence to Active Inference principles."
        prompt = self.construct_prompt([gnn_content], task_desc)
        return self.get_llm_response(prompt, max_tokens=2000)
    
    async def _async_validate_gnn(self, gnn_content: str) -> str:
        """Async version using new analysis system."""
        if not await self._ensure_initialized():
            raise Exception("Processor not initialized")
        
        response = await self.processor.analyze_gnn(
            gnn_content=gnn_content,
            analysis_type=AnalysisType.VALIDATION
        )
        
        return response.content
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        if self.use_legacy:
            return ["openai"] if self.client else []
        
        if self.processor:
            return [p.value for p in self.processor.get_available_providers()]
        
        return []
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the LLM processor."""
        if self.use_legacy:
            return {
                "mode": "legacy",
                "providers": ["openai"] if self.client else [],
                "initialized": self.client is not None
            }
        
        if self.processor:
            return {
                "mode": "multi-provider",
                "providers": [p.value for p in self.processor.get_available_providers()],
                "initialized": self._initialized,
                "provider_info": self.processor.get_provider_info()
            }
        
        return {"mode": "uninitialized"}

# Global instance for easy access - now using multi-provider by default
llm_ops = LLMOperations(use_legacy=False)

# Convenience functions for backward compatibility
def construct_prompt(content_parts: List[str], task_description: str) -> str:
    """Convenience function for prompt construction."""
    return llm_ops.construct_prompt(content_parts, task_description)

def get_llm_response(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Convenience function for getting LLM response."""
    return llm_ops.get_llm_response(prompt, model, max_tokens)

def load_api_key() -> Optional[str]:
    """
    Load API key from environment or return None.
    
    Returns:
        API key string or None if not available
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        logger.info("OpenAI API key loaded from environment")
        return api_key
    return None

# Additional convenience functions for new capabilities
def summarize_gnn(gnn_content: str, max_length: int = 500) -> str:
    """Convenience function for GNN summarization."""
    return llm_ops.summarize_gnn(gnn_content, max_length)

def analyze_gnn_structure(gnn_content: str) -> str:
    """Convenience function for GNN structure analysis."""
    return llm_ops.analyze_gnn_structure(gnn_content)

def generate_questions(gnn_content: str, num_questions: int = 5) -> List[str]:
    """Convenience function for question generation."""
    return llm_ops.generate_questions(gnn_content, num_questions)

def enhance_gnn(gnn_content: str) -> str:
    """Convenience function for GNN enhancement."""
    return llm_ops.enhance_gnn(gnn_content)

def validate_gnn(gnn_content: str) -> str:
    """Convenience function for GNN validation."""
    return llm_ops.validate_gnn(gnn_content)

if __name__ == '__main__':
    # Example Usage (requires .env file with OPENAI_API_KEY)
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing llm_operations.py...")
    try:
        load_api_key()
        
        example_gnn_content = """
## ModelName
MyExampleGNN

## StateSpaceBlock
# States
S1: [Location] dimension(3) # {Location_A, Location_B, Location_C}
O1: [Observation] dimension(2) # {Obs_Hot, Obs_Cold}

## Connections
S1 -> O1
        """
        
        contexts = [
            "Context: This is a GNN file describing a simple model.",
            f"GNN File Content:\n{example_gnn_content}"
        ]
        
        task_summary = "Provide a concise summary of this GNN model, highlighting its key components."
        prompt_summary = construct_prompt(contexts, task_summary)
        print(f"\n--- Prompt for Summary ---\n{prompt_summary}")
        
        summary = get_llm_response(prompt_summary)
        print(f"\n--- LLM Summary ---\n{summary}")

        task_explanation = "Explain the purpose of the StateSpaceBlock in this GNN file in simple terms."
        prompt_explanation = construct_prompt([f"GNN File Content:\n{example_gnn_content}"], task_explanation)
        print(f"\n--- Prompt for Explanation ---\n{prompt_explanation}")
        explanation = get_llm_response(prompt_explanation)
        print(f"\n--- LLM Explanation ---\n{explanation}")

    except ValueError as ve:
        print(f"Setup Error: {ve}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}") 