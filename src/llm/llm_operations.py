#!/usr/bin/env python3
"""
LLM Operations Module for GNN Pipeline

This module provides Large Language Model operations for analyzing
and processing GNN files, including summarization, analysis, and enhancement.
"""

import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# LLM Configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1000

class LLMOperations:
    """Main class for LLM operations on GNN content."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM operations.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from env)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI library not available")
        else:
            logger.warning("No OpenAI API key provided")
    
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
                temperature=0.3  # Lower temperature for more consistent responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return f"Error: LLM API call failed - {str(e)}"
    
    def summarize_gnn(self, gnn_content: str, max_length: int = 500) -> str:
        """
        Generate a summary of GNN content.
        
        Args:
            gnn_content: The GNN file content to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        task_desc = f"Provide a concise summary (max {max_length} words) of this GNN model, highlighting key components and purpose."
        prompt = self.construct_prompt([gnn_content], task_desc)
        
        return self.get_llm_response(prompt, max_tokens=max_length*2)
    
    def analyze_gnn_structure(self, gnn_content: str) -> str:
        """
        Analyze the structure and components of a GNN model.
        
        Args:
            gnn_content: The GNN file content to analyze
            
        Returns:
            Structured analysis
        """
        task_desc = """Analyze this GNN model structure and provide:
        1. Model purpose and domain
        2. Key state variables and observations
        3. Control factors and actions
        4. Matrix relationships (A, B, C, D)
        5. Notable features or complexity"""
        
        prompt = self.construct_prompt([gnn_content], task_desc)
        
        return self.get_llm_response(prompt, max_tokens=2000)
    
    def generate_questions(self, gnn_content: str, num_questions: int = 5) -> List[str]:
        """
        Generate relevant questions about a GNN model.
        
        Args:
            gnn_content: The GNN file content
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        task_desc = f"""Generate {num_questions} insightful questions about this GNN model that would help understand:
        - Model behavior and dynamics
        - Implementation challenges
        - Potential applications
        - Active Inference principles involved
        
        Format as a numbered list."""
        
        prompt = self.construct_prompt([gnn_content], task_desc)
        response = self.get_llm_response(prompt, max_tokens=1000)
        
        # Extract questions from response
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

# Global instance for easy access
llm_ops = LLMOperations()

# Convenience functions for backward compatibility
def construct_prompt(content_parts: List[str], task_description: str) -> str:
    """Convenience function for prompt construction."""
    return llm_ops.construct_prompt(content_parts, task_description)

def get_llm_response(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Convenience function for getting LLM response."""
    return llm_ops.get_llm_response(prompt, model, max_tokens)

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