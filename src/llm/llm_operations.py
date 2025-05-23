import os
import openai
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def load_api_key():
    """Loads the OpenAI API key from the .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in .env file or environment variables.")
        raise ValueError("OPENAI_API_KEY not found.")
    openai.api_key = api_key
    logger.info("OpenAI API key loaded successfully.")
    return api_key

def construct_prompt(contexts: list[str], task_description: str) -> str:
    """
    Constructs a prompt for the LLM by concatenating various context strings
    and a specific task description.

    Args:
        contexts (list[str]): A list of strings, where each string is a piece of context
                              (e.g., GNN file content, user query, previous responses).
        task_description (str): A string describing the specific task for the LLM.

    Returns:
        str: The fully constructed prompt.
    """
    full_context = "\n\n---\n\n".join(contexts)
    prompt = f"""{full_context}

---\n
Based on the context above, please perform the following task:
{task_description}
"""
    logger.debug(f"Constructed prompt: {prompt[:500]}...") # Log a preview
    return prompt

def get_llm_response(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 8000, temperature: float = 0.7, request_timeout: float = 30.0) -> str:
    """
    Sends a prompt to the specified OpenAI model and returns the response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4o-mini").
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): Controls randomness (0.0 to 2.0). Lower values are more deterministic.
        request_timeout (float): Timeout in seconds for the API request.

    Returns:
        str: The LLM's response or an error message.
    """
    try:
        logger.info(f"Sending prompt to OpenAI model: {model} (Timeout: {request_timeout}s, Max Tokens: {max_tokens}, Temperature: {temperature})")
        logger.debug(f"Full prompt for {model} (first 500 chars): {prompt[:500]}...")
        # Using the new client syntax for OpenAI API version >= 1.0
        client = openai.OpenAI()
        logger.info(f"Attempting to call OpenAI API with model {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specializing in analyzing and summarizing technical documents, particularly GNN (Generalized Notation Notation) files."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None,
            timeout=request_timeout
        )
        llm_response = response.choices[0].message.content.strip()
        response_length = len(llm_response)
        # Log token usage if available
        token_usage = response.usage
        if token_usage:
            logger.info(
                f"LLM call successful. Response length: {response_length} chars. "
                f"Token usage: Prompt={token_usage.prompt_tokens}, Completion={token_usage.completion_tokens}, Total={token_usage.total_tokens}"
            )
        else:
            logger.info(f"LLM call successful. Response length: {response_length} chars. Token usage not available in response.")
        logger.debug(f"LLM Response preview: {llm_response[:500]}...")
        return llm_response
    except openai.APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error after {request_timeout}s: {e}", exc_info=True)
        return f"Error: OpenAI API Timeout Error after {request_timeout}s - {e}"
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {e}", exc_info=True)
        return f"Error: OpenAI API Error - {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting LLM response: {e}", exc_info=True)
        return f"Error: An unexpected error occurred - {e}"

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