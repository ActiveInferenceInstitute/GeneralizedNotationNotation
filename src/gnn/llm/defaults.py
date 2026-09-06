"""
Central defaults for local Ollama usage.

Override precedence (highest first):
``OLLAMA_MODEL`` env → ``input/config.yaml`` key ``llm.model`` (read in
``llm.processor``) → this constant.
"""

DEFAULT_OLLAMA_MODEL = "smollm2:135m-instruct-q4_K_S"
