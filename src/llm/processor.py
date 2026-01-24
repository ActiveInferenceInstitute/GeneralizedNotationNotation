#!/usr/bin/env python3
"""
LLM processor module for GNN analysis.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import re
from datetime import datetime
import os
import subprocess
import shutil
import asyncio

try:
    from utils.pipeline_template import (
        log_step_start,
        log_step_success,
        log_step_error,
        log_step_warning
    )
except Exception:
    def log_step_start(logger, msg): logger.info(f"🚀 {msg}")
    def log_step_success(logger, msg): logger.info(f"✅ {msg}")
    def log_step_error(logger, msg): logger.error(f"❌ {msg}")
    def log_step_warning(logger, msg): logger.warning(f"⚠️ {msg}")

def _check_and_start_ollama(logger) -> tuple[bool, list[str]]:
    """
    Check if Ollama is available and running with enhanced detection.

    Returns:
        Tuple of (is_available, list_of_models)
    """
    try:
        # Check if ollama command exists
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            logger.info("ℹ️ Ollama not found in PATH - LLM analysis will use fallback")
            return False, []

        logger.info(f"🔍 Found Ollama at: {ollama_path}")

        # Check if Ollama is already running by trying 'ollama list'
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10  # Increased timeout
            )

            if result.returncode == 0:
                logger.info("✅ Ollama is running and ready")
                # Parse available models
                models = []
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Skip header
                        for line in lines[1:]:
                            parts = line.split()
                            if parts:
                                model_name = parts[0]
                                models.append(model_name)

                if models:
                    logger.info(f"📦 Available Ollama models ({len(models)}): {', '.join(models[:5])}")
                    if len(models) > 5:
                        logger.info(f"   ... and {len(models) - 5} more models")
                else:
                    logger.warning("⚠️ Ollama is running but no models are installed")
                    logger.info("To install a model, run: ollama pull smollm2:135m")

                return True, models

        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Ollama list command timed out (>10s)")
        except Exception as e:
            logger.debug(f"Ollama list check failed: {e}")

        # Try to start Ollama if it's not running
        logger.info("🔄 Attempting to start Ollama...")
        try:
            # Try to start Ollama in background using subprocess.Popen for non-blocking
            import subprocess
            import signal
            import os

            # Start Ollama serve in background
            ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group on Unix
            )

            logger.info(f"✅ Ollama started with PID {ollama_process.pid}")

            # Give it a moment to start up
            import time
            time.sleep(3)

            # Try to check again
            try:
                result = subprocess.run(
                    ['ollama', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    models = []
                    if result.stdout:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            for line in lines[1:]:
                                parts = line.split()
                                if parts:
                                    model_name = parts[0]
                                    models.append(model_name)

                    if models:
                        logger.info(f"📦 Available Ollama models: {', '.join(models)}")
                    else:
                        logger.warning("⚠️ Ollama started but no models are installed")
                        # Try to install the default model
                        logger.info("📥 Installing default model...")
                        install_result = subprocess.run(
                            ['ollama', 'pull', 'ministral-3:3b'],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if install_result.returncode == 0:
                            logger.info("✅ Default model installed successfully")
                            models = ['ministral-3:3b']
                        else:
                            logger.warning(f"⚠️ Failed to install default model: {install_result.stderr}")

                    return True, models

            except Exception as e:
                logger.debug(f"Post-start check failed: {e}")
                logger.info("ℹ️ Ollama may be starting up, but not ready yet")

        except Exception as e:
            logger.debug(f"Failed to start Ollama: {e}")
            logger.warning("⚠️ Could not start Ollama automatically")

        # If 'ollama list' failed, try to check if the service is running
        # by attempting a simple API endpoint check
        logger.info("🔄 Attempting to check Ollama serve status via API...")
        try:
            # Check if Ollama is serving by testing the API endpoint
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 11434))  # Default Ollama port
            sock.close()

            if result == 0:
                logger.info("✅ Ollama server is running on localhost:11434")
                logger.warning("⚠️ Could not list models, but server is responsive")
                return True, []
            else:
                logger.info("ℹ️ Ollama server not responding on localhost:11434")
        except Exception as e:
            logger.debug(f"Socket check failed: {e}")

        # Ollama exists but may not be running - provide helpful instructions
        logger.warning("⚠️ Ollama is installed but may not be running")
        logger.info("📝 To start Ollama, run in a separate terminal:")
        logger.info("   $ ollama serve")
        logger.info("📝 To install a lightweight model for testing:")
        logger.info("   $ ollama pull smollm2:135m")
        logger.info("   $ ollama pull tinyllama")
        logger.info("ℹ️ LLM analysis will use fallback mode without live model interaction")
        return False, []

    except Exception as e:
        logger.debug(f"Error checking Ollama availability: {e}")
        return False, []

def _select_best_ollama_model(available_models: list[str], logger) -> str:
    """
    Select the best available Ollama model for GNN analysis.
    
    Prioritizes small, fast models for quick analysis and reliability.
    """
    # Preference order: prioritize smaller/faster models for reliability
    preferred_models = [
        'smollm2:135m-instruct-q4_K_S',
        'smollm2:135m',
        'smollm2:360m',
        'tinyllama',
        'smollm2:1.7b',
        'smollm2',
        'gemma2:2b',
        'ministral-3:3b',
        'mistral:7b',
        'llama2:7b',
        'phi3',
        'llama2',
        'mistral'
    ]
    
    # Check environment variable override
    env_model = os.getenv('OLLAMA_MODEL') or os.getenv('OLLAMA_TEST_MODEL')
    if env_model:
        logger.info(f"🎯 Using model from environment: {env_model}")
        return env_model
    
    # Find first available model from preference list
    for preferred in preferred_models:
        for available in available_models:
            if available.startswith(preferred):
                logger.info(f"🎯 Selected model: {available}")
                return available
    
    # Fallback to first available model
    if available_models:
        model = available_models[0]
        logger.info(f"🎯 Using first available model: {model}")
        return model
    
    # Ultimate fallback - use the smallest model
    default_model = 'smollm2:135m-instruct-q4_K_S'
    logger.warning(f"⚠️ No models found, defaulting to: {default_model}")
    logger.info(f"   Note: You may need to run: ollama pull {default_model}")
    return default_model

from .analyzer import analyze_gnn_file_with_llm
from .generator import (
    generate_model_insights,
    generate_code_suggestions,
    generate_documentation,
    generate_llm_summary,
)
from .prompts import get_default_prompt_sequence, get_prompt, PromptType
from .llm_processor import LLMProcessor, ProviderType
from .providers.base_provider import LLMMessage, LLMConfig

def process_llm(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process GNN files with LLM-enhanced analysis.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    import asyncio
    
    # Run the async implementation in a single event loop
    try:
        return asyncio.run(_process_llm_async(target_dir, output_dir, verbose, **kwargs))
    except Exception as e:
        logger = logging.getLogger("llm")
        log_step_error(logger, f"LLM processing failed: {e}")
        return False

async def _process_llm_async(
    target_dir: Path, 
    output_dir: Path, 
    verbose: bool, 
    **kwargs
) -> bool:
    """Async implementation of process_llm."""
    logger = logging.getLogger("llm")
    
    # Initialize processor variable for cleanup in finally block
    processor = None
    
    try:
        log_step_start(logger, "Processing LLM with enhanced Ollama integration")
        
        # Check if Ollama is available and running with model detection
        ollama_available, ollama_models = _check_and_start_ollama(logger)
        
        # Select best model if Ollama is available
        selected_model = None
        if ollama_available and ollama_models:
            selected_model = _select_best_ollama_model(ollama_models, logger)
        elif ollama_available:
            # Ollama running but no models listed - use default
            selected_model = _select_best_ollama_model([], logger)
        else:
            logger.info("ℹ️ Proceeding with fallback LLM analysis (no live model interaction)")
        
        # Create results directory
        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "provider_matrix": {
                "ollama": {
                    "available": ollama_available,
                    "models": ollama_models,
                    "selected_model": selected_model
                },
                "openai": {
                    "available": bool(os.getenv("OPENAI_API_KEY")),
                    "models": ["gpt-4", "gpt-3.5-turbo"] if os.getenv("OPENAI_API_KEY") else [],
                    "selected_model": None
                },
                "anthropic": {
                    "available": bool(os.getenv("ANTHROPIC_API_KEY")),
                    "models": ["claude-3", "claude-2"] if os.getenv("ANTHROPIC_API_KEY") else [],
                    "selected_model": None
                }
            },
            "analysis_results": [],
            "model_insights": [],
            "code_suggestions": [],
            "documentation_generated": [],
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for LLM processing")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)
            
            # Initialize LLM processor (prioritize Ollama)
            processor_initialized = False
            try:
                # Create processor with Ollama prioritized
                processor = LLMProcessor(preferred_providers=[ProviderType.OLLAMA, ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.PERPLEXITY])
                processor_initialized = await processor.initialize()

                if not processor_initialized:
                    logger.warning("LLM processor initialization failed - using fallback analysis")
                else:
                    logger.info(f"LLM processor initialized with providers: {[p.value for p in processor.get_available_providers()]}")
            except Exception as e:
                logger.warning(f"LLM processor initialization failed: {e} - using fallback analysis")
                processor_initialized = False
                processor = None

            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Await the coroutine since we're in an async context
                    file_analysis = await analyze_gnn_file_with_llm(gnn_file, verbose)
                    results["analysis_results"].append(file_analysis)
                    
                    # Generate insights
                    insights = generate_model_insights(file_analysis)
                    results["model_insights"].append(insights)
                    
                    # Generate code suggestions
                    suggestions = generate_code_suggestions(file_analysis)
                    results["code_suggestions"].append(suggestions)
                    
                    # Generate documentation
                    docs = generate_documentation(file_analysis)
                    results["documentation_generated"].append(docs)

                    # If LLM processor is available, run structured prompts and save outputs
                    if processor_initialized and processor:
                        with open(gnn_file, 'r') as f:
                            gnn_content = f.read()
                        # Build custom prompt sequence including user-requested prompts
                        prompt_sequence = [
                            PromptType.SUMMARIZE_CONTENT,
                            PromptType.EXPLAIN_MODEL,
                            PromptType.IDENTIFY_COMPONENTS,
                            PromptType.ANALYZE_STRUCTURE,
                            PromptType.EXTRACT_PARAMETERS,
                            PromptType.PRACTICAL_APPLICATIONS,
                        ]

                        # Allow custom_prompts to be overridden via kwargs
                        custom_prompts = kwargs.get('custom_prompts', [
                            ("technical_description", "Describe this GNN model comprehensively, in technical detail."),
                            ("nontechnical_description", "Describe this GNN model comprehensively, in non-technical language suitable for a broad audience."),
                            ("runtime_behavior", "Describe what happens when this GNN model runs and how it would behave in different settings or domains."),
                        ])

                        per_file_dir = results_dir / f"prompts_{gnn_file.stem}"
                        per_file_dir.mkdir(parents=True, exist_ok=True)

                        prompt_outputs = {}
                        # Use selected model or fallback
                        ollama_model = selected_model if selected_model else 'ministral-3:3b'
                        logger.info(f"🤖 Using model '{ollama_model}' for LLM prompts")

                        # Ensure model is available, install if needed
                        if ollama_available and ollama_model not in ollama_models:
                            logger.info(f"📥 Installing model '{ollama_model}'...")
                            try:
                                install_result = subprocess.run(
                                    ['ollama', 'pull', ollama_model],
                                    capture_output=True,
                                    text=True,
                                    timeout=60
                                )
                                if install_result.returncode == 0:
                                    logger.info(f"✅ Model '{ollama_model}' installed successfully")
                                else:
                                    logger.warning(f"⚠️ Failed to install model '{ollama_model}': {install_result.stderr}")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not install model '{ollama_model}': {e}")

                        for idx, ptype in enumerate(prompt_sequence, start=1):
                            prompt_cfg = get_prompt(ptype, gnn_content)
                            messages = [
                                LLMMessage(role="system", content=prompt_cfg["system_message"]),
                                LLMMessage(role="user", content=prompt_cfg["user_prompt"]),
                            ]

                            # Log progress
                            logger.info(f"  📝 Running prompt {idx}/{len(prompt_sequence)}: {ptype.value}")

                            # Get max prompt timeout from kwargs - default to 300s (5m) to allow for provider fallbacks
                            max_prompt_timeout = kwargs.get('max_prompt_timeout', 300)

                            # Execute prompt
                            try:
                                # Use wait_for for timeout control
                                resp = await asyncio.wait_for(
                                    processor.get_response(
                                        messages=messages,
                                        max_tokens=min(512, prompt_cfg.get("max_tokens", 512)),
                                        temperature=0.2,
                                        config=LLMConfig(timeout=60)
                                    ), timeout=max_prompt_timeout
                                )
                                content = resp.content if hasattr(resp, 'content') else str(resp)

                                # Ensure we have some content, even if it's an error message
                                if not content or content.strip() == "":
                                    content = f"No response generated for prompt {ptype.value}. This may indicate that the LLM provider is not available or not responding."
                                prompt_outputs[ptype.value] = content
                                logger.debug(f"  ✅ Prompt completed successfully")
                            except asyncio.TimeoutError:
                                error_msg = f"Prompt execution timed out after {max_prompt_timeout} seconds"
                                logger.error(f"  ❌ {error_msg}")
                                prompt_outputs[ptype.value] = error_msg
                            except Exception as e:
                                error_msg = f"Prompt execution failed: {e}"
                                logger.error(f"  ❌ {error_msg}")
                                # Provide a meaningful fallback response
                                fallback_content = f"LLM analysis for {ptype.value} was not available. Please ensure that Ollama is running and the required model is installed."
                                prompt_outputs[ptype.value] = fallback_content

                            # Write to file
                            out_path = per_file_dir / f"{ptype.value}.md"
                            with open(out_path, 'w') as outf:
                                outf.write(f"# {ptype.name}\n\n")
                                outf.write(prompt_outputs[ptype.value] or "")

                        # Run custom free-form prompts
                        for cust_idx, (key, user_prompt) in enumerate(custom_prompts, start=1):
                            logger.info(f"  📝 Running custom prompt {cust_idx}/{len(custom_prompts)}: {key}")
                            
                            messages = [
                                LLMMessage(role="system", content="You are an expert in Active Inference and GNN specifications."),
                                LLMMessage(role="user", content=f"{user_prompt}\n\nGNN Model Content:\n{gnn_content}"),
                            ]

                            try:
                                # Use configurable timeout
                                resp = await asyncio.wait_for(
                                    processor.get_response(
                                        messages=messages,
                                        model_name=ollama_model,
                                        max_tokens=512,
                                        temperature=0.2,
                                        config=LLMConfig(timeout=60)
                                    ), timeout=max_prompt_timeout
                                )
                                content = resp.content if hasattr(resp, 'content') else str(resp)

                                # Ensure we have some content, even if it's an error message
                                if not content or content.strip() == "":
                                    content = f"No response generated for custom prompt {key}. This may indicate that the LLM provider is not available or not responding."
                                prompt_outputs[key] = content
                                logger.debug(f"  ✅ Custom prompt completed successfully")
                            except asyncio.TimeoutError:
                                error_msg = f"Prompt execution timed out after {max_prompt_timeout} seconds"
                                logger.error(f"  ❌ {error_msg}")
                                prompt_outputs[key] = error_msg
                            except Exception as e:
                                error_msg = f"Custom prompt execution failed: {e}"
                                logger.error(f"  ❌ {error_msg}")
                                # Provide a meaningful fallback response
                                fallback_content = f"LLM analysis for custom prompt {key} was not available. Please ensure that Ollama is running and the required model is installed."
                                prompt_outputs[key] = fallback_content
                            
                            out_path = per_file_dir / f"{key}.md"
                            with open(out_path, 'w') as outf:
                                title = key.replace('_', ' ').title()
                                outf.write(f"# {title}\n\n")
                                outf.write(f"Prompt:\n\n> {user_prompt}\n\n")
                                outf.write("Response:\n\n")
                                outf.write(prompt_outputs[key] or "")

                        # Attach to file analysis record
                        file_analysis["llm_prompt_outputs"] = prompt_outputs
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
        
        # Save detailed results
        results_file = results_dir / "llm_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary = generate_llm_summary(results)
        summary_file = results_dir / "llm_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        if results["success"]:
            log_step_success(logger, "LLM processing completed successfully")
        else:
            log_step_error(logger, "LLM processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, f"LLM processing failed: {e}")
        return False
    finally:
        # Close processor connections
        if processor:
            await processor.close()

