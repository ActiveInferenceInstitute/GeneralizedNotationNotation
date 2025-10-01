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

def _check_and_start_ollama(logger) -> bool:
    """
    Check if Ollama is available and running. Start it if available but not running.
    
    Returns:
        True if Ollama is available and running, False otherwise
    """
    try:
        # Check if ollama command exists
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            logger.info("Ollama not found in PATH - LLM analysis will use fallback")
            return False
        
        # Check if Ollama is already running by trying 'ollama list'
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info("✅ Ollama is running and ready")
                # Log available models
                if result.stdout:
                    models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
                    if models:
                        logger.info(f"Available Ollama models: {', '.join(models[:3])}")
                return True
        except subprocess.TimeoutExpired:
            logger.warning("Ollama list command timed out")
        except Exception as e:
            logger.debug(f"Ollama list check failed: {e}")
        
        # If 'ollama list' failed, try to check if the service is running
        # by attempting a simple serve check (non-blocking)
        logger.info("Attempting to check Ollama serve status...")
        try:
            # Check if Ollama is serving by testing the API endpoint
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 11434))  # Default Ollama port
            sock.close()
            
            if result == 0:
                logger.info("✅ Ollama server is running on localhost:11434")
                return True
            else:
                logger.info("Ollama server not responding on localhost:11434")
        except Exception as e:
            logger.debug(f"Socket check failed: {e}")
        
        # Ollama exists but may not be running - inform user
        logger.warning("⚠️ Ollama is installed but may not be running")
        logger.info("To use Ollama for LLM analysis, please run: ollama serve")
        logger.info("LLM analysis will use fallback mode without live model interaction")
        return False
        
    except Exception as e:
        logger.debug(f"Error checking Ollama availability: {e}")
        return False

from .analyzer import analyze_gnn_file_with_llm
from .generator import (
    generate_model_insights,
    generate_code_suggestions,
    generate_documentation,
    generate_llm_summary,
)
from .prompts import get_default_prompt_sequence, get_prompt, PromptType
from .llm_processor import LLMProcessor
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
    logger = logging.getLogger("llm")
    
    try:
        log_step_start(logger, "Processing LLM")
        
        # Check if Ollama is available and running
        ollama_available = _check_and_start_ollama(logger)
        
        # Create results directory
        results_dir = output_dir / "llm_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "analysis_results": [],
            "model_insights": [],
            "code_suggestions": [],
            "documentation_generated": [],
            "ollama_available": ollama_available
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for LLM processing")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)
            
            # Initialize LLM processor (will prefer any available provider; Ollama if present)
            processor = LLMProcessor()
            try:
                import asyncio
                async def _init():
                    return await processor.initialize()
                processor_initialized = asyncio.run(_init())
            except Exception:
                processor_initialized = False
                logger.warning("LLM processor initialization failed - using fallback analysis")

            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    file_analysis = analyze_gnn_file_with_llm(gnn_file, verbose)
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
                    if processor_initialized:
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

                        custom_prompts = [
                            ("technical_description", "Describe this GNN model comprehensively, in technical detail."),
                            ("nontechnical_description", "Describe this GNN model comprehensively, in non-technical language suitable for a broad audience."),
                            ("runtime_behavior", "Describe what happens when this GNN model runs and how it would behave in different settings or domains."),
                        ]

                        per_file_dir = results_dir / f"prompts_{gnn_file.stem}"
                        per_file_dir.mkdir(parents=True, exist_ok=True)

                        prompt_outputs = {}
                        # Optional model override for Ollama; prefer smallest fast default
                        ollama_model = os.getenv('OLLAMA_MODEL', os.getenv('OLLAMA_TEST_MODEL', 'smollm2:135m-instruct-q4_K_S'))

                        for idx, ptype in enumerate(prompt_sequence, start=1):
                            prompt_cfg = get_prompt(ptype, gnn_content)
                            messages = [
                                LLMMessage(role="system", content=prompt_cfg["system_message"]),
                                LLMMessage(role="user", content=prompt_cfg["user_prompt"]),
                            ]

                            # Run generation with enhanced timeout handling
                            def _run_prompt():
                                async def _inner():
                                    try:
                                        # Use timeout manager for LLM calls
                                        from utils.timeout_manager import get_llm_timeout_manager, TimeoutConfig
                                        timeout_manager = get_llm_timeout_manager()
                                        
                                        # Configure timeout based on prompt complexity
                                        prompt_length = len(prompt_cfg["user_prompt"])
                                        base_timeout = min(120, max(30, prompt_length / 100))  # Dynamic timeout
                                        
                                        timeout_config = TimeoutConfig(
                                            base_timeout=base_timeout,
                                            max_timeout=300,
                                            max_retries=2,
                                            retry_delay=3.0
                                        )
                                        
                                        async def llm_operation():
                                            return await processor.get_response(
                                                messages=messages,
                                                model_name=ollama_model,
                                                max_tokens=min(512, prompt_cfg.get("max_tokens", 512)),
                                                temperature=0.2,
                                                config=LLMConfig(timeout=base_timeout)
                                            )
                                        
                                        result = await timeout_manager._execute_with_timeout_async(
                                            f"llm_prompt_{ptype.value}", timeout_config, llm_operation
                                        )
                                        
                                        if result.success:
                                            if result.used_fallback:
                                                logger.warning(f"Using fallback result for prompt {ptype.value}")
                                            return result.result.content if hasattr(result.result, 'content') else str(result.result)
                                        else:
                                            return f"Prompt execution failed after {result.attempts} attempts: {result.error}"
                                            
                                    except ImportError:
                                        # Fallback to original timeout handling if timeout manager not available
                                        logger.warning("Timeout manager not available, using basic timeout")
                                        try:
                                            resp = await asyncio.wait_for(
                                                processor.get_response(
                                                    messages=messages,
                                                    model_name=ollama_model,
                                                    max_tokens=min(512, prompt_cfg.get("max_tokens", 512)),
                                                    temperature=0.2,
                                                    config=LLMConfig(timeout=60)
                                                ), timeout=120
                                            )
                                            return resp.content
                                        except asyncio.TimeoutError:
                                            return f"Prompt execution timed out after 120 seconds"
                                        except Exception as e:
                                            return f"Prompt execution failed: {e}"
                                    except Exception as e:
                                        return f"Prompt execution failed: {e}"
                                return asyncio.run(_inner())

                            # Log which prompt we're running
                            logger.info(f"Running structured prompt {idx}/{len(prompt_sequence)}: {ptype.value} for {gnn_file.name}")
                            content = _run_prompt()
                            prompt_outputs[ptype.value] = content

                            # Write to file
                            out_path = per_file_dir / f"{ptype.value}.md"
                            with open(out_path, 'w') as outf:
                                outf.write(f"# {ptype.name}\n\n")
                                outf.write(content or "")

                        # Run custom free-form prompts
                        for key, user_prompt in custom_prompts:
                            messages = [
                                LLMMessage(role="system", content="You are an expert in Active Inference and GNN specifications."),
                                LLMMessage(role="user", content=f"{user_prompt}\n\nGNN Model Content:\n{gnn_content}"),
                            ]

                            def _run_custom():
                                async def _inner():
                                    try:
                                        resp = await processor.get_response(
                                            messages=messages,
                                            model_name=ollama_model,
                                            max_tokens=512,
                                            temperature=0.2,
                                            config=LLMConfig(timeout=60)
                                        )
                                        return resp.content
                                    except Exception as e:
                                        return f"Prompt execution failed: {e}"
                                return asyncio.run(_inner())

                            logger.info(f"Running custom prompt: {key} for {gnn_file.name}")
                            content = _run_custom()
                            prompt_outputs[key] = content
                            out_path = per_file_dir / f"{key}.md"
                            with open(out_path, 'w') as outf:
                                title = key.replace('_', ' ').title()
                                outf.write(f"# {title}\n\n")
                                outf.write(f"Prompt:\n\n> {user_prompt}\n\n")
                                outf.write("Response:\n\n")
                                outf.write(content or "")

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
