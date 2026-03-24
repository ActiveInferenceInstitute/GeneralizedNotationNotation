#!/usr/bin/env python3
"""
LLM processor module for GNN analysis.
"""

from pathlib import Path
import logging
import json
from datetime import datetime
import os
import subprocess  # nosec B404 -- subprocess calls with controlled/trusted input
import shutil
from typing import Dict, Any, Tuple, List, Optional

try:
    import yaml
except ImportError:  # PyYAML is a project dependency; keep processor import-safe
    yaml = None  # type: ignore[assignment]

_logger = logging.getLogger(__name__)

try:
    from pipeline.config import get_pipeline_config
except ImportError:
    def get_pipeline_config() -> Dict[str, Any]:
        return {}

def _get_llm_config() -> dict:
    """Read LLM configuration from input/config.yaml, with pipeline config recovery."""
    try:
        if os.getenv('GNN_TESTING_NO_LLM_CONFIG'):
            return {}
        
        # Resolve input/config.yaml relative to project root (src/../input/config.yaml)
        config_path = Path(__file__).resolve().parent.parent.parent / "input" / "config.yaml"
        if config_path.exists():
            if yaml is None:
                _logger.debug("PyYAML not available; skipping input/config.yaml LLM section")
                return {}
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f) or {}
            return full_config.get("llm", {})
    except Exception as e:
        _logger.debug("LLM config YAML load failed: %s", e)
    # Recovery to pipeline config system
    try:
        config = get_pipeline_config()
        return config.get("llm", {})
    except Exception as e:
        _logger.debug("LLM pipeline config recovery failed: %s", e)
        return {}

def _model_is_cached(model_name: str, logger: logging.Logger) -> bool:
    """Check if an Ollama model is already cached locally using 'ollama show'."""
    try:
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            ['ollama', 'show', model_name, '--modelfile'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"✅ Model '{model_name}' is already cached locally")
            return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Model cache check failed for '{model_name}': {e}")
        return False
import asyncio

from utils.logging.logging_utils import log_step_start, log_step_success, log_step_error, log_step_warning

from .defaults import DEFAULT_OLLAMA_MODEL

def _start_ollama_if_needed(logger) -> tuple[bool, list[str]]:
    """
    Check if Ollama is available and running with enhanced detection.

    Returns:
        Tuple of (is_available, list_of_models)
    """
    try:
        # Check if ollama command exists
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            logger.info("ℹ️ Ollama not found in PATH - LLM analysis will use recovery")
            return False, []

        logger.info(f"🔍 Found Ollama at: {ollama_path}")

        # Check if Ollama is already running by trying 'ollama list'
        try:
            result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
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
                    logger.info(f"To install a model, run: ollama pull {DEFAULT_OLLAMA_MODEL}")

                return True, models

        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Ollama list command timed out (>10s)")
        except Exception as e:
            logger.debug(f"Ollama list check failed: {e}")

        # Try to start Ollama if it's not running
        logger.info("🔄 Attempting to start Ollama...")
        try:
            # Try to start Ollama in background using subprocess.Popen for non-blocking
            # Start Ollama serve in background
            ollama_process = subprocess.Popen(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
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
                result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
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
                        install_result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
                            ['ollama', 'pull', DEFAULT_OLLAMA_MODEL],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if install_result.returncode == 0:
                            logger.info("✅ Default model installed successfully")
                            models = [DEFAULT_OLLAMA_MODEL]
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
        logger.info(f"   $ ollama pull {DEFAULT_OLLAMA_MODEL}")
        logger.info("   $ ollama pull tinyllama")
        logger.info("ℹ️ LLM analysis will use recovery mode without live model interaction")
        return False, []

    except Exception as e:
        logger.debug(f"Error checking Ollama availability: {e}")
        return False, []

def _select_best_ollama_model(available_models: List[str], logger: logging.Logger) -> str:
    """
    Select the best available Ollama model for GNN analysis.
    
    Priority: environment variable > configured model (if installed) >
    preference list > first available > configured/default fallback.
    """
    # 1. Check environment variable override
    env_model = os.getenv('OLLAMA_MODEL') or os.getenv('OLLAMA_TEST_MODEL')
    if env_model:
        logger.info(f"🎯 Using model from environment: {env_model}")
        return env_model

    # 2. Respect config.yaml when that model is available locally
    llm_config = _get_llm_config()
    config_model = llm_config.get("model")
    if config_model:
        for available in available_models:
            if available.startswith(config_model):
                logger.info(f"🎯 Using model from config.yaml: {available}")
                return available

    # 3. Preference order: prioritize smaller/faster models for reliability
    preferred_models = [
        'smollm2',
        'tinyllama',
        'gemma3:4b',
        'gemma2:2b',
        'ministral-3:3b',
        'mistral:7b',
        'llama2:7b',
        'phi3',
        'llama2',
        'mistral',
    ]

    # Find first available model from preference list
    for preferred in preferred_models:
        for available in available_models:
            if available.startswith(preferred):
                logger.info(f"🎯 Selected model: {available}")
                return available

    # 4. Recovery to first available model
    if available_models:
        model = available_models[0]
        logger.info(f"🎯 Using first available model: {model}")
        return model

    # 5. Ultimate recovery
    default_model = llm_config.get("model", DEFAULT_OLLAMA_MODEL)
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
from .prompts import get_prompt, PromptType
from .llm_processor import LLMProcessor, ProviderType
from .providers.base_provider import LLMMessage, LLMConfig
from .cache import LLMCache

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
    import time as _time
    logger = logging.getLogger("llm")

    # Initialize processor variable for cleanup in finally block
    processor = None

    # Initialize LLM response cache
    cache = LLMCache(cache_dir=output_dir / ".cache")

    # Total budget: read from config, recovery to kwargs, recovery to 600s
    llm_config = _get_llm_config()
    TOTAL_BUDGET_SECONDS = kwargs.get('total_budget', llm_config.get('timeout_seconds', 600))
    budget_start = _time.monotonic()

    def _budget_remaining() -> float:
        return max(0.0, TOTAL_BUDGET_SECONDS - (_time.monotonic() - budget_start))

    try:
        log_step_start(logger, "Processing LLM with enhanced Ollama integration")

        # Check if Ollama is available and running with model detection
        ollama_available, ollama_models = _start_ollama_if_needed(logger)

        # Select best model if Ollama is available
        selected_model = None
        if ollama_available and ollama_models:
            selected_model = _select_best_ollama_model(ollama_models, logger)
        elif ollama_available:
            # Ollama running but no models listed - use default
            selected_model = _select_best_ollama_model([], logger)
        else:
            logger.info("ℹ️ Proceeding with recovery LLM analysis (no live model interaction)")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "auth_errors": [],
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

        # Track providers with auth failures to fail-fast on subsequent calls
        failed_auth_providers = set()

        # Find GNN files (recursive to handle subdirectory structure)
        gnn_files = list(target_dir.rglob("*.md"))
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
                    logger.warning("LLM processor initialization failed - using recovery analysis")
                else:
                    logger.info(f"LLM processor initialized with providers: {[p.value for p in processor.get_available_providers()]}")
            except Exception as e:
                logger.warning(f"LLM processor initialization failed: {e} - using recovery analysis")
                processor_initialized = False
                processor = None

            # Process each GNN file
            for file_idx, gnn_file in enumerate(gnn_files, 1):
                # Check total budget before starting a new file
                remaining = _budget_remaining()
                if remaining < 30:
                    logger.warning(f"⏱️ Budget exhausted after {file_idx-1}/{len(gnn_files)} files — skipping remaining")
                    break
                logger.info(f"📄 File {file_idx}/{len(gnn_files)}: {gnn_file.name} (budget: {remaining:.0f}s remaining)")
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
                        # Use selected model or recovery
                        ollama_model = selected_model if selected_model else DEFAULT_OLLAMA_MODEL
                        logger.info(f"🤖 Using model '{ollama_model}' for LLM prompts")

                        # Ensure model is available — use pre-pull guard to skip if cached
                        if ollama_available and ollama_model not in ollama_models:
                            if _model_is_cached(ollama_model, logger):
                                logger.info(f"⏭️ Skipping pull — '{ollama_model}' already cached")
                            else:
                                logger.info(f"📥 Pulling model '{ollama_model}' (not cached)...")
                                try:
                                    install_result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
                                        ['ollama', 'pull', ollama_model],
                                        capture_output=True,
                                        text=True,
                                        timeout=120
                                    )
                                    if install_result.returncode == 0:
                                        logger.info(f"✅ Model '{ollama_model}' pulled successfully")
                                    else:
                                        logger.warning(f"⚠️ Failed to pull model '{ollama_model}': {install_result.stderr}")
                                except subprocess.TimeoutExpired:
                                    logger.warning("⚠️ Model pull timed out after 120s — continuing with recovery")
                                except Exception as e:
                                    logger.warning(f"⚠️ Could not pull model '{ollama_model}': {e}")

                        for idx, ptype in enumerate(prompt_sequence, start=1):
                            # Check budget before each prompt
                            if _budget_remaining() < 30:
                                logger.warning(f"⏱️ Budget exhausted ({TOTAL_BUDGET_SECONDS}s), skipping remaining prompts for {gnn_file.name}")
                                break
                            prompt_cfg = get_prompt(ptype, gnn_content)
                            messages = [
                                LLMMessage(role="system", content=prompt_cfg["system_message"]),
                                LLMMessage(role="user", content=prompt_cfg["user_prompt"]),
                            ]

                            # Log progress
                            logger.info(f"  📝 Running prompt {idx}/{len(prompt_sequence)}: {ptype.value}")

                            # Per-prompt timeout: config-driven, recovery to kwargs
                            max_prompt_timeout = kwargs.get('max_prompt_timeout', llm_config.get('prompt_timeout', 45))

                            # Execute prompt — check cache first
                            prompt_text = prompt_cfg["user_prompt"]
                            cached = cache.get(gnn_content, ollama_model, prompt_text)
                            if cached is not None:
                                logger.info(f"  ⚡ Cache HIT for {ptype.value}")
                                prompt_outputs[ptype.value] = cached
                            else:
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
                                    cache.put(gnn_content, ollama_model, prompt_text, content)
                                    logger.debug("  ✅ Prompt completed successfully")
                                except asyncio.TimeoutError:
                                    error_msg = f"Prompt execution timed out after {max_prompt_timeout} seconds"
                                    logger.error(f"  ❌ {error_msg}")
                                    prompt_outputs[ptype.value] = error_msg
                                except Exception as e:
                                    error_str = str(e)
                                    # Fail-fast on auth errors (401/403) to avoid burning time
                                    if any(code in error_str for code in ["401", "403", "invalid_api_key", "Incorrect API key"]):
                                        auth_err = {"provider": "unknown", "error": error_str[:200]}
                                        # Detect which provider failed
                                        for prov_name in ["openai", "anthropic", "openrouter"]:
                                            if prov_name in error_str.lower():
                                                auth_err["provider"] = prov_name
                                                break
                                        if auth_err["provider"] not in failed_auth_providers:
                                            results["auth_errors"].append(auth_err)
                                            failed_auth_providers.add(auth_err["provider"])
                                            logger.error(f"  🔑 Auth failure for {auth_err['provider']} — skipping future calls to this provider")
                                        error_msg = f"Auth error ({auth_err['provider']}): {error_str[:100]}"
                                    else:
                                        error_msg = f"Prompt execution failed: {e}"
                                    logger.error(f"  ❌ {error_msg}")
                                    # Provide a meaningful recovery response
                                    fallback_content = f"LLM analysis for {ptype.value} was not available. Please ensure that Ollama is running and the required model is installed."
                                    prompt_outputs[ptype.value] = fallback_content

                            # Write to file
                            out_path = per_file_dir / f"{ptype.value}.md"
                            with open(out_path, 'w') as outf:
                                outf.write(f"# {ptype.name}\n\n")
                                outf.write(prompt_outputs[ptype.value] or "")

                        # Run custom free-form prompts
                        for cust_idx, (key, user_prompt) in enumerate(custom_prompts, start=1):
                            # Check budget before each custom prompt
                            if _budget_remaining() < 30:
                                logger.warning(f"⏱️ Budget exhausted ({TOTAL_BUDGET_SECONDS}s), skipping remaining custom prompts for {gnn_file.name}")
                                break
                            logger.info(f"  📝 Running custom prompt {cust_idx}/{len(custom_prompts)}: {key}")

                            messages = [
                                LLMMessage(role="system", content="You are an expert in Active Inference and GNN specifications."),
                                LLMMessage(role="user", content=f"{user_prompt}\n\nGNN Model Content:\n{gnn_content}"),
                            ]

                            # Cache check for custom prompts
                            cached = cache.get(gnn_content, ollama_model, user_prompt)
                            if cached is not None:
                                logger.info(f"  ⚡ Cache HIT for custom prompt {key}")
                                prompt_outputs[key] = cached
                            else:
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
                                    cache.put(gnn_content, ollama_model, user_prompt, content)
                                    logger.debug("  ✅ Custom prompt completed successfully")
                                except asyncio.TimeoutError:
                                    error_msg = f"Prompt execution timed out after {max_prompt_timeout} seconds"
                                    logger.error(f"  ❌ {error_msg}")
                                    prompt_outputs[key] = error_msg
                                except Exception as e:
                                    error_str = str(e)
                                    # Fail-fast on auth errors (401/403)
                                    if any(code in error_str for code in ["401", "403", "invalid_api_key", "Incorrect API key"]):
                                        auth_err = {"provider": "unknown", "error": error_str[:200]}
                                        for prov_name in ["openai", "anthropic", "openrouter"]:
                                            if prov_name in error_str.lower():
                                                auth_err["provider"] = prov_name
                                                break
                                        if auth_err["provider"] not in failed_auth_providers:
                                            results["auth_errors"].append(auth_err)
                                            failed_auth_providers.add(auth_err["provider"])
                                            logger.error(f"  🔑 Auth failure for {auth_err['provider']} — skipping future calls to this provider")
                                        error_msg = f"Auth error ({auth_err['provider']}): {error_str[:100]}"
                                    else:
                                        error_msg = f"Custom prompt execution failed: {e}"
                                    logger.error(f"  ❌ {error_msg}")
                                    # Provide a meaningful recovery response
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

        # Save detailed results (include cache stats)
        results["cache_stats"] = cache.summary()
        results_file = results_dir / "llm_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate summary report
        summary = generate_llm_summary(results)
        summary_file = results_dir / "llm_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)

        # Log cache summary
        cs = cache.summary()
        logger.info(f"📦 Cache: {cs['hits']} hits, {cs['misses']} misses, {cs['writes']} new entries ({cs['hit_ratio_pct']}% hit ratio)")

        # Surface auth errors in final status
        if results["auth_errors"]:
            providers_failed = [e["provider"] for e in results["auth_errors"]]
            log_step_warning(logger, f"LLM processing completed with auth errors for: {', '.join(providers_failed)}")
            logger.warning("💡 Check your API keys — invalid keys waste time on retries")
            # Mark as not fully successful so pipeline reports SUCCESS_WITH_WARNINGS
            return False
        elif results["success"]:
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

