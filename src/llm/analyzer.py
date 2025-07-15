from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
import time
import json

# Import LLM functionality
try:
    from .providers.openai_provider import OpenAIProvider
    from .providers.openrouter_provider import OpenRouterProvider
    from .providers.perplexity_provider import PerplexityProvider
    LLM_PROVIDERS_AVAILABLE = True
except ImportError as e:
    LLM_PROVIDERS_AVAILABLE = False

def analyze_gnn_files(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Analyze GNN files using LLM providers.
    
    Args:
        target_dir: Directory containing GNN files to analyze
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional analysis options (e.g., llm_tasks, llm_timeout)
        
    Returns:
        True if analysis succeeded, False otherwise
    """
    log_step_start(logger, "Analyzing GNN files with LLM providers")
    
    # Use centralized output directory configuration
    llm_output_dir = get_output_dir_for_script("11_llm.py", output_dir)
    llm_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not LLM_PROVIDERS_AVAILABLE:
        log_step_error(logger, "LLM providers not available")
        return False
    
    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir}")
        return True
    
    logger.info(f"Found {len(gnn_files)} GNN files to analyze")
    
    # Initialize LLM providers
    providers = {}
    try:
        providers['openai'] = OpenAIProvider()
        providers['openrouter'] = OpenRouterProvider()
        providers['perplexity'] = PerplexityProvider()
        logger.debug("LLM providers initialized successfully")
    except Exception as e:
        log_step_warning(logger, f"Some LLM providers failed to initialize: {e}")
    
    if not providers:
        log_step_error(logger, "No LLM providers available")
        return False
    
    successful_analyses = 0
    failed_analyses = 0
    
    try:
        with performance_tracker.track_operation("analyze_all_gnn_files"):
            for gnn_file in gnn_files:
                try:
                    logger.debug(f"Analyzing file: {gnn_file}")
                    
                    # Read GNN file content
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        gnn_content = f.read()
                    
                    # Create file-specific output directory
                    file_output_dir = llm_output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Analyze with each provider
                    file_success = True
                    analysis_results = {
                        "file": str(gnn_file),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "providers": {}
                    }
                    
                    for provider_name, provider in providers.items():
                        try:
                            with performance_tracker.track_operation(f"analyze_{provider_name}_{gnn_file.name}"):
                                # Generate analysis prompt
                                prompt = f"""Analyze the following GNN (Generalized Notation Notation) model specification:

{gnn_content}

Please provide:
1. A clear explanation of what this model represents
2. Key components and their relationships
3. Potential applications or use cases
4. Any potential issues or improvements
5. Suggestions for optimization

Focus on the Active Inference aspects and mathematical relationships."""
                                
                                # Get analysis from provider
                                response = provider.analyze(prompt, max_tokens=2000)
                                
                                if response and response.strip():
                                    # Save individual provider analysis
                                    provider_file = file_output_dir / f"{provider_name}_analysis.md"
                                    with open(provider_file, 'w', encoding='utf-8') as f:
                                        f.write(f"# {provider_name.upper()} Analysis of {gnn_file.name}\n\n")
                                        f.write(f"**Generated:** {analysis_results['timestamp']}\n\n")
                                        f.write(response)
                                    
                                    analysis_results["providers"][provider_name] = {
                                        "status": "SUCCESS",
                                        "file": str(provider_file),
                                        "response_length": len(response)
                                    }
                                    logger.debug(f"{provider_name} analysis completed for {gnn_file.name}")
                                else:
                                    analysis_results["providers"][provider_name] = {
                                        "status": "FAILED",
                                        "error": "Empty response"
                                    }
                                    file_success = False
                                    
                        except Exception as e:
                            analysis_results["providers"][provider_name] = {
                                "status": "ERROR",
                                "error": str(e)
                            }
                            file_success = False
                            log_step_warning(logger, f"{provider_name} analysis failed for {gnn_file.name}: {e}")
                    
                    # Generate combined analysis report
                    combined_file = file_output_dir / "combined_analysis.md"
                    with open(combined_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Combined LLM Analysis of {gnn_file.name}\n\n")
                        f.write(f"**Generated:** {analysis_results['timestamp']}\n\n")
                        f.write("## Analysis Summary\n\n")
                        
                        successful_providers = [name for name, result in analysis_results["providers"].items() 
                                              if result.get("status") == "SUCCESS"]
                        failed_providers = [name for name, result in analysis_results["providers"].items() 
                                          if result.get("status") != "SUCCESS"]
                        
                        f.write(f"- **Successful Providers:** {', '.join(successful_providers) if successful_providers else 'None'}\n")
                        f.write(f"- **Failed Providers:** {', '.join(failed_providers) if failed_providers else 'None'}\n\n")
                        
                        # Include successful analyses
                        for provider_name, result in analysis_results["providers"].items():
                            if result.get("status") == "SUCCESS":
                                f.write(f"## {provider_name.upper()} Analysis\n\n")
                                provider_file = Path(result["file"])
                                if provider_file.exists():
                                    with open(provider_file, 'r', encoding='utf-8') as pf:
                                        content = pf.read()
                                        # Extract the analysis content (skip header)
                                        lines = content.split('\n')
                                        analysis_content = '\n'.join(lines[4:])  # Skip first 4 lines (header)
                                        f.write(analysis_content)
                                        f.write("\n\n---\n\n")
                    
                    analysis_results["combined_file"] = str(combined_file)
                    analysis_results["success"] = file_success
                    
                    # Save analysis metadata
                    metadata_file = file_output_dir / "analysis_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_results, f, indent=2)
                    
                    if file_success:
                        successful_analyses += 1
                        logger.debug(f"Analysis completed for {gnn_file.name}")
                    else:
                        failed_analyses += 1
                        log_step_warning(logger, f"Some analyses failed for {gnn_file.name}")
                        
                except Exception as e:
                    failed_analyses += 1
                    log_step_error(logger, f"Failed to analyze {gnn_file.name}: {e}")
        
        # Generate overall summary
        summary_file = llm_output_dir / "llm_analysis_summary.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(gnn_files),
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "success_rate": successful_analyses / len(gnn_files) * 100 if gnn_files else 0,
            "available_providers": list(providers.keys())
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Log results summary
        if successful_analyses == len(gnn_files):
            log_step_success(logger, f"All {len(gnn_files)} files analyzed successfully")
            return True
        elif successful_analyses > 0:
            log_step_warning(logger, f"Partial success: {successful_analyses}/{len(gnn_files)} files analyzed successfully")
            return True
        else:
            log_step_error(logger, "No files were analyzed successfully")
            return False
        
    except Exception as e:
        log_step_error(logger, f"LLM analysis failed: {e}")
        return False 