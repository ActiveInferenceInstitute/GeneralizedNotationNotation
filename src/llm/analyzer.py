from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
import time
import json
import asyncio
from typing import Dict, List, Any, Optional

# Import LLM functionality with better error handling
try:
    from .providers.openai_provider import OpenAIProvider
    OPENAI_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).debug(f"OpenAI provider not available: {e}")
    OpenAIProvider = None
    OPENAI_AVAILABLE = False

try:
    from .providers.openrouter_provider import OpenRouterProvider
    OPENROUTER_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).debug(f"OpenRouter provider not available: {e}")
    OpenRouterProvider = None
    OPENROUTER_AVAILABLE = False

try:
    from .providers.perplexity_provider import PerplexityProvider
    PERPLEXITY_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).debug(f"Perplexity provider not available: {e}")
    PerplexityProvider = None
    PERPLEXITY_AVAILABLE = False

LLM_PROVIDERS_AVAILABLE = any([OPENAI_AVAILABLE, OPENROUTER_AVAILABLE, PERPLEXITY_AVAILABLE])

def get_analysis_prompts() -> Dict[str, str]:
    """Get predefined analysis prompts for different analysis types."""
    return {
        "explain_model": """Analyze the following GNN (Generalized Notation Notation) model specification and provide a clear explanation of what this model represents, its purpose, and key characteristics:

{content}

Please provide:
1. A concise explanation of the model's purpose
2. Key components and their roles
3. The Active Inference framework elements present
4. How this model would be used in practice""",

        "analyze_structure": """Analyze the structural components of this GNN model specification:

{content}

Please provide:
1. Model architecture and hierarchy
2. Variable relationships and dependencies
3. Information flow patterns
4. Structural strengths and potential weaknesses""",

        "summarize_content": """Provide a comprehensive summary of this GNN model:

{content}

Please provide:
1. Executive summary (2-3 sentences)
2. Key variables and parameters
3. Main equations and relationships
4. Practical applications""",

        "mathematical_analysis": """Perform a mathematical analysis of this GNN model:

{content}

Please provide:
1. Mathematical foundations and formulations
2. Equation analysis and relationships
3. Parameter significance
4. Numerical considerations and constraints""",

        "practical_applications": """Analyze the practical applications and use cases for this GNN model:

{content}

Please provide:
1. Real-world applications
2. Domain-specific use cases
3. Implementation considerations
4. Potential impact and benefits""",

        "suggest_improvements": """Review this GNN model and suggest improvements:

{content}

Please provide:
1. Potential optimizations
2. Missing components or considerations
3. Alternative formulations
4. Best practices recommendations"""
    }

def check_provider_availability(logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Check which LLM providers are available and their status."""
    providers_status = {
        "openai": {"available": OPENAI_AVAILABLE, "instance": None, "error": None},
        "openrouter": {"available": OPENROUTER_AVAILABLE, "instance": None, "error": None},
        "perplexity": {"available": PERPLEXITY_AVAILABLE, "instance": None, "error": None}
    }
    
    # Try to initialize available providers
    if OPENAI_AVAILABLE and OpenAIProvider:
        try:
            providers_status["openai"]["instance"] = OpenAIProvider()
            logger.debug("OpenAI provider initialized successfully")
        except Exception as e:
            providers_status["openai"]["available"] = False
            providers_status["openai"]["error"] = str(e)
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
    
    if OPENROUTER_AVAILABLE and OpenRouterProvider:
        try:
            providers_status["openrouter"]["instance"] = OpenRouterProvider()
            logger.debug("OpenRouter provider initialized successfully")
        except Exception as e:
            providers_status["openrouter"]["available"] = False
            providers_status["openrouter"]["error"] = str(e)
            logger.warning(f"Failed to initialize OpenRouter provider: {e}")
    
    if PERPLEXITY_AVAILABLE and PerplexityProvider:
        try:
            providers_status["perplexity"]["instance"] = PerplexityProvider()
            logger.debug("Perplexity provider initialized successfully")
        except Exception as e:
            providers_status["perplexity"]["available"] = False
            providers_status["perplexity"]["error"] = str(e)
            logger.warning(f"Failed to initialize Perplexity provider: {e}")
    
    return providers_status

def analyze_gnn_files(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced GNN file analysis using LLM providers with comprehensive reporting.
    
    Args:
        target_dir: Directory containing GNN files to analyze
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional analysis options (e.g., llm_tasks, llm_timeout)
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    log_step_start(logger, "Enhanced LLM analysis of GNN files with comprehensive reporting")
    
    # Use centralized output directory configuration
    llm_output_dir = get_output_dir_for_script("11_llm.py", output_dir)
    llm_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check provider availability
    with performance_tracker.track_operation("check_llm_providers"):
        providers_status = check_provider_availability(logger)
        available_providers = {name: status for name, status in providers_status.items() 
                             if status["available"] and status["instance"]}
    
    if not available_providers:
        log_step_warning(logger, "No LLM providers available - creating placeholder report")
        
        # Create placeholder report
        placeholder_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "no_providers_available",
            "total_files": 0,
            "files_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_analyses": 0,
            "success": False,
            "providers_status": providers_status,
            "analysis_tasks": [],
            "message": "No LLM providers available for analysis"
        }
        
        summary_file = llm_output_dir / "llm_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(placeholder_summary, f, indent=2)
        
        # Create markdown report
        report_file = llm_output_dir / "llm_analysis_report.md"
        with open(report_file, 'w') as f:
            f.write("# LLM Analysis Report\n\n")
            f.write("**Status**: No LLM providers available\n\n")
            f.write("## Provider Status\n\n")
            for name, status in providers_status.items():
                available_icon = "✅" if status["available"] else "❌"
                f.write(f"- {available_icon} **{name.title()}**")
                if status["error"]:
                    f.write(f": {status['error']}")
                f.write("\n")
            f.write("\n## Recommendations\n\n")
            f.write("1. Check LLM provider API keys and configuration\n")
            f.write("2. Install required dependencies for LLM providers\n")
            f.write("3. Verify network connectivity for API access\n")
        
        logger.info("Created LLM placeholder report")
        return placeholder_summary
    
    # Get analysis tasks from kwargs
    analysis_tasks = kwargs.get('llm_tasks', 'all')
    if isinstance(analysis_tasks, str):
        if analysis_tasks.lower() == 'all':
            analysis_tasks = ['explain_model', 'analyze_structure', 'summarize_content']
        else:
            analysis_tasks = [task.strip() for task in analysis_tasks.split(',')]
    
    # Get analysis prompts
    prompts = get_analysis_prompts()
    valid_tasks = [task for task in analysis_tasks if task in prompts]
    
    if not valid_tasks:
        valid_tasks = ['explain_model', 'summarize_content']  # Default fallback
    
    logger.info(f"Using analysis tasks: {', '.join(valid_tasks)}")
    
    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir}")
        empty_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "no_files_found",
            "total_files": 0,
            "files_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_analyses": 0,
            "success": True,
            "providers_status": providers_status,
            "analysis_tasks": valid_tasks
        }
        return empty_summary
    
    logger.info(f"Found {len(gnn_files)} GNN files to analyze with {len(available_providers)} providers")
    
    files_processed = 0
    successful_analyses = 0
    failed_analyses = 0
    total_analyses = 0
    file_results = []
    
    try:
        with performance_tracker.track_operation("analyze_all_gnn_files"):
            for gnn_file in gnn_files:
                try:
                    logger.info(f"Analyzing file: {gnn_file.name}")
                    
                    # Read GNN file content
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        gnn_content = f.read()
                    
                    if not gnn_content.strip():
                        logger.warning(f"Empty file: {gnn_file.name}")
                        continue
                    
                    # Create file-specific output directory
                    file_output_dir = llm_output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # File analysis results
                    file_analysis = {
                        "file": str(gnn_file),
                        "file_name": gnn_file.name,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "content_length": len(gnn_content),
                        "output_directory": str(file_output_dir),
                        "providers": {},
                        "tasks": {},
                        "success": True
                    }
                    
                    file_successful_analyses = 0
                    file_total_analyses = 0
                    
                    # Analyze with each provider and task
                    for provider_name, provider_info in available_providers.items():
                        provider = provider_info["instance"]
                        provider_results = {"tasks": {}, "overall_success": True}
                        
                        for task_name in valid_tasks:
                            file_total_analyses += 1
                            total_analyses += 1
                            
                            try:
                                with performance_tracker.track_operation(f"analyze_{provider_name}_{task_name}_{gnn_file.name}"):
                                    # Generate task-specific prompt
                                    prompt = prompts[task_name].format(content=gnn_content)
                                    
                                    # Get analysis from provider
                                    timeout = kwargs.get('llm_timeout', 360)
                                    response = provider.analyze(prompt, max_tokens=2000, timeout=timeout)
                                    
                                    if response and response.strip():
                                        # Save task-specific analysis
                                        task_file = file_output_dir / f"{provider_name}_{task_name}.md"
                                        with open(task_file, 'w', encoding='utf-8') as f:
                                            f.write(f"# {task_name.replace('_', ' ').title()} Analysis\n")
                                            f.write(f"**Provider**: {provider_name.title()}\n")
                                            f.write(f"**File**: {gnn_file.name}\n")
                                            f.write(f"**Generated**: {file_analysis['timestamp']}\n\n")
                                            f.write(response)
                                        
                                        provider_results["tasks"][task_name] = {
                                            "status": "SUCCESS",
                                            "file": str(task_file),
                                            "response_length": len(response)
                                        }
                                        file_successful_analyses += 1
                                        successful_analyses += 1
                                        
                                        logger.debug(f"Completed {provider_name} {task_name} analysis for {gnn_file.name}")
                                    else:
                                        provider_results["tasks"][task_name] = {
                                            "status": "FAILED",
                                            "error": "Empty response"
                                        }
                                        failed_analyses += 1
                                        provider_results["overall_success"] = False
                                        
                            except Exception as e:
                                provider_results["tasks"][task_name] = {
                                    "status": "ERROR",
                                    "error": str(e)
                                }
                                failed_analyses += 1
                                provider_results["overall_success"] = False
                                log_step_warning(logger, f"{provider_name} {task_name} analysis failed for {gnn_file.name}: {e}")
                        
                        file_analysis["providers"][provider_name] = provider_results
                    
                    # Generate combined analysis report for this file
                    _generate_combined_file_report(file_output_dir, file_analysis, gnn_file)
                    
                    # Update file success status
                    file_analysis["successful_analyses"] = file_successful_analyses
                    file_analysis["total_analyses"] = file_total_analyses
                    file_analysis["success"] = file_successful_analyses > 0
                    
                    # Save file analysis metadata
                    metadata_file = file_output_dir / "analysis_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(file_analysis, f, indent=2)
                    
                    file_results.append(file_analysis)
                    files_processed += 1
                    
                    if file_analysis["success"]:
                        logger.info(f"✅ Analysis completed for {gnn_file.name} ({file_successful_analyses}/{file_total_analyses} tasks successful)")
                    else:
                        logger.warning(f"⚠️ Partial analysis for {gnn_file.name} ({file_successful_analyses}/{file_total_analyses} tasks successful)")
                        
                except Exception as e:
                    failed_analyses += len(valid_tasks) * len(available_providers)
                    total_analyses += len(valid_tasks) * len(available_providers)
                    log_step_error(logger, f"Failed to analyze {gnn_file.name}: {e}")
        
        # Generate comprehensive summary
        overall_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed",
            "total_files": len(gnn_files),
            "files_processed": files_processed,
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "total_analyses": total_analyses,
            "success": successful_analyses > 0,
            "providers_status": providers_status,
            "available_providers": list(available_providers.keys()),
            "analysis_tasks": valid_tasks,
            "file_results": file_results,
            "success_rate": (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
        }
        
        # Save comprehensive summary
        summary_file = llm_output_dir / "llm_analysis_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2)
        
        # Generate comprehensive markdown report
        _generate_comprehensive_report(llm_output_dir, overall_results)
        
        # Log final results
        if successful_analyses > 0:
            success_rate = (successful_analyses / total_analyses) * 100
            log_step_success(logger, 
                f"LLM analysis completed: {files_processed} files processed, "
                f"{successful_analyses}/{total_analyses} analyses successful ({success_rate:.1f}%)")
        else:
            log_step_warning(logger, "No analyses completed successfully")
        
        return overall_results
        
    except Exception as e:
        log_step_error(logger, f"LLM analysis failed: {e}")
        error_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error",
            "error": str(e),
            "total_files": len(gnn_files),
            "files_processed": files_processed,
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "total_analyses": total_analyses,
            "success": False,
            "providers_status": providers_status
        }
        return error_summary


def _generate_combined_file_report(output_dir: Path, file_analysis: Dict[str, Any], gnn_file: Path):
    """Generate a combined analysis report for a single file."""
    combined_file = output_dir / "combined_analysis.md"
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(f"# Combined LLM Analysis: {file_analysis['file_name']}\n\n")
        f.write(f"**Generated**: {file_analysis['timestamp']}\n")
        f.write(f"**File Size**: {file_analysis['content_length']} characters\n\n")
        
        # Analysis summary
        f.write("## Analysis Summary\n\n")
        successful_count = file_analysis.get('successful_analyses', 0)
        total_count = file_analysis.get('total_analyses', 0)
        f.write(f"- **Success Rate**: {(successful_count/total_count*100):.1f}% ({successful_count}/{total_count})\n")
        
        # Provider results
        for provider_name, provider_data in file_analysis["providers"].items():
            successful_tasks = [task for task, result in provider_data["tasks"].items() 
                              if result.get("status") == "SUCCESS"]
            f.write(f"- **{provider_name.title()}**: {len(successful_tasks)} successful tasks\n")
        f.write("\n")
        
        # Include successful analyses
        for provider_name, provider_data in file_analysis["providers"].items():
            for task_name, task_result in provider_data["tasks"].items():
                if task_result.get("status") == "SUCCESS" and "file" in task_result:
                    f.write(f"## {provider_name.title()} - {task_name.replace('_', ' ').title()}\n\n")
                    
                    task_file = Path(task_result["file"])
                    if task_file.exists():
                        with open(task_file, 'r', encoding='utf-8') as tf:
                            content = tf.read()
                            # Extract content after the header
                            lines = content.split('\n')
                            analysis_content = '\n'.join(lines[5:])  # Skip header lines
                            f.write(analysis_content)
                            f.write("\n\n---\n\n")


def _generate_comprehensive_report(output_dir: Path, results: Dict[str, Any]):
    """Generate a comprehensive markdown report for all analyses."""
    report_file = output_dir / "llm_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive LLM Analysis Report\n\n")
        f.write(f"**Generated**: {results['timestamp']}\n")
        f.write(f"**Status**: {results['status'].upper()}\n")
        f.write(f"**Overall Success Rate**: {results['success_rate']:.1f}%\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Files**: {results['total_files']}\n")
        f.write(f"- **Files Processed**: {results['files_processed']}\n")
        f.write(f"- **Total Analyses**: {results['total_analyses']}\n")
        f.write(f"- **Successful Analyses**: {results['successful_analyses']}\n")
        f.write(f"- **Failed Analyses**: {results['failed_analyses']}\n\n")
        
        # Provider status
        f.write("## Provider Status\n\n")
        providers_status = results.get('providers_status', {})
        for name, status in providers_status.items():
            available_icon = "✅" if status["available"] else "❌"
            f.write(f"- {available_icon} **{name.title()}**")
            if status.get("error"):
                f.write(f": {status['error']}")
            f.write("\n")
        f.write("\n")
        
        # Analysis tasks
        f.write("## Analysis Tasks Performed\n\n")
        for task in results.get('analysis_tasks', []):
            f.write(f"- {task.replace('_', ' ').title()}\n")
        f.write("\n")
        
        # File-by-file results
        if 'file_results' in results:
            f.write("## File Analysis Results\n\n")
            for file_result in results['file_results']:
                file_name = file_result['file_name']
                file_success_rate = (file_result.get('successful_analyses', 0) / 
                                   file_result.get('total_analyses', 1)) * 100
                status_icon = "✅" if file_result.get('success') else "❌"
                
                f.write(f"### {status_icon} {file_name}\n\n")
                f.write(f"- **Success Rate**: {file_success_rate:.1f}%\n")
                f.write(f"- **Output Directory**: `{file_result['output_directory']}`\n")
                
                # Provider breakdown
                for provider_name, provider_data in file_result.get('providers', {}).items():
                    successful_tasks = [task for task, result in provider_data.get('tasks', {}).items() 
                                      if result.get('status') == 'SUCCESS']
                    f.write(f"- **{provider_name.title()}**: {len(successful_tasks)} successful tasks\n")
                f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        if results['successful_analyses'] == 0:
            f.write("1. Check LLM provider configurations and API keys\n")
            f.write("2. Verify network connectivity\n")
            f.write("3. Review input file formats and content\n")
        elif results['success_rate'] < 80:
            f.write("1. Review failed analyses for common error patterns\n")
            f.write("2. Consider adjusting timeout settings\n")
            f.write("3. Check for provider-specific issues\n")
        else:
            f.write("1. Analysis completed successfully\n")
            f.write("2. Review generated analyses for insights\n")
            f.write("3. Consider additional analysis tasks if needed\n") 