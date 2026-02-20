#!/usr/bin/env python3
"""
Intelligent Analysis processor module for GNN pipeline analysis.

This module provides the main processing logic for intelligent pipeline analysis,
including LLM-powered insights and executive report generation with per-step
breakdowns, yellow/red flag detection, and actionable recommendations.
"""

import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)


@dataclass
class StepAnalysis:
    """Detailed analysis of a single pipeline step."""
    step_number: int
    script_name: str
    description: str
    status: str
    duration_seconds: float
    memory_mb: float
    exit_code: int
    flags: List[str] = field(default_factory=list)
    flag_type: str = "none"  # "none", "yellow", "red"
    summary: str = ""
    stdout_snippet: str = ""
    stderr_snippet: str = ""


def analyze_pipeline_summary(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze pipeline summary data to extract key insights.

    Args:
        summary_data: Pipeline execution summary dictionary

    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        "overall_status": summary_data.get("overall_status", "UNKNOWN"),
        "total_duration": summary_data.get("total_duration_seconds", 0),
        "step_count": len(summary_data.get("steps", [])),
        "failures": [],
        "warnings": [],
        "performance_metrics": {},
        "health_score": 0.0
    }

    steps = summary_data.get("steps", [])
    performance = summary_data.get("performance_summary", {})

    # Extract failures
    for step in steps:
        if step.get("status") == "FAILED":
            analysis["failures"].append({
                "step": step.get("script_name"),
                "error": step.get("stderr", "")[-1000:] if step.get("stderr") else "No error captured",
                "duration": step.get("duration_seconds", 0),
                "exit_code": step.get("exit_code", -1)
            })
        elif "WARNING" in step.get("status", ""):
            analysis["warnings"].append({
                "step": step.get("script_name"),
                "message": step.get("stdout", "")[-500:] if step.get("stdout") else "No output captured"
            })

    # Calculate performance metrics
    analysis["performance_metrics"] = {
        "peak_memory_mb": performance.get("peak_memory_mb", 0),
        "successful_steps": performance.get("successful_steps", 0),
        "failed_steps": performance.get("failed_steps", 0),
        "warning_count": performance.get("warnings", 0)
    }

    # Calculate health score (0-100)
    total_steps = analysis["step_count"]
    if total_steps > 0:
        success_ratio = (total_steps - len(analysis["failures"])) / total_steps
        warning_penalty = min(len(analysis["warnings"]) * 0.05, 0.2)
        analysis["health_score"] = max(0, min(100, (success_ratio - warning_penalty) * 100))

    return analysis


def analyze_individual_steps(summary_data: Dict[str, Any]) -> Tuple[List[StepAnalysis], Dict[str, List[StepAnalysis]]]:
    """
    Perform detailed analysis of each pipeline step.

    Args:
        summary_data: Pipeline execution summary dictionary

    Returns:
        Tuple of (list of StepAnalysis objects, dict of flags by type)
    """
    steps = summary_data.get("steps", [])
    step_analyses = []

    # Calculate averages for comparison
    durations = [s.get("duration_seconds", 0) for s in steps if s.get("duration_seconds")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    memories = [s.get("peak_memory_mb", 0) for s in steps if s.get("peak_memory_mb")]
    avg_memory = sum(memories) / len(memories) if memories else 0

    # Thresholds for flags
    SLOW_THRESHOLD = 60.0  # seconds
    VERY_SLOW_THRESHOLD = 120.0  # seconds
    HIGH_MEMORY_THRESHOLD = 500.0  # MB
    CRITICAL_MEMORY_THRESHOLD = 1000.0  # MB

    flags_by_type = {"red": [], "yellow": [], "green": []}

    for step in steps:
        duration = step.get("duration_seconds", 0)
        memory = step.get("peak_memory_mb", 0)
        status = step.get("status", "UNKNOWN")
        exit_code = step.get("exit_code", 0)

        flags = []
        flag_type = "none"

        # Determine flags and severity
        if status == "FAILED" or exit_code != 0:
            flag_type = "red"
            flags.append(f"FAILED with exit code {exit_code}")
            if step.get("stderr"):
                flags.append("Error output captured")

        # Performance flags
        if duration > VERY_SLOW_THRESHOLD:
            flags.append(f"Very slow: {duration:.1f}s (>{VERY_SLOW_THRESHOLD}s threshold)")
            if flag_type != "red":
                flag_type = "yellow"
        elif duration > SLOW_THRESHOLD:
            flags.append(f"Slow: {duration:.1f}s (>{SLOW_THRESHOLD}s threshold)")
            if flag_type != "red":
                flag_type = "yellow"
        elif duration > avg_duration * 3:
            flags.append(f"Significantly above average: {duration:.1f}s ({duration/avg_duration:.1f}x avg)")
            if flag_type != "red":
                flag_type = "yellow"

        # Memory flags
        if memory > CRITICAL_MEMORY_THRESHOLD:
            flags.append(f"Critical memory: {memory:.0f}MB (>{CRITICAL_MEMORY_THRESHOLD}MB)")
            if flag_type != "red":
                flag_type = "yellow"
        elif memory > HIGH_MEMORY_THRESHOLD:
            flags.append(f"High memory: {memory:.0f}MB (>{HIGH_MEMORY_THRESHOLD}MB)")
            if flag_type != "red":
                flag_type = "yellow"

        # Warning in status
        if "WARNING" in status:
            flags.append("Step completed with warnings")
            if flag_type == "none":
                flag_type = "yellow"

        # Retry flags
        retry_count = step.get("retry_count", 0)
        if retry_count > 0:
            flags.append(f"Required {retry_count} retries")
            if flag_type == "none":
                flag_type = "yellow"

        # Dependency warnings
        dep_warnings = step.get("dependency_warnings", [])
        if dep_warnings:
            flags.append(f"{len(dep_warnings)} dependency warning(s)")
            if flag_type == "none":
                flag_type = "yellow"

        # Generate summary
        if status == "SUCCESS" and not flags:
            summary = f"Completed successfully in {duration:.2f}s"
        elif status == "SUCCESS" and flags:
            summary = f"Completed with {len(flags)} flag(s) in {duration:.2f}s"
        elif status == "FAILED":
            summary = f"FAILED after {duration:.2f}s"
        else:
            summary = f"Status: {status} ({duration:.2f}s)"

        # Extract output snippets
        stdout = step.get("stdout", "")
        stderr = step.get("stderr", "")

        # Get meaningful snippets
        stdout_snippet = _extract_meaningful_snippet(stdout)
        stderr_snippet = _extract_meaningful_snippet(stderr) if stderr else ""

        step_analysis = StepAnalysis(
            step_number=step.get("step_number", 0),
            script_name=step.get("script_name", "unknown"),
            description=step.get("description", ""),
            status=status,
            duration_seconds=duration,
            memory_mb=memory,
            exit_code=exit_code,
            flags=flags,
            flag_type=flag_type,
            summary=summary,
            stdout_snippet=stdout_snippet,
            stderr_snippet=stderr_snippet
        )

        step_analyses.append(step_analysis)

        # Categorize by flag type
        if flag_type == "red":
            flags_by_type["red"].append(step_analysis)
        elif flag_type == "yellow":
            flags_by_type["yellow"].append(step_analysis)
        else:
            flags_by_type["green"].append(step_analysis)

    return step_analyses, flags_by_type


def _extract_meaningful_snippet(output: str, max_lines: int = 5, max_chars: int = 500) -> str:
    """Extract meaningful snippet from step output."""
    if not output:
        return ""

    lines = output.strip().split('\n')

    # Look for important patterns
    important_patterns = ['ERROR', 'WARN', 'FAIL', 'SUCCESS', 'Generated', 'Processed', 'Completed']
    important_lines = []

    for line in lines:
        if any(pattern in line.upper() for pattern in important_patterns):
            important_lines.append(line.strip())

    if important_lines:
        snippet = '\n'.join(important_lines[:max_lines])
    else:
        # Take last few lines if no important patterns found
        snippet = '\n'.join(lines[-max_lines:])

    return snippet[:max_chars]


def identify_bottlenecks(summary_data: Dict[str, Any], threshold_seconds: float = 60.0) -> List[Dict[str, Any]]:
    """
    Identify performance bottlenecks in pipeline execution.

    Args:
        summary_data: Pipeline execution summary dictionary
        threshold_seconds: Duration threshold for flagging slow steps

    Returns:
        List of bottleneck descriptions
    """
    bottlenecks = []
    steps = summary_data.get("steps", [])

    # Calculate average duration
    durations = [s.get("duration_seconds", 0) for s in steps if s.get("duration_seconds")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    for step in steps:
        duration = step.get("duration_seconds", 0)
        if duration > threshold_seconds or duration > avg_duration * 2:
            bottlenecks.append({
                "step": step.get("script_name"),
                "duration_seconds": duration,
                "threshold_exceeded": duration > threshold_seconds,
                "above_average_ratio": duration / avg_duration if avg_duration > 0 else 0,
                "memory_mb": step.get("peak_memory_mb", 0)
            })

    # Sort by duration descending
    bottlenecks.sort(key=lambda x: x["duration_seconds"], reverse=True)
    return bottlenecks


def extract_failure_context(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detailed context about failures for root cause analysis.

    Args:
        summary_data: Pipeline execution summary dictionary

    Returns:
        List of failure contexts with diagnostic information
    """
    failures = []
    steps = summary_data.get("steps", [])

    for i, step in enumerate(steps):
        if step.get("status") == "FAILED":
            # Get preceding step for context
            preceding_step = steps[i - 1] if i > 0 else None

            failure_context = {
                "step_number": step.get("step_number"),
                "step_name": step.get("script_name"),
                "description": step.get("description"),
                "exit_code": step.get("exit_code", -1),
                "error_output": step.get("stderr", "")[-2000:] if step.get("stderr") else None,
                "stdout_tail": step.get("stdout", "")[-1000:] if step.get("stdout") else None,
                "duration": step.get("duration_seconds"),
                "memory_at_failure": step.get("peak_memory_mb"),
                "preceding_step": {
                    "name": preceding_step.get("script_name") if preceding_step else None,
                    "status": preceding_step.get("status") if preceding_step else None
                } if preceding_step else None,
                "dependency_warnings": step.get("dependency_warnings", []),
                "prerequisite_check_passed": step.get("prerequisite_check", True)
            }
            failures.append(failure_context)

    return failures


def generate_recommendations(analysis: Dict[str, Any], bottlenecks: List[Dict[str, Any]], flags_by_type: Dict[str, List]) -> List[str]:
    """
    Generate actionable recommendations based on analysis.

    Args:
        analysis: Analysis results dictionary
        bottlenecks: List of identified bottlenecks
        flags_by_type: Dictionary of steps grouped by flag type

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Red flag recommendations (critical)
    red_flags = flags_by_type.get("red", [])
    if red_flags:
        recommendations.append(
            f"🔴 **CRITICAL**: {len(red_flags)} step(s) have red flags requiring immediate attention."
        )
        for step in red_flags[:3]:  # Top 3
            recommendations.append(
                f"   ↳ **{step.script_name}**: {', '.join(step.flags[:2])}"
            )

    # Yellow flag recommendations (warnings)
    yellow_flags = flags_by_type.get("yellow", [])
    if yellow_flags:
        recommendations.append(
            f"🟡 **WARNINGS**: {len(yellow_flags)} step(s) have yellow flags that should be reviewed."
        )
        for step in yellow_flags[:3]:  # Top 3
            recommendations.append(
                f"   ↳ **{step.script_name}**: {', '.join(step.flags[:2])}"
            )

    # Performance-based recommendations
    if bottlenecks:
        slowest = bottlenecks[0]
        recommendations.append(
            f"⚡ **Performance**: Slowest step is **{slowest['step']}** ({slowest['duration_seconds']:.1f}s). "
            "Consider parallelization or caching."
        )

    # Memory-based recommendations
    peak_memory = analysis["performance_metrics"].get("peak_memory_mb", 0)
    if peak_memory > 2048:
        recommendations.append(
            f"💾 **Memory**: Peak usage {peak_memory:.0f}MB exceeds 2GB. "
            "Consider memory optimization."
        )
    elif peak_memory > 1024:
        recommendations.append(
            f"💾 **Memory**: Peak usage {peak_memory:.0f}MB is elevated. "
            "Monitor for resource constraints."
        )

    # Health score recommendations
    health_score = analysis["health_score"]
    if health_score == 100:
        recommendations.append(
            "✅ **Health**: Pipeline is healthy (100/100). All systems nominal."
        )
    elif health_score >= 90:
        recommendations.append(
            f"✅ **Health**: Pipeline health is good ({health_score:.0f}/100)."
        )
    elif health_score >= 70:
        recommendations.append(
            f"⚠️ **Health**: Pipeline health needs attention ({health_score:.0f}/100)."
        )
    else:
        recommendations.append(
            f"🔴 **Health**: Pipeline health is critical ({health_score:.0f}/100). "
            "Address failures before production use."
        )

    return recommendations


async def _run_llm_analysis(
    context: Dict[str, Any],
    step_analyses: List[StepAnalysis],
    flags_by_type: Dict[str, List],
    logger: logging.Logger
) -> str:
    """
    Run LLM-powered analysis on pipeline context.

    Args:
        context: Analysis context dictionary
        step_analyses: List of per-step analysis objects
        flags_by_type: Steps grouped by flag type
        logger: Logger instance

    Returns:
        LLM-generated analysis report as markdown string
    """
    try:
        from llm.llm_processor import initialize_global_processor
        from llm.providers.base_provider import LLMMessage
        processor = await initialize_global_processor()
    except Exception as e:
        logger.warning(f"Failed to initialize LLM processor: {e}")
        return _generate_rule_based_summary(context, step_analyses, flags_by_type)

    if not processor:
        return _generate_rule_based_summary(context, step_analyses, flags_by_type)

    # Construct comprehensive prompt
    status_emoji = "✅" if context['overall_status'] == "SUCCESS" else "❌" if context['overall_status'] == "FAILED" else "⚠️"
    duration = context.get('total_duration', 0)
    peak_memory = context.get('performance_metrics', {}).get('peak_memory_mb', 0)

    # Build step summaries for LLM
    step_summaries = []
    for sa in step_analyses:
        flag_indicator = "🔴" if sa.flag_type == "red" else "🟡" if sa.flag_type == "yellow" else "✅"
        step_summaries.append(
            f"{flag_indicator} **{sa.script_name}** ({sa.description}): {sa.summary}"
            + (f" | Flags: {', '.join(sa.flags)}" if sa.flags else "")
        )

    red_count = len(flags_by_type.get("red", []))
    yellow_count = len(flags_by_type.get("yellow", []))

    prompt = f"""You are an expert DevOps analyst reviewing a pipeline execution report. Provide a comprehensive but concise analysis.

## Pipeline Overview
- **Status**: {status_emoji} {context['overall_status']}
- **Duration**: {duration:.2f}s
- **Peak Memory**: {peak_memory:.2f} MB
- **Health Score**: {context.get('health_score', 0):.1f}/100
- **Total Steps**: {len(step_analyses)}
- **Red Flags**: {red_count}
- **Yellow Flags**: {yellow_count}

## Per-Step Results
{chr(10).join(step_summaries)}

## Failures
{json.dumps(context.get('failures', []), indent=2) if context.get('failures') else "None"}

## Warnings
{json.dumps(context.get('warnings', []), indent=2) if context.get('warnings') else "None"}

---

Please provide analysis in EXACTLY this format:

### Executive Summary
[2-3 sentences summarizing the overall pipeline health and key findings]

### Red Flags (Critical Issues)
[List any critical issues that need immediate attention, or "None" if all clear]

### Yellow Flags (Warnings)
[List concerning patterns or performance issues, or "None" if all clear]

### Root Cause Analysis
[If failures exist, explain likely causes. If successful, note any concerning patterns]

### Optimization Opportunities
[Specific, actionable suggestions for improving pipeline performance]

### Action Items
[Prioritized bullet list of what the user should do next]
"""

    try:
        messages = [LLMMessage(role="user", content=prompt)]
        response = await processor.get_response(
            messages=messages,
            model_name="gemma3:4b",
            max_tokens=2500
        )
        return response.content
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return _generate_rule_based_summary(context, step_analyses, flags_by_type)


def _generate_rule_based_summary(
    context: Dict[str, Any],
    step_analyses: List[StepAnalysis],
    flags_by_type: Dict[str, List]
) -> str:
    """Generate a rule-based summary when LLM is unavailable."""

    red_flags = flags_by_type.get("red", [])
    yellow_flags = flags_by_type.get("yellow", [])

    parts = []

    # Executive Summary
    parts.append("### Executive Summary\n")
    if context["overall_status"] == "SUCCESS" and not red_flags:
        parts.append(f"Pipeline completed successfully with a health score of {context['health_score']:.0f}/100. ")
        if yellow_flags:
            parts.append(f"There are {len(yellow_flags)} yellow flag(s) to review for optimization opportunities.")
        else:
            parts.append("All systems nominal with no flags raised.")
    elif context["overall_status"] == "SUCCESS" and red_flags:
        parts.append(f"Pipeline completed but {len(red_flags)} critical issue(s) detected. Review required.")
    else:
        parts.append(f"Pipeline {context['overall_status']} with {len(red_flags)} critical and {len(yellow_flags)} warning flags.")
    parts.append("\n\n")

    # Red Flags
    parts.append("### Red Flags (Critical Issues)\n")
    if red_flags:
        for step in red_flags:
            parts.append(f"- **{step.script_name}**: {', '.join(step.flags)}\n")
    else:
        parts.append("None - No critical issues detected.\n")
    parts.append("\n")

    # Yellow Flags
    parts.append("### Yellow Flags (Warnings)\n")
    if yellow_flags:
        for step in yellow_flags[:5]:  # Top 5
            parts.append(f"- **{step.script_name}**: {', '.join(step.flags)}\n")
        if len(yellow_flags) > 5:
            parts.append(f"- ...and {len(yellow_flags) - 5} more\n")
    else:
        parts.append("None - No warnings detected.\n")
    parts.append("\n")

    # Action Items
    parts.append("### Action Items\n")
    if red_flags:
        parts.append("1. **Immediate**: Address red flag issues before proceeding\n")
    if yellow_flags:
        parts.append(f"{'2' if red_flags else '1'}. **Review**: Investigate yellow flag warnings\n")
    if context.get("performance_metrics", {}).get("peak_memory_mb", 0) > 1024:
        parts.append(f"{'3' if red_flags and yellow_flags else '2' if red_flags or yellow_flags else '1'}. **Monitor**: Track memory usage in production\n")
    if not red_flags and not yellow_flags:
        parts.append("1. Continue normal operations - pipeline is healthy\n")
        parts.append("2. Consider setting up monitoring for regression detection\n")

    return "".join(parts)


def generate_executive_report(
    analysis: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    recommendations: List[str],
    step_analyses: List[StepAnalysis],
    flags_by_type: Dict[str, List],
    llm_analysis: Optional[str] = None,
    summary_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a comprehensive executive report with per-step analysis.

    Args:
        analysis: Analysis results
        bottlenecks: Identified bottlenecks
        failures: Failure contexts
        recommendations: Generated recommendations
        step_analyses: Per-step analysis objects
        flags_by_type: Steps grouped by flag type
        llm_analysis: Optional LLM-generated analysis
        summary_data: Original pipeline summary data

    Returns:
        Markdown formatted executive report
    """
    report_parts = []

    # Header
    status = analysis["overall_status"]
    status_emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"

    red_count = len(flags_by_type.get("red", []))
    yellow_count = len(flags_by_type.get("yellow", []))
    green_count = len(flags_by_type.get("green", []))

    report_parts.append(f"# Pipeline Intelligent Analysis Report\n")
    report_parts.append(f"**Generated**: {datetime.now().isoformat()}\n")
    report_parts.append(f"**Status**: {status_emoji} {status}\n")
    report_parts.append(f"**Health Score**: {analysis['health_score']:.1f}/100\n")
    report_parts.append("")

    # Quick Stats Box
    report_parts.append("## Quick Overview\n")
    report_parts.append("| Metric | Value |")
    report_parts.append("|--------|-------|")
    report_parts.append(f"| Total Steps | {analysis['step_count']} |")
    report_parts.append(f"| Duration | {analysis['total_duration']:.2f}s |")
    report_parts.append(f"| Peak Memory | {analysis['performance_metrics'].get('peak_memory_mb', 0):.1f} MB |")
    report_parts.append(f"| 🔴 Red Flags | {red_count} |")
    report_parts.append(f"| 🟡 Yellow Flags | {yellow_count} |")
    report_parts.append(f"| ✅ Green (Clean) | {green_count} |")
    report_parts.append("")

    # AI-Powered Analysis (if available)
    if llm_analysis and "Unavailable" not in llm_analysis and "Error" not in llm_analysis:
        report_parts.append("## AI-Powered Analysis\n")
        report_parts.append(llm_analysis)
        report_parts.append("")
    elif llm_analysis:
        # Show rule-based analysis
        report_parts.append("## Analysis Summary\n")
        report_parts.append(llm_analysis)
        report_parts.append("")

    # Red Flags Section
    red_flags = flags_by_type.get("red", [])
    if red_flags:
        report_parts.append("## 🔴 Red Flags (Critical)\n")
        for step in red_flags:
            report_parts.append(f"### {step.script_name}\n")
            report_parts.append(f"- **Status**: {step.status}")
            report_parts.append(f"- **Exit Code**: {step.exit_code}")
            report_parts.append(f"- **Duration**: {step.duration_seconds:.2f}s")
            report_parts.append(f"- **Issues**: {', '.join(step.flags)}")
            if step.stderr_snippet:
                report_parts.append(f"\n**Error Output**:\n```\n{step.stderr_snippet}\n```")
            report_parts.append("")

    # Yellow Flags Section
    yellow_flags = flags_by_type.get("yellow", [])
    if yellow_flags:
        report_parts.append("## 🟡 Yellow Flags (Warnings)\n")
        report_parts.append("| Step | Duration | Memory | Issues |")
        report_parts.append("|------|----------|--------|--------|")
        for step in yellow_flags:
            issues = '; '.join(step.flags[:2]) if step.flags else "N/A"
            report_parts.append(
                f"| {step.script_name} | {step.duration_seconds:.1f}s | "
                f"{step.memory_mb:.0f}MB | {issues} |"
            )
        report_parts.append("")

    # Per-Step Breakdown
    report_parts.append("## Per-Step Execution Details\n")
    report_parts.append("| # | Step | Status | Duration | Memory | Flags |")
    report_parts.append("|---|------|--------|----------|--------|-------|")

    for step in step_analyses:
        flag_emoji = "🔴" if step.flag_type == "red" else "🟡" if step.flag_type == "yellow" else "✅"
        status_display = f"{flag_emoji} {step.status}"
        flags_display = len(step.flags) if step.flags else "-"
        report_parts.append(
            f"| {step.step_number} | {step.script_name} | {status_display} | "
            f"{step.duration_seconds:.2f}s | {step.memory_mb:.0f}MB | {flags_display} |"
        )
    report_parts.append("")

    # Detailed Step Output (for flagged steps only)
    flagged_steps = [s for s in step_analyses if s.flag_type != "none"]
    if flagged_steps:
        report_parts.append("## Detailed Step Output (Flagged Steps)\n")
        for step in flagged_steps:
            report_parts.append(f"### {step.script_name}\n")
            report_parts.append(f"**{step.description}**\n")
            report_parts.append(f"- Status: {step.status}")
            report_parts.append(f"- Duration: {step.duration_seconds:.2f}s")
            report_parts.append(f"- Memory: {step.memory_mb:.0f}MB")
            if step.flags:
                report_parts.append(f"- Flags: {', '.join(step.flags)}")
            if step.stdout_snippet:
                report_parts.append(f"\n**Output Snippet**:\n```\n{step.stdout_snippet}\n```")
            if step.stderr_snippet:
                report_parts.append(f"\n**Error Output**:\n```\n{step.stderr_snippet}\n```")
            report_parts.append("")

    # Performance Bottlenecks
    if bottlenecks:
        report_parts.append("## Performance Bottlenecks\n")
        report_parts.append("| Step | Duration (s) | Memory (MB) | Above Avg Ratio |")
        report_parts.append("|------|-------------|-------------|-----------------|")
        for bn in bottlenecks[:5]:  # Top 5
            report_parts.append(
                f"| {bn['step']} | {bn['duration_seconds']:.1f} | "
                f"{bn.get('memory_mb', 0):.0f} | {bn['above_average_ratio']:.1f}x |"
            )
        report_parts.append("")

    # Recommendations Section
    report_parts.append("## Recommendations\n")
    for rec in recommendations:
        report_parts.append(f"- {rec}")
    report_parts.append("")

    # Pipeline Arguments (for context)
    if summary_data and summary_data.get("arguments"):
        args = summary_data["arguments"]
        report_parts.append("## Pipeline Configuration\n")
        report_parts.append("```json")
        # Show relevant config only
        relevant_args = {k: v for k, v in args.items() if k in [
            "target_dir", "output_dir", "verbose", "strict",
            "only_steps", "skip_steps", "frameworks"
        ]}
        report_parts.append(json.dumps(relevant_args, indent=2))
        report_parts.append("```")
        report_parts.append("")

    return "\n".join(report_parts)


def process_intelligent_analysis(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    **kwargs
) -> bool:
    """
    Perform intelligent analysis of the pipeline execution.

    Args:
        target_dir: Directory containing input files (not used directly here)
        output_dir: Output directory for generated artifacts
        logger: Logger instance
        **kwargs: Additional arguments

    Returns:
        True if analysis succeeded, False if analysis itself failed
    """
    log_step_start(logger, "Intelligent Pipeline Analysis")

    # 1. Locate Pipeline Summary
    summary_path = output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
    if not summary_path.exists():
        # Try parent directory
        summary_path = output_dir.parent / "00_pipeline_summary" / "pipeline_execution_summary.json"

    if not summary_path.exists():
        log_step_error(logger, f"Pipeline summary not found at {summary_path}")
        return False

    try:
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
    except Exception as e:
        log_step_error(logger, f"Failed to load pipeline summary: {e}")
        return False

    # 2. Perform Analysis
    logger.info("Analyzing pipeline execution data...")
    analysis = analyze_pipeline_summary(summary_data)

    logger.info(f"Analysis complete: Status={analysis['overall_status']}, "
                f"Failures={len(analysis['failures'])}, Health={analysis['health_score']:.1f}")

    # 3. Analyze Individual Steps
    logger.info("Performing per-step analysis...")
    step_analyses, flags_by_type = analyze_individual_steps(summary_data)

    red_count = len(flags_by_type.get("red", []))
    yellow_count = len(flags_by_type.get("yellow", []))
    logger.info(f"Step analysis: {red_count} red flags, {yellow_count} yellow flags")

    # 4. Identify Bottlenecks
    bottlenecks = identify_bottlenecks(summary_data)
    if bottlenecks:
        logger.info(f"Identified {len(bottlenecks)} performance bottlenecks")

    # 5. Extract Failure Context
    failures = extract_failure_context(summary_data)

    # 6. Generate Recommendations
    recommendations = generate_recommendations(analysis, bottlenecks, flags_by_type)

    # 7. Run LLM Analysis
    llm_analysis = None
    try:
        llm_analysis = asyncio.run(_run_llm_analysis(analysis, step_analyses, flags_by_type, logger))
        logger.info("LLM analysis completed")
    except Exception as e:
        logger.warning(f"LLM analysis skipped: {e}")
        llm_analysis = _generate_rule_based_summary(analysis, step_analyses, flags_by_type)

    # 8. Generate Executive Report
    report_content = generate_executive_report(
        analysis, bottlenecks, failures, recommendations,
        step_analyses, flags_by_type, llm_analysis, summary_data
    )

    # 9. Save Outputs
    if output_dir.name == "24_intelligent_analysis_output":
        analysis_output_dir = output_dir
    else:
        analysis_output_dir = output_dir / "24_intelligent_analysis_output"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # Save markdown report
    report_path = analysis_output_dir / "intelligent_analysis_report.md"
    try:
        with open(report_path, 'w') as f:
            f.write(report_content)
        logger.info(f"Analysis report saved to {report_path}")
    except Exception as e:
        log_step_error(logger, f"Failed to save report: {e}")
        return False

    # Save JSON analysis data
    analysis_data_path = analysis_output_dir / "analysis_data.json"
    try:
        # Convert StepAnalysis objects to dicts
        step_analyses_dict = [
            {
                "step_number": sa.step_number,
                "script_name": sa.script_name,
                "description": sa.description,
                "status": sa.status,
                "duration_seconds": sa.duration_seconds,
                "memory_mb": sa.memory_mb,
                "exit_code": sa.exit_code,
                "flags": sa.flags,
                "flag_type": sa.flag_type,
                "summary": sa.summary
            }
            for sa in step_analyses
        ]

        with open(analysis_data_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "step_analyses": step_analyses_dict,
                "flags_summary": {
                    "red_count": red_count,
                    "yellow_count": yellow_count,
                    "green_count": len(flags_by_type.get("green", []))
                },
                "bottlenecks": bottlenecks,
                "failures": failures,
                "recommendations": recommendations,
                "llm_analysis_available": llm_analysis is not None and "Unavailable" not in llm_analysis
            }, f, indent=2)
        logger.info(f"Analysis data saved to {analysis_data_path}")
    except Exception as e:
        logger.warning(f"Failed to save analysis data: {e}")

    # Save summary
    summary_output_path = analysis_output_dir / "intelligent_analysis_summary.json"
    try:
        with open(summary_output_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "overall_status": analysis["overall_status"],
                "health_score": analysis["health_score"],
                "failure_count": len(failures),
                "red_flag_count": red_count,
                "yellow_flag_count": yellow_count,
                "bottleneck_count": len(bottlenecks),
                "recommendation_count": len(recommendations),
                "report_file": str(report_path),
                "data_file": str(analysis_data_path)
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save summary: {e}")

    log_step_success(logger, "Intelligent analysis completed successfully")
    return True
