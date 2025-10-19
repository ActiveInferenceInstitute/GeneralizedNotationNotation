#!/usr/bin/env python3
"""
Enhanced Visual Logging System for GNN Pipeline

Provides comprehensive visual accessibility improvements including:
- Progress bars and indicators
- Color-coded status messages
- Structured output formatting
- Screen reader friendly messages
- Animated progress indicators
- Comprehensive error reporting with recovery suggestions
"""

import sys
import time
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json

# Try to import rich for enhanced terminal output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Progress = None
    Panel = None

# Terminal capability detection
try:
    import shutil
    TERMINAL_SIZE = shutil.get_terminal_size()
    TERMINAL_WIDTH = TERMINAL_SIZE.columns
    TERMINAL_HEIGHT = TERMINAL_SIZE.lines
except (OSError, AttributeError):
    TERMINAL_WIDTH = 80
    TERMINAL_HEIGHT = 24

# Color and emoji definitions for visual accessibility
STATUS_ICONS = {
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    "progress": "🔄",
    "complete": "🎯",
    "loading": "⏳",
    "rocket": "🚀",
    "test": "🧪",
    "file": "📄",
    "folder": "📁",
    "gear": "⚙️",
    "check": "✔️",
    "cross": "✗",
    "star": "⭐",
    "arrow": "→",
    "bullet": "•",
    "pipe": "│",
    "corner": "└",
    "step": "🔢",
}

PROGRESS_CHARS = {
    "full": "█",
    "empty": "░",
    "partial": ["▏", "▎", "▍", "▌", "▋", "▊", "▉"],
}

@dataclass
class VisualConfig:
    """Configuration for visual logging features."""
    enable_colors: bool = True
    enable_progress_bars: bool = True
    enable_emoji: bool = True
    enable_animation: bool = True
    max_width: int = TERMINAL_WIDTH
    show_timestamps: bool = False
    show_correlation_ids: bool = True
    compact_mode: bool = False

class VisualLogger:
    """Enhanced visual logger with accessibility features."""

    def __init__(self, name: str, config: Optional[VisualConfig] = None):
        self.name = name
        self.config = config or VisualConfig()
        self.console = Console() if RICH_AVAILABLE else None
        self._correlation_id = None
        self._start_time = time.time()

        # Setup standard logger
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for tracking across pipeline steps."""
        self._correlation_id = correlation_id

    def format_message(self, message: str, level: str = "info", **kwargs) -> str:
        """Format message with visual enhancements."""
        if not self.config.enable_emoji:
            # Remove emoji for screen readers
            for icon in STATUS_ICONS.values():
                message = message.replace(icon, "")

        # Add correlation ID if configured
        if self.config.show_correlation_ids and self._correlation_id:
            message = f"[{self._correlation_id}] {message}"

        # Add timestamp if configured
        if self.config.show_timestamps:
            timestamp = datetime.now().strftime("%H:%M:%S")
            message = f"[{timestamp}] {message}"

        return message

    def print_header(self, title: str, subtitle: str = "", width: int = None):
        """Print a formatted header."""
        if width is None:
            width = min(self.config.max_width, 80)

        # Create border
        border = "=" * width

        if self.console and RICH_AVAILABLE:
            self.console.print(Panel.fit(
                f"[bold blue]{title}[/bold blue]\n[dim]{subtitle}[/dim]" if subtitle else f"[bold blue]{title}[/bold blue]",
                border_style="blue"
            ))
        else:
            print(f"\n{border}")
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print(f"{border}\n")

    def print_step_header(self, step_num: int, description: str, total_steps: int = 24):
        """Print a formatted step header with progress indicator."""
        progress_text = f"Step {step_num}/{total_steps}"
        bar_width = min(30, self.config.max_width - len(progress_text) - 10)

        if self.console and RICH_AVAILABLE:
            # Create progress bar
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console
            )

            with progress:
                task = progress.add_task(
                    f"[cyan]Step {step_num}: {description}[/cyan]",
                    total=1
                )
                progress.update(task, advance=1)
        else:
            # Fallback text-based progress
            bar = self._create_text_progress_bar(step_num, total_steps, bar_width)
            print(f"\n🔢 {progress_text} - {description}")
            print(f"   {bar}")

    def _create_text_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """Create a text-based progress bar."""
        if total == 0:
            return ""

        filled = int((current / total) * width)
        bar = PROGRESS_CHARS["full"] * filled
        remaining = width - filled
        bar += PROGRESS_CHARS["empty"] * remaining

        return f"[{bar}] {current}/{total}"

    def print_status(self, message: str, status: str = "info", **kwargs):
        """Print a status message with appropriate visual styling."""
        icon = STATUS_ICONS.get(status, STATUS_ICONS["info"])

        if self.console and RICH_AVAILABLE:
            color_map = {
                "success": "green",
                "warning": "yellow",
                "error": "red",
                "info": "blue",
                "progress": "cyan"
            }
            color = color_map.get(status, "white")

            self.console.print(f"[{color}]{icon} {message}[/{color}]")
        else:
            print(f"{icon} {message}")

    def print_progress(self, current: int, total: int, description: str = "", width: int = None):
        """Print a progress indicator."""
        if width is None:
            width = min(self.config.max_width, 60)

        if self.console and RICH_AVAILABLE:
            progress = Progress(
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("• {task.description}"),
                console=self.console
            )

            with progress:
                task = progress.add_task(description or "Progress", total=total)
                progress.update(task, advance=current)
        else:
            # Text-based progress
            percentage = (current / total * 100) if total > 0 else 0
            bar = self._create_text_progress_bar(current, total, width - 15)
            print(f"🔄 {description}: {bar} ({percentage:.1f})")

    def print_summary(self, title: str, data: Dict[str, Any]):
        """Print a formatted summary table."""
        if self.console and RICH_AVAILABLE:
            table = Table(title=title, show_header=True, header_style="bold magenta")

            for key, value in data.items():
                table.add_column(key, style="cyan", no_wrap=True)
                table.add_column(str(value), style="white")

            self.console.print(table)
        else:
            print(f"\n📊 {title}:")
            for key, value in data.items():
                print(f"  {STATUS_ICONS['bullet']} {key}: {value}")

    def print_error_with_recovery(self, error: str, recovery_suggestions: List[str]):
        """Print an error message with recovery suggestions."""
        if self.console and RICH_AVAILABLE:
            # Create error panel
            error_text = Text(error, style="red")
            self.console.print(Panel(error_text, title="❌ Error", border_style="red"))

            if recovery_suggestions:
                suggestions_panel = Panel(
                    "\n".join(f"• {suggestion}" for suggestion in recovery_suggestions),
                    title="💡 Recovery Suggestions",
                    border_style="yellow"
                )
                self.console.print(suggestions_panel)
        else:
            print(f"\n❌ ERROR: {error}")
            if recovery_suggestions:
                print("💡 Recovery Suggestions:")
                for suggestion in recovery_suggestions:
                    print(f"  {STATUS_ICONS['bullet']} {suggestion}")

    def print_completion_banner(self, success: bool, duration: float, stats: Dict[str, Any]):
        """Print a completion banner with statistics."""
        status = "SUCCESS" if success else "FAILED"
        icon = STATUS_ICONS["success"] if success else STATUS_ICONS["error"]

        if self.console and RICH_AVAILABLE:
            # Create completion panel
            stats_text = "\n".join(f"{key}: {value}" for key, value in stats.items())
            completion_panel = Panel(
                f"[bold]{icon} Pipeline {status}[/bold]\n\n{stats_text}",
                title=f"🎯 Pipeline Complete ({duration:.1f}s)",
                border_style="green" if success else "red"
            )
            self.console.print(completion_panel)
        else:
            print(f"\n{'='*60}")
            print(f"{icon} PIPELINE {status} (Duration: {duration:.1f}s)")
            print(f"{'='*60}")
            for key, value in stats.items():
                print(f"  {STATUS_ICONS['bullet']} {key}: {value}")

    def create_step_progress_display(self, current_step: int, total_steps: int, description: str):
        """Create a visual step progress display."""
        if self.console and RICH_AVAILABLE:
            layout = Layout()

            # Header
            header = Panel(f"[bold cyan]Step {current_step}/{total_steps}: {description}[/bold cyan]",
                          border_style="blue")

            # Progress bar
            progress = Progress(
                BarColumn(complete_style="green"),
                MofNCompleteColumn(),
                TextColumn("• {task.description}"),
                console=self.console
            )

            layout.split_column(
                header,
                progress
            )

            return layout
        else:
            return None

def create_visual_logger(name: str, config: Optional[VisualConfig] = None) -> VisualLogger:
    """Factory function to create a visual logger."""
    return VisualLogger(name, config)

# Utility functions for consistent visual output across pipeline
def format_step_header(step_num: int, description: str, total_steps: int = 24) -> str:
    """Format a consistent step header."""
    progress_text = f"Step {step_num:2d}/{total_steps}"
    return f"🔢 {progress_text} - {description}"

def format_status_message(message: str, status: str = "info") -> str:
    """Format a status message with appropriate icon."""
    icon = STATUS_ICONS.get(status, STATUS_ICONS["info"])
    return f"{icon} {message}"

def format_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Create a text-based progress bar."""
    if total == 0:
        return ""

    filled = int((current / total) * width)
    bar = PROGRESS_CHARS["full"] * filled
    remaining = width - filled
    bar += PROGRESS_CHARS["empty"] * remaining

    percentage = (current / total * 100) if total > 0 else 0
    return f"[{bar}] {percentage:5.1f}"

def print_pipeline_banner(title: str, subtitle: str = ""):
    """Print a formatted pipeline banner."""
    if RICH_AVAILABLE and Console():
        console = Console()
        console.print(Panel.fit(
            f"[bold blue]{title}[/bold blue]\n[dim]{subtitle}[/dim]" if subtitle else f"[bold blue]{title}[/bold blue]",
            border_style="blue"
        ))
    else:
        width = min(TERMINAL_WIDTH, 80)
        border = "=" * width
        print(f"\n{border}")
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print(f"{border}\n")

def print_step_summary(step_num: int, description: str, status: str, duration: float, stats: Dict[str, Any]):
    """Print a formatted step completion summary."""
    status_icon = STATUS_ICONS.get(status, STATUS_ICONS["info"])

    if RICH_AVAILABLE and Console():
        console = Console()
        # Create summary panel
        stats_text = "\n".join(f"{key}: {value}" for key, value in stats.items())
        summary_panel = Panel(
            f"[bold]{status_icon} Step {step_num}: {description}[/bold]\n\n{stats_text}",
            title=f"⏱️ Complete ({duration:.2f}s)",
            border_style="green" if status == "success" else "yellow"
        )
        console.print(summary_panel)
    else:
        print(f"\n{status_icon} Step {step_num}: {description}")
        print(f"⏱️  Duration: {duration:.2f}s")
        for key, value in stats.items():
            print(f"   {STATUS_ICONS['bullet']} {key}: {value}")

def print_completion_summary(success: bool, total_duration: float, stats: Dict[str, Any]):
    """Print a comprehensive pipeline completion summary."""
    status = "SUCCESS" if success else "COMPLETED WITH ISSUES"
    status_icon = STATUS_ICONS["success"] if success else STATUS_ICONS["warning"]

    if RICH_AVAILABLE and Console():
        console = Console()
        # Create completion panel
        stats_text = "\n".join(f"{key}: {value}" for key, value in stats.items())
        completion_panel = Panel(
            f"[bold]{status_icon} Pipeline {status}[/bold]\n\n{stats_text}",
            title=f"🎯 Pipeline Complete ({total_duration:.1f}s)",
            border_style="green" if success else "yellow"
        )
        console.print(completion_panel)
    else:
        print(f"\n{'='*60}")
        print(f"{status_icon} PIPELINE {status} (Duration: {total_duration:.1f}s)")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"  {STATUS_ICONS['bullet']} {key}: {value}")

# Accessibility helpers
def strip_visual_elements(text: str) -> str:
    """Remove visual elements for screen readers."""
    for icon in STATUS_ICONS.values():
        text = text.replace(icon, "")
    return text

def ensure_minimum_width(text: str, min_width: int = 40) -> str:
    """Ensure text meets minimum width for readability."""
    if len(text) < min_width:
        text += " " * (min_width - len(text))
    return text

def format_accessible_message(message: str, level: str = "info") -> str:
    """Format message for screen reader accessibility."""
    # Add level indicator for screen readers
    level_indicators = {
        "error": "Error: ",
        "warning": "Warning: ",
        "info": "Info: ",
        "success": "Success: "
    }
    prefix = level_indicators.get(level, "")
    return f"{prefix}{strip_visual_elements(message)}"
