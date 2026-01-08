#!/usr/bin/env python3
"""
Base Processor - Abstract Base Class for Pipeline Step Processors

This module provides a standardized base class for all pipeline step processors,
reducing code duplication and ensuring consistent patterns across the codebase.

Usage:
    from utils.base_processor import BaseProcessor, ProcessingResult
    
    class MyProcessor(BaseProcessor):
        def process_single_file(self, file_path: Path, output_dir: Path, **kwargs) -> bool:
            # Implement file processing logic
            return True
"""

import logging
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .step_logging import (
    log_step_start,
    log_step_success,
    log_step_warning,
    log_step_error,
    setup_step_logging
)


@dataclass
class ProcessingResult:
    """Standardized result from processing operations."""
    success: bool
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)
    
    def save_to_json(self, output_path: Path) -> None:
        """Save result to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class BaseProcessor(ABC):
    """
    Abstract base class for all pipeline step processors.
    
    Subclasses must implement `process_single_file()` method.
    Provides standardized patterns for:
    - File discovery
    - Error handling
    - Logging
    - Result aggregation
    - Report generation
    """
    
    def __init__(self, step_name: str, logger: Optional[logging.Logger] = None, verbose: bool = False):
        """
        Initialize the processor.
        
        Args:
            step_name: Name of the pipeline step (e.g., "export", "visualization")
            logger: Optional logger instance. If None, creates one using step_name.
            verbose: Enable verbose logging
        """
        self.step_name = step_name
        self.logger = logger or setup_step_logging(step_name, verbose)
        self.verbose = verbose
    
    @abstractmethod
    def process_single_file(self, file_path: Path, output_dir: Path, **kwargs) -> bool:
        """
        Process a single file. Must be implemented by subclasses.
        
        Args:
            file_path: Path to the file to process
            output_dir: Output directory for results
            **kwargs: Additional processing options
            
        Returns:
            True if processing succeeded, False otherwise
        """
        pass
    
    def find_files(self, target_dir: Path, recursive: bool = False, 
                   extensions: List[str] = None, pattern: str = None) -> List[Path]:
        """
        Find files to process in the target directory.
        
        Args:
            target_dir: Directory to search
            recursive: Search recursively
            extensions: File extensions to include (e.g., ['.md', '.gnn'])
            pattern: Glob pattern to match (e.g., '*.md')
            
        Returns:
            List of file paths matching the criteria
        """
        if extensions is None:
            extensions = ['.md', '.gnn', '.json', '.yaml', '.yml']
        
        files = []
        
        if pattern:
            if recursive:
                files = list(target_dir.rglob(pattern))
            else:
                files = list(target_dir.glob(pattern))
        else:
            for ext in extensions:
                if recursive:
                    files.extend(target_dir.rglob(f"*{ext}"))
                else:
                    files.extend(target_dir.glob(f"*{ext}"))
        
        # Filter out hidden files and directories
        files = [f for f in files if not any(part.startswith('.') for part in f.parts)]
        
        return sorted(set(files))
    
    def process(self, target_dir: Path, output_dir: Path, 
                recursive: bool = False, **kwargs) -> ProcessingResult:
        """
        Main processing method. Finds files and processes each one.
        
        Args:
            target_dir: Directory containing input files
            output_dir: Directory for output files
            recursive: Process files recursively
            **kwargs: Additional processing options
            
        Returns:
            ProcessingResult with aggregated results
        """
        import time
        start_time = time.time()
        
        result = ProcessingResult(success=True)
        
        try:
            log_step_start(self.logger, f"Starting {self.step_name} processing")
            
            # Validate directories
            if not target_dir.exists():
                log_step_error(self.logger, f"Target directory does not exist: {target_dir}")
                result.success = False
                result.errors.append(f"Target directory not found: {target_dir}")
                return result
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find files to process
            files = self.find_files(target_dir, recursive=recursive, **kwargs)
            
            if not files:
                log_step_warning(self.logger, f"No files found in {target_dir}")
                result.warnings.append("No files found to process")
                return result
            
            self.logger.info(f"Found {len(files)} files to process")
            
            # Process each file
            for file_path in files:
                try:
                    if self.verbose:
                        self.logger.debug(f"Processing: {file_path}")
                    
                    success = self.process_single_file(file_path, output_dir, **kwargs)
                    
                    if success:
                        result.files_processed += 1
                    else:
                        result.files_failed += 1
                        result.errors.append(f"Failed to process: {file_path}")
                        
                except Exception as e:
                    result.files_failed += 1
                    result.errors.append(f"Error processing {file_path}: {str(e)}")
                    self.logger.error(f"Error processing {file_path}: {e}")
            
            # Determine overall success
            result.success = result.files_failed == 0
            result.execution_time = time.time() - start_time
            
            # Log summary
            self._log_summary(result)
            
            # Save report
            self._save_report(result, output_dir)
            
            return result
            
        except Exception as e:
            result.success = False
            result.errors.append(f"Processing failed: {str(e)}")
            result.execution_time = time.time() - start_time
            log_step_error(self.logger, f"{self.step_name} processing failed: {e}")
            return result
    
    def _log_summary(self, result: ProcessingResult) -> None:
        """Log processing summary."""
        total = result.files_processed + result.files_failed + result.files_skipped
        
        if result.success:
            log_step_success(
                self.logger, 
                f"{self.step_name} completed: {result.files_processed}/{total} files processed"
            )
        elif result.files_processed > 0:
            log_step_warning(
                self.logger,
                f"{self.step_name} completed with errors: {result.files_processed}/{total} files processed, {result.files_failed} failed"
            )
        else:
            log_step_error(
                self.logger,
                f"{self.step_name} failed: {result.files_failed}/{total} files failed"
            )
    
    def _save_report(self, result: ProcessingResult, output_dir: Path) -> None:
        """Save processing report to JSON."""
        report_path = output_dir / f"{self.step_name}_processing_report.json"
        result.save_to_json(report_path)
        self.logger.debug(f"Saved processing report to {report_path}")


def create_processor(step_name: str, process_func: Callable[[Path, Path], bool], 
                     logger: Optional[logging.Logger] = None, verbose: bool = False) -> BaseProcessor:
    """
    Factory function to create a processor from a simple processing function.
    
    Args:
        step_name: Name of the step
        process_func: Function that takes (file_path, output_dir) and returns bool
        logger: Optional logger
        verbose: Enable verbose logging
        
    Returns:
        BaseProcessor instance wrapping the function
    """
    class FunctionProcessor(BaseProcessor):
        def __init__(self):
            super().__init__(step_name, logger, verbose)
            self._process_func = process_func
            
        def process_single_file(self, file_path: Path, output_dir: Path, **kwargs) -> bool:
            return self._process_func(file_path, output_dir)
    
    return FunctionProcessor()


__all__ = [
    "BaseProcessor",
    "ProcessingResult", 
    "create_processor",
]
