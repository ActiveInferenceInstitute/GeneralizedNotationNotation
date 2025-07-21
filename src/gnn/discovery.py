#!/usr/bin/env python3
"""
GNN File Discovery Strategy Module

This module provides intelligent file discovery with content analysis
and validation for GNN models.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class DiscoveryResult:
    """Result of file discovery operation."""
    files_found: List[Path]
    analysis_time: float
    file_types: Dict[str, int]
    potential_gnn_files: List[Path]
    metadata: Dict[str, Any]


class FileDiscoveryStrategy:
    """
    Intelligent file discovery strategy for GNN models.
    
    Performs content-aware discovery and basic analysis
    to identify potential GNN files.
    """
    
    def __init__(self):
        self.target_extensions = ['.md', '.json', '.xml', '.yaml', '.pkl']
        self.gnn_indicators = [
            '## ModelName',
            '## StateSpaceBlock', 
            '## Connections',
            'GNN',
            'model_name',
            'variables',
            'connections'
        ]
    
    def configure(self, target_extensions: Optional[List[str]] = None, **kwargs):
        """Configure discovery parameters."""
        if target_extensions:
            self.target_extensions = target_extensions
    
    def discover(self, target_dir: Path) -> List[Path]:
        """
        Discover GNN files in target directory.
        
        Args:
            target_dir: Directory to search
            
        Returns:
            List of discovered file paths
        """
        logger.info(f"Starting file discovery in {target_dir}")
        start_time = time.time()
        
        # Basic file discovery
        all_files = []
        potential_gnn_files = []
        
        # Recursive search for target extensions
        for ext in self.target_extensions:
            pattern = f"**/*{ext}"
            found_files = list(target_dir.rglob(pattern))
            all_files.extend(found_files)
            logger.debug(f"Found {len(found_files)} {ext} files")
        
        # Content analysis for GNN indicators
        for file_path in all_files:
            if self._analyze_file_content(file_path):
                potential_gnn_files.append(file_path)
        
        discovery_time = time.time() - start_time
        logger.info(f"Discovery completed: {len(potential_gnn_files)}/{len(all_files)} potential GNN files in {discovery_time:.3f}s")
        
        return potential_gnn_files
    
    def _analyze_file_content(self, file_path: Path) -> bool:
        """
        Analyze file content for GNN indicators.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            bool: Whether file appears to be a GNN file
        """
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check for GNN indicators
            indicators_found = 0
            for indicator in self.gnn_indicators:
                if indicator.lower() in content.lower():
                    indicators_found += 1
            
            # Require at least 2 indicators for GNN classification
            return indicators_found >= 2
            
        except Exception as e:
            logger.debug(f"Could not analyze {file_path}: {e}")
            return False 