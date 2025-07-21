#!/usr/bin/env python3
"""
GNN File Discovery Module

This module provides intelligent file discovery strategies for GNN models,
supporting multiple file formats and content-based detection.
"""

import logging
import os
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DiscoveryStrategy(Enum):
    """File discovery strategies."""
    EXTENSION_BASED = "extension"
    CONTENT_BASED = "content"
    HYBRID = "hybrid"


@dataclass
class FileAnalysis:
    """Analysis result for a discovered file."""
    file_path: Path
    is_gnn_file: bool
    confidence: float
    detected_format: str
    file_size: int
    analysis_notes: List[str]


class FileDiscoveryStrategy:
    """
    Intelligent file discovery with multiple strategies.
    
    Supports extension-based, content-based, and hybrid discovery
    with configurable confidence thresholds.
    """
    
    def __init__(self, strategy: DiscoveryStrategy = DiscoveryStrategy.HYBRID):
        self.strategy = strategy
        self.recursive = False
        self.target_extensions: Set[str] = {'.md', '.json', '.xml', '.yaml', '.yml', '.pkl', '.gnn', '.txt'}
        self.confidence_threshold = 0.7
        
        # GNN content markers for detection
        self.gnn_markers = {
            'strong': [
                'generalized notation notation',
                'gnn model',
                'gnn section',
                'modelname',
                'statespaceblock',
                'initialparameterization'
            ],
            'moderate': [
                'active inference',
                'model specification',
                'pomdp',
                'belief state',
                'generative model'
            ],
            'weak': [
                'variables',
                'connections',
                'parameters',
                'equations'
            ]
        }
    
    def configure(self, recursive: bool = False, 
                 target_extensions: Optional[List[str]] = None,
                 confidence_threshold: float = 0.7):
        """Configure discovery parameters."""
        self.recursive = recursive
        if target_extensions:
            self.target_extensions = set(target_extensions)
        self.confidence_threshold = confidence_threshold
    
    def discover(self, target_dir: Path) -> List[Path]:
        """
        Discover GNN files using configured strategy.
        
        Args:
            target_dir: Directory to search
            
        Returns:
            List of discovered GNN file paths
        """
        logger.info(f"Starting file discovery in {target_dir}")
        logger.debug(f"Strategy: {self.strategy.value}, Recursive: {self.recursive}")
        
        if not target_dir.exists():
            logger.error(f"Target directory does not exist: {target_dir}")
            return []
        
        # Get candidate files
        candidate_files = self._get_candidate_files(target_dir)
        logger.debug(f"Found {len(candidate_files)} candidate files")
        
        # Analyze and filter files
        discovered_files = []
        for file_path in candidate_files:
            analysis = self._analyze_file(file_path)
            
            if analysis.is_gnn_file and analysis.confidence >= self.confidence_threshold:
                discovered_files.append(file_path)
                logger.debug(f"Discovered GNN file: {file_path} (confidence: {analysis.confidence:.2f})")
        
        logger.info(f"Discovered {len(discovered_files)} GNN files")
        return discovered_files
    
    def _get_candidate_files(self, target_dir: Path) -> List[Path]:
        """Get candidate files based on search strategy."""
        candidate_files = []
        
        if self.recursive:
            for root, _, files in os.walk(target_dir):
                for file in files:
                    file_path = Path(root) / file
                    if self._is_candidate_file(file_path):
                        candidate_files.append(file_path)
        else:
            for file_path in target_dir.iterdir():
                if file_path.is_file() and self._is_candidate_file(file_path):
                    candidate_files.append(file_path)
        
        return candidate_files
    
    def _is_candidate_file(self, file_path: Path) -> bool:
        """Check if file is a candidate for GNN analysis."""
        # Always check files with target extensions
        if file_path.suffix.lower() in self.target_extensions:
            return True
        
        # For hybrid strategy, also check files without extensions
        if self.strategy == DiscoveryStrategy.HYBRID:
            return not file_path.suffix or file_path.suffix.lower() in {'.txt', '.text'}
        
        return False
    
    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a file to determine if it's a GNN file."""
        analysis = FileAnalysis(
            file_path=file_path,
            is_gnn_file=False,
            confidence=0.0,
            detected_format="unknown",
            file_size=0,
            analysis_notes=[]
        )
        
        try:
            analysis.file_size = file_path.stat().st_size
            
            # Extension-based analysis
            extension_score = self._analyze_extension(file_path, analysis)
            
            # Content-based analysis
            content_score = self._analyze_content(file_path, analysis)
            
            # Combine scores based on strategy
            if self.strategy == DiscoveryStrategy.EXTENSION_BASED:
                analysis.confidence = extension_score
            elif self.strategy == DiscoveryStrategy.CONTENT_BASED:
                analysis.confidence = content_score
            else:  # HYBRID
                analysis.confidence = max(extension_score * 0.3 + content_score * 0.7, 
                                        max(extension_score, content_score))
            
            analysis.is_gnn_file = analysis.confidence >= self.confidence_threshold
            
        except Exception as e:
            analysis.analysis_notes.append(f"Analysis error: {e}")
            logger.debug(f"Error analyzing {file_path}: {e}")
        
        return analysis
    
    def _analyze_extension(self, file_path: Path, analysis: FileAnalysis) -> float:
        """Analyze file extension for GNN likelihood."""
        ext = file_path.suffix.lower()
        
        # High confidence extensions
        if ext in {'.gnn', '.md'}:
            analysis.detected_format = 'markdown' if ext == '.md' else 'gnn'
            analysis.analysis_notes.append(f"High-confidence extension: {ext}")
            return 0.9
        
        # Medium confidence extensions
        if ext in {'.json', '.xml', '.yaml', '.yml'}:
            analysis.detected_format = ext[1:]  # Remove dot
            analysis.analysis_notes.append(f"Medium-confidence extension: {ext}")
            return 0.6
        
        # Low confidence extensions
        if ext in {'.pkl', '.pickle', '.txt'}:
            analysis.detected_format = 'pickle' if 'pkl' in ext else 'text'
            analysis.analysis_notes.append(f"Low-confidence extension: {ext}")
            return 0.3
        
        analysis.analysis_notes.append(f"Unknown extension: {ext}")
        return 0.1
    
    def _analyze_content(self, file_path: Path, analysis: FileAnalysis) -> float:
        """Analyze file content for GNN markers."""
        try:
            # Handle binary files
            if self._is_binary_file(file_path):
                return self._analyze_binary_content(file_path, analysis)
            
            # Read text content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2000).lower()  # Read first 2KB
            
            return self._score_text_content(content, analysis)
        
        except Exception as e:
            analysis.analysis_notes.append(f"Content analysis error: {e}")
            return 0.0
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(512)
                return b'\x00' in chunk or not chunk.replace(b'\r', b'').replace(b'\n', b'').replace(b'\t', b'').decode('ascii', errors='ignore').isprintable()
        except:
            return True
    
    def _analyze_binary_content(self, file_path: Path, analysis: FileAnalysis) -> float:
        """Analyze binary file content."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(50)
            
            # Check for pickle signatures
            if header.startswith(b'\x80\x03') or b'pickle' in header:
                analysis.detected_format = 'pickle'
                analysis.analysis_notes.append("Detected pickle format")
                return 0.5  # Moderate confidence for pickle files
            
            analysis.analysis_notes.append("Unknown binary format")
            return 0.1
        
        except Exception as e:
            analysis.analysis_notes.append(f"Binary analysis error: {e}")
            return 0.0
    
    def _score_text_content(self, content: str, analysis: FileAnalysis) -> float:
        """Score text content for GNN markers."""
        score = 0.0
        markers_found = []
        
        # Check for strong markers
        for marker in self.gnn_markers['strong']:
            if marker in content:
                score += 0.3
                markers_found.append(f"strong:{marker}")
        
        # Check for moderate markers
        for marker in self.gnn_markers['moderate']:
            if marker in content:
                score += 0.15
                markers_found.append(f"moderate:{marker}")
        
        # Check for weak markers
        weak_count = 0
        for marker in self.gnn_markers['weak']:
            if marker in content:
                weak_count += 1
                markers_found.append(f"weak:{marker}")
        
        # Bonus for multiple weak markers
        if weak_count >= 3:
            score += 0.2
        elif weak_count >= 2:
            score += 0.1
        
        # Check for structural markers
        if self._has_gnn_structure(content):
            score += 0.2
            markers_found.append("structural:gnn_sections")
        
        # Format-specific detection
        detected_format = self._detect_text_format(content)
        if detected_format != 'unknown':
            analysis.detected_format = detected_format
            score += 0.1
        
        # Cap score at 1.0
        score = min(score, 1.0)
        
        analysis.analysis_notes.append(f"Content markers: {markers_found}")
        analysis.analysis_notes.append(f"Content score: {score:.2f}")
        
        return score
    
    def _has_gnn_structure(self, content: str) -> bool:
        """Check if content has GNN structural markers."""
        # Look for section headers
        gnn_sections = [
            '## gnn', '## model', '## state', '## connection', 
            '## parameter', '## equation', '## time'
        ]
        
        section_count = sum(1 for section in gnn_sections if section in content)
        return section_count >= 2
    
    def _detect_text_format(self, content: str) -> str:
        """Detect text format from content."""
        content_strip = content.strip()
        
        # JSON detection
        if content_strip.startswith('{') and '"model_name"' in content:
            return 'json'
        
        # XML detection
        if content_strip.startswith('<?xml') or content_strip.startswith('<gnn'):
            return 'xml'
        
        # YAML detection
        if content.startswith('---') or ('model_name:' in content and 'variables:' in content):
            return 'yaml'
        
        # Markdown detection (GNN markdown has specific structure)
        if '##' in content and any(marker in content for marker in ['GNNSection', 'ModelName', 'StateSpaceBlock']):
            return 'markdown'
        
        return 'unknown'
    
    def get_discovery_stats(self, discovered_files: List[Path]) -> Dict[str, Any]:
        """Get statistics about discovered files."""
        stats = {
            'total_discovered': len(discovered_files),
            'formats': {},
            'sizes': [],
            'average_confidence': 0.0
        }
        
        total_confidence = 0.0
        for file_path in discovered_files:
            analysis = self._analyze_file(file_path)
            
            # Format distribution
            fmt = analysis.detected_format
            stats['formats'][fmt] = stats['formats'].get(fmt, 0) + 1
            
            # File sizes
            stats['sizes'].append(analysis.file_size)
            
            # Confidence
            total_confidence += analysis.confidence
        
        if discovered_files:
            stats['average_confidence'] = total_confidence / len(discovered_files)
            stats['average_size'] = sum(stats['sizes']) / len(stats['sizes'])
            stats['size_range'] = (min(stats['sizes']), max(stats['sizes']))
        
        return stats 