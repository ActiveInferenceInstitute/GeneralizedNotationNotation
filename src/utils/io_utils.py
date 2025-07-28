"""
I/O utilities for the GNN Processing Pipeline.

This module provides file I/O utilities for batch operations,
performance monitoring, and file system operations.
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil

logger = logging.getLogger(__name__)

def batch_write_files(files_data: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """
    Write multiple files in batch with performance tracking.
    
    Args:
        files_data: List of dictionaries with 'path' and 'content' keys
        output_dir: Directory to write files to
        
    Returns:
        Dictionary with write performance metrics
    """
    start_time = time.time()
    results = []
    
    for file_data in files_data:
        file_path = output_dir / file_data['path']
        content = file_data['content']
        
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            if isinstance(content, str):
                file_path.write_text(content, encoding='utf-8')
            elif isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                # Assume JSON-serializable
                file_path.write_text(json.dumps(content, indent=2), encoding='utf-8')
            
            results.append({
                'path': str(file_path),
                'success': True,
                'size': file_path.stat().st_size if file_path.exists() else 0
            })
            
        except Exception as e:
            results.append({
                'path': str(file_path),
                'success': False,
                'error': str(e)
            })
    
    end_time = time.time()
    
    successful_writes = [r for r in results if r['success']]
    failed_writes = [r for r in results if not r['success']]
    
    total_size = sum(r.get('size', 0) for r in successful_writes)
    
    return {
        'total_files': len(files_data),
        'successful_writes': len(successful_writes),
        'failed_writes': len(failed_writes),
        'total_size_bytes': total_size,
        'write_time_seconds': end_time - start_time,
        'throughput_mbps': (total_size / (1024 * 1024)) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
        'results': results
    }

def batch_read_files(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Read multiple files in batch with performance tracking.
    
    Args:
        file_paths: List of file paths to read
        
    Returns:
        Dictionary with read performance metrics
    """
    start_time = time.time()
    results = []
    
    for file_path in file_paths:
        try:
            if file_path.exists():
                # Try to read as text first
                try:
                    content = file_path.read_text(encoding='utf-8')
                    content_type = 'text'
                except UnicodeDecodeError:
                    # Fall back to binary
                    content = file_path.read_bytes()
                    content_type = 'binary'
                
                results.append({
                    'path': str(file_path),
                    'success': True,
                    'size': file_path.stat().st_size,
                    'content_type': content_type,
                    'content_length': len(content)
                })
            else:
                results.append({
                    'path': str(file_path),
                    'success': False,
                    'error': 'File not found'
                })
                
        except Exception as e:
            results.append({
                'path': str(file_path),
                'success': False,
                'error': str(e)
            })
    
    end_time = time.time()
    
    successful_reads = [r for r in results if r['success']]
    failed_reads = [r for r in results if not r['success']]
    
    total_size = sum(r.get('size', 0) for r in successful_reads)
    
    return {
        'total_files': len(file_paths),
        'successful_reads': len(successful_reads),
        'failed_reads': len(failed_reads),
        'total_size_bytes': total_size,
        'read_time_seconds': end_time - start_time,
        'throughput_mbps': (total_size / (1024 * 1024)) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
        'results': results
    }

def get_file_performance_metrics(file_path: Path) -> Dict[str, Any]:
    """
    Get performance metrics for file operations.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Dictionary with file performance metrics
    """
    if not file_path.exists():
        return {
            'exists': False,
            'error': 'File not found'
        }
    
    try:
        stat = file_path.stat()
        
        # Test read performance
        read_start = time.time()
        with open(file_path, 'rb') as f:
            content = f.read()
        read_time = time.time() - read_start
        
        return {
            'exists': True,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'read_time_seconds': read_time,
            'read_throughput_mbps': (stat.st_size / (1024 * 1024)) / read_time if read_time > 0 else 0,
            'modified_time': stat.st_mtime,
            'created_time': stat.st_ctime
        }
        
    except Exception as e:
        return {
            'exists': True,
            'error': str(e)
        }

def create_temp_file_with_content(content: Union[str, bytes], suffix: str = '.tmp') -> Path:
    """
    Create a temporary file with content and return its path.
    
    Args:
        content: Content to write to the file
        suffix: File suffix
        
    Returns:
        Path to the created temporary file
    """
    with tempfile.NamedTemporaryFile(mode='w' if isinstance(content, str) else 'wb', 
                                    suffix=suffix, delete=False) as f:
        if isinstance(content, str):
            f.write(content)
        else:
            f.write(content)
        return Path(f.name)

def cleanup_temp_files(temp_files: List[Path]) -> Dict[str, Any]:
    """
    Clean up temporary files and return cleanup metrics.
    
    Args:
        temp_files: List of temporary file paths to clean up
        
    Returns:
        Dictionary with cleanup metrics
    """
    start_time = time.time()
    results = []
    
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                size = temp_file.stat().st_size
                temp_file.unlink()
                results.append({
                    'path': str(temp_file),
                    'success': True,
                    'size_bytes': size
                })
            else:
                results.append({
                    'path': str(temp_file),
                    'success': True,
                    'size_bytes': 0
                })
        except Exception as e:
            results.append({
                'path': str(temp_file),
                'success': False,
                'error': str(e)
            })
    
    end_time = time.time()
    
    successful_cleanups = [r for r in results if r['success']]
    total_size = sum(r.get('size_bytes', 0) for r in successful_cleanups)
    
    return {
        'total_files': len(temp_files),
        'successful_cleanups': len(successful_cleanups),
        'failed_cleanups': len(results) - len(successful_cleanups),
        'total_size_bytes': total_size,
        'cleanup_time_seconds': end_time - start_time,
        'results': results
    } 