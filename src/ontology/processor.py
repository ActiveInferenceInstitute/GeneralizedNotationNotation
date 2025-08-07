#!/usr/bin/env python3
"""
Ontology Processor Module

This module provides core ontology processing functionality for GNN files.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import time
from collections import Counter

def extract_ontology_terms(content: str) -> List[str]:
    """
    Extract ontology terms from GNN content.
    
    Args:
        content: GNN file content
        
    Returns:
        List of ontology terms found
    """
    terms = []
    
    # Extract terms from different sections
    sections = content.split('\n## ')
    
    for section in sections:
        if section.strip():
            # Extract section name as potential term
            lines = section.split('\n')
            if lines:
                section_name = lines[0].strip()
                if section_name:
                    terms.append(section_name.lower())
    
    # Extract specific ontology terms
    ontology_keywords = [
        'state', 'observation', 'action', 'reward', 'belief', 'policy',
        'transition', 'likelihood', 'prior', 'posterior', 'evidence',
        'free energy', 'surprise', 'entropy', 'information', 'uncertainty'
    ]
    
    content_lower = content.lower()
    for keyword in ontology_keywords:
        if keyword in content_lower:
            terms.append(keyword)
    
    return list(set(terms))  # Remove duplicates

def validate_ontology_compliance(terms: List[str]) -> Dict[str, Any]:
    """
    Validate ontology compliance for extracted terms.
    
    Args:
        terms: List of ontology terms
        
    Returns:
        Validation result dictionary
    """
    # Define Active Inference ontology standards
    required_terms = ['state', 'observation', 'action']
    recommended_terms = ['belief', 'policy', 'free energy']
    
    validation_result = {
        "compliance_score": 0.0,
        "required_terms_found": [],
        "required_terms_missing": [],
        "recommended_terms_found": [],
        "recommended_terms_missing": [],
        "compliance_level": "unknown"
    }
    
    # Check required terms
    for term in required_terms:
        if term in terms:
            validation_result["required_terms_found"].append(term)
        else:
            validation_result["required_terms_missing"].append(term)
    
    # Check recommended terms
    for term in recommended_terms:
        if term in terms:
            validation_result["recommended_terms_found"].append(term)
        else:
            validation_result["recommended_terms_missing"].append(term)
    
    # Calculate compliance score
    required_found = len(validation_result["required_terms_found"])
    required_total = len(required_terms)
    
    if required_total > 0:
        validation_result["compliance_score"] = required_found / required_total
    
    # Determine compliance level
    if validation_result["compliance_score"] >= 0.8:
        validation_result["compliance_level"] = "high"
    elif validation_result["compliance_score"] >= 0.5:
        validation_result["compliance_level"] = "medium"
    else:
        validation_result["compliance_level"] = "low"
    
    return validation_result

def generate_ontology_mapping(terms: List[str]) -> Dict[str, Any]:
    """
    Generate ontology mapping for extracted terms.
    
    Args:
        terms: List of ontology terms
        
    Returns:
        Mapping result dictionary
    """
    # Define ontology relationships
    ontology_relationships = {
        "state": ["observation", "belief", "transition"],
        "observation": ["state", "likelihood", "evidence"],
        "action": ["policy", "reward", "transition"],
        "belief": ["state", "prior", "posterior"],
        "policy": ["action", "reward", "free energy"],
        "free energy": ["surprise", "entropy", "uncertainty"]
    }
    
    mapping_result = {
        "terms": terms,
        "relationships": {},
        "hierarchical_structure": {},
        "semantic_clusters": []
    }
    
    # Generate relationships for found terms
    for term in terms:
        if term in ontology_relationships:
            mapping_result["relationships"][term] = ontology_relationships[term]
    
    # Create semantic clusters
    clusters = {
        "perception": ["observation", "evidence", "likelihood"],
        "cognition": ["belief", "state", "uncertainty"],
        "action": ["action", "policy", "reward"],
        "learning": ["free energy", "surprise", "entropy"]
    }
    
    for cluster_name, cluster_terms in clusters.items():
        found_terms = [term for term in terms if term in cluster_terms]
        if found_terms:
            mapping_result["semantic_clusters"].append({
                "cluster": cluster_name,
                "terms": found_terms
            })
    
    return mapping_result

def process_ontology_file(gnn_file: Path, output_dir: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Process ontology for a single GNN file.
    
    Args:
        gnn_file: Path to the GNN file
        output_dir: Output directory for results
        logger: Logger instance
        
    Returns:
        Dictionary with ontology processing results or None if failed
    """
    try:
        # Read GNN file content
        with open(gnn_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract ontology terms from content
        ontology_terms = extract_ontology_terms(content)
        
        # Validate ontology compliance
        validation_result = validate_ontology_compliance(ontology_terms)
        
        # Generate ontology mapping
        mapping_result = generate_ontology_mapping(ontology_terms)
        
        # Create result structure
        result = {
            "file": str(gnn_file),
            "ontology_terms": ontology_terms,
            "validation": validation_result,
            "mapping": mapping_result,
            "timestamp": time.time()
        }
        
        # Save individual file result
        output_file = output_dir / f"{gnn_file.stem}_ontology.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        logger.debug(f"Saved ontology result to: {output_file}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process ontology for {gnn_file}: {e}")
        return None

def generate_ontology_report(processed_files: List[Path], ontology_results: List[Dict[str, Any]], 
                           output_dir: Path, logger: logging.Logger) -> None:
    """
    Generate comprehensive ontology report.
    
    Args:
        processed_files: List of processed files
        ontology_results: List of ontology results
        output_dir: Output directory
        logger: Logger instance
    """
    try:
        # Aggregate results
        total_files = len(processed_files)
        total_terms = sum(len(result["ontology_terms"]) for result in ontology_results)
        
        # Calculate compliance statistics
        compliance_scores = [result["validation"]["compliance_score"] for result in ontology_results]
        avg_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
        
        # Generate report
        report = {
            "summary": {
                "total_files_processed": total_files,
                "total_ontology_terms": total_terms,
                "average_compliance_score": avg_compliance,
                "processing_timestamp": time.time()
            },
            "compliance_distribution": {
                "high": len([r for r in ontology_results if r["validation"]["compliance_level"] == "high"]),
                "medium": len([r for r in ontology_results if r["validation"]["compliance_level"] == "medium"]),
                "low": len([r for r in ontology_results if r["validation"]["compliance_level"] == "low"])
            },
            "common_terms": {},
            "semantic_clusters": {},
            "files": [str(f) for f in processed_files]
        }
        
        # Count common terms
        all_terms = []
        for result in ontology_results:
            all_terms.extend(result["ontology_terms"])
        
        term_counts = Counter(all_terms)
        report["common_terms"] = dict(term_counts.most_common(10))
        
        # Aggregate semantic clusters
        all_clusters = {}
        for result in ontology_results:
            for cluster in result["mapping"]["semantic_clusters"]:
                cluster_name = cluster["cluster"]
                if cluster_name not in all_clusters:
                    all_clusters[cluster_name] = []
                all_clusters[cluster_name].extend(cluster["terms"])
        
        # Remove duplicates from clusters
        for cluster_name in all_clusters:
            all_clusters[cluster_name] = list(set(all_clusters[cluster_name]))
        
        report["semantic_clusters"] = all_clusters
        
        # Save report
        report_file = output_dir / "ontology_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated comprehensive ontology report: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate ontology report: {e}")

def process_ontology_fallback(gnn_files: List[Path], output_dir: Path, logger: logging.Logger) -> bool:
    """
    Fallback ontology processing when ontology module is not available.
    
    Args:
        gnn_files: List of GNN files to process
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info("Using fallback ontology processing")
        
        # Basic ontology processing
        processed_count = 0
        
        for gnn_file in gnn_files:
            try:
                # Read file content
                with open(gnn_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic ontology extraction
                terms = extract_ontology_terms(content)
                validation = validate_ontology_compliance(terms)
                mapping = generate_ontology_mapping(terms)
                
                # Create basic result
                result = {
                    "file": str(gnn_file),
                    "ontology_terms": terms,
                    "validation": validation,
                    "mapping": mapping,
                    "timestamp": time.time(),
                    "processing_mode": "fallback"
                }
                
                # Save result
                output_file = output_dir / f"{gnn_file.stem}_ontology_fallback.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {gnn_file} in fallback mode: {e}")
                continue
        
        logger.info(f"Fallback processing completed: {processed_count} files processed")
        return processed_count > 0
        
    except Exception as e:
        logger.error(f"Fallback ontology processing failed: {e}")
        return False 