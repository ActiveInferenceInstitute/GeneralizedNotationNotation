"""
GNN Parser Module

This module provides functionality to parse GNN files and convert them into a structured format
for visualization and analysis.
"""

import re
import csv
import io
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GNNParser:
    """Parser for GNN files in various formats."""
    
    def __init__(self):
        """Initialize the GNN parser."""
        self.sections = {}
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a GNN file and return the structured content.
        
        Args:
            file_path: Path to the GNN file to parse
            
        Returns:
            Dictionary containing parsed GNN sections
        """
        path = Path(file_path)
        content = path.read_text()
        
        # Determine file format based on file content
        if "GNNSection," in content:
            return self._parse_csv_format(file_path, content)
        else:
            return self._parse_markdown_format(content)
    
    def _parse_csv_format(self, file_path: str, content: str) -> Dict[str, Any]:
        """Parse GNN file in CSV format."""
        sections = {}
        
        # Extract header comments
        header_lines = []
        content_lines = content.splitlines()
        line_index = 0
        
        while line_index < len(content_lines) and (content_lines[line_index].startswith('#') or not content_lines[line_index].strip()):
            if content_lines[line_index].startswith('#'):
                header_lines.append(content_lines[line_index])
            line_index += 1
        
        # Add header information to sections
        if header_lines:
            sections['ModelName'] = header_lines[0].replace('#', '').strip()
            if len(header_lines) > 1:
                sections['ModelAnnotation'] = '\n'.join([h.replace('#', '').strip() for h in header_lines[1:]])
        
        # Re-parse the raw file content to extract all sections
        self._extract_gnn_csv_sections(content, sections)
        
        # Process state space and connections
        logger.debug(f"Before processing, StateSpaceBlock length: {len(sections.get('StateSpaceBlock', ''))}")
        self._process_state_space(sections)
        self._process_connections(sections)
        
        return sections
    
    def _extract_gnn_csv_sections(self, content: str, sections: Dict[str, Any]) -> None:
        """Extract sections from GNN CSV format, handling quoted multiline values."""
        # Extract each section with its quoted content
        section_pattern = r'(\w+),(?:"([^"]*)"|(.*?)(?=\n\w+,|\Z))'
        
        for match in re.finditer(section_pattern, content, re.DOTALL):
            section_name = match.group(1)
            # Get the content from either the quoted group or unquoted group
            section_content = match.group(2) if match.group(2) is not None else match.group(3)
            
            if section_content is not None:
                # Clean up the section content
                if section_name in ['StateSpaceBlock', 'Connections', 'InitialParameterization',
                                    'Equations', 'Time', 'ActInfOntologyAnnotation']:
                    # Remove leading markdown header if present
                    if section_content.startswith('## '):
                        section_content = section_content.split('\n', 1)[1] if '\n' in section_content else ''
                
                sections[section_name] = section_content.strip()
    
    def _parse_markdown_format(self, content: str) -> Dict[str, Any]:
        """Parse GNN file in Markdown format."""
        lines = content.splitlines()
        sections: Dict[str, Any] = {}
        current_section_name: str | None = None
        current_section_content: List[str] = []
        
        # Preserve header comments (lines starting with # not followed by ##)
        header_comments = []
        line_idx = 0
        while line_idx < len(lines) and lines[line_idx].startswith('#') and not lines[line_idx].startswith('##'):
            header_comments.append(lines[line_idx])
            line_idx +=1
        if header_comments:
            sections['_HeaderComments'] = "\n".join(header_comments)

        # Heuristic for ModelName and ModelAnnotation if not explicitly sectioned early
        # This is based on the observation that the first few non-comment, non-section lines might be these.
        if lines[0].startswith('# ') and not lines[0].startswith('## '):
            potential_model_name = lines[0][2:].strip()
            if 'ModelName' not in sections:
                 sections['ModelName'] = potential_model_name

        for line in lines[line_idx:]:
            stripped_line = line.strip()
            if stripped_line.startswith('## '):
                if current_section_name and current_section_content:
                    sections[current_section_name] = '\n'.join(current_section_content).strip()
                
                current_section_name = stripped_line[3:].strip()
                current_section_content = []
                # Handle cases where section name might have a colon (as seen in gnn_file_structure.md)
                if ':' in current_section_name:
                    current_section_name = current_section_name.split(':',1)[0].strip()

            elif current_section_name:
                current_section_content.append(line)
        
        # Save the last section
        if current_section_name and current_section_content:
            sections[current_section_name] = '\n'.join(current_section_content).strip()
        
        # Process specific GNN sections for structured data
        self._process_state_space(sections)
        self._process_connections(sections)
        
        # Attempt to populate ModelName from header if not found as a section
        if 'ModelName' not in sections and '_HeaderComments' in sections:
            header_lines = sections['_HeaderComments'].split('\n')
            if header_lines and header_lines[0].startswith('# GNN Example: '):
                sections['ModelName'] = header_lines[0].replace('# GNN Example: ', '').strip()
            elif header_lines and header_lines[0].startswith('# '):
                 sections['ModelName'] = header_lines[0][2:].strip()

        return sections
    
    def _process_state_space(self, sections: Dict[str, Any]) -> None:
        """Process the StateSpaceBlock to extract variables and their dimensions."""
        if 'StateSpaceBlock' not in sections:
            logger.debug("StateSpaceBlock section not found")
            return
            
        state_space_content = sections['StateSpaceBlock']
        variables = {}
        
        logger.debug("Processing state space block...")
        
        lines = state_space_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('##'): # Ignore empty lines or sub-headers within content
                continue
                
            # First, check if there's a comment in the line
            comment = ""
            hash_index = line.find('#')
            if hash_index != -1:
                comment = line[hash_index+1:].strip()
                line = line[:hash_index].strip()  # Remove the comment part for cleaner regex matching
            
            # Match the variable definition
            match = re.match(r'(\w+(?:\^\w+)?(?:_\w+)?)\s*\[([^\]]+)\]', line)
            if not match:
                match = re.match(r'(\w+(?:\^\w+)?(?:_\w+)?)\s*([\w,=\[\]]+)', line)
            
            if match:
                var_name = match.group(1)
                dimensions_str = match.group(2)
                
                logger.debug(f"Found variable: {var_name}, dimensions: {dimensions_str}, comment: {comment}")
                
                dimensions = []
                var_type = None
                
                # Handle simple dimension strings like "len(π)" or "[2]"
                if re.fullmatch(r'len\(\w+\)', dimensions_str) or re.fullmatch(r'\[\d+\]', dimensions_str):
                    dimensions.append(dimensions_str) # Keep as string for such cases
                else:
                    for dim_part in dimensions_str.split(','):
                        dim_part = dim_part.strip()
                        if 'type=' in dim_part:
                            var_type = dim_part.split('=')[1]
                        else:
                            try:
                                dimensions.append(int(dim_part))
                            except ValueError:
                                dimensions.append(dim_part) # Keep as string if not int
                
                variables[var_name] = {
                    'dimensions': dimensions,
                    'type': var_type,
                    'comment': comment
                }
        
        if variables:
            logger.debug(f"Extracted {len(variables)} variables with comments: {[(k, v.get('comment', '')) for k, v in variables.items()]}")
            sections['Variables'] = variables
        else:
            logger.debug("No variables could be extracted from state space")
    
    def _process_connections(self, sections: Dict[str, Any]) -> None:
        """Process the Connections section to extract graph structure."""
        if 'Connections' not in sections:
            return
            
        connections_content = sections['Connections']
        edges = []
        
        pattern = r'(\w+(?:\^\w+)?(?:_\w+)?(?:\+\d+)?)\s*([>\-])\s*(\w+(?:\^\w+)?(?:_\w+)?(?:\+\d+)?)\s*(?:=\s*([^#]*))?(?:#(.*))?'
        
        lines = connections_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('##'): # Ignore empty lines or sub-headers
                continue
                
            match = re.match(pattern, line)
            if match:
                source = match.group(1)
                edge_type = match.group(2)
                target = match.group(3)
                constraint = match.group(4).strip() if match.group(4) else None
                comment = match.group(5).strip() if match.group(5) else None
                
                logger.debug(f"Found edge: {source} {edge_type} {target}")
                
                edges.append({
                    'source': source,
                    'target': target,
                    'directed': edge_type == '>',
                    'constraint': constraint,
                    'comment': comment
                })
        
        if edges:
            sections['Edges'] = edges
    
    def extract_sections(self, file_path: str) -> Dict[str, str]:
        """
        Extract all sections from a GNN file without detailed parsing.
        
        Args:
            file_path: Path to the GNN file
            
        Returns:
            Dictionary with section names as keys and their raw content as values
        """
        return self.parse_file(file_path) 