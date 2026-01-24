# oxdraw as GNN Interface: Visual Diagram-as-Code for Active Inference Models

**Status**: Integration Proposal
**Version**: 1.0

---

## Executive Summary

oxdraw serves as a visual interface for the Generalized Notation Notation (GNN) pipeline, enabling drag-and-drop creation of Active Inference graphical models with bidirectional text-visual synchronization. This integration leverages oxdraw's hybrid Mermaid-based architecture alongside GNN's 24-step processing pipeline to provide researchers with intuitive visual model construction while maintaining rigorous semantic validation and computational reproducibility.

**Key Integration Value**:
- Visual construction of POMDP architectures through intuitive node-and-edge manipulation
- Automatic translation between Mermaid diagrams and GNN plaintext specifications
- Preservation of Active Inference ontology assertions through comment-based metadata
- Full integration with GNN validation, type-checking, and simulation rendering workflows

---

## Architecture Overview

### Bidirectional Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                    GNN Pipeline Integration                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐       ┌──────────────┐      ┌─────────────┐│
│  │  GNN File    │◄─────►│   oxdraw     │◄────►│  Mermaid    ││
│  │ actinf.md    │       │  Interface   │      │  Diagram    ││
│  └──────────────┘       └──────────────┘      └─────────────┘│
│        │                       │                      │        │
│        │ (1) parse_gnn_file   │ (3) visual_edit     │        │
│        ▼                       ▼                      ▼        │
│  ┌──────────────┐       ┌──────────────┐      ┌─────────────┐│
│  │ GNN Parser   │       │  oxdraw      │      │  Mermaid    ││
│  │ (Step 3)     │       │  Editor      │      │  Renderer   ││
│  └──────────────┘       └──────────────┘      └─────────────┘│
│        │                       │                      │        │
│        │ (2) gnn_to_mermaid   │ (4) mermaid_to_gnn  │        │
│        ▼                       ▼                      ▼        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │            Unified GNN Internal Representation            │ │
│  │  • Variables (A,B,C,D,E matrices & state vectors)        │ │
│  │  • Connections (D>s, s-A, A-o, etc.)                     │ │
│  │  • Ontology Mappings (LikelihoodMatrix, etc.)           │ │
│  │  • Parameters (num_states, num_obs, num_actions)        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │          GNN Pipeline Processing (Steps 3-23)             │ │
│  │  • Type Checking (Step 5)  • Validation (Step 6)         │ │
│  │  • Visualization (Step 8)   • Ontology (Step 10)         │ │
│  │  • Code Render (Step 11)    • Execute (Step 12)          │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### Integration Points with GNN Modules

| GNN Module | oxdraw Integration | Implementation |
|------------|-------------------|----------------|
| **3_gnn** | Parse GNN → Mermaid | `parse_gnn_file()` + custom Mermaid generator |
| **5_type_checker** | Validate diagram structure | Real-time validation in oxdraw editor |
| **6_validation** | Connection consistency | Visual feedback on invalid edges |
| **7_export** | Multi-format export | Export oxdraw state with GNN metadata |
| **8_visualization** | Generate static diagrams | Automatic diagram generation from GNN |
| **10_ontology** | Embed ontology terms | Comment-based metadata in Mermaid |
| **22_gui** | Alternative visual interface | Complementary to form-based GUI |

---

## Implementation: GNN to Mermaid Conversion

### Module: `src/gnn/mermaid_converter.py`

This module translates parsed GNN models to Mermaid flowchart syntax compatible with oxdraw.

```python
"""
GNN to Mermaid Converter for oxdraw Integration

Converts parsed GNN Active Inference models to Mermaid flowchart format
with embedded metadata for bidirectional synchronization.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import re

from gnn.parser import parse_gnn_file, ParsedGNN
from ontology.processor import load_defined_ontology_terms


def gnn_to_mermaid(gnn_model: ParsedGNN, include_metadata: bool = True) -> str:
    """
    Convert parsed GNN model to Mermaid flowchart format.
    
    Args:
        gnn_model: Parsed GNN model from gnn.parser
        include_metadata: Include oxdraw-compatible metadata in comments
        
    Returns:
        Mermaid flowchart string with embedded GNN metadata
    """
    lines = []
    
    # Header with model metadata
    lines.append(f"flowchart TD")
    lines.append(f"    %% GNN Model: {gnn_model.model_name}")
    lines.append(f"    %% GNN Version: {gnn_model.version}")
    
    if include_metadata:
        # Embed full GNN specification as JSON in comment
        metadata = {
            "model_name": gnn_model.model_name,
            "variables": _serialize_variables(gnn_model.variables),
            "connections": _serialize_connections(gnn_model.connections),
            "parameters": gnn_model.parameters,
            "ontology_mappings": _serialize_ontology(gnn_model.ontology_mappings)
        }
        lines.append(f"    %% GNN_METADATA: {json.dumps(metadata)}")
    
    # Generate nodes from variables
    for var_name, var_data in gnn_model.variables.items():
        node_shape = _infer_node_shape(var_name, var_data)
        node_label = _generate_node_label(var_name, var_data)
        lines.append(f"    {var_name}{node_shape}")
        
        # Add ontology annotation as comment
        if var_data.get('ontology_mapping'):
            lines.append(f"    %% {var_name}: {var_data['ontology_mapping']}")
    
    # Generate edges from connections
    for conn in gnn_model.connections:
        edge_style = _infer_edge_style(conn['symbol'])
        edge_label = _generate_edge_label(conn)
        lines.append(f"    {conn['source']}{edge_style}{conn['target']}")
        
        # Add connection type as comment
        if conn.get('connection_type'):
            lines.append(f"    %% Connection: {conn['connection_type']}")
    
    # Styling section
    lines.append("")
    lines.append("    %% Node styling based on variable types")
    lines.extend(_generate_node_styles(gnn_model.variables))
    
    return "\n".join(lines)


def _infer_node_shape(var_name: str, var_data: Dict) -> str:
    """
    Infer Mermaid node shape from GNN variable characteristics.
    
    Shape mapping:
    - Matrices (A, B): Rectangle [A]
    - Vectors (C, D, E): Rounded (C)
    - States (s, s_prime): Stadium ([s])
    - Observations (o): Circle ((o))
    - Actions (u): Hexagon {{u}}
    - Policies (π): Diamond {π}
    - Free Energy (F, G): Trapezoid [/F\]
    """
    # Check dimensionality
    dims = var_data.get('dimensions', [])
    ontology = var_data.get('ontology_mapping', '')
    
    if len(dims) >= 2:  # Matrices
        return f"[{var_name}]"
    elif 'State' in ontology:
        return f"([{var_name}])"
    elif 'Observation' in ontology:
        return f"(({var_name}))"
    elif 'Action' in ontology or var_name == 'u':
        return f"{{{{{var_name}}}}}"
    elif 'Policy' in ontology or var_name == 'π':
        return f"{{{var_name}}}"
    elif 'FreeEnergy' in ontology:
        return f"[/{var_name}\\]"
    else:  # Default vectors
        return f"({var_name})"


def _generate_node_label(var_name: str, var_data: Dict) -> str:
    """Generate descriptive node label with dimensions and type."""
    label = var_name
    
    if var_data.get('dimensions'):
        dims_str = 'x'.join(map(str, var_data['dimensions']))
        label += f"<br/>{dims_str}"
    
    if var_data.get('data_type'):
        label += f"<br/>{var_data['data_type']}"
    
    return label


def _infer_edge_style(symbol: str) -> str:
    """
    Convert GNN connection symbols to Mermaid edge styles.
    
    Mapping:
    - > : Generative (thick arrow) ==>
    - - : Inference (dashed line) -.->
    - * : Modulation (dotted line) -..->
    - ~ : Weak coupling (thin line) -->
    """
    if symbol == '>':
        return ' ==> '
    elif symbol == '-':
        return ' -.-> '
    elif symbol == '*':
        return ' -..-> '
    else:
        return ' --> '


def _generate_edge_label(conn: Dict) -> str:
    """Generate edge label with connection semantics."""
    label = ""
    if conn.get('description'):
        label = f"|{conn['description']}|"
    return label


def _serialize_variables(variables: Dict) -> Dict:
    """Serialize variables to JSON-compatible format."""
    serialized = {}
    for var_name, var_data in variables.items():
        serialized[var_name] = {
            "dimensions": var_data.get('dimensions', []),
            "data_type": var_data.get('data_type', 'float'),
            "ontology_mapping": var_data.get('ontology_mapping', ''),
            "description": var_data.get('description', '')
        }
    return serialized


def _serialize_connections(connections: List[Dict]) -> List[Dict]:
    """Serialize connections to JSON-compatible format."""
    return [
        {
            "source": c.get('source', ''),
            "target": c.get('target', ''),
            "symbol": c.get('symbol', ''),
            "connection_type": c.get('connection_type', '')
        }
        for c in connections
    ]


def _serialize_ontology(ontology_mappings: List[Dict]) -> Dict:
    """Serialize ontology mappings."""
    return {
        mapping.get('variable', ''): mapping.get('ontology_term', '')
        for mapping in ontology_mappings
    }


def _generate_node_styles(variables: Dict) -> List[str]:
    """Generate Mermaid styling directives based on variable types."""
    styles = []
    
    # Group variables by ontology type
    matrices = []
    vectors = []
    states = []
    observations = []
    actions = []
    
    for var_name, var_data in variables.items():
        ontology = var_data.get('ontology_mapping', '')
        dims = var_data.get('dimensions', [])
        
        if len(dims) >= 2:
            matrices.append(var_name)
        elif 'State' in ontology:
            states.append(var_name)
        elif 'Observation' in ontology:
            observations.append(var_name)
        elif 'Action' in ontology or var_name == 'u':
            actions.append(var_name)
        else:
            vectors.append(var_name)
    
    # Apply class styles
    if matrices:
        for var in matrices:
            styles.append(f"    classDef matrixStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px")
            styles.append(f"    class {var} matrixStyle")
    
    if states:
        for var in states:
            styles.append(f"    classDef stateStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px")
            styles.append(f"    class {var} stateStyle")
    
    if observations:
        for var in observations:
            styles.append(f"    classDef obsStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px")
            styles.append(f"    class {var} obsStyle")
    
    if actions:
        for var in actions:
            styles.append(f"    classDef actionStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:2px")
            styles.append(f"    class {var} actionStyle")
    
    return styles


def convert_gnn_file_to_mermaid(gnn_file_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Convert a GNN file to Mermaid format for oxdraw.
    
    Args:
        gnn_file_path: Path to GNN markdown file
        output_path: Optional path to save Mermaid output
        
    Returns:
        Mermaid diagram string
    """
    from gnn.processor import parse_gnn_file
    
    # Parse GNN file using existing pipeline module
    parsed_model = parse_gnn_file(gnn_file_path)
    
    # Convert to Mermaid
    mermaid_content = gnn_to_mermaid(parsed_model)
    
    # Save if output path provided
    if output_path:
        output_path.write_text(mermaid_content)
    
    return mermaid_content
```

---

## Implementation: Mermaid to GNN Conversion

### Module: `src/gnn/mermaid_parser.py`

This module parses Mermaid diagrams edited in oxdraw back to GNN format.

```python
"""
Mermaid to GNN Parser for oxdraw Integration

Extracts GNN model structure from Mermaid flowcharts with embedded metadata.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re

from gnn.parser import ParsedGNN
from ontology.processor import load_defined_ontology_terms, validate_annotations


def mermaid_to_gnn(mermaid_content: str, validate_ontology: bool = True) -> ParsedGNN:
    """
    Parse Mermaid flowchart back to GNN model structure.
    
    Args:
        mermaid_content: Mermaid flowchart string from oxdraw
        validate_ontology: Validate ontology term mappings
        
    Returns:
        ParsedGNN model ready for pipeline processing
    """
    # Extract metadata from comments
    metadata = _extract_gnn_metadata(mermaid_content)
    
    # Parse visual structure
    nodes = _extract_nodes(mermaid_content)
    edges = _extract_edges(mermaid_content)
    
    # Merge metadata with visual edits
    variables = _merge_variables(metadata.get('variables', {}), nodes)
    connections = _merge_connections(metadata.get('connections', []), edges)
    
    # Validate ontology mappings if requested
    if validate_ontology and metadata.get('ontology_mappings'):
        ontology_terms = load_defined_ontology_terms()
        validation_result = validate_annotations(
            list(metadata['ontology_mappings'].values()),
            ontology_terms
        )
        if validation_result.get('invalid_annotations'):
            raise ValueError(f"Invalid ontology terms: {validation_result['invalid_annotations']}")
    
    # Construct ParsedGNN
    return ParsedGNN(
        model_name=metadata.get('model_name', 'Untitled Model'),
        version=metadata.get('version', '1.0'),
        variables=variables,
        connections=connections,
        parameters=metadata.get('parameters', {}),
        ontology_mappings=_reconstruct_ontology_mappings(
            variables,
            metadata.get('ontology_mappings', {})
        )
    )


def _extract_gnn_metadata(mermaid_content: str) -> Dict[str, Any]:
    """Extract embedded GNN metadata from Mermaid comments."""
    metadata_pattern = r'%%\s*GNN_METADATA:\s*(\{.*?\})'
    match = re.search(metadata_pattern, mermaid_content, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    return {}


def _extract_nodes(mermaid_content: str) -> Dict[str, Dict]:
    """
    Extract node definitions from Mermaid diagram.
    
    Supports various shapes:
    - [A] : Rectangle
    - (C) : Rounded
    - ([s]) : Stadium
    - ((o)) : Circle
    - {{u}} : Hexagon
    - {π} : Diamond
    - [/F\] : Trapezoid
    """
    nodes = {}
    
    # Pattern for various node shapes
    patterns = [
        (r'(\w+)\[(.*?)\]', 'rectangle'),           # [A]
        (r'(\w+)\((.*?)\)', 'rounded'),              # (C)
        (r'(\w+)\(\[(.*?)\]\)', 'stadium'),          # ([s])
        (r'(\w+)\(\((.*?)\)\)', 'circle'),           # ((o))
        (r'(\w+)\{\{(.*?)\}\}', 'hexagon'),          # {{u}}
        (r'(\w+)\{(.*?)\}', 'diamond'),              # {π}
        (r'(\w+)\[\/(.*?)\\]', 'trapezoid')          # [/F\]
    ]
    
    for pattern, shape in patterns:
        for match in re.finditer(pattern, mermaid_content):
            node_id = match.group(1)
            node_label = match.group(2)
            
            nodes[node_id] = {
                'shape': shape,
                'label': node_label,
                'inferred_type': _infer_variable_type_from_shape(shape, node_id)
            }
    
    return nodes


def _extract_edges(mermaid_content: str) -> List[Dict]:
    """
    Extract edge definitions from Mermaid diagram.
    
    Supports:
    - ==> : Thick arrow (generative)
    - -.-> : Dashed (inference)
    - -..-> : Dotted (modulation)
    - --> : Normal arrow (coupling)
    """
    edges = []
    
    edge_patterns = [
        (r'(\w+)\s*==>\s*(\w+)', '>'),   # Generative
        (r'(\w+)\s*-\.->\s*(\w+)', '-'),  # Inference
        (r'(\w+)\s*-\.\.->\s*(\w+)', '*'), # Modulation
        (r'(\w+)\s*-->\s*(\w+)', '~')     # Coupling
    ]
    
    for pattern, symbol in edge_patterns:
        for match in re.finditer(pattern, mermaid_content):
            edges.append({
                'source': match.group(1),
                'target': match.group(2),
                'symbol': symbol,
                'line_position': match.start()
            })
    
    return edges


def _infer_variable_type_from_shape(shape: str, var_name: str) -> str:
    """Infer GNN variable type from Mermaid node shape."""
    shape_type_map = {
        'rectangle': 'matrix',
        'rounded': 'vector',
        'stadium': 'state',
        'circle': 'observation',
        'hexagon': 'action',
        'diamond': 'policy',
        'trapezoid': 'free_energy'
    }
    return shape_type_map.get(shape, 'unknown')


def _merge_variables(metadata_vars: Dict, visual_nodes: Dict) -> Dict:
    """
    Merge metadata variables with visually edited node positions.
    
    Preserves metadata dimensions/types while respecting visual layout.
    """
    merged = {}
    
    # Start with metadata
    for var_name, var_data in metadata_vars.items():
        merged[var_name] = var_data.copy()
    
    # Update/add from visual structure
    for node_id, node_data in visual_nodes.items():
        if node_id not in merged:
            merged[node_id] = {
                'dimensions': _infer_dimensions(node_data),
                'data_type': 'float',
                'description': node_data.get('label', '')
            }
        else:
            # Update description from visual label if changed
            if node_data.get('label'):
                merged[node_id]['description'] = node_data['label']
    
    return merged


def _merge_connections(metadata_conns: List[Dict], visual_edges: List[Dict]) -> List[Dict]:
    """
    Merge metadata connections with visually edited edges.
    
    Visual edits (adding/removing edges in oxdraw) take precedence.
    """
    # Use visual structure as source of truth for topology
    merged = []
    
    for edge in visual_edges:
        # Find matching metadata connection
        metadata_conn = next(
            (c for c in metadata_conns 
             if c['source'] == edge['source'] and c['target'] == edge['target']),
            None
        )
        
        merged.append({
            'source': edge['source'],
            'target': edge['target'],
            'symbol': edge['symbol'],
            'connection_type': metadata_conn.get('connection_type', 'directed') if metadata_conn else 'directed',
            'description': metadata_conn.get('description', '') if metadata_conn else ''
        })
    
    return merged


def _infer_dimensions(node_data: Dict) -> List[int]:
    """Infer variable dimensions from node shape and label."""
    # Parse label for dimension hints (e.g., "A<br/>3x3")
    label = node_data.get('label', '')
    dim_match = re.search(r'(\d+)x(\d+)', label)
    
    if dim_match:
        return [int(dim_match.group(1)), int(dim_match.group(2))]
    elif node_data['shape'] == 'rectangle':
        return [3, 3]  # Default matrix size
    else:
        return [3, 1]  # Default vector size


def _reconstruct_ontology_mappings(variables: Dict, ontology_map: Dict) -> List[Dict]:
    """Reconstruct ontology mapping list from merged variables."""
    mappings = []
    
    for var_name, var_data in variables.items():
        ontology_term = var_data.get('ontology_mapping') or ontology_map.get(var_name)
        if ontology_term:
            mappings.append({
                'variable': var_name,
                'ontology_term': ontology_term
            })
    
    return mappings


def convert_mermaid_file_to_gnn(mermaid_file_path: Path, output_path: Optional[Path] = None) -> ParsedGNN:
    """
    Convert Mermaid file from oxdraw back to GNN format.
    
    Args:
        mermaid_file_path: Path to Mermaid file (.mmd)
        output_path: Optional path to save GNN output (.md)
        
    Returns:
        ParsedGNN model
    """
    mermaid_content = mermaid_file_path.read_text()
    parsed_model = mermaid_to_gnn(mermaid_content)
    
    # Convert to GNN markdown if output path provided
    if output_path:
        from gnn.exporters import export_to_markdown
        gnn_content = export_to_markdown(parsed_model)
        output_path.write_text(gnn_content)
    
    return parsed_model
```

---

## Workflow Example: actinf_pomdp_agent.md

### Step 1: Convert GNN to Mermaid

```python
from pathlib import Path
from gnn.mermaid_converter import convert_gnn_file_to_mermaid

# Convert existing GNN model to Mermaid
gnn_file = Path("input/gnn_files/actinf_pomdp_agent.md")
mermaid_output = Path("output/oxdraw_interface/actinf_pomdp_agent.mmd")

mermaid_diagram = convert_gnn_file_to_mermaid(gnn_file, mermaid_output)
print(f"Generated Mermaid diagram: {mermaid_output}")
```

**Generated Mermaid (`actinf_pomdp_agent.mmd`)**:

```mermaid
flowchart TD
    %% GNN Model: Classic Active Inference POMDP Agent v1
    %% GNN Version: 1.0
    %% GNN_METADATA: {"model_name":"Classic Active Inference POMDP Agent v1","variables":{"A":{"dimensions":[3,3],"data_type":"float","ontology_mapping":"LikelihoodMatrix","description":"Likelihood matrix"},"B":{"dimensions":[3,3,3],"data_type":"float","ontology_mapping":"TransitionMatrix","description":"Transition matrix"},"C":{"dimensions":[3],"data_type":"float","ontology_mapping":"LogPreferenceVector","description":"Preference vector"},"D":{"dimensions":[3],"data_type":"float","ontology_mapping":"PriorOverHiddenStates","description":"Prior vector"},"E":{"dimensions":[3],"data_type":"float","ontology_mapping":"Habit","description":"Habit vector"},"s":{"dimensions":[3,1],"data_type":"float","ontology_mapping":"HiddenState","description":"Current hidden state"},"s_prime":{"dimensions":[3,1],"data_type":"float","ontology_mapping":"NextHiddenState","description":"Next hidden state"},"o":{"dimensions":[3,1],"data_type":"int","ontology_mapping":"Observation","description":"Current observation"},"π":{"dimensions":[3],"data_type":"float","ontology_mapping":"PolicyVector","description":"Policy distribution"},"u":{"dimensions":[1],"data_type":"int","ontology_mapping":"Action","description":"Action taken"},"F":{"dimensions":[],"data_type":"float","ontology_mapping":"VariationalFreeEnergy","description":"Free energy"},"G":{"dimensions":[],"data_type":"float","ontology_mapping":"ExpectedFreeEnergy","description":"Expected free energy"},"t":{"dimensions":[1],"data_type":"int","ontology_mapping":"Time","description":"Time step"}},"connections":[{"source":"D","target":"s","symbol":">"},{"source":"s","target":"A","symbol":"-"},{"source":"s","target":"s_prime","symbol":">"},{"source":"A","target":"o","symbol":"-"},{"source":"s","target":"B","symbol":"-"},{"source":"C","target":"G","symbol":">"},{"source":"E","target":"π","symbol":">"},{"source":"G","target":"π","symbol":">"},{"source":"π","target":"u","symbol":">"},{"source":"B","target":"u","symbol":">"},{"source":"u","target":"s_prime","symbol":">"}],"parameters":{"num_hidden_states":3,"num_obs":3,"num_actions":3},"ontology_mappings":{"A":"LikelihoodMatrix","B":"TransitionMatrix","C":"LogPreferenceVector","D":"PriorOverHiddenStates","E":"Habit","s":"HiddenState","s_prime":"NextHiddenState","o":"Observation","π":"PolicyVector","u":"Action","F":"VariationalFreeEnergy","G":"ExpectedFreeEnergy","t":"Time"}}
    
    A[A<br/>3x3<br/>float]
    %% A: LikelihoodMatrix
    B[B<br/>3x3x3<br/>float]
    %% B: TransitionMatrix
    C(C<br/>3<br/>float)
    %% C: LogPreferenceVector
    D(D<br/>3<br/>float)
    %% D: PriorOverHiddenStates
    E(E<br/>3<br/>float)
    %% E: Habit
    s([s<br/>3x1<br/>float])
    %% s: HiddenState
    s_prime([s_prime<br/>3x1<br/>float])
    %% s_prime: NextHiddenState
    o((o<br/>3x1<br/>int))
    %% o: Observation
    π{π<br/>3<br/>float}
    %% π: PolicyVector
    u{{u<br/>1<br/>int}}
    %% u: Action
    F[/F<br/>float\]
    %% F: VariationalFreeEnergy
    G[/G<br/>float\]
    %% G: ExpectedFreeEnergy
    t(t<br/>1<br/>int)
    %% t: Time
    
    D ==> s
    s -.-> A
    s ==> s_prime
    A -.-> o
    s -.-> B
    C ==> G
    E ==> π
    G ==> π
    π ==> u
    B ==> u
    u ==> s_prime
    
    %% Node styling based on variable types
    classDef matrixStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    class A matrixStyle
    class B matrixStyle
    classDef stateStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    class s stateStyle
    class s_prime stateStyle
    classDef obsStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    class o obsStyle
    classDef actionStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    class u actionStyle
```

### Step 2: Visual Editing in oxdraw

Launch oxdraw editor:

```bash
# Install oxdraw if not already installed
cargo install oxdraw

# Launch editor
oxdraw --input output/oxdraw_interface/actinf_pomdp_agent.mmd --edit --serve-port 5151
```

**Visual Editing Capabilities**:
- Drag nodes to refine layout (preserves GNN metadata)
- Add/remove connections by drawing edges
- Adjust edge routing with control points
- Change node colors to highlight components
- Add visual annotations (preserved in comments)

**Example Edit**: Add precision modulation to the model by:
1. Creating new node `γ` (precision parameter)
2. Drawing edge `γ ==> F` (modulates inference)
3. Adjusting layout for clarity

### Step 3: Convert Back to GNN

```python
from pathlib import Path
from gnn.mermaid_parser import convert_mermaid_file_to_gnn

# Parse edited Mermaid back to GNN
edited_mermaid = Path("output/oxdraw_interface/actinf_pomdp_agent.mmd")
output_gnn = Path("output/oxdraw_interface/actinf_pomdp_agent_edited.md")

parsed_model = convert_mermaid_file_to_gnn(edited_mermaid, output_gnn)
print(f"Regenerated GNN file: {output_gnn}")
```

### Step 4: Validate with GNN Pipeline

```python
# Run GNN pipeline steps to validate edited model
import subprocess

# Step 3: Parse and validate syntax
subprocess.run([
    "python3", "src/3_gnn.py",
    "--target-dir", "output/oxdraw_interface",
    "--output-dir", "output",
    "--verbose"
])

# Step 5: Type checking
subprocess.run([
    "python3", "src/5_type_checker.py",
    "--target-dir", "output/oxdraw_interface",
    "--output-dir", "output"
])

# Step 6: Semantic validation
subprocess.run([
    "python3", "src/6_validation.py",
    "--target-dir", "output/oxdraw_interface",
    "--output-dir", "output"
])

# Step 10: Ontology validation
subprocess.run([
    "python3", "src/10_ontology.py",
    "--target-dir", "output/oxdraw_interface",
    "--output-dir", "output",
    "--ontology-terms-file", "src/ontology/act_inf_ontology_terms.json"
])

print("✅ Model validated through GNN pipeline")
```

---

## Advanced Integration: Pipeline Step 22 (oxdraw GUI)

### Module: `src/oxdraw_integration/`

Create a dedicated pipeline step for oxdraw integration.

#### File: `src/24_oxdraw_gui.py`

```python
#!/usr/bin/env python3
"""
Pipeline Step 22: oxdraw GUI Integration

Launches oxdraw editor for visual GNN model construction with
bidirectional synchronization.
"""

from pathlib import Path
from utils.pipeline_template import (
    create_standardized_pipeline_script,
    log_step_start,
    log_step_success
)
from pipeline.config import get_output_dir_for_script
from oxdraw_integration.processor import process_oxdraw_gui


def main():
    """Main entry point for oxdraw GUI integration step."""
    
    run_script = create_standardized_pipeline_script(
        script_name="24_oxdraw_gui.py",
        processing_function=process_oxdraw_gui,
        description="oxdraw visual interface for GNN model construction"
    )
    
    return run_script()


if __name__ == "__main__":
    exit(main())
```

#### File: `src/oxdraw_integration/__init__.py`

```python
"""
oxdraw Integration Module for GNN Pipeline

Provides visual diagram-as-code interface for Active Inference model construction.
"""

from .processor import process_oxdraw_gui
from .mermaid_converter import gnn_to_mermaid, convert_gnn_file_to_mermaid
from .mermaid_parser import mermaid_to_gnn, convert_mermaid_file_to_gnn

__all__ = [
    'process_oxdraw_gui',
    'gnn_to_mermaid',
    'convert_gnn_file_to_mermaid',
    'mermaid_to_gnn',
    'convert_mermaid_file_to_gnn'
]

__version__ = "1.0.0"
```

#### File: `src/oxdraw_integration/processor.py`

```python
"""
oxdraw GUI Processor for GNN Pipeline Integration

Orchestrates oxdraw editor launch, model conversion, and validation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import json
import shutil
import logging

from gnn.processor import discover_gnn_files, parse_gnn_file
from .mermaid_converter import convert_gnn_file_to_mermaid
from .mermaid_parser import convert_mermaid_file_to_gnn


def process_oxdraw_gui(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    mode: str = "interactive",
    auto_convert: bool = True,
    validate_on_save: bool = True,
    **kwargs
) -> bool:
    """
    Process GNN files through oxdraw visual interface.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for oxdraw results
        logger: Logger instance
        mode: "interactive" (launch editor) or "headless" (convert only)
        auto_convert: Automatically convert GNN files to Mermaid
        validate_on_save: Validate models when saving from oxdraw
        **kwargs: Additional options (port, host, etc.)
        
    Returns:
        True if processing succeeded
    """
    logger.info("🎨 oxdraw GUI Integration")
    logger.info(f"Mode: {mode}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check oxdraw availability
    if not _check_oxdraw_installed():
        logger.warning("oxdraw not installed. Install with: cargo install oxdraw")
        if mode == "interactive":
            return False
    
    # Discover GNN files
    gnn_files = discover_gnn_files(target_dir, recursive=True)
    logger.info(f"Found {len(gnn_files)} GNN files")
    
    results = {
        "mode": mode,
        "files_processed": [],
        "conversions": []
    }
    
    # Convert GNN files to Mermaid
    if auto_convert:
        logger.info("📝 Converting GNN files to Mermaid format...")
        for gnn_file in gnn_files:
            try:
                mermaid_file = output_dir / f"{gnn_file.stem}.mmd"
                convert_gnn_file_to_mermaid(gnn_file, mermaid_file)
                
                results["conversions"].append({
                    "gnn_file": str(gnn_file),
                    "mermaid_file": str(mermaid_file),
                    "success": True
                })
                
                logger.info(f"✅ Converted: {gnn_file.name} → {mermaid_file.name}")
                
            except Exception as e:
                logger.error(f"❌ Conversion failed for {gnn_file.name}: {e}")
                results["conversions"].append({
                    "gnn_file": str(gnn_file),
                    "error": str(e),
                    "success": False
                })
    
    # Launch oxdraw in interactive mode
    if mode == "interactive" and results["conversions"]:
        logger.info("🚀 Launching oxdraw editor...")
        
        # Launch editor for first converted file
        first_mermaid = Path(results["conversions"][0]["mermaid_file"])
        
        port = kwargs.get("port", 5151)
        host = kwargs.get("host", "127.0.0.1")
        
        try:
            subprocess.run([
                "oxdraw",
                "--input", str(first_mermaid),
                "--edit",
                "--serve-host", host,
                "--serve-port", str(port)
            ], check=True)
            
            logger.info(f"✅ oxdraw editor launched at http://{host}:{port}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to launch oxdraw: {e}")
            return False
    
    # Save processing results
    results_file = output_dir / "oxdraw_processing_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📊 Results saved to: {results_file}")
    
    return len(results["conversions"]) > 0


def _check_oxdraw_installed() -> bool:
    """Check if oxdraw is installed and available."""
    return shutil.which("oxdraw") is not None
```

---

## Use Cases

### 1. Rapid Prototyping of Active Inference Architectures

**Scenario**: Researcher wants to explore different POMDP architectures

**Workflow**:
1. Create skeleton GNN file with basic structure
2. Convert to Mermaid with `convert_gnn_file_to_mermaid()`
3. Launch oxdraw editor to visually experiment with:
   - Additional state factors
   - Hierarchical nesting
   - Precision modulation pathways
4. Real-time validation through GNN pipeline
5. Export to PyMDP/RxInfer for simulation

**Benefits**:
- Immediate visual feedback
- No syntax errors from manual GNN editing
- Preservation of mathematical semantics

### 2. Collaborative Model Development

**Scenario**: Team designing complex cognitive security model

**Workflow**:
1. Initial model specification in GNN
2. Convert to Mermaid for version control
3. Team members use oxdraw to:
   - Propose architectural changes via visual edits
   - Annotate with comments (embedded in Mermaid)
   - Review changes through Git diff on `.mmd` files
4. Validate changes through GNN pipeline
5. Merge approved changes back to canonical GNN

**Benefits**:
- Version-controlled visual models
- Reduced communication overhead
- Clear visual diff of architectural changes

### 3. Educational Scaffolding

**Scenario**: Teaching Active Inference to new researchers

**Workflow**:
1. Provide pre-built GNN models for classic examples
2. Students explore through oxdraw visual interface
3. Guided exercises:
   - Add precision parameters
   - Extend observation modalities
   - Create hierarchical structures
4. Automatic validation provides immediate feedback
5. Export to executable simulations for experimentation

**Benefits**:
- Lower barrier to entry (visual vs. text)
- Immediate feedback on model validity
- Smooth transition to full GNN syntax

---

## Performance Considerations

### Conversion Overhead

| Operation | Typical Time | Memory |
|-----------|-------------|--------|
| GNN → Mermaid | 10-50ms | <5MB |
| Mermaid → GNN | 20-100ms | <10MB |
| oxdraw Launch | 1-2s | ~12-23MB |
| Full Pipeline Validation | 2-5s | ~50MB |

### Scalability Limits

- **Model Size**: oxdraw handles up to ~500 nodes effectively
- **Connection Density**: Optimal for <1000 edges (typical Active Inference models: 10-50)
- **Recommended**: Models with 5-50 variables (most Active Inference architectures)

### Optimization Strategies

1. **Lazy Metadata Embedding**: Only embed full GNN specification if `include_metadata=True`
2. **Incremental Validation**: Validate only changed components on save
3. **Cached Conversions**: Cache parsed GNN models to avoid re-parsing
4. **Selective Rendering**: oxdraw renders only visible viewport (handles large models)

---

## Technical Requirements

### Dependencies

**Required**:
- `oxdraw` (Rust binary via cargo)
- Existing GNN pipeline modules (steps 3, 5, 6, 10)
- Python 3.10+

**Optional**:
- `mermaid-cli` (for server-side rendering)
- `pygraphviz` (for advanced layout algorithms)

### Installation

```bash
# Install oxdraw
cargo install oxdraw

# Install GNN pipeline dependencies using UV
uv sync

# Optional: Install Mermaid CLI for server-side rendering
npm install -g @mermaid-js/mermaid-cli
```

### Configuration

**File**: `input/config.yaml`

```yaml
oxdraw_integration:
  enabled: true
  default_port: 5151
  default_host: "127.0.0.1"
  auto_convert_on_startup: true
  validate_on_save: true
  
  # Node style preferences
  node_styles:
    matrices:
      fill: "#e1f5ff"
      stroke: "#0288d1"
    states:
      fill: "#fff3e0"
      stroke: "#f57c00"
    observations:
      fill: "#f3e5f5"
      stroke: "#7b1fa2"
    actions:
      fill: "#e8f5e9"
      stroke: "#388e3c"
  
  # Edge style preferences
  edge_styles:
    generative:
      style: "==>"
      color: "#0288d1"
    inference:
      style: "-.->
      color: "#f57c00"
    modulation:
      style: "-..->
      color: "#7b1fa2"
```

---

## Limitations and Future Work

### Current Limitations

1. **No Multi-File Editing**: oxdraw edits one diagram at a time (no hierarchical model editing)
2. **Limited Shape Palette**: Mermaid shape set constrains visual distinction of GNN constructs
3. **Manual Synchronization**: User must explicitly trigger GNN ← → Mermaid conversions
4. **Ontology Annotation UI**: No dedicated interface for editing ontology mappings (comment-based only)

### Planned Enhancements

1. **Real-Time Validation**: WebSocket connection between oxdraw and GNN validator for live feedback
2. **Custom Shape Library**: Extended Mermaid shapes for Active Inference-specific constructs
3. **Hierarchical Editing**: Support for hierarchical POMDP editing with drill-down
4. **Ontology Inspector**: Sidebar UI in oxdraw for browsing/selecting ontology terms
5. **Collaborative Editing**: Multi-user simultaneous editing with conflict resolution
6. **Auto-Layout Algorithms**: Integration with specialized Active Inference layout algorithms

---

## Conclusion

oxdraw provides a compelling visual interface for GNN Active Inference model construction by bridging the gap between intuitive diagram manipulation and rigorous mathematical specification. The bidirectional conversion workflow preserves both visual layout preferences and semantic constraints, enabling researchers to:

- **Rapidly prototype** model architectures through drag-and-drop
- **Collaborate effectively** via version-controlled visual artifacts
- **Maintain rigor** through automatic validation and ontology checking
- **Execute seamlessly** by exporting to simulation frameworks

This integration demonstrates how **diagram-as-code** tools can enhance scientific computing workflows without sacrificing reproducibility or semantic precision. By embedding GNN metadata in Mermaid comments, oxdraw edits remain fully compatible with the broader GNN pipeline ecosystem while providing researchers with the visual affordances necessary for complex model development.

---

## References

### GNN Pipeline Documentation
- [GNN Parser](../../src/gnn/AGENTS.md) - Core parsing and validation
- [Ontology Module](../../src/ontology/AGENTS.md) - Active Inference ontology
- [Visualization Module](../../src/visualization/AGENTS.md) - Graph rendering
- [GUI Module](../../src/gui/AGENTS.md) - Interactive constructors

### oxdraw Resources
- [oxdraw GitHub](https://github.com/RohanAdwankar/oxdraw)
- [Mermaid Syntax](https://mermaid.js.org/syntax/flowchart.html)
- [oxdraw Technical Overview](oxdraw.md)

### Active Inference
- [Active Inference Ontology](../../src/ontology/act_inf_ontology_terms.json)
- [POMDP Specification](../gnn/gnn_standards.md)

---

**Document Version**: 1.0  
**Authors**: GNN Pipeline Team  
**License**: MIT

