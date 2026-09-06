# oxdraw: Comprehensive Technical Overview and Setup Documentation

**oxdraw** is a sophisticated diagram-as-code tool that bridges the gap between declarative syntax-based diagramming (like Mermaid) and direct-manipulation visual editors (like Lucidchart) through a hybrid Rust CLI and React web interface architecture. The project represents an innovative approach to maintaining high-quality technical diagrams with both reproducibility and visual refinement capabilities.[1][2]

## Project Architecture and Technical Foundation

### Core Technology Stack

oxdraw employs a **dual-layer architecture** consisting of a Rust-based CLI backend and a React-based frontend, with the following composition:[2][1]

- **Rust (56.1%)**: Backend CLI, rendering engine, and SVG/PNG compilation
- **TypeScript (39.5%)**: Frontend interactive editor interface
- **CSS (3.4%)**: Styling and visual presentation
- **Other (1.0%)**: Build configurations and assets

The codebase comprises approximately **4,000 source lines of code (SLoC)** with a binary size of **3MB** and includes embedded WOFF font assets for consistent typography.[2]

### Architectural Design Philosophy

The system operates on a **bidirectional synchronization model** where visual manipulations in the web interface are persisted back to the source `.mmd` file as **declarative metadata stored in comments**. This design ensures that diagrams remain fully compatible with standard Mermaid parsers while preserving oxdraw-specific customizations such as node positions, edge routing, and styling overrides.[1][2]

## Installation and Setup Requirements

### Method 1: Install from Cargo (Recommended for End Users)

The simplest installation method leverages Rust's package manager:[1]

```bash
cargo install oxdraw
```

**Prerequisites:**
- Rust toolchain (Cargo) installed on the system
- Sufficient system resources (~12-23MB runtime memory footprint)[2]

### Method 2: Build from Source (Development Setup)

For development or customization purposes:[2]

```bash
# Clone the repository
git clone https://github.com/RohanAdwankar/oxdraw.git
cd oxdraw

# Build the CLI in release mode
cargo build --release

# The compiled binary will be located at:
# ./target/release/oxdraw
```

### Frontend Development Setup

For contributors working on the React-based web interface:[3]

```bash
# Install Node.js dependencies
npm install

# Build the frontend assets
npm run build
```

**Requirements:**
- Node.js >= 14.0[4]
- npm or Yarn package manager
- Compatible with modern web browsers supporting ES6+

### System Dependencies

**For Windows users**, the Visual C++ Build Tools are required for Rust compilation:[5]
1. Download Visual Studio Build Tools
2. Install the "C++ Build Tools" workload (~1.63 GB)
3. Proceed with Rust installation using `rustup-init.exe`

## CLI Usage Patterns and Features

### Basic Operations

#### Rendering Static Diagrams

Generate SVG or PNG output from Mermaid source files:[1]

```bash
# Default SVG output
oxdraw --input flow.mmd

# Explicit PNG output with custom scaling
oxdraw --input flow.mmd --png --scale 15.0

# Custom output path
oxdraw --input diagram.mmd --output /path/to/output.svg

# Stream SVG to stdout
oxdraw --input diagram.mmd --output -
```

#### Interactive Editor Mode

Launch the web-based editor for visual manipulation:[1]

```bash
# Launch editor on default port 5151
oxdraw --input flow.mmd --edit

# Custom host and port binding
oxdraw --input diagram.mmd --edit --serve-host 0.0.0.0 --serve-port 8080
```

### Complete CLI Flag Reference

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input <PATH>` | Mermaid source file path; `-` for stdin | Required |
| `-o, --output <PATH>` | Output destination; `-` for stdout | `<input>.svg` |
| `-e, --output-format <svg\|png>` | Explicit format specification | `svg` |
| `--png` | Shorthand for `--output-format png` | - |
| `--scale <FACTOR>` | PNG rasterization multiplier (>0.0) | `10.0` |
| `--edit` | Launch interactive editor | - |
| `--serve-host <ADDR>` | Editor HTTP bind address | `127.0.0.1` |
| `--serve-port <PORT>` | Editor HTTP port | `5151` |
| `-b, --background-color <COLOR>` | SVG background fill color | Transparent |
| `-q, --quiet` | Suppress informational output | - |

[2][1]

## Web Interface Features and Interaction Model

### Node Manipulation Capabilities

The interactive editor provides **granular control over diagram aesthetics**:[1]

**Positioning and Layout:**
- **Drag-and-drop positioning** with automatic grid snapping
- **Live alignment guides** during node movement for visual consistency
- **Keyboard nudging** using Shift+Arrow keys for precise grid-based adjustments
- **Subgraph container dragging** to move entire groups while preserving internal relationships

**Visual Styling:**
- Per-node color overrides for **fill, stroke, and text** via color pickers
- **Double-click to clear** individual node styling overrides
- Dedicated "Reset node style" button to remove all customizations
- Node deletion via Delete/Backspace keys or right-click context menu

### Edge Customization System

**Routing and Path Control:**
- **Draggable edge handles** for manual path refinement
- **Control point insertion** via "Add control point" button or double-clicking edges
- **Handle removal** by double-clicking existing control points
- **Label handle dragging** for positioning edge annotations
- **Double-click edge clearing** to revert to automatic routing

**Styling Options:**
- Edge color picker for stroke customization
- Line style toggle between **solid and dashed** strokes
- Arrow direction selector supporting **forward/backward/bidirectional/none** configurations
- "Reset edge style" function to restore default appearance

[2][1]

### Editor Interface Components

**Canvas Interactions:**
- Grid-based snapping system for consistent spacing
- Live alignment feedback during drag operations
- Right-click context menus for node/edge operations
- Selection persistence across editing sessions

**Source Panel:**
- Real-time Mermaid code mirroring with **syntax highlighting**
- **Auto-save functionality** after idle periods (implementation-dependent)
- Pending/saving/error state indicators
- Current selection context display

**Toolbar Status:**
- File path of currently edited diagram
- Loading/saving operation indicators
- Global actions including "Delete selected" and reset functions

[1][2]

## Path Algorithm and Routing Strategy

The edge routing algorithm represents a **pragmatic balance between aesthetic preferences**:[1]

**Design Considerations:**
- **Strong edges preferred**: Uses orthogonal routing with clear angular transitions rather than smooth curves to enhance diagram clarity
- **Selective overlap tolerance**: Allows edge crossings when necessary to prevent excessive path elongation that would compromise diagram compactness
- **Dynamic recomputation**: Automatically recalculates paths when nodes are repositioned via keyboard or mouse manipulation
- **Iterative refinement**: Acknowledged as an evolving component with planned improvements to handle complex topologies

This approach distinguishes oxdraw from pure auto-layout systems that prioritize mathematical optimality over practical readability in technical diagrams.[1]

## Mermaid Syntax Compatibility

oxdraw fully supports **standard Mermaid syntax** for diagram definitions, including:[6][1]

**Supported Diagram Types:**
- Flowcharts (`flowchart`, `graph`)
- Sequence diagrams
- State diagrams
- Class diagrams
- Entity-relationship diagrams
- Gantt charts
- Pie charts
- Git graphs

**Shape Notation Examples**:[6]
- `[Rectangle]` - Standard rectangular nodes
- `(Rounded Rectangle)` - Nodes with rounded corners
- `((Circle))` - Circular nodes
- `{Diamond}` - Decision node diamonds

**Connector Styles**:[6]
- `A-->B` - Directional arrow
- `A-.->B` - Dotted arrow
- `A---B` - Line without arrow
- `A-->|label|B` - Labeled connection

**Metadata Persistence:**
All oxdraw-specific customizations (positions, colors, paths) are **stored as comments** within the `.mmd` file, ensuring that diagrams remain parseable by other Mermaid tools without degradation.[2][1]

## Development Context and Use Cases

### Original Motivation

The project emerged from the creator's workflow challenges when using AI-generated Mermaid diagrams for **architecture visualization and codebase comprehension**. The typical pattern involved:[1]

1. Generating `.mmd` files programmatically (via AI tools or scripts)
2. Encountering limitations with automatic layout quality
3. Migrating to visual editors like Lucidchart for refinement
4. Losing the benefits of version control and reproducibility

oxdraw solves this by enabling **iterative refinement without abandoning code-first workflows**.[1]

### Target User Profiles

**Software Architects**: Creating and maintaining system architecture diagrams with version control integration[7]

**Researchers**: Documenting complex systems (like Active Inference frameworks) with precise visual communication requirements

**Development Teams**: Generating diagrams from codebases that require periodic updates and collaborative refinement

**Technical Writers**: Producing documentation diagrams that balance aesthetic quality with maintainability

## Project Status and Community

### Current Release Status

- **Latest version**: 0.1.0 (released October 25, 2025)[2]
- **Stability**: Unstable/preview release (1 unstable release)
- **License**: MIT License (permissive open-source)[1]
- **Language Edition**: Rust 2024 (leveraging latest language features)[2]

### Community Engagement

- **GitHub Stars**: 944 (indicating strong community interest)[1]
- **Forks**: 22 (active derivative development)[1]
- **Contributors**: 3 core developers (Rohan Adwankar, Hong Jiarong, Jorgen Cani)[1]
- **Downloads**: 88 downloads per month via crates.io[2]
- **Visibility**: Featured on Hacker News (Show HN) with community discussion[3]

### Known Limitations

- **No stable releases**: Project is in active early development with API/format changes expected
- **Algorithm refinement ongoing**: Path routing behavior continues to evolve[1]
- **Limited diagram type testing**: Primary focus appears to be flowcharts and architecture diagrams
- **Documentation gaps**: No formal API documentation or comprehensive guides beyond README

## Integration Considerations for Cognitive Security Research

For your **Active Inference and Cognitive Security research workflows**, oxdraw offers particular advantages:

**Version Control Integration**: All diagram modifications remain in Git-trackable text files, enabling collaborative research documentation with full change history.

**AI-Assisted Diagram Generation**: The Mermaid compatibility enables LLM-generated diagram scaffolding (e.g., representing belief propagation networks or threat model architectures) with subsequent manual refinement.

**Reproducibility**: Diagrams can be regenerated from source without requiring proprietary software licenses, critical for academic publication and replication.

**Workflow Automation**: The CLI interface supports scripted diagram generation pipelines for systematic documentation of experimental setups or model architectures.

**Cognitive Load Management**: The hybrid approach reduces cognitive switching costs between code-based definition and visual refinement, particularly valuable for complex systems like Active Inference hierarchies.

## Advanced Configuration and Customization

### Background Color Customization

For presentation or publication requirements:[1]

```bash
# Apply background color to SVG output
oxdraw --input diagram.mmd --background-color "#FFFFFF"

# In editor mode (applies to preview)
oxdraw --input diagram.mmd --edit --background-color "#F5F5F5"
```

### Stdin/Stdout Processing

For pipeline integration:[1]

```bash
# Generate Mermaid code and render in one pipeline
echo "graph TD; A-->B; B-->C;" | oxdraw --input - --output -

# Process multiple diagrams with scripts
cat diagram_list.txt | xargs -I {} oxdraw --input {}
```

### Network Accessibility

For remote editing or team collaboration:[1]

```bash
# Bind to all network interfaces
oxdraw --input team_diagram.mmd --edit --serve-host 0.0.0.0

# Access from other machines at http://<server-ip>:5151
```

## Comparison with Alternative Tools

**vs. Standard Mermaid**: oxdraw adds persistent visual refinement and manual layout control while maintaining full syntax compatibility.[8][1]

**vs. Lucidchart/draw.io**: oxdraw provides version-controllable text-based source with visual editing, whereas traditional tools use proprietary binary formats.[6][1]

**vs. PlantUML**: Similar code-first philosophy but oxdraw emphasizes interactive post-generation editing rather than pure declarative control.[1]

**vs. D2/Graphviz**: oxdraw uses the more accessible Mermaid syntax and provides GUI refinement unavailable in pure CLI tools.[1]

This technical overview synthesizes installation procedures, architectural design, feature specifications, and usage patterns to provide a complete reference for adopting oxdraw in professional research and development environments.[3][2][1]

[1](https://github.com/RohanAdwankar/oxdraw)
[2](https://lib.rs/crates/oxdraw)
[3](https://news.ycombinator.com/item?id=45706792)
[4](https://create-react-app.dev/docs/getting-started/)
[5](https://www.youtube.com/watch?v=enk0o7eWNsc)
[6](https://www.drawio.com/blog/mermaid-diagrams)
[7](https://indiebase.io/tools/oxdraw)
[8](https://docs.mermaidchart.com/mermaid-oss/intro/getting-started.html)
[9](https://github.com/RohanAdwankar)
[10](https://www.reddit.com/r/rust/comments/1ofo1li/media_oxdraw_cli_for_declarative_diagram/)
[11](https://www.linkedin.com/in/rohanadwankar)
[12](https://mermaid.js.org/config/configuration.html)
[13](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)
[14](https://doc.rust-lang.org/cargo/guide/dependencies.html)
[15](https://users.rust-lang.org/t/psa-please-specify-precise-dependency-versions-in-cargo-toml/71277)
[16](https://www.youtube.com/watch?v=9CcDy2HCNnU)
[17](https://dev.to/rijultp/getting-started-with-dependency-management-in-rust-using-cargotoml-54oo)
[18](https://www.reddit.com/r/rust/comments/skckkl/psa_please_specify_precise_dependency_versions_in/)
[19](https://create-react-app.dev/docs/deployment/)
[20](https://community.lucid.co/inspiration-5/what-syntax-can-i-use-to-diagram-as-code-with-mermaid-9665)