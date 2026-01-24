# GNN Comprehensive API Reference

> **📋 Document Metadata**  
> **Type**: API Reference | **Audience**: Developers & Integrators | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [Pipeline Architecture](../gnn/gnn_tools.md) | [Framework Integration](../gnn/framework_integration_guide.md)

This comprehensive reference documents all programmatic interfaces for integrating with the GeneralizedNotationNotation (GNN) system.

## 🎯 API Overview

### **📚 API Categories**

1. **🔍 Core Parsing API**: GNN file parsing and validation
2. **⚙️ Pipeline API**: Pipeline execution and orchestration  
3. **🔧 Framework Integration API**: PyMDP, RxInfer, DisCoPy interfaces
4. **📊 Visualization API**: Programmatic visualization generation
5. **🤖 LLM Integration API**: AI-enhanced model analysis
6. **📡 MCP API**: Model Context Protocol integration
7. **⚡ Performance API**: Monitoring and optimization interfaces

### **🚀 Quick Start Example**

```python
# Complete GNN API usage example
from gnn import GNNParser, Pipeline, Visualizer

# Parse GNN model
parser = GNNParser()
model = parser.parse_file("examples/navigation_agent.md")

# Execute pipeline
pipeline = Pipeline(config="production")
results = pipeline.process(model, steps=[1, 4, 6, 9])

# Generate visualizations
viz = Visualizer()
diagrams = viz.create_all_visualizations(model, results)

print(f"✅ Model processed: {len(diagrams)} visualizations generated")
```

## 🔍 Core Parsing API

### **📄 GNNParser Class**

The primary interface for parsing GNN files into structured model objects.

```python
from gnn.parsing import GNNParser, ParseConfig, ValidationLevel

class GNNParser:
    """High-level GNN file parser with validation and error handling."""
    
    def __init__(self, 
                 config: ParseConfig = None,
                 validation_level: ValidationLevel = ValidationLevel.STRICT):
        """
        Initialize GNN parser.
        
        Args:
            config: Parsing configuration options
            validation_level: STRICT, NORMAL, PERMISSIVE, or DISABLED
        """
    
    def parse_file(self, filepath: str) -> GNNModel:
        """
        Parse a single GNN file.
        
        Args:
            filepath: Path to GNN markdown file
            
        Returns:
            GNNModel: Parsed and validated model object
            
        Raises:
            ParseError: If file cannot be parsed
            ValidationError: If model validation fails
        """
    
    def parse_directory(self, dirpath: str, 
                       pattern: str = "*.md",
                       recursive: bool = True) -> List[GNNModel]:
        """
        Parse all GNN files in a directory.
        
        Args:
            dirpath: Directory path to scan
            pattern: File pattern to match
            recursive: Whether to scan subdirectories
            
        Returns:
            List[GNNModel]: List of parsed models
        """
    
    def parse_string(self, content: str, 
                    model_name: str = "unnamed") -> GNNModel:
        """
        Parse GNN content from string.
        
        Args:
            content: GNN model content
            model_name: Name for the model
            
        Returns:
            GNNModel: Parsed model object
        """
    
    def validate_model(self, model: GNNModel) -> ValidationResult:
        """
        Validate a parsed model.
        
        Args:
            model: GNN model to validate
            
        Returns:
            ValidationResult: Validation report with errors/warnings
        """

# Usage examples:
parser = GNNParser(validation_level=ValidationLevel.STRICT)

# Parse single file
model = parser.parse_file("path/to/model.md")

# Parse directory  
models = parser.parse_directory("examples/", recursive=True)

# Parse from string
gnn_content = """
## ModelName
TestModel

## StateSpaceBlock
s_f0[2,1,type=categorical]
o_m0[2,1,type=categorical]
"""
model = parser.parse_string(gnn_content, "TestModel")

# Validate model
validation = parser.validate_model(model)
if validation.is_valid:
    print("✅ Model is valid")
else:
    print(f"❌ Validation errors: {validation.errors}")
```

### **📊 GNNModel Class**

Structured representation of parsed GNN models.

```python
from gnn.model import GNNModel, StateSpace, Connections, Parameters

class GNNModel:
    """Structured representation of a GNN model."""
    
    def __init__(self, name: str):
        self.name = name
        self.annotation = ""
        self.state_space = StateSpace()
        self.connections = Connections()
        self.parameters = Parameters()
        self.equations = []
        self.time_config = {}
        self.ontology_annotations = {}
        self.metadata = {}
    
    @property
    def complexity_score(self) -> float:
        """Compute model complexity score."""
        return self._calculate_complexity()
    
    @property
    def state_dimensions(self) -> Dict[str, Tuple[int, ...]]:
        """Get dimensions of all state variables."""
        return {var.name: var.dimensions 
                for var in self.state_space.variables 
                if var.name.startswith('s_')}
    
    @property
    def observation_dimensions(self) -> Dict[str, Tuple[int, ...]]:
        """Get dimensions of all observation variables."""
        return {var.name: var.dimensions 
                for var in self.state_space.variables 
                if var.name.startswith('o_')}
    
    def get_matrix(self, matrix_name: str) -> np.ndarray:
        """Get parameter matrix by name."""
        return self.parameters.get(matrix_name)
    
    def set_matrix(self, matrix_name: str, matrix: np.ndarray):
        """Set parameter matrix."""
        self.parameters.set(matrix_name, matrix)
    
    def to_dict(self) -> dict:
        """Convert model to dictionary representation."""
        return {
            'name': self.name,
            'annotation': self.annotation,
            'state_space': self.state_space.to_dict(),
            'connections': self.connections.to_dict(),
            'parameters': self.parameters.to_dict(),
            'equations': self.equations,
            'time_config': self.time_config,
            'ontology_annotations': self.ontology_annotations,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GNNModel':
        """Create model from dictionary representation."""
        model = cls(data['name'])
        model.annotation = data.get('annotation', '')
        model.state_space = StateSpace.from_dict(data['state_space'])
        model.connections = Connections.from_dict(data['connections'])
        model.parameters = Parameters.from_dict(data['parameters'])
        model.equations = data.get('equations', [])
        model.time_config = data.get('time_config', {})
        model.ontology_annotations = data.get('ontology_annotations', {})
        model.metadata = data.get('metadata', {})
        return model

# Usage examples:
model = parser.parse_file("navigation_agent.md")

# Access model properties
print(f"Model: {model.name}")
print(f"Complexity: {model.complexity_score}")
print(f"States: {model.state_dimensions}")
print(f"Observations: {model.observation_dimensions}")

# Access matrices
A_matrix = model.get_matrix('A_m0')
print(f"A matrix shape: {A_matrix.shape}")

# Modify matrices
model.set_matrix('C_m0', np.array([0.0, 1.0]))

# Serialize/deserialize
model_dict = model.to_dict()
model_restored = GNNModel.from_dict(model_dict)
```

## ⚙️ Pipeline API

### **🚀 Pipeline Class**

High-level pipeline execution interface.

```python
from gnn.pipeline import Pipeline, PipelineConfig, StepResult

class Pipeline:
    """GNN processing pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration options
        """
    
    def process(self, 
                model: GNNModel,
                steps: List[int] = None,
                output_dir: str = None,
                **kwargs) -> PipelineResult:
        """
        Process model through pipeline steps.
        
        Args:
            model: GNN model to process
            steps: List of step numbers to execute (default: all)
            output_dir: Output directory for results
            **kwargs: Additional step-specific parameters
            
        Returns:
            PipelineResult: Results from all executed steps
        """
    
    def process_batch(self,
                     models: List[GNNModel],
                     steps: List[int] = None,
                     parallel: bool = True,
                     max_workers: int = None) -> List[PipelineResult]:
        """
        Process multiple models in batch.
        
        Args:
            models: List of models to process
            steps: Steps to execute
            parallel: Whether to use parallel processing
            max_workers: Maximum number of worker processes
            
        Returns:
            List[PipelineResult]: Results for each model
        """
    
    def get_step_info(self, step_number: int) -> StepInfo:
        """Get information about a specific pipeline step."""
        
    def list_available_steps(self) -> List[StepInfo]:
        """List all available pipeline steps."""

# Usage examples:
from gnn.pipeline import Pipeline, PipelineConfig

# Create pipeline with custom configuration
config = PipelineConfig(
    parallel=True,
    max_workers=4,
    cache_enabled=True,
    output_format=['json', 'xml'],
    verbose=True
)
pipeline = Pipeline(config)

# Process single model
model = parser.parse_file("agent.md")
result = pipeline.process(model, steps=[1, 4, 6, 9])

# Process multiple models in parallel
models = parser.parse_directory("examples/")
results = pipeline.process_batch(models, parallel=True)

# Check results
for result in results:
    if result.success:
        print(f"✅ {result.model_name}: {result.execution_time:.2f}s")
    else:
        print(f"❌ {result.model_name}: {result.error}")
```

### **📊 Individual Step APIs**

Direct access to individual pipeline steps.

```python
# Step 1: GNN Parsing
from gnn.steps import GNNParsingStep

parser_step = GNNParsingStep()
parse_result = parser_step.execute("model.md")

# Step 4: Type Checking  
from gnn.steps import TypeCheckingStep

checker = TypeCheckingStep(strict_mode=True)
check_result = checker.execute(model)

# Step 6: Visualization
from gnn.steps import VisualizationStep

visualizer = VisualizationStep(output_format='png')
viz_result = visualizer.execute(model, output_dir="./output/")

# Step 9: Code Rendering
from gnn.steps import RenderingStep

renderer = RenderingStep(target_framework='pymdp')
render_result = renderer.execute(model, template_dir="./templates/")
```

## 🔧 Framework Integration API

### **🐍 PyMDP Integration**

```python
from gnn.frameworks.pymdp import PyMDPConverter, PyMDPAgent

class PyMDPConverter:
    """Convert GNN models to PyMDP agents."""
    
    def __init__(self, optimization_level: str = 'standard'):
        """
        Initialize PyMDP converter.
        
        Args:
            optimization_level: 'minimal', 'standard', 'aggressive'
        """
    
    def convert_model(self, model: GNNModel) -> PyMDPAgent:
        """
        Convert GNN model to PyMDP agent.
        
        Args:
            model: GNN model to convert
            
        Returns:
            PyMDPAgent: Configured PyMDP agent
        """
    
    def generate_code(self, model: GNNModel, 
                     output_file: str = None) -> str:
        """
        Generate PyMDP Python code.
        
        Args:
            model: GNN model to convert
            output_file: Optional file to write code
            
        Returns:
            str: Generated Python code
        """

# Usage example:
converter = PyMDPConverter(optimization_level='aggressive')
agent = converter.convert_model(model)

# Generate executable code
code = converter.generate_code(model, "generated_agent.py")

# Run simulation
observations = [0, 1, 0, 1, 1]
for obs in observations:
    beliefs = agent.infer_states([obs])
    action = agent.sample_action()
    print(f"Obs: {obs}, Belief: {beliefs[0]}, Action: {action[0]}")
```

### **🔢 RxInfer.jl Integration**

```python
from gnn.frameworks.rxinfer import RxInferConverter, JuliaRunner

class RxInferConverter:
    """Convert GNN models to RxInfer.jl code."""
    
    def convert_model(self, model: GNNModel) -> str:
        """
        Convert to RxInfer.jl model code.
        
        Args:
            model: GNN model to convert
            
        Returns:
            str: Julia/RxInfer code
        """
    
    def generate_inference_script(self, 
                                 model: GNNModel,
                                 data_file: str = None) -> str:
        """Generate complete inference script."""

# Usage example:
converter = RxInferConverter()
julia_code = converter.convert_model(model)

# Execute Julia code
runner = JuliaRunner()
results = runner.execute_inference(julia_code, data="observations.csv")
```

### **🎨 DisCoPy Integration**

```python
from gnn.frameworks.discopy import DisCoPyConverter, CategoryDiagram

class DisCoPyConverter:
    """Convert GNN models to categorical diagrams."""
    
    def convert_model(self, model: GNNModel) -> CategoryDiagram:
        """Convert to DisCoPy categorical diagram."""
    
    def evaluate_with_jax(self, 
                         diagram: CategoryDiagram,
                         backend: str = 'jax') -> np.ndarray:
        """Evaluate diagram with JAX backend."""

# Usage example:
converter = DisCoPyConverter()
diagram = converter.convert_model(model)
result = converter.evaluate_with_jax(diagram)
```

## 📊 Visualization API

### **🎨 Visualizer Class**

```python
from gnn.visualization import Visualizer, VisualizationConfig

class Visualizer:
    """GNN model visualization generator."""
    
    def __init__(self, config: VisualizationConfig = None):
        """Initialize visualizer with configuration."""
    
    def create_network_diagram(self, 
                              model: GNNModel,
                              layout: str = 'spring',
                              output_file: str = None) -> Figure:
        """
        Create network diagram of model structure.
        
        Args:
            model: GNN model to visualize
            layout: 'spring', 'circular', 'hierarchical', 'force'
            output_file: Optional file to save diagram
            
        Returns:
            Figure: Matplotlib figure object
        """
    
    def create_matrix_heatmaps(self, 
                              model: GNNModel,
                              matrices: List[str] = None) -> Dict[str, Figure]:
        """Create heatmap visualizations of parameter matrices."""
    
    def create_belief_landscape(self, 
                               model: GNNModel,
                               beliefs: np.ndarray) -> Figure:
        """Create 3D visualization of belief landscapes."""
    
    def create_interactive_explorer(self, 
                                   model: GNNModel) -> str:
        """Create interactive HTML model explorer."""

# Usage examples:
viz = Visualizer()

# Create network diagram
network_fig = viz.create_network_diagram(model, layout='hierarchical')
network_fig.savefig('model_network.png', dpi=300)

# Create matrix heatmaps
heatmaps = viz.create_matrix_heatmaps(model, matrices=['A_m0', 'B_f0'])
for name, fig in heatmaps.items():
    fig.savefig(f'{name}_heatmap.png')

# Create interactive explorer
explorer_html = viz.create_interactive_explorer(model)
with open('model_explorer.html', 'w') as f:
    f.write(explorer_html)
```

## 🤖 LLM Integration API

### **🧠 LLMAnalyzer Class**

```python
from gnn.llm import LLMAnalyzer, AnalysisConfig

class LLMAnalyzer:
    """AI-enhanced model analysis using language models."""
    
    def __init__(self, 
                 provider: str = 'openai',
                 model: str = 'gpt-4',
                 api_key: str = None):
        """
        Initialize LLM analyzer.
        
        Args:
            provider: 'openai', 'anthropic', 'local'
            model: Model name/identifier
            api_key: API key for external providers
        """
    
    def analyze_model_structure(self, model: GNNModel) -> AnalysisReport:
        """
        Analyze model structure and suggest improvements.
        
        Args:
            model: GNN model to analyze
            
        Returns:
            AnalysisReport: Detailed analysis with suggestions
        """
    
    def explain_model(self, 
                     model: GNNModel,
                     audience: str = 'general') -> str:
        """
        Generate natural language explanation of model.
        
        Args:
            model: GNN model to explain
            audience: 'general', 'technical', 'academic'
            
        Returns:
            str: Natural language explanation
        """
    
    def suggest_optimizations(self, model: GNNModel) -> List[Optimization]:
        """Suggest performance and design optimizations."""
    
    def generate_research_questions(self, model: GNNModel) -> List[str]:
        """Generate research questions based on model."""

# Usage examples:
analyzer = LLMAnalyzer(provider='openai', model='gpt-4')

# Analyze model structure
analysis = analyzer.analyze_model_structure(model)
print(f"Complexity Score: {analysis.complexity_score}")
print(f"Suggestions: {analysis.optimization_suggestions}")

# Generate explanation
explanation = analyzer.explain_model(model, audience='general')
print(f"Model Explanation:\n{explanation}")

# Get optimization suggestions
optimizations = analyzer.suggest_optimizations(model)
for opt in optimizations:
    print(f"🎯 {opt.type}: {opt.description}")
    print(f"   Expected improvement: {opt.expected_improvement}")
```

## 📡 MCP API

### **🔌 MCP Integration**

```python
from gnn.mcp import MCPServer, GNNTool

class MCPServer:
    """Model Context Protocol server for GNN integration."""
    
    def __init__(self, name: str = "GNN-MCP-Server"):
        """Initialize MCP server."""
    
    def register_tool(self, tool: GNNTool):
        """Register a GNN tool with MCP."""
    
    def start_server(self, port: int = 8080):
        """Start MCP server on specified port."""

# Register GNN tools with MCP
server = MCPServer()

@server.tool("parse_gnn_model")
def parse_gnn_model(filepath: str) -> dict:
    """Parse GNN model and return structured representation."""
    parser = GNNParser()
    model = parser.parse_file(filepath)
    return model.to_dict()

@server.tool("visualize_model")
def visualize_model(model_data: dict, viz_type: str) -> str:
    """Create model visualization."""
    model = GNNModel.from_dict(model_data)
    viz = Visualizer()
    
    if viz_type == 'network':
        fig = viz.create_network_diagram(model)
        return save_figure_to_base64(fig)
    elif viz_type == 'interactive':
        return viz.create_interactive_explorer(model)

# Start MCP server
server.start_server(port=8080)
```

## ⚡ Performance API

### **📊 Performance Monitor**

```python
from gnn.performance import PerformanceMonitor, BenchmarkSuite

class PerformanceMonitor:
    """Real-time performance monitoring for GNN operations."""
    
    def __init__(self):
        """Initialize performance monitor."""
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
    
    def measure_operation(self, operation_name: str):
        """Decorator for measuring operation performance."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.measure(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_metrics(self, operation: str = None) -> PerformanceMetrics:
        """Get performance metrics for operations."""
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate performance report."""

# Usage examples:
monitor = PerformanceMonitor()
monitor.start_monitoring()

@monitor.measure_operation("model_parsing")
def parse_model(filepath):
    parser = GNNParser()
    return parser.parse_file(filepath)

# Get performance metrics
metrics = monitor.get_metrics("model_parsing")
print(f"Average time: {metrics.avg_time:.3f}s")
print(f"Memory usage: {metrics.avg_memory:.1f}MB")

# Generate report
report = monitor.generate_report("performance_report.html")
```

## 🔧 Configuration API

### **⚙️ Configuration Management**

```python
from gnn.config import ConfigManager, GNNConfig

class ConfigManager:
    """Centralized configuration management for GNN system."""
    
    def __init__(self, config_file: str = None):
        """Load configuration from file or defaults."""
    
    def get_config(self, section: str = None) -> GNNConfig:
        """Get configuration section or full config."""
    
    def set_config(self, section: str, key: str, value: Any):
        """Set configuration value."""
    
    def save_config(self, output_file: str = None):
        """Save current configuration to file."""

# Configuration example:
config_manager = ConfigManager("gnn_config.yaml")

# Get parsing configuration
parse_config = config_manager.get_config("parsing")
parser = GNNParser(config=parse_config)

# Modify pipeline configuration
config_manager.set_config("pipeline", "parallel", True)
config_manager.set_config("pipeline", "max_workers", 8)
config_manager.save_config()
```

## 🎯 Complete Integration Example

```python
"""
Complete example showing full GNN API integration
for building a custom Active Inference research workflow.
"""
from gnn import (
    GNNParser, Pipeline, Visualizer, LLMAnalyzer,
    PyMDPConverter, PerformanceMonitor
)

class ActiveInferenceWorkflow:
    """Complete Active Inference research workflow."""
    
    def __init__(self):
        # Initialize components
        self.parser = GNNParser(validation_level="STRICT")
        self.pipeline = Pipeline()
        self.visualizer = Visualizer()
        self.llm_analyzer = LLMAnalyzer()
        self.pymdp_converter = PyMDPConverter()
        self.monitor = PerformanceMonitor()
    
    def process_research_model(self, model_file: str) -> ResearchReport:
        """Process a model through complete research workflow."""
        
        # 1. Parse and validate model
        model = self.parser.parse_file(model_file)
        
        # 2. Run basic pipeline steps
        pipeline_result = self.pipeline.process(
            model, 
            steps=[1, 4, 6, 9],  # Parse, validate, visualize, render
            output_dir=f"./output/{model.name}/"
        )
        
        # 3. Generate PyMDP implementation
        pymdp_agent = self.pymdp_converter.convert_model(model)
        
        # 4. Create visualizations
        network_fig = self.visualizer.create_network_diagram(model)
        heatmaps = self.visualizer.create_matrix_heatmaps(model)
        
        # 5. AI analysis
        analysis = self.llm_analyzer.analyze_model_structure(model)
        explanation = self.llm_analyzer.explain_model(model, "academic")
        research_questions = self.llm_analyzer.generate_research_questions(model)
        
        # 6. Performance benchmarking
        performance = self.monitor.get_metrics()
        
        # 7. Compile research report
        return ResearchReport(
            model=model,
            pipeline_results=pipeline_result,
            pymdp_agent=pymdp_agent,
            visualizations={'network': network_fig, 'heatmaps': heatmaps},
            ai_analysis=analysis,
            explanation=explanation,
            research_questions=research_questions,
            performance_metrics=performance
        )

# Usage
workflow = ActiveInferenceWorkflow()
report = workflow.process_research_model("cognitive_navigation_model.md")

# Access results
print(f"✅ Model: {report.model.name}")
print(f"📊 Complexity: {report.ai_analysis.complexity_score}")
print(f"🎯 Research Questions: {len(report.research_questions)}")
print(f"⚡ Processing Time: {report.performance_metrics.total_time:.2f}s")
```

---

**🔌 API Integration**: This comprehensive API enables seamless integration of GNN capabilities into research workflows, production systems, and custom applications.

**📚 Documentation**: All APIs include comprehensive docstrings, type hints, and usage examples for immediate productivity.

---

**Status**: Production- **Start Here**: [Overview](../../README.md)
- **Examples**: [Model Examples](../../doc/gnn/gnn_examples_doc.md)
- **Development**: [Contribution Guide](../../CONTRIBUTING.md)