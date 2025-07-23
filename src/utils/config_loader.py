"""
Configuration Loader for GNN Pipeline

Handles loading and validation of YAML configuration files for the pipeline.
"""

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the GNN pipeline."""
    
    # Core directories
    target_dir: Path = field(default_factory=lambda: Path("input/gnn_files"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # Processing options
    recursive: bool = True
    verbose: bool = True
    
    # Enhanced validation options (enabled by default for comprehensive testing)
    enable_round_trip: bool = True      # Enable round-trip testing across all 21 formats
    enable_cross_format: bool = True    # Enable cross-format consistency validation
    
    # Step control
    skip_steps: List[str] = field(default_factory=list)
    only_steps: List[str] = field(default_factory=list)
    
    # Pipeline summary file
    pipeline_summary_file: Optional[Path] = None
    
    def __post_init__(self):
        """Post-initialization validation and path resolution."""
        # Ensure Path objects
        if isinstance(self.target_dir, str):
            self.target_dir = Path(self.target_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.pipeline_summary_file, str):
            self.pipeline_summary_file = Path(self.pipeline_summary_file)
        elif self.pipeline_summary_file is None:
            self.pipeline_summary_file = self.output_dir / "pipeline_execution_summary.json"

@dataclass
class TypeCheckerConfig:
    """Configuration for type checking step."""
    strict: bool = False
    estimate_resources: bool = True

@dataclass
class OntologyConfig:
    """Configuration for ontology processing."""
    terms_file: Path = field(default_factory=lambda: Path("src/ontology/act_inf_ontology_terms.json"))
    
    def __post_init__(self):
        if isinstance(self.terms_file, str):
            self.terms_file = Path(self.terms_file)

@dataclass
class LLMConfig:
    """Configuration for LLM processing."""
    tasks: str = "all"
    timeout: int = 360

@dataclass
class WebsiteConfig:
    """Configuration for website generation."""
    html_filename: str = "gnn_pipeline_summary_website.html"

@dataclass
class SetupConfig:
    """Configuration for setup step."""
    recreate_venv: bool = False
    dev: bool = False

@dataclass
class SAPFConfig:
    """Configuration for SAPF audio generation."""
    duration: float = 30.0

@dataclass
class ModelConfig:
    """Configuration for model-specific settings."""
    global_settings: Dict[str, Any] = field(default_factory=dict)
    model_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class GNNPipelineConfig:
    """Complete configuration for the GNN pipeline."""
    
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    type_checker: TypeCheckerConfig = field(default_factory=TypeCheckerConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    website: WebsiteConfig = field(default_factory=WebsiteConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
    sapf: SAPFConfig = field(default_factory=SAPFConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'GNNPipelineConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, returning default configuration")
            return cls()
            
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return cls._from_dict(config_data)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def _from_dict(cls, config_data: Dict[str, Any]) -> 'GNNPipelineConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Load pipeline configuration
        if 'pipeline' in config_data:
            pipeline_data = config_data['pipeline']
            config.pipeline.target_dir = Path(pipeline_data.get('target_dir', 'input/gnn_files'))
            config.pipeline.output_dir = Path(pipeline_data.get('output_dir', 'output'))
            config.pipeline.recursive = pipeline_data.get('recursive', True)
            config.pipeline.verbose = pipeline_data.get('verbose', True)
            config.pipeline.enable_round_trip = pipeline_data.get('enable_round_trip', True)
            config.pipeline.enable_cross_format = pipeline_data.get('enable_cross_format', True)
            config.pipeline.skip_steps = pipeline_data.get('skip_steps', [])
            config.pipeline.only_steps = pipeline_data.get('only_steps', [])
            if 'pipeline_summary_file' in pipeline_data:
                config.pipeline.pipeline_summary_file = Path(pipeline_data['pipeline_summary_file'])
        
        # Load type checker configuration
        if 'type_checker' in config_data:
            tc_data = config_data['type_checker']
            config.type_checker.strict = tc_data.get('strict', False)
            config.type_checker.estimate_resources = tc_data.get('estimate_resources', True)
        
        # Load ontology configuration
        if 'ontology' in config_data:
            ontology_data = config_data['ontology']
            config.ontology.terms_file = Path(ontology_data.get('terms_file', 'src/ontology/act_inf_ontology_terms.json'))
        
        # Load LLM configuration
        if 'llm' in config_data:
            llm_data = config_data['llm']
            config.llm.tasks = llm_data.get('tasks', 'all')
            config.llm.timeout = llm_data.get('timeout', 360)
        
        # Load website configuration
        if 'website' in config_data:
            website_data = config_data['website']
            config.website.html_filename = website_data.get('html_filename', 'gnn_pipeline_summary_website.html')
        
        # Load setup configuration
        if 'setup' in config_data:
            setup_data = config_data['setup']
            config.setup.recreate_venv = setup_data.get('recreate_venv', False)
            config.setup.dev = setup_data.get('dev', False)
        
        # Load SAPF configuration
        if 'sapf' in config_data:
            sapf_data = config_data['sapf']
            config.sapf.duration = sapf_data.get('duration', 30.0)
        
        # Load model configuration
        if 'models' in config_data:
            models_data = config_data['models']
            config.models.global_settings = models_data.get('global', {})
            config.models.model_overrides = models_data.get('model_overrides', {})
        
        return config
    
    def to_pipeline_arguments(self) -> Dict[str, Any]:
        """Convert configuration to pipeline arguments dictionary."""
        return {
            'target_dir': self.pipeline.target_dir,
            'output_dir': self.pipeline.output_dir,
            'recursive': self.pipeline.recursive,
            'verbose': self.pipeline.verbose,
            'enable_round_trip': self.pipeline.enable_round_trip,
            'enable_cross_format': self.pipeline.enable_cross_format,
            'skip_steps': ','.join(self.pipeline.skip_steps) if self.pipeline.skip_steps else None,
            'only_steps': ','.join(self.pipeline.only_steps) if self.pipeline.only_steps else None,
            'pipeline_summary_file': self.pipeline.pipeline_summary_file,
            'strict': self.type_checker.strict,
            'estimate_resources': self.type_checker.estimate_resources,
            'ontology_terms_file': self.ontology.terms_file,
            'llm_tasks': self.llm.tasks,
            'llm_timeout': self.llm.timeout,
            'website_html_filename': self.website.html_filename,
            'recreate_venv': self.setup.recreate_venv,
            'dev': self.setup.dev,
            'duration': self.sapf.duration
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Get project root for relative path resolution
        project_root = Path(__file__).parent.parent.parent
        
        # Validate target directory exists (only if it's an absolute path or we can resolve it)
        target_dir = self.pipeline.target_dir
        if target_dir.is_absolute() and not target_dir.exists():
            errors.append(f"Target directory does not exist: {target_dir}")
        elif not target_dir.is_absolute():
            # For relative paths, check if they exist relative to project root
            resolved_target = project_root / target_dir
            if not resolved_target.exists():
                errors.append(f"Target directory does not exist: {resolved_target}")
        
        # Validate ontology terms file exists
        ontology_file = self.ontology.terms_file
        if not ontology_file.is_absolute():
            ontology_file = project_root / ontology_file
        if not ontology_file.exists():
            errors.append(f"Ontology terms file does not exist: {ontology_file}")
        
        # Validate LLM timeout
        if self.llm.timeout <= 0:
            errors.append(f"LLM timeout must be positive: {self.llm.timeout}")
        
        # Validate SAPF duration
        if self.sapf.duration <= 0:
            errors.append(f"SAPF duration must be positive: {self.sapf.duration}")
        
        return errors

def load_config(config_path: Optional[Path] = None) -> GNNPipelineConfig:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file. If None, looks for config.yaml in input/
        
    Returns:
        Loaded configuration
    """
    if config_path is None:
        # Look for config.yaml in input directory
        input_dir = Path("input")
        config_path = input_dir / "config.yaml"
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found at {config_path}, using defaults")
            return GNNPipelineConfig()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    config = GNNPipelineConfig.load_from_file(config_path)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError("Configuration validation failed")
    
    logger.info("Configuration loaded successfully")
    return config 

def save_config(config: dict, file_path: Path) -> None:
    """
    Save a configuration dictionary to a file (YAML or JSON based on extension).
    Args:
        config: Configuration dictionary to save
        file_path: Path to the output file (should end with .yaml/.yml or .json)
    Raises:
        ValueError: If file extension is not supported
    """
    file_path = Path(file_path)
    try:
        if file_path.suffix.lower() in {'.yaml', '.yml'}:
            if not YAML_AVAILABLE:
                logger.warning("PyYAML not available, saving as JSON instead")
                file_path = file_path.with_suffix('.json')
                import json
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                with open(file_path, 'w') as f:
                    yaml.safe_dump(config, f)
        elif file_path.suffix.lower() == '.json':
            import json
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file extension: {file_path.suffix}")
        logger.info(f"Configuration saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {file_path}: {e}")
        raise 

def validate_config(config: dict) -> bool:
    """
    Validate a configuration dictionary for required structure.
    Returns True if valid, False otherwise.
    For demonstration, require at least one section with at least one key.
    """
    if not isinstance(config, dict) or not config:
        logger.warning("Config is not a dictionary or is empty.")
        return False
    for section, section_data in config.items():
        if isinstance(section_data, dict) and section_data:
            return True
    logger.warning("No valid section with keys found in config.")
    return False

def get_config_value(config: dict, key: str):
    """
    Retrieve a value from a nested config dictionary using dot notation.
    Example: get_config_value(config, 'section.key')
    """
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        logger.warning(f"Key '{key}' not found in config.")
        return None

def set_config_value(config: dict, key: str, value):
    """
    Set a value in a nested config dictionary using dot notation.
    Example: set_config_value(config, 'section.key', value)
    """
    keys = key.split('.')
    d = config
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
    logger.info(f"Set config value: {key} = {value}") 