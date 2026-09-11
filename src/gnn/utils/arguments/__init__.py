"""Argument-handling concern package (S2-33 Step 2): canonical home of the
argument parsing, definitions, pipeline arguments, step configuration and
path-conversion code that used to live in the ``gnn/utils`` top-level grab-bag.

Re-exports the family's public names as real objects (not lazy — design
§4.3.1), so intra-family and consumer reads use
``from gnn.utils.arguments import parse_arguments``. Import-weight note (I1):
this package is NOT imported by ``import gnn.utils`` — the top-level facade
stays lazy through its PEP 562 map (guarded by tests/tests/test_light_import.py).
Importing this package eagerly imports every leaf, which is the same cost the
old ``import gnn.utils.arg_parsing`` paid.

Leaf inventory:
- arg_definitions: ``ArgumentDefinition`` and the argument contract table
- pipeline_arguments: ``PipelineArguments`` + ontology-term file constant
- step_config: declarative ``StepConfiguration``
- arg_parsing: ``ArgumentParser``/``StepAwareArgumentParser`` and the parse
  helpers (``parse_arguments``, ``build_step_command_args``, ...)
- path_conversion: path/argument normalization and configuration validation
- pipeline_config_merge: ``apply_input_config_defaults`` (YAML defaults)

Cross-family imports go through leaf modules, never through any facade (I5).
The old top-level paths (``gnn/utils/arg_parsing.py`` etc. and the
``argument_utils.py`` entry facade) are deprecation facades over this package.
"""

from gnn.utils.arguments.arg_definitions import ArgumentDefinition
from gnn.utils.arguments.arg_parsing import (
    ArgumentParser,
    PipelineStepInfo,
    StepAwareArgumentParser,
    audit_step_contracts,
    build_step_command_args,
    fallback_default_for,
    get_pipeline_step_info,
    parse_arguments,
    parse_step_arguments,
    parse_step_list,
    validate_arguments,
)
from gnn.utils.arguments.path_conversion import (
    convert_path_arguments,
    validate_and_convert_paths,
    validate_pipeline_configuration,
)
from gnn.utils.arguments.pipeline_arguments import (
    DEFAULT_ONTOLOGY_TERMS_FILE,
    PipelineArguments,
)
from gnn.utils.arguments.pipeline_config_merge import apply_input_config_defaults
from gnn.utils.arguments.step_config import StepConfiguration

__all__: list[str] = [
    "ArgumentDefinition",
    "ArgumentParser",
    "DEFAULT_ONTOLOGY_TERMS_FILE",
    "PipelineArguments",
    "PipelineStepInfo",
    "StepAwareArgumentParser",
    "StepConfiguration",
    "apply_input_config_defaults",
    "audit_step_contracts",
    "build_step_command_args",
    "convert_path_arguments",
    "fallback_default_for",
    "get_pipeline_step_info",
    "parse_arguments",
    "parse_step_arguments",
    "parse_step_list",
    "validate_and_convert_paths",
    "validate_arguments",
    "validate_pipeline_configuration",
]
