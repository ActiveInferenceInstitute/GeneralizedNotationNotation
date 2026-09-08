#!/usr/bin/env python3
"""Old import path for the pipeline runtime integration tester.

This module was renamed to ``gnn.pipeline.pipeline_runtime_validator`` to
disambiguate it from ``gnn.utils.pipeline_validator`` (the pre-execution
prerequisite checker). The old import path still works: it emits a
``DeprecationWarning`` and re-exports the same symbols with identical
behavior.

See also:
- pipeline/pipeline_runtime_validator.py: Runtime integration tester (runs
  pipeline via subprocess)
- pipeline/pipeline_validation.py: Static code analysis (checks import
  patterns, naming)
- utils/pipeline_validator.py: Pre-execution prerequisite checker (checks
  step outputs exist)
"""

import warnings

warnings.warn(
    "gnn.pipeline.pipeline_validator is an old name; "
    "use gnn.pipeline.pipeline_runtime_validator instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gnn.pipeline.pipeline_runtime_validator import (  # noqa: E402,F401
    PipelineValidator,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
