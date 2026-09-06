"""
Argument definitions for the GNN processing pipeline.

Provides the ArgumentDefinition dataclass describing individual pipeline
arguments with metadata (flag, type, default, help text, choices, action,
nargs, dest, suppression behavior) and helpers to add them to parsers.
"""

import argparse
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Type


@dataclass
class ArgumentDefinition:
    """Definition of a pipeline argument with metadata."""

    flag: str
    arg_type: Type = str
    default: Any = None
    required: bool = False
    help_text: str = ""
    choices: Optional[List[str]] = None
    action: str | type[argparse.Action] | None = None
    nargs: str | int | None = None
    dest: Optional[str] = None
    # When True with store_true/store_false, default is SUPPRESS so YAML/config can supply values
    use_suppress: bool = False

    def with_default(self, default: Any) -> "ArgumentDefinition":
        """Return a copy with a different default while preserving all metadata."""
        return replace(self, default=default)

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add this argument to an ArgumentParser."""
        kwargs: Dict[str, Any] = {"help": self.help_text}
        if self.dest is not None:
            kwargs["dest"] = self.dest

        if isinstance(self.action, str):
            kwargs["action"] = self.action
            if self.action in ("store_true", "store_false"):
                if self.use_suppress:
                    kwargs["default"] = argparse.SUPPRESS
                elif self.default is not None:
                    kwargs["default"] = self.default
                else:
                    kwargs["default"] = False if self.action == "store_true" else True
            else:
                kwargs["type"] = self.arg_type
                if self.default is not None:
                    kwargs["default"] = self.default
        elif self.action is argparse.BooleanOptionalAction:
            kwargs["action"] = self.action
            kwargs["default"] = self.default
        else:
            kwargs["type"] = self.arg_type
            kwargs["default"] = self.default

        if self.required:
            kwargs["required"] = True

        if self.choices:
            kwargs["choices"] = self.choices

        if self.nargs is not None:
            kwargs["nargs"] = self.nargs

        parser.add_argument(self.flag, **kwargs)
