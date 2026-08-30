#!/usr/bin/env python3
"""Add module docstrings to files that lack them, composing from existing class/function docstrings.

Inserts a PEP 257 module docstring as the first statement (after shebang/encoding
comment lines, before `from __future__` imports). Content is derived from the
file's own top-level classes and functions so it stays accurate.

Usage:
    python scripts/add_module_docstrings.py [ROOT] [--dry-run]

Pass ``--dry-run`` to report which files would change without writing them.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

TARGETS = []


def leading_comment_lines(text: str) -> tuple[list[str], int]:
    """Return (leading comment lines, count) — shebang/encoding/vim lines at file top."""
    lines = text.splitlines(keepends=True)
    leading = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("#!"):
            leading.append(lines[i])
            i += 1
            continue
        if stripped.startswith("#"):
            # Only keep encoding/vim/emacs marker comments; drop prose comment blocks.
            if re.match(r"#\s*[-\*]", stripped) or re.match(
                r"#\s*(?:-\*-\s*coding|vim:|emacs)", stripped
            ):
                leading.append(lines[i])
                i += 1
                continue
            # Treat a run of comment lines as a header block worth keeping only if
            # it looks like a license/attribution header.
            if (
                "license" in stripped.lower()
                or "copyright" in stripped.lower()
                or "author" in stripped.lower()
            ):
                leading.append(lines[i])
                i += 1
                continue
            break
        break
    return leading, i


def describe_module(path: pathlib.Path, tree: ast.Module) -> str:
    """Build a 1-3 line module description from the file's own content."""
    if path.name == "__init__.py":
        pkg = path.parent.name
        exports = [a for a in tree.body if isinstance(a, (ast.Import, ast.ImportFrom))]
        # Collect imported names that look like re-exports.
        imported: list[str] = []
        for node in exports:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name != "*":
                        imported.append(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.asname or alias.name.split(".")[0])
        if imported:
            joined = ", ".join(imported[:8])
            more = f" and {len(imported) - 8} more" if len(imported) > 8 else ""
            return (
                f"Public API for the {pkg} package.\n\n"
                f"Re-exports {joined}{more} from submodules."
            )
        return f"Package marker for the {pkg} package."

    classes = []
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            classes.append((node.name, doc))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            funcs.append((node.name, doc))

    parts = []
    if classes:
        names = ", ".join(c[0] for c in classes[:4])
        if len(classes) > 4:
            names += f", and {len(classes) - 4} more"
        primary_doc = classes[0][1] if classes[0][1] else ""
        first_sentence = (
            re.split(r"[.\n]", primary_doc)[0].strip() if primary_doc else ""
        )
        if first_sentence and len(first_sentence) > 12:
            parts.append(first_sentence.rstrip(".") + ".")
        else:
            parts.append(f"Defines {names}.")
    elif funcs:
        names = ", ".join(f[0] for f in funcs[:4])
        if len(funcs) > 4:
            names += f", and {len(funcs) - 4} more"
        parts.append(f"Provides helper functions: {names}.")
    else:
        parts.append(f"Module for {path.parent.name} functionality.")

    # Second line: public API listing
    if classes:
        parts.append(f"Public classes: {', '.join(c[0] for c in classes)}")
    if funcs and (not classes or len(funcs) <= 8):
        parts.append(f"Public functions: {', '.join(f[0] for f in funcs)}")

    return "\n\n".join(parts)


def add_docstring(path: pathlib.Path, write: bool = True) -> str | None:
    """Compose a module docstring for ``path``; write it unless ``write=False``.

    Returns the composed docstring when the file would change, else ``None``.
    """
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(f"  SKIP (syntax): {path} — {e}")
        return None

    # Skip if any top-level string-literal expression exists (already documented).
    if any(
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
        for stmt in tree.body
    ):
        return None

    description = describe_module(path, tree)
    docstring = f'"""{description}\n"""'

    leading, idx = leading_comment_lines(text)
    lines = text.splitlines(keepends=True)
    new_lines = leading + [docstring + "\n", "\n"] + lines[idx:]
    new_text = "".join(new_lines)

    # Verify it still parses and the docstring is now first statement.
    tree2 = ast.parse(new_text)
    first = tree2.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant), (
        f"docstring not first statement in {path}"
    )
    if write:
        path.write_text(new_text)
    return docstring


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="src",
        help="Directory tree to process (default: src)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would change without writing them",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root)
    py_files = sorted(
        p
        for p in root.rglob("*.py")
        if ".venv" not in str(p) and "__pycache__" not in str(p)
    )
    added = 0
    for p in py_files:
        result = add_docstring(p, write=not args.dry_run)
        if result is not None:
            print(f"  {'~' if args.dry_run else '+'} {p}")
            added += 1
    mode = "would receive" if args.dry_run else "added"
    print(f"\n{added} docstrings {mode}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
