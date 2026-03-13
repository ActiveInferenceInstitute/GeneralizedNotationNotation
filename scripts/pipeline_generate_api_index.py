#!/usr/bin/env python3
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_JSON = PROJECT_ROOT / "doc" / "api" / "api_index.json"

EXCLUDE_DIRS = {
    "__pycache__",
    ".venv",
    "venv",
    "output",
    "tests",
}


def discover_python_files(base: Path) -> List[Path]:
    py_files: List[Path] = []
    for path in base.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        py_files.append(path)
    return py_files


def safe_unparse(node: Any) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def ast_to_api(path: Path) -> Dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return {"file": str(path.relative_to(PROJECT_ROOT)), "error": "read_error"}

    try:
        tree = ast.parse(source)
    except Exception as e:
        return {"file": str(path.relative_to(PROJECT_ROOT)), "error": f"parse_error: {e}"}

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "lineno": node.lineno,
                "args": [a.arg for a in node.args.args],
                "returns": safe_unparse(node.returns) if getattr(node, 'returns', None) else None,
                "decorators": [safe_unparse(d) for d in getattr(node, 'decorator_list', [])],
                "doc": ast.get_docstring(node) or ""
            })
        elif isinstance(node, ast.ClassDef):
            methods: List[Dict[str, Any]] = []
            bases = [safe_unparse(b) for b in getattr(node, 'bases', [])]
            for cnode in node.body:
                if isinstance(cnode, ast.FunctionDef):
                    methods.append({
                        "name": cnode.name,
                        "lineno": cnode.lineno,
                        "args": [a.arg for a in cnode.args.args],
                        "returns": safe_unparse(cnode.returns) if getattr(cnode, 'returns', None) else None,
                        "decorators": [safe_unparse(d) for d in getattr(cnode, 'decorator_list', [])],
                        "doc": ast.get_docstring(cnode) or ""
                    })
            classes.append({
                "name": node.name,
                "lineno": node.lineno,
                "bases": bases,
                "doc": ast.get_docstring(node) or "",
                "methods": methods
            })

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "module": str(path.relative_to(SRC_DIR)).replace("/", ".")[:-3] if str(path).endswith(".py") else None,
        "functions": functions,
        "classes": classes,
    }


def build_index() -> Dict[str, Any]:
    files = discover_python_files(SRC_DIR)
    entries: List[Dict[str, Any]] = []
    for f in sorted(files):
        entries.append(ast_to_api(f))
    return {"generated_from": str(SRC_DIR), "count": len(entries), "entries": entries}


def main() -> int:
    index = build_index()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote API index: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
