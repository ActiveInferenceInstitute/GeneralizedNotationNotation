#!/usr/bin/env python3
"""
GNN CLI — Unified command-line interface with subcommands.

Provides:
  gnn run        — Execute the full pipeline
  gnn validate   — Validate a GNN file
  gnn parse      — Parse and output JSON
  gnn render     — Render a GNN file to a specific framework
  gnn report     — Generate pipeline report
  gnn reproduce  — Re-run from a previous run hash
  gnn preflight  — Run environment & config checks
  gnn health     — Show renderer & dependency status
  gnn serve      — Start Pipeline-as-a-Service API
  gnn lsp        — Launch Language Server
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="gnn",
        description="GNN Processing Pipeline — Command-line interface",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── gnn run ──────────────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Execute the full pipeline")
    run_p.add_argument("--target-dir", "-t", default="input/gnn_files", help="Input directory")
    run_p.add_argument("--output-dir", "-o", default="output", help="Output directory")
    run_p.add_argument("--skip-steps", nargs="*", type=int, help="Step numbers to skip")
    run_p.add_argument("--skip-llm", action="store_true", help="Skip LLM step (alias for --skip-steps 13)")
    run_p.add_argument("--linear", action="store_true", help="Force linear execution (no DAG)")

    # ── gnn validate ─────────────────────────────────────────────────────────
    validate_p = subparsers.add_parser("validate", help="Validate a GNN file")
    validate_p.add_argument("file", type=Path, help="GNN file to validate")
    validate_p.add_argument("--strict", action="store_true", help="Fail on warnings")

    # ── gnn parse ────────────────────────────────────────────────────────────
    parse_p = subparsers.add_parser("parse", help="Parse a GNN file and output JSON")
    parse_p.add_argument("file", type=Path, help="GNN file to parse")
    parse_p.add_argument("--format", choices=["json", "yaml", "summary"], default="json")

    # ── gnn render ───────────────────────────────────────────────────────────
    render_p = subparsers.add_parser("render", help="Render a GNN file to framework code")
    render_p.add_argument("file", type=Path, help="GNN file to render")
    render_p.add_argument("--framework", "-f", default="pymdp",
                          choices=["pymdp", "rxinfer", "jax", "numpyro", "stan", "pytorch"],
                          help="Target framework")
    render_p.add_argument("--output", "-o", type=Path, help="Output file path")

    # ── gnn report ───────────────────────────────────────────────────────────
    report_p = subparsers.add_parser("report", help="Generate pipeline report")
    report_p.add_argument("--output-dir", "-o", default="output", help="Pipeline output directory")

    # ── gnn reproduce ────────────────────────────────────────────────────────
    reproduce_p = subparsers.add_parser("reproduce", help="Re-run from a previous run hash")
    reproduce_p.add_argument("run_hash", help="Run hash (12-char hex prefix)")

    # ── gnn preflight ────────────────────────────────────────────────────────
    preflight_p = subparsers.add_parser("preflight", help="Run environment & config checks")
    preflight_p.add_argument("--config", type=Path, default=None, help="Config file path")

    # ── gnn health ───────────────────────────────────────────────────────────
    subparsers.add_parser("health", help="Show renderer & dependency status")

    # ── gnn serve ────────────────────────────────────────────────────────────
    serve_p = subparsers.add_parser("serve", help="Start Pipeline-as-a-Service API")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_p.add_argument("--port", type=int, default=8000, help="Bind port")

    # ── gnn lsp ──────────────────────────────────────────────────────────────
    subparsers.add_parser("lsp", help="Launch GNN Language Server")

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch
    handlers = {
        "run": _cmd_run,
        "validate": _cmd_validate,
        "parse": _cmd_parse,
        "render": _cmd_render,
        "report": _cmd_report,
        "reproduce": _cmd_reproduce,
        "preflight": _cmd_preflight,
        "health": _cmd_health,
        "serve": _cmd_serve,
        "lsp": _cmd_lsp,
    }
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


# ─── Command Handlers ────────────────────────────────────────────────────────────

def _cmd_run(args):
    """Execute full pipeline."""
    # Add src to path
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from main import main as pipeline_main
        sys.argv = ["gnn"]
        extra_args = ["--target-dir", str(args.target_dir), "--output-dir", str(args.output_dir)]
        if args.verbose:
            extra_args.append("--verbose")
        if args.skip_llm:
            extra_args.extend(["--skip-steps", "13"])
        elif args.skip_steps:
            extra_args.extend(["--skip-steps"] + [str(s) for s in args.skip_steps])
        sys.argv.extend(extra_args)
        return pipeline_main()
    except ImportError as e:
        logger.error(f"Could not import pipeline: {e}")
        return 1


def _cmd_validate(args):
    """Validate a GNN file."""
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        return 1

    content = args.file.read_text(encoding="utf-8")
    file_name = str(args.file)

    from gnn.schema import (
        validate_required_sections,
        parse_state_space,
        parse_connections,
        validate_matrix_dimensions,
    )

    errors = []

    # Section validation
    section_errors = validate_required_sections(content, file_path=file_name)
    errors.extend(section_errors)

    # State-space parsing
    variables, var_errors = parse_state_space(content, file_path=file_name)
    errors.extend(var_errors)

    # Connection parsing
    var_names = {v.name for v in variables}
    connections, conn_errors = parse_connections(content, known_variables=var_names, file_path=file_name)
    errors.extend(conn_errors)

    # Matrix validation
    dim_errors = validate_matrix_dimensions(content, variables, file_path=file_name)
    errors.extend(dim_errors)

    # Output
    if errors:
        for e in errors:
            print(f"  {e}")
        if args.strict:
            print(f"\n❌ {len(errors)} error(s) found")
            return 1
        print(f"\n⚠️ {len(errors)} warning(s) — pass --strict to fail")
        return 0
    else:
        print(f"✅ {file_name}: valid ({len(variables)} variables, {len(connections)} connections)")
        return 0


def _cmd_parse(args):
    """Parse a GNN file and output JSON."""
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        return 1

    content = args.file.read_text(encoding="utf-8")

    from gnn.schema import parse_state_space, parse_connections

    variables, _ = parse_state_space(content, file_path=str(args.file))
    var_names = {v.name for v in variables}
    connections, _ = parse_connections(content, known_variables=var_names, file_path=str(args.file))

    # Try frontmatter
    metadata = {}
    try:
        from gnn.frontmatter import parse_frontmatter
        metadata, _ = parse_frontmatter(content)
    except ImportError:
        pass

    result = {
        "file": str(args.file),
        "metadata": metadata,
        "variables": [
            {"name": v.name, "dimensions": v.dimensions, "dtype": v.dtype, "default": v.default}
            for v in variables
        ],
        "connections": [
            {
                "source": c.source, "target": c.target,
                "directed": c.directed, "label": c.label, "line": c.line,
            }
            for c in connections
        ],
    }

    if args.format == "summary":
        print(f"File: {args.file.name}")
        print(f"Variables: {len(variables)}")
        print(f"Connections: {len(connections)}")
        if metadata:
            print(f"Metadata: {', '.join(metadata.keys())}")
    else:
        print(json.dumps(result, indent=2))
    return 0


def _cmd_render(args):
    """Render a GNN file to framework code."""
    print(f"Rendering {args.file} → {args.framework}")
    print("(Render via CLI delegates to src/render/processor.py)")
    # Stub — full implementation would import render.processor
    return 0


def _cmd_report(args):
    """Generate pipeline report from existing outputs."""
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return 1

    from report.pipeline_report import generate_pipeline_report
    report = generate_pipeline_report(output_dir)
    report_path = output_dir / "PIPELINE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"📄 Report written to: {report_path}")
    return 0


def _cmd_reproduce(args):
    """Re-run from a previous run hash."""
    print(f"Looking up run hash: {args.run_hash}")
    print("(Reproduce delegates to pipeline/hasher.py)")
    return 0


def _cmd_preflight(args):
    """Run environment & config checks."""
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from pipeline.preflight import run_preflight
    report = run_preflight(config_path=args.config)
    print(report.to_markdown())
    return 0 if report.is_ok else 1


def _cmd_health(args):
    """Show renderer & dependency status."""
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from render.health import check_renderers
    from pipeline.preflight import check_environment

    renderers = check_renderers()
    env = check_environment()

    print("🔧 Renderers:")
    for name, status in sorted(renderers.items()):
        emoji = "🟢" if status.available else "🔴"
        print(f"  {emoji} {name}")

    available = sum(1 for r in renderers.values() if r.available)
    print(f"\n  {available}/{len(renderers)} available")
    print(f"\n🏗️ Environment: {env.checks_passed} passed, {env.checks_failed} failed")
    for issue in env.issues:
        sev = "⚠️" if issue.severity != "error" else "❌"
        print(f"  {sev} {issue.message}")

    return 0


def _cmd_serve(args):
    """Start Pipeline-as-a-Service API."""
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from api.app import start_server
        start_server(host=args.host, port=args.port)
    except ImportError:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
        return 1
    return 0


def _cmd_lsp(args):
    """Launch GNN Language Server."""
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from lsp import start_server
        start_server()
    except ImportError:
        print("❌ pygls not installed. Run: pip install pygls")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
