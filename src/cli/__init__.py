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
    run_p.add_argument("--log-format", choices=["human", "json"], default="human", help="Output format for pipeline logs")

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
    reproduce_p.add_argument("--history-dir", type=Path, default=Path("output/00_pipeline_summary/.history"), 
                             help="Directory containing index.json")

    # ── gnn preflight ────────────────────────────────────────────────────────
    preflight_p = subparsers.add_parser("preflight", help="Run environment & config checks")
    preflight_p.add_argument("--config", type=Path, default=None, help="Config file path")

    # ── gnn health ───────────────────────────────────────────────────────────
    subparsers.add_parser("health", help="Show renderer & dependency status")

    # ── gnn serve ────────────────────────────────────────────────────────────
    serve_p = subparsers.add_parser("serve", help="Start Pipeline-as-a-Service API")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_p.add_argument("--port", type=int, default=8000, help="Bind port")

    # ── gnn watch ────────────────────────────────────────────────────────────
    watch_p = subparsers.add_parser("watch", help="Monitor directory and live-reparse on change")
    watch_p.add_argument("dir", type=Path, help="Directory to monitor (e.g. input/gnn_files/)")

    # ── gnn graph ────────────────────────────────────────────────────────────
    graph_p = subparsers.add_parser("graph", help="Generate dependency graph from multi-model files")
    graph_p.add_argument("file", type=Path, help="GNN file to render")
    graph_p.add_argument("--format", choices=["mermaid", "text"], default="mermaid", help="Output format")

    # ── gnn lsp ──────────────────────────────────────────────────────────────
    subparsers.add_parser("lsp", help="Launch GNN Language Server")

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Ensure src/ is on sys.path for all subcommands
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

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
        "watch": _cmd_watch,
        "graph": _cmd_graph,
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
    try:
        from main import main as pipeline_main
        sys.argv = ["gnn"]
        extra_args = ["--target-dir", str(args.target_dir), "--output-dir", str(args.output_dir)]
        if args.verbose:
            extra_args.append("--verbose")
        if args.log_format == "json":
            extra_args.extend(["--log-format", "json"])
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
        pass  # frontmatter parsing is optional

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
    # Placeholder — full implementation would import render.processor
    return 0


def _cmd_report(args):
    """Generate pipeline report from existing outputs."""
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
    import logging
    logger = logging.getLogger(__name__)
    
    from pipeline.hasher import lookup_run
    history_dir = args.history_dir
    
    run_entry = lookup_run(args.run_hash, history_dir)
    if not run_entry:
        print(f"❌ Run hash not found: {args.run_hash}")
        return 1
        
    print(f"🔄 Reproducing run: {args.run_hash}")
    config = run_entry.get("config", {})
    run_args_dict = config.get("args", {})
    pipeline_settings = config.get("pipeline", {})
    
    try:
        from main import main as pipeline_main
        from utils.argument_utils import PipelineArguments
        
        # Reconstruct PipelineArguments
        # Some paths might need to be converted back to Path objects
        if "target_dir" in run_args_dict:
            run_args_dict["target_dir"] = Path(run_args_dict["target_dir"])
        if "output_dir" in run_args_dict:
            run_args_dict["output_dir"] = Path(run_args_dict["output_dir"])
            
        reproduced_args = PipelineArguments(**run_args_dict)
        
        # Trigger execution bypassing normal CLI arg parsing
        print(f"🚀 Bypassing CLI parser, running with reconstructed config")
        
        # Pass the full config structure that main() expects back in override_config
        full_config_override = {"pipeline": pipeline_settings}
        
        return pipeline_main(override_args=reproduced_args, override_config=full_config_override)
        
    except ImportError as e:
        logger.error(f"Could not import pipeline for reproduction: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to reproduce run: {e}")
        return 1


def _cmd_preflight(args):
    """Run environment & config checks."""
    from pipeline.preflight import run_preflight
    report = run_preflight(config_path=args.config)
    print(report.to_markdown())
    return 0 if report.is_ok else 1


def _cmd_health(args):
    """Show renderer & dependency status."""
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
    try:
        from api.app import start_server
        start_server(host=args.host, port=args.port)
    except ImportError:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
        return 1
    return 0


def _cmd_lsp(args):
    """Launch GNN Language Server."""
    try:
        from cli.lsp import start_lsp
        start_lsp()
    except ImportError as e:
        print(f"❌ Could not start LSP server: {e}")
        return 1
    return 0


def _cmd_watch(args):
    """Monitor directory and live-reparse on change."""
    try:
        from gnn.watcher import GNNWatcher
        watcher = GNNWatcher(watch_dir=args.dir)
        watcher.start()
    except ImportError as e:
        logger.error(f"Could not import watcher: {e}")
        return 1
    return 0


def _cmd_graph(args):
    """Generate dependency graph from multi-model files."""
    if not args.file.exists():
        logger.error(f"File not found: {args.file}")
        return 1

    try:
        from gnn.dep_graph import render_graph_from_file
        output = render_graph_from_file(str(args.file), output_format=args.format)
        print(output)
    except ImportError as e:
        logger.error(f"Could not import graph generator: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
