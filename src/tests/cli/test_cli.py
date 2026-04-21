"""Tests for the GNN CLI module."""

import sys
from io import StringIO
from pathlib import Path

import pytest

# Import the CLI main function
try:
    import cli
    from cli import main
except ImportError:
    try:
        import src.cli as cli
        from src.cli import main
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent.parent))
        import cli
        from cli import main

class CallTracker:
    def __init__(self):
        self.called = False
        self.call_args = None
        
    def __call__(self, *args, **kwargs):
        self.called = True
        self.call_args = args

def test_cli_help():
    """Test that 'gnn --help' works and returns success."""
    orig_stdout = sys.stdout
    fake_out = StringIO()
    sys.stdout = fake_out
    try:
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        
        assert exit_info.value.code == 0
        output = fake_out.getvalue()
        assert "GNN Processing Pipeline" in output
        assert "Available commands" in output
        assert "run" in output
        assert "validate" in output
    finally:
        sys.stdout = orig_stdout

def test_cli_invalid_command():
    """Test that an invalid command returns an error or help."""
    orig_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        with pytest.raises(SystemExit):
            main(["nonexistent-command"])
    finally:
        sys.stderr = orig_stderr
    
def test_cli_validate_parser(tmp_path):
    """Test the 'validate' subcommand parser."""
    test_file = tmp_path / "test.md"
    test_file.touch()
    
    orig_validate = getattr(cli, "_cmd_validate", None)
    tracker = CallTracker()
    cli._cmd_validate = tracker
    
    try:
        main(["validate", str(test_file)])
        assert tracker.called is True
    finally:
        if orig_validate:
            cli._cmd_validate = orig_validate

def test_cli_verbose_flag(tmp_path):
    """Test that the --verbose flag is correctly handled."""
    orig_health = getattr(cli, "_cmd_health", None)
    tracker = CallTracker()
    cli._cmd_health = tracker
    
    try:
        main(["--verbose", "health"])
        args = tracker.call_args[0]
        assert getattr(args, "verbose", False) is True
    finally:
        if orig_health:
            cli._cmd_health = orig_health

def test_cli_subcommand_routing(tmp_path):
    """Test that subcommands are routed to the correct handlers."""
    commands = [
        ("run", "_cmd_run"),
        ("validate", "_cmd_validate"),
        ("parse", "_cmd_parse"),
        ("render", "_cmd_render"),
        ("report", "_cmd_report"),
        ("health", "_cmd_health"),
    ]
    
    test_file = tmp_path / "test.md"
    test_file.touch()
    str_test_file = str(test_file)
    
    for cmd_name, handler_name in commands:
        orig_handler = getattr(cli, handler_name, None)
        tracker = CallTracker()
        setattr(cli, handler_name, tracker)
        
        try:
            if cmd_name in ["validate", "parse", "render", "graph"]:
                main([cmd_name, str_test_file])
            elif cmd_name == "reproduce":
                main([cmd_name, "abc123def456"])
            elif cmd_name == "watch":
                main([cmd_name, "."])
            else:
                main([cmd_name])
            
            assert tracker.called is True
        finally:
            if orig_handler:
                setattr(cli, handler_name, orig_handler)
