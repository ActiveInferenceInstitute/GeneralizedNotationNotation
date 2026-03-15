"""Tests for the GNN CLI module."""

import pytest
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch

# Import the CLI main function
# Adjust import path if necessary based on project structure
try:
    from cli import main
except ImportError:
    # Try absolute import if relative fails
    try:
        from src.cli import main
    except ImportError:
        # Fallback for different execution environments
        sys.path.append(str(Path(__file__).parent.parent))
        from cli import main

def test_cli_help():
    """Test that 'gnn --help' works and returns success."""
    with patch('sys.stdout', new=StringIO()) as fake_out:
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        
        assert exit_info.value.code == 0
        output = fake_out.getvalue()
        assert "GNN Processing Pipeline" in output
        assert "Available commands" in output
        assert "run" in output
        assert "validate" in output

def test_cli_invalid_command():
    """Test that an invalid command returns an error or help."""
    with patch('sys.stderr', new=StringIO()):
        with pytest.raises(SystemExit):
            main(["nonexistent-command"])
    
    # argparse usually exits with 2 for invalid arguments
    # and prints "invalid choice" to stderr

def test_cli_validate_parser():
    """Test the 'validate' subcommand parser."""
    with patch('cli.Path.exists', return_value=True):
        with patch('cli.Path.is_file', return_value=True):
            # Mock the validation command handler directly
            with patch('cli._cmd_validate') as mock_validate:
                main(["validate", "test.md"])
                mock_validate.assert_called_once()

def test_cli_verbose_flag():
    """Test that the --verbose flag is correctly handled."""
    with patch('cli._cmd_health') as mock_health:
        main(["--verbose", "health"])
        # Check if the args object passed to command handler has verbose=True
        args = mock_health.call_args[0][0]
        assert args.verbose is True

def test_cli_subcommand_routing():
    """Test that subcommands are routed to the correct handlers."""
    commands = [
        ("run", "_cmd_run"),
        ("validate", "_cmd_validate"),
        ("parse", "_cmd_parse"),
        ("render", "_cmd_render"),
        ("report", "_cmd_report"),
        ("health", "_cmd_health"),
    ]
    
    for cmd_name, handler_name in commands:
        with patch(f'cli.{handler_name}') as mock_handler:
            # Provide dummy arguments where needed by the parser
            if cmd_name in ["validate", "parse", "render", "graph"]:
                main([cmd_name, "dummy.md"])
            elif cmd_name == "reproduce":
                main([cmd_name, "abc123def456"])
            elif cmd_name == "watch":
                main([cmd_name, "."])
            else:
                main([cmd_name])
            
            mock_handler.assert_called_once()
