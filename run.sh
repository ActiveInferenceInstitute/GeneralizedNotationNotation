#!/bin/bash
# GNN Processing Pipeline Runner
# 
# This script provides an easy way to run the Generalized Notation Notation (GNN)
# processing pipeline with sensible defaults and configuration options.
#
# Usage: ./run.sh [options]
# 
# Examples:
#   ./run.sh                    # Run full pipeline with default settings
#   ./run.sh --verbose          # Run with verbose output
#   ./run.sh --quick            # Run quick test (steps 0-3 only)
#   ./run.sh --steps "0,1,2,3" # Run specific steps
#   ./run.sh --help             # Show detailed help

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SRC_DIR="$PROJECT_ROOT/src"
MAIN_SCRIPT="$SRC_DIR/main.py"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_EXECUTABLE="python3"

# Default configuration
DEFAULT_TARGET_DIR="input/gnn_files"
DEFAULT_OUTPUT_DIR="output"
DEFAULT_VERBOSE="false"
DEFAULT_QUICK_MODE="false"
DEFAULT_STEPS=""
DEFAULT_SKIP_STEPS=""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Display API key guidance
display_api_key_guidance() {
    echo
    log_info "=========================================="
    log_info "API KEY CONFIGURATION GUIDANCE"
    log_info "=========================================="
    log_info "To enable LLM features, configure API keys:"
    echo
    log_info "Method 1 - Environment Variables (Recommended):"
    log_info "  export OPENAI_API_KEY='your_openai_key_here'"
    log_info "  export ANTHROPIC_API_KEY='your_anthropic_key_here'"
    log_info "  export OPENROUTER_API_KEY='your_openrouter_key_here'"
    log_info "  export PERPLEXITY_API_KEY='your_perplexity_key_here'"
    echo
    log_info "Method 2 - .env file:"
    log_info "  1. Copy api_keys_template.env to .env"
    log_info "  2. Edit .env with your actual API keys"
    log_info "  3. Never commit .env files to version control"
    echo
    log_info "For local LLM (Ollama):"
    log_info "  1. Install Ollama: https://ollama.ai"
    log_info "  2. Pull a model: ollama pull llama2"
    log_info "  3. Set OLLAMA_BASE_URL=http://localhost:11434"
    echo
    log_info "Security Best Practices:"
    log_info "  - Use environment variables in production"
    log_info "  - Rotate API keys regularly"
    log_info "  - Use least privilege principle"
    log_info "  - Monitor API key usage"
    log_info "=========================================="
    echo
}

# Create API key template file
create_api_key_template() {
    local template_file="$PROJECT_ROOT/api_keys_template.env"
    local env_file="$PROJECT_ROOT/.env"
    
    # Create template file if it doesn't exist
    if [[ ! -f "$template_file" ]]; then
        log_info "Creating API key template file..."
        cat > "$template_file" << 'EOF'
# API Keys Configuration Template
# Copy this file to .env and fill in your API keys
# Never commit .env files to version control!

# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenRouter API Key (for multiple providers)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Perplexity API Key
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Environment Variables (for shell export)
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
export PERPLEXITY_API_KEY="your_perplexity_api_key_here"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama2"
EOF
        log_success "Created API key template: $template_file"
    fi
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$env_file" ]]; then
        log_info "Creating .env file..."
        echo "# Add your API keys here" > "$env_file"
        log_success "Created .env file: $env_file"
    fi
}

# Help function
show_help() {
    cat << EOF
GNN Processing Pipeline Runner

USAGE:
    ./run.sh [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -q, --quick             Run quick test (steps 0-3 only)
    -s, --steps STEPS       Run specific steps (comma-separated, e.g., "0,1,2,3")
    -k, --skip-steps STEPS  Skip specific steps (comma-separated)
    -t, --target-dir DIR    Target directory for GNN files (default: input/gnn_files)
    -o, --output-dir DIR    Output directory for results (default: output)
    --setup-only            Run only the setup step (step 1)
    --test-only             Run only the test step (step 2)
    --validate-only         Run validation steps only (steps 0-6)
    --render-only           Run rendering steps only (steps 0-3,11)
    --full-pipeline         Run the complete 24-step pipeline (default)
    --dry-run               Show what would be executed without running
    --check-deps            Check dependencies and environment
    --clean                 Clean output directories before running

EXAMPLES:
    ./run.sh                                    # Run full pipeline
    ./run.sh --verbose                          # Run with verbose output
    ./run.sh --quick                            # Quick test (steps 0-3)
    ./run.sh --steps "0,1,2,3" --verbose       # Run specific steps
    ./run.sh --skip-steps "15,16"               # Skip audio and analysis steps
    ./run.sh --validate-only                    # Run validation only
    ./run.sh --render-only                      # Run rendering only
    ./run.sh --dry-run                          # Show what would run
    ./run.sh --check-deps                       # Check environment

CONFIGURATION:
    The script uses the following default paths:
    - Project root: $PROJECT_ROOT
    - Source directory: $SRC_DIR
    - Main script: $MAIN_SCRIPT
    - Virtual environment: $VENV_DIR
    - Target directory: $DEFAULT_TARGET_DIR
    - Output directory: $DEFAULT_OUTPUT_DIR

    Configuration can be overridden via command-line options or by modifying
    the variables at the top of this script.

ENVIRONMENT:
    The script will automatically:
    1. Check for Python 3 installation
    2. Use virtual environment if available (.venv/)
    3. Fall back to system Python if no venv found
    4. Set appropriate environment variables

For more information, see:
- README.md: Project overview
- src/README.md: Pipeline documentation
- input/config.yaml: Configuration file
EOF
}

# Check if Python 3 is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Using Python $python_version"
}

# Check if UV is available and set up environment
check_uv() {
    if command -v uv &> /dev/null; then
        log_info "UV is available: $(uv --version 2>&1 | head -n1)"
        
        # Check if UV environment exists
        if [[ -d "$VENV_DIR" ]]; then
            if [[ -f "$VENV_DIR/bin/python" ]]; then
                PYTHON_EXECUTABLE="uv run python"
                log_info "Using UV environment: $VENV_DIR"
                return 0
            else
                log_warning "UV environment directory exists but is invalid"
            fi
        fi
        
        # Try to create UV environment
        log_info "Creating UV environment..."
        if uv venv --python 3.12; then
            PYTHON_EXECUTABLE="uv run python"
            log_info "UV environment created successfully"
            return 0
        else
            log_warning "Failed to create UV environment, falling back to system Python"
        fi
    else
        log_warning "UV not available, falling back to system Python"
    fi
    
    # Fallback to system Python
    log_info "Using system Python: $(which python3)"
    PYTHON_EXECUTABLE="python3"
}

# Check if virtual environment exists and is valid (legacy)
check_venv() {
    check_uv
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    check_python
    
    # Check virtual environment
    check_venv
    
    # Sync UV dependencies if using UV
    if [[ "$PYTHON_EXECUTABLE" == "uv run python" ]]; then
        log_info "Syncing UV dependencies..."
        if uv sync --extra dev; then
            log_success "UV dependencies synced successfully"
        else
            log_warning "Failed to sync UV dependencies, continuing anyway"
        fi
    fi
    
    # Check if main script exists
    if [[ ! -f "$MAIN_SCRIPT" ]]; then
        log_error "Main script not found: $MAIN_SCRIPT"
        exit 1
    fi
    
    # Check if target directory exists
    if [[ ! -d "$PROJECT_ROOT/$DEFAULT_TARGET_DIR" ]]; then
        log_warning "Target directory not found: $PROJECT_ROOT/$DEFAULT_TARGET_DIR"
        log_info "Creating target directory..."
        mkdir -p "$PROJECT_ROOT/$DEFAULT_TARGET_DIR"
    fi
    
    # Check if input files exist
    local gnn_files
    gnn_files=$(find "$PROJECT_ROOT/$DEFAULT_TARGET_DIR" -name "*.md" -o -name "*.gnn" 2>/dev/null | wc -l)
    if [[ $gnn_files -eq 0 ]]; then
        log_warning "No GNN files found in target directory"
        log_info "Add .md or .gnn files to $PROJECT_ROOT/$DEFAULT_TARGET_DIR"
    else
        log_info "Found $gnn_files GNN files in target directory"
    fi
    
    # Check for missing dependencies and provide guidance
    log_info "Checking for missing dependencies..."
    if $PYTHON_EXECUTABLE -c "
import sys
sys.path.insert(0, 'src')
try:
    from setup.dependency_installer import DependencyInstaller
    installer = DependencyInstaller('$PROJECT_ROOT', verbose=False)
    status = installer.check_dependencies()
    missing = [dep for dep, available in status.items() if not available]
    if missing:
        print(f'MISSING_DEPS: {missing}')
    else:
        print('ALL_DEPS_AVAILABLE')
except Exception as e:
    print(f'DEP_CHECK_ERROR: {e}')
" 2>/dev/null; then
        local dep_check_result
        dep_check_result=$($PYTHON_EXECUTABLE -c "
import sys
sys.path.insert(0, 'src')
from setup.dependency_installer import DependencyInstaller
installer = DependencyInstaller('$PROJECT_ROOT', verbose=False)
status = installer.check_dependencies()
missing = [dep for dep, available in status.items() if not available]
if missing:
    print(' '.join(missing))
else:
    print('none')
" 2>/dev/null)
        
        if [[ "$dep_check_result" != "none" ]]; then
            log_warning "Missing dependencies detected: $dep_check_result"
            log_info "The pipeline will attempt to install missing dependencies automatically"
            log_info "For manual installation, see the API key guidance below"
        fi
    fi
    
    # Create API key template if it doesn't exist
    create_api_key_template
    
    # Display API key guidance
    display_api_key_guidance
    
    log_success "Dependency check completed"
}

# Clean output directories
clean_output() {
    log_info "Cleaning output directories..."
    
    local dirs_to_clean=("$PROJECT_ROOT/output" "$PROJECT_ROOT/temp" "$PROJECT_ROOT/logs")
    
    for dir in "${dirs_to_clean[@]}"; do
        if [[ -d "$dir" ]]; then
            log_info "Cleaning $dir"
            rm -rf "$dir"/*
        fi
    done
    
    log_success "Output directories cleaned"
}

# Build command arguments
build_command() {
    local args=()
    
    # Add target directory
    args+=("--target-dir" "$PROJECT_ROOT/$DEFAULT_TARGET_DIR")
    
    # Add output directory
    args+=("--output-dir" "$PROJECT_ROOT/$DEFAULT_OUTPUT_DIR")
    
    # Add verbose flag if requested
    if [[ "$VERBOSE" == "true" ]]; then
        args+=("--verbose")
    fi
    
    # Add steps if specified
    if [[ -n "$STEPS" ]]; then
        args+=("--only-steps" "$STEPS")
    fi
    
    # Add skip steps if specified
    if [[ -n "$SKIP_STEPS" ]]; then
        args+=("--skip-steps" "$SKIP_STEPS")
    fi
    
    # Add pipeline summary file
    args+=("--pipeline-summary-file" "$PROJECT_ROOT/output/pipeline_execution_summary.json")
    
    echo "${args[@]}"
}

# Parse command line arguments
parse_args() {
    VERBOSE="$DEFAULT_VERBOSE"
    QUICK_MODE="$DEFAULT_QUICK_MODE"
    STEPS="$DEFAULT_STEPS"
    SKIP_STEPS="$DEFAULT_SKIP_STEPS"
    TARGET_DIR="$DEFAULT_TARGET_DIR"
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    DRY_RUN="false"
    CHECK_DEPS_ONLY="false"
    CLEAN_BEFORE_RUN="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -q|--quick)
                QUICK_MODE="true"
                STEPS="0,1,2,3"
                shift
                ;;
            -s|--steps)
                STEPS="$2"
                shift 2
                ;;
            -k|--skip-steps)
                SKIP_STEPS="$2"
                shift 2
                ;;
            -t|--target-dir)
                TARGET_DIR="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --setup-only)
                STEPS="1"
                shift
                ;;
            --test-only)
                STEPS="2"
                shift
                ;;
            --validate-only)
                STEPS="0,1,2,3,4,5,6"
                shift
                ;;
            --render-only)
                STEPS="0,1,2,3,11"
                shift
                ;;
            --full-pipeline)
                STEPS=""
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --check-deps)
                CHECK_DEPS_ONLY="true"
                shift
                ;;
            --clean)
                CLEAN_BEFORE_RUN="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Main execution function
main() {
    log_info "GNN Processing Pipeline Runner"
    log_info "================================"
    
    # Parse arguments
    parse_args "$@"
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Check dependencies
    check_dependencies
    
    # Exit if only checking dependencies
    if [[ "$CHECK_DEPS_ONLY" == "true" ]]; then
        exit 0
    fi
    
    # Clean output if requested
    if [[ "$CLEAN_BEFORE_RUN" == "true" ]]; then
        clean_output
    fi
    
    # Build command
    local cmd_args
    cmd_args=$(build_command)
    
    # Show what will be executed
    log_info "Command to execute:"
    echo "  $PYTHON_EXECUTABLE $MAIN_SCRIPT $cmd_args"
    echo
    
    # Exit if dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run completed - no actual execution"
        exit 0
    fi
    
    # Execute the pipeline
    log_info "Starting GNN Processing Pipeline..."
    log_info "Working directory: $(pwd)"
    echo
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    
    # Check if we should run individual steps or the full pipeline
    if [[ -n "$STEPS" ]]; then
        log_info "Running specific steps: $STEPS"
        # Run individual steps by calling the main script with step selection
        if $PYTHON_EXECUTABLE "$MAIN_SCRIPT" $cmd_args; then
            log_success "Selected steps completed successfully"
            exit 0
        else
            local exit_code=$?
            log_error "Selected steps failed with exit code $exit_code"
            exit $exit_code
        fi
    else
        log_info "Running full 24-step pipeline..."
        # Run the complete pipeline
        if $PYTHON_EXECUTABLE "$MAIN_SCRIPT" $cmd_args; then
            log_success "Full pipeline completed successfully"
            exit 0
        else
            local exit_code=$?
            log_error "Pipeline failed with exit code $exit_code"
            exit $exit_code
        fi
    fi
}

# Run main function with all arguments
main "$@"
