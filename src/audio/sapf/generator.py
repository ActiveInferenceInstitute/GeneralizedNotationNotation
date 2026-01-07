from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker

# Import SAPF functionality
try:
    from .audio_generators import SAPFAudioGenerator
    SAPF_AVAILABLE = True
except ImportError as e:
    SAPF_AVAILABLE = False

def generate_sapf_audio(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Generate SAPF (Sound As Pure Form) audio from GNN models.
    
    Args:
        target_dir: Directory containing GNN files to generate audio for
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional audio generation options (e.g., duration)
        
    Returns:
        True if audio generation succeeded, False otherwise
    """
    log_step_start(logger, "Generating SAPF audio from GNN models")
    
    # Use centralized output directory configuration
    sapf_output_dir = get_output_dir_for_script("15_audio.py", output_dir)
    sapf_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not SAPF_AVAILABLE:
        log_step_error(logger, "SAPF audio generation not available")
        return False
    
    # Get duration from kwargs or use default
    duration = kwargs.get('duration', 30)
    
    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir}")
        return True
    
    logger.info(f"Found {len(gnn_files)} GNN files to generate audio for")
    
    successful_generations = 0
    failed_generations = 0
    
    try:
        # Initialize SAPF audio generator
        audio_generator = SAPFAudioGenerator()
        
        with performance_tracker.track_operation("generate_sapf_audio"):
            for gnn_file in gnn_files:
                try:
                    logger.debug(f"Generating audio for file: {gnn_file}")
                    
                    # Read GNN file content
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        gnn_content = f.read()
                    
                    # Create file-specific output directory
                    file_output_dir = sapf_output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate audio
                    audio_file = file_output_dir / f"{gnn_file.stem}_sapf.wav"
                    
                    with performance_tracker.track_operation(f"generate_audio_{gnn_file.name}"):
                        success = audio_generator.generate_audio(
                            gnn_content=gnn_content,
                            output_file=str(audio_file),
                            duration=duration
                        )
                    
                    if success:
                        successful_generations += 1
                        logger.debug(f"Audio generated successfully for {gnn_file.name}: {audio_file}")
                    else:
                        failed_generations += 1
                        log_step_warning(logger, f"Audio generation failed for {gnn_file.name}")
                        
                except Exception as e:
                    failed_generations += 1
                    log_step_error(logger, f"Failed to generate audio for {gnn_file.name}: {e}")
        
        # Log results summary
        total_files = len(gnn_files)
        if successful_generations == total_files:
            log_step_success(logger, f"All {total_files} files generated audio successfully")
            return True
        elif successful_generations > 0:
            log_step_warning(logger, f"Partial success: {successful_generations}/{total_files} files generated audio successfully")
            return True
        else:
            log_step_error(logger, "No audio files were generated successfully")
            return False
        
    except Exception as e:
        log_step_error(logger, f"SAPF audio generation failed: {e}")
        return False 