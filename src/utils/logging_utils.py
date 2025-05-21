import logging
import sys

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = None # Uses logging default: YYYY-MM-DD HH:MM:SS,ms

class LevelFilter(logging.Filter):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def filter(self, record):
        return self.low <= record.levelno <= self.high

def setup_standalone_logging(level=logging.INFO, logger_name=None):
    '''
    Configures basic logging for standalone script execution, splitting output
    to stdout (INFO, DEBUG) and stderr (WARNING, ERROR, CRITICAL).

    This function should be called only if no root handlers are already configured,
    typically within an 'if __name__ == "__main__":' block of a script that
    can also be run as part of a larger pipeline (which would configure logging).

    Args:
        level: The minimum logging level to set for the root logger and handlers
               (e.g., logging.INFO, logging.DEBUG).
        logger_name: Optional. If provided, also get and set the level for this specific logger.
                     Otherwise, only basicConfig is called for the root logger.
    '''
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        root_logger.setLevel(level) # Set root logger level

        # Handler for INFO and DEBUG to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG) # Capture from DEBUG up to INFO
        stdout_handler.addFilter(LevelFilter(logging.DEBUG, logging.INFO))
        stdout_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        stdout_handler.setFormatter(stdout_formatter)
        root_logger.addHandler(stdout_handler)

        # Handler for WARNING and above to stderr
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING) # Capture from WARNING up
        stderr_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        stderr_handler.setFormatter(stderr_formatter)
        root_logger.addHandler(stderr_handler)
        
        config_logger_name = "StandaloneLoggingSetup"
        if logger_name:
            config_logger_name = f"{logger_name}.StandaloneLoggingSetup"
        
        setup_logger = logging.getLogger(config_logger_name)
        # This message will go to stdout if level is INFO/DEBUG, or stderr if WARNING+
        # For initial setup, INFO is typical, so it should go to stdout.
        setup_logger.info(f"Dual-stream logging (stdout/stderr) configured by setup_standalone_logging at root level {logging.getLevelName(level)}.")

    if logger_name:
        script_logger = logging.getLogger(logger_name)
        script_logger.setLevel(level)
        # This debug message will go to stdout if level is DEBUG
        script_logger.debug(f"Logger '{logger_name}' level set to {logging.getLevelName(level)} by setup_standalone_logging.") 