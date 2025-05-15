import logging
import sys

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = None # Uses logging default: YYYY-MM-DD HH:MM:SS,ms

def setup_standalone_logging(level=logging.INFO, logger_name=None):
    '''
    Configures basic logging for standalone script execution.

    This function should be called only if no root handlers are already configured,
    typically within an 'if __name__ == "__main__":' block of a script that
    can also be run as part of a larger pipeline (which would configure logging).

    Args:
        level: The logging level to set (e.g., logging.INFO, logging.DEBUG).
        logger_name: Optional. If provided, also get and set the level for this specific logger.
                     Otherwise, only basicConfig is called for the root logger.
    '''
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=level,
            format=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
            stream=sys.stdout
        )
        # Get root logger to log that basicConfig was called by this utility
        # This helps in tracing how logging was set up.
        # Using a more specific logger name here to avoid confusion with the script's own __name__ logger
        # if logger_name is the same as __name__ when __name__ is '__main__'.
        config_logger_name = "StandaloneLoggingSetup"
        if logger_name:
            # e.g. MyScript.__main__.StandaloneLoggingSetup
            config_logger_name = f"{logger_name}.StandaloneLoggingSetup"
        
        setup_logger = logging.getLogger(config_logger_name)
        setup_logger.info(f"Basic logging configured by setup_standalone_logging at level {logging.getLevelName(level)}.")

    # If a specific logger name is given, also set its level.
    # This is useful if the script's main logger (e.g., logging.getLogger(__name__))
    # needs its level set independently after basicConfig (or if basicConfig was skipped
    # because handlers already existed).
    if logger_name:
        script_logger = logging.getLogger(logger_name)
        script_logger.setLevel(level)
        script_logger.debug(f"Logger '{logger_name}' level set to {logging.getLevelName(level)} by setup_standalone_logging.") 