import logging

# Define a logging wrapper class
class Logger:
    def __init__(self, name="ApplicationLogger", level=logging.INFO):
        """
        Initialize the logger with a name and level.
        :param name: Name of the logger.
        :param level: Logging level (e.g., INFO, ERROR).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Define a basic console handler and formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def success(self, message):
        """Custom success log (between INFO and WARNING)."""
        success_level = 25  # Custom log level
        logging.addLevelName(success_level, "SUCCESS")
        self.logger.log(success_level, message)