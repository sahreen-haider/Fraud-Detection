import logging

# Create a logger
logger = logging.getLogger("model_logger")
logger.setLevel(logging.DEBUG)  # Capture all log levels

# Define log format
formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(module)s - %(funcName)s - %(message)s")

# Create a console handler (only logs INFO and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console will show INFO and above
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# Function to set log file path dynamically
def set_log_file(file_path="model.log"):
    """Changes the log file path dynamically and logs everything in the file."""
    # Remove existing file handlers if any
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Create a new file handler (logs everything)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)  # Logs everything to file
    file_handler.setFormatter(formatter)

    # Add the new file handler
    logger.addHandler(file_handler)
    logger.info(f"Log file set to: {file_path}")
