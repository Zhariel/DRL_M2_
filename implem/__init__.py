import logging
import os

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a log directory if it doesn't exist
log_dir = os.path.join("..", "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a file handler to log to a file
file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
file_handler.setLevel(logging.DEBUG)

# Create a formatter for the log entries
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
