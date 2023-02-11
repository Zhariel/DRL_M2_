import logging
import os

def write_log(filename, message):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Open a file in the log directory for writing
    file_path = os.path.join("logs", filename)
    with open(file_path, "w") as log_file:
        log_file.write(message)