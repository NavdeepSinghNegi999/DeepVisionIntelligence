import logging
import os
from datetime import datetime


def setup_logger(log_name:str = "training", name: str = "project-1-visual_intelligence", log_dir: str = "logs", level: int = logging.INFO):
    """
    Set up logging to console + file.

    Args:
        name (str): Logger name
        log_dir (str): Directory to save log files
        level (int): Logging level

    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    log_file = os.path.join(log_dir,f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
