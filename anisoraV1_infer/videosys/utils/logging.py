import logging

import torch.distributed as dist
from rich.logging import RichHandler


def create_logger():
    """
    Create a logger that writes to a log file and stdout.
    """
    logger = logging.getLogger(__name__)
    return logger


def init_dist_logger():
    """
    Update the logger to write to a log file.
    """
    global logger
    if dist.get_rank() == 0:
        logger = logging.getLogger(__name__)
        handler = RichHandler(show_path=False, markup=True, rich_tracebacks=True)
        formatter = logging.Formatter("VideoSys - %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

logger = create_logger()
