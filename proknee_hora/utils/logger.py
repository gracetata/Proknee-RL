"""Logging utilities for ProKnee-Hora."""

import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "proknee_hora",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup and configure logger.
    
    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path for logging.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "proknee_hora") -> logging.Logger:
    """Get existing logger or create a new one.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize training logger.
        
        Args:
            log_dir: Directory for logs.
            experiment_name: Name of the experiment.
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.logger = get_logger(f"proknee_hora.{experiment_name}")
        self.metrics_history = []
        
    def log_metrics(self, epoch: int, metrics: dict) -> None:
        """Log training metrics.
        
        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric names and values.
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")
        self.metrics_history.append({"epoch": epoch, **metrics})
        
    def log_checkpoint(self, checkpoint_path: str) -> None:
        """Log checkpoint save.
        
        Args:
            checkpoint_path: Path where checkpoint was saved.
        """
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
