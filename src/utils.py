"""
Utility functions — logging, timing, memory profiling, config loading.
"""

import os
import time
import logging
import functools
from pathlib import Path

import yaml
import psutil
import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "iot_ids", log_dir: str = "results/logs") -> logging.Logger:
    """Create a configured logger that writes to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "experiment.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logger()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    """Load the YAML experiment configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timed(func):
    """Decorator that logs the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------

def log_memory_usage(tag: str = "") -> dict:
    """Log current memory usage and return stats dict."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    stats = {
        "rss_mb": mem_info.rss / (1024 ** 2),
        "vms_mb": mem_info.vms / (1024 ** 2),
        "percent": process.memory_percent(),
    }
    logger.info(
        f"Memory [{tag}]: RSS={stats['rss_mb']:.1f} MB, "
        f"VMS={stats['vms_mb']:.1f} MB, {stats['percent']:.1f}%"
    )
    return stats


def dataframe_memory_mb(df) -> float:
    """Return memory usage of a DataFrame in MB."""
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    return round(mem_mb, 2)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dirs(config: dict):
    """Create all output directories from the config."""
    for key, path in config.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Ensured directory: {path}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}")
