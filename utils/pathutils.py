"""Path utils for NLP project."""
import argparse
import datetime
from typing import Dict
from pathlib import Path

LOG_PATH = Path("./logs")

def create_dir_hierarchy() -> argparse.Namespace:
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_kwargs: Dict[str, Path] = {}
    log_dir = LOG_PATH / current_date
    
    dir_kwargs["log_dir"] = log_dir
    dir_kwargs["log_training"] = log_dir / "training"
    dir_kwargs["log_cv"] = log_dir / "cross_validation"
    dir_kwargs["log_inference"] = log_dir / "inference"

    return argparse.Namespace(**dir_kwargs)

