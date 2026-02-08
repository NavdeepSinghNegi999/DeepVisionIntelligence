# src/utils/run_utils.py
import os
from datetime import datetime

def create_run_dir(base_dir: str, prefix: str = "run"):
    """
    Create a unique run directory like:
    run-001-YYYY-MM-DD_HH-MM-SS
    """
    runs_dir = os.path.join(base_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(runs_dir)
        if d.startswith(prefix)
    ]
    run_id = len(existing) + 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{prefix}-{run_id:03d}-{timestamp}"

    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


