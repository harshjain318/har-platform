"""Experiment tracking: always writes files under results/<run_id>/, additionally logs to MLflow if installed."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.settings import RESULTS_DIR

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:   # pragma: no cover
    mlflow, HAS_MLFLOW = None, False


class Tracker:
    def __init__(self, name: str = "har_experiment", root: Optional[Path] = None, use_mlflow: bool = True):
        self.run_id = f"{name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.dir = Path(root or RESULTS_DIR) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow and HAS_MLFLOW
        if self.use_mlflow:
            mlflow.set_tracking_uri((Path(root or RESULTS_DIR) / "mlruns").resolve().as_uri())
            mlflow.set_experiment(name)
            mlflow.start_run(run_name=self.run_id)

    def log_params(self, params: Dict) -> None:
        (self.dir / "config.json").write_text(json.dumps(params, indent=2, default=str))
        if self.use_mlflow:
            flat = {k: (str(v)[:250]) for k, v in params.items()}
            mlflow.log_params(flat)

    def log_table(self, name: str, df: pd.DataFrame) -> Path:
        p = self.dir / f"{name}.csv"
        df.to_csv(p, index=False)
        if self.use_mlflow:
            mlflow.log_artifact(str(p))
        return p

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.use_mlflow:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v}, step=step)

    def end(self) -> None:
        if self.use_mlflow:
            mlflow.end_run()
