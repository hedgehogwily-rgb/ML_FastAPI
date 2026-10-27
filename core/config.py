import logging
from pathlib import Path

DATASET_PATH = "data/churn_dataset.csv"
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
INFO_PATH = MODELS_DIR / "churn_model_info.pkl"
METRICS_PATH = MODELS_DIR / "churn_model_metrics.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
