import pandas as pd
import pytest

from dataset_service import ChurnDatasetService, EmptyDatasetError
from model_service import evaluate_churn_model, train_churn_model
from schemas import TrainingConfigChurn


@pytest.fixture
def tiny_csv(tmp_path):
    rows = []
    for i in range(40):
        rows.append({
            "monthly_fee": 10 + i,
            "usage_hours": float(i % 20),
            "support_requests": i % 5,
            "account_age_months": i % 24,
            "failed_payments": i % 3,
            "region": ["europe", "asia", "america"][i % 3],
            "device_type": ["mobile", "desktop"][i % 2],
            "payment_method": "card",
            "autopay_enabled": i % 2,
            "churn": i % 2,
        })
    path = tmp_path / "churn.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_prepare_splits_features_and_target(tiny_csv):
    prepared = ChurnDatasetService(str(tiny_csv)).prepare_data()
    assert "churn" not in prepared.X.columns
    assert set(prepared.y.unique()) == {0, 1}
    assert len(prepared.X) == 40


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(EmptyDatasetError):
        ChurnDatasetService(str(path)).prepare_data()


def test_train_returns_metrics_between_zero_and_one(tiny_csv):
    service = ChurnDatasetService(str(tiny_csv))
    train_data, test_data = service.split_data(test_size=0.25, random_state=42)
    pipeline = train_churn_model(
        TrainingConfigChurn(
            model_type="logreg",
            hyperparameters={"max_iter": 200, "random_state": 42},
        ),
        train_data,
    )
    metrics = evaluate_churn_model(pipeline, test_data)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.f1 <= 1
    assert 0 <= metrics.roc_auc <= 1
