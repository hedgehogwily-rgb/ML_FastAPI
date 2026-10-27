import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
import ml.pipeline as model_service

CLIENT = {
    "monthly_fee": 9.99,
    "usage_hours": 27.92,
    "support_requests": 1,
    "account_age_months": 14,
    "failed_payments": 1,
    "region": "america",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1,
}


@pytest.fixture
def client(tmp_path):
    models = Path("models")
    backup = tmp_path / "models_backup"
    had_models = models.exists()
    if had_models:
        shutil.copytree(models, backup)
    yield TestClient(main.app)
    if models.exists():
        shutil.rmtree(models)
    if had_models:
        shutil.copytree(backup, models)
    main.load_saved_model()


def test_train_status_and_predict(client):
    info = client.get("/dataset/info")
    assert info.status_code == 200
    assert info.json()["rows_count"] > 0

    trained = client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"max_iter": 300, "random_state": 42},
        },
    )
    assert trained.status_code == 200
    assert {"accuracy", "f1", "roc_auc"} <= trained.json().keys()

    status = client.get("/model/status")
    assert status.status_code == 200
    assert status.json()["is_trained"] is True
    assert status.json()["model_type"] == "logreg"

    predicted = client.post("/predict", json=CLIENT)
    assert predicted.status_code == 200
    assert predicted.json()["prediction"] in (0, 1)
    assert set(predicted.json()["probabilities"]) == {"0", "1"}


def test_predict_without_model(client):
    model_service._clear_loaded_model()
    response = client.post("/predict", json=CLIENT)
    assert response.status_code == 400
    assert response.json()["code"] == "model_not_trained"
