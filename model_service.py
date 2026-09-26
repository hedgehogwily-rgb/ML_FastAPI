from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_service import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PreparedData
from schemas import TrainMetricsResponse


def build_churn_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    if numeric_features is None:
        numeric_features = list(NUMERIC_FEATURES)
    if categorical_features is None:
        categorical_features = list(CATEGORICAL_FEATURES)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_churn_model(train_data: PreparedData) -> Pipeline:
    pipeline = build_churn_pipeline(
        numeric_features=train_data.numeric_features,
        categorical_features=train_data.categorical_features,
    )
    pipeline.fit(train_data.X, train_data.y)
    return pipeline


def evaluate_churn_model(pipeline: Pipeline, test_data: PreparedData) -> TrainMetricsResponse:
    y_pred = pipeline.predict(test_data.X)
    return TrainMetricsResponse(
        accuracy=float(accuracy_score(test_data.y, y_pred)),
        f1=float(f1_score(test_data.y, y_pred, zero_division=0)),
    )
