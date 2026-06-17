import csv
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from schemas import DatasetRowChurn, SplitInfoResponse
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = ["monthly_fee", "usage_hours", "support_requests", "account_age_months", "failed_payments", "autopay_enabled"]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method"]
TARGET_COLUMN = "churn"


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


class ChurnDatasetService:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.rows: list[dict] = []
        self.columns: list[str] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            self.columns = reader.fieldnames or []

            parsed_rows: list[dict] = []
            for row in reader:
                parsed_row = DatasetRowChurn(
                    monthly_fee=float(row["monthly_fee"]),
                    usage_hours=float(row["usage_hours"]),
                    support_requests=int(row["support_requests"]),
                    account_age_months=int(row["account_age_months"]),
                    failed_payments=int(row["failed_payments"]),
                    region=row["region"],
                    device_type=row["device_type"],
                    payment_method=row["payment_method"],
                    autopay_enabled=int(row["autopay_enabled"]),
                    churn=int(row["churn"]),
                )
                parsed_rows.append(parsed_row.model_dump())

            self.rows = parsed_rows
            self._loaded = True

    def preview(self, limit: int = 5) -> list[dict]:
        self._ensure_loaded()
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return self.rows[:limit]

    def info(self) -> dict:
        self._ensure_loaded()
        churn_distribution: dict[str, int] = {}
        feature_columns = [column for column in self.columns if column != TARGET_COLUMN]
        for row in self.rows:
            churn_value = str(row["churn"])
            churn_distribution[churn_value] = churn_distribution.get(churn_value, 0) + 1

        return {
            "rows_count": len(self.rows),
            "columns_count": len(self.columns),
            "feature_names": feature_columns,
            "churn_distribution": churn_distribution,
        }

    def prepare_data(self) -> PreparedData:
        self._ensure_loaded()
        X = pd.DataFrame(self.rows)
        y = X.pop(TARGET_COLUMN)
        for col in NUMERIC_FEATURES:
            X[col] = X[col].fillna(X[col].median())
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].fillna("unknown")
        return PreparedData(X=X, y=y, numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES)

    @staticmethod
    def _churn_distribution(y: pd.Series) -> dict[str, float]:
        return {str(k): round(v, 4) for k, v in y.value_counts(normalize=True).items()}

    def _split(
        self, test_size: float, random_state: int
    ) -> tuple[PreparedData, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        prepared_data = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(
            prepared_data.X,
            prepared_data.y,
            test_size=test_size,
            random_state=random_state,
            stratify=prepared_data.y,
        )
        return prepared_data, X_train, X_test, y_train, y_test

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[PreparedData, PreparedData]:
        prepared_data, X_train, X_test, y_train, y_test = self._split(test_size, random_state)
        return (
            PreparedData(X=X_train, y=y_train, numeric_features=prepared_data.numeric_features, categorical_features=prepared_data.categorical_features),
            PreparedData(X=X_test, y=y_test, numeric_features=prepared_data.numeric_features, categorical_features=prepared_data.categorical_features),
        )

    def split_info(self, test_size: float = 0.2, random_state: int = 42) -> SplitInfoResponse:
        _, X_train, X_test, y_train, y_test = self._split(test_size, random_state)
        return SplitInfoResponse(
            train_size=len(X_train),
            test_size=len(X_test),
            train_churn_distribution=self._churn_distribution(y_train),
            test_churn_distribution=self._churn_distribution(y_test),
        )