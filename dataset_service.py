from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from schemas import DatasetRowChurn, SplitInfoResponse
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = [
    "monthly_fee", 
    "usage_hours", 
    "support_requests", 
    "account_age_months", 
    "failed_payments", 
    "autopay_enabled",
]
CATEGORICAL_FEATURES = [
    "region", 
    "device_type", 
    "payment_method",
]
TARGET_COLUMN = "churn"
INTEGER_NUMERIC_FEATURES = [
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


class ChurnDatasetService:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        self.df = pd.read_csv(
            self.csv_path,
            na_values=["", "None", "nan", "null", "NA", "NaN"],
        )

        self._loaded = True

    def _clean_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()

        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

        for col in INTEGER_NUMERIC_FEATURES:
            df[col] = df[col].astype(int)

        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("unknown")

        y = df.pop(TARGET_COLUMN)
        y = pd.to_numeric(y, errors="coerce")

        mask = y.notna()
        df, y = df[mask], y[mask]
        y = y.astype(int)
        return df, y

    def preview(self, limit: int = 5) -> list[DatasetRowChurn]:
        self._ensure_loaded()
        if limit < 0:
            raise ValueError("limit must be non-negative")
        X, y = self._clean_dataframe(self.df)
        df = X.copy()
        df[TARGET_COLUMN] = y
        return [
            DatasetRowChurn(**row)
            for row in df.head(limit).to_dict(orient="records")
        ]

    def info(self) -> dict:
        self._ensure_loaded()
        feature_columns = [c for c in self.df.columns if c != TARGET_COLUMN]
        churn_distribution = {
            str(k): int(v)
            for k, v in self.df[TARGET_COLUMN].value_counts(dropna=False).items()
        }
        return {
            "rows_count": len(self.df),
            "columns_count": len(self.df.columns),
            "feature_names": feature_columns,
            "churn_distribution": churn_distribution,
        }

    def prepare_data(self) -> PreparedData:
        self._ensure_loaded()
        X, y = self._clean_dataframe(self.df)
        return PreparedData(
            X=X, 
            y=y, 
            numeric_features=NUMERIC_FEATURES, 
            categorical_features=CATEGORICAL_FEATURES,
        )

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
            PreparedData(
                X=X_train, 
                y=y_train, 
                numeric_features=prepared_data.numeric_features, 
                categorical_features=prepared_data.categorical_features,
            ),
            PreparedData(
                X=X_test, 
                y=y_test, 
                numeric_features=prepared_data.numeric_features, 
                categorical_features=prepared_data.categorical_features,
            ),
        )

    def split_info(self, test_size: float = 0.2, random_state: int = 42) -> SplitInfoResponse:
        _, X_train, X_test, y_train, y_test = self._split(test_size, random_state)
        return SplitInfoResponse(
            train_size=len(X_train),
            test_size=len(X_test),
            train_churn_distribution=self._churn_distribution(y_train),
            test_churn_distribution=self._churn_distribution(y_test),
        )