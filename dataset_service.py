import csv
from pathlib import Path

from schemas import DatasetRowChurn


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
        safe_limit = max(1, limit)
        return self.rows[:safe_limit]

    def info(self) -> dict:
        self._ensure_loaded()
        churn_distribution: dict[str, int] = {}
        for row in self.rows:
            churn_value = str(row["churn"])
            churn_distribution[churn_value] = churn_distribution.get(churn_value, 0) + 1

        return {
            "rows_count": len(self.rows),
            "columns_count": len(self.columns),
            "feature_names": self.columns,
            "churn_distribution": churn_distribution,
        }
