from pathlib import Path
import shutil
from typing import Annotated
import pandas as pd
import typer

from .loader import load_dataset

class Inspector:
    def __init__(self,
        path: str,
        target: str,
        task: str,
        baseline: bool,
        json: bool,
        output: str,
        jupyter: Annotated[
            bool | None,
            typer.Option(
                "--jupyter/--no-jupyter",
                "-j/-nj",
                help="Generate Jupyter notebook"
            )
        ] = None
    ):
        self.path = path
        self.target = target
        self.task = task
        self.jupyter = jupyter
        self.baseline = baseline
        self.json = json
        self.output = output

    def rollback(self):
        """
        Remove partially generated report
        if generation failed.
        """
        if self.path.exists():
            shutil.rmtree(
                self.path,
                ignore_errors=True,
            )

    def inspect(self):
        try:
            df = load_dataset(path=self.path)

            target_summary = (
                self.get_target_summary(df)
                if self.target
                else None
            )

            return {
                "shape": self.get_shape(df),
                "duplicates": self.get_duplicates(df),
                "missing": self.get_missing(df),
                "memory": self.get_memory(df),
                "columns_summary": self.get_columns_summary(df),
                "numeric_summary": self.get_numeric_summary(df),
                "categorical_summary": self.get_categorical_summary(df),
                "target": target_summary,
                "warnings": self.get_warnings(
                    df,
                    target_summary,
                ),
            }

        except Exception:
            self.rollback()
            raise

    def get_task_type(self, df: pd.DataFrame):
        series = df[self.target].dropna()

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or series.nunique() <= 20
        ):
            return "classification"

        return "regression"

    def get_shape(self, df: pd.DataFrame):
        return {
            "rows": len(df),
            "columns": len(df.columns),
        }

    def get_duplicates(self, df: pd.DataFrame):
        duplicates = df.duplicated().sum()
        rows = len(df)

        return {
            "count": duplicates,
            "percent": duplicates / rows * 100 if rows else 0,
        }

    def get_missing(self, df: pd.DataFrame):
        return df.isna().sum()

    def get_memory(self, df: pd.DataFrame):
        return df.memory_usage(deep=True).sum()

    def get_columns_summary(self, df: pd.DataFrame):
        rows = len(df)

        summary = []

        for column in df.columns:
            filled = df[column].notna().sum()
            missing = df[column].isna().sum()
            unique = df[column].nunique(dropna=True)

            missing_percent = (
                missing / rows * 100
                if rows > 0 else 0
            )

            unique_percent = (
                unique / filled * 100
                if filled > 0 else 0
            )

            summary.append({
                "name": column,
                "dtype": str(df[column].dtype),
                "filled": filled,
                "missing": missing,
                "missing_percent": missing_percent,
                "unique": unique,
                "unique_percent": unique_percent,
            })

        return summary

    def get_numeric_summary(self, df: pd.DataFrame):
        numeric_df = df.select_dtypes(include="number")

        summary = []

        for column in numeric_df.columns:
            series = numeric_df[column].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (
                (series < lower_bound) |
                (series > upper_bound)
            ).sum()

            summary.append({
                "name": column,
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "outliers": outliers,
            })

        return summary

    def get_categorical_summary(self, df: pd.DataFrame):
        categorical_df = df.select_dtypes(
            include=["object", "string", "category", "bool"]
        )

        summary = []

        for column in categorical_df.columns:
            series = df[column].dropna()

            if series.empty:
                continue

            value_counts = series.value_counts()

            most_frequent = value_counts.index[0]
            most_frequent_count = value_counts.iloc[0]

            top_values = value_counts.head(3)

            summary.append({
                "name": column,
                "categories": series.nunique(),
                "most_frequent": str(most_frequent),
                "most_frequent_count": int(most_frequent_count),
                "top_values": {
                    str(value): int(count)
                    for value, count in top_values.items()
                },
            })

        return summary

    def get_target_summary(self, df: pd.DataFrame):
        if self.target is None:
            return None

        if self.target not in df.columns:
            raise ValueError(
                f"Target column '{self.target}' not found"
            )

        series = df[self.target]

        missing = series.isna().sum()
        missing_percent = (
            missing / len(df) * 100
            if len(df) > 0 else 0
        )

        task = self.get_task_type(df)

        result = {
            "name": self.target,
            "task": task,
            "missing": int(missing),
            "missing_percent": missing_percent,
        }

        if task == "classification":
            clean_series = series.dropna()

            counts = clean_series.value_counts()

            distribution = []

            for value, count in counts.items():
                distribution.append({
                    "class": str(value),
                    "count": int(count),
                    "percent": (
                        count / len(clean_series) * 100
                        if len(clean_series) > 0 else 0
                    ),
                })

            result["distribution"] = distribution

            if len(counts) > 1:
                min_count = counts.min()
                max_count = counts.max()

                result["imbalance_ratio"] = (
                    min_count / max_count
                    if max_count > 0 else 0
                )
            else:
                result["imbalance_ratio"] = 0

        else:
            clean_series = series.dropna()

            result["statistics"] = {
                "min": clean_series.min(),
                "max": clean_series.max(),
                "mean": clean_series.mean(),
                "median": clean_series.median(),
                "std": clean_series.std(),
            }

        return result

    def get_warnings(
        self,
        df: pd.DataFrame,
        target_summary=None,
    ):
        warnings = []

        rows = len(df)

        if rows == 0:
            warnings.append({
                "type": "empty_dataset",
                "column": None,
                "message": "Dataset contains no rows.",
            })

            return warnings

        for column in df.columns:
            series = df[column]

            # Completely empty
            if series.isna().all():
                warnings.append({
                    "type": "empty_column",
                    "column": column,
                    "message": (
                        f"Column '{column}' is completely empty."
                    ),
                })

                continue

            # Missing values
            missing_percent = (
                series.isna().sum() / rows * 100
            )

            if missing_percent >= 30:
                warnings.append({
                    "type": "high_missing",
                    "column": column,
                    "message": (
                        f"Column '{column}' contains "
                        f"{missing_percent:.2f}% missing values."
                    ),
                })

            # Unique values
            unique = series.nunique(dropna=True)
            filled = series.notna().sum()

            # Constant column
            if unique == 1:
                warnings.append({
                    "type": "constant",
                    "column": column,
                    "message": (
                        f"Column '{column}' contains "
                        "only one unique value."
                    ),
                })

            # Possible ID
            unique_ratio = (
                unique / filled
                if filled > 0 else 0
            )

            if (
                unique_ratio >= 0.95
                and unique > 20
            ):
                warnings.append({
                    "type": "possible_id",
                    "column": column,
                    "message": (
                        f"Column '{column}' is "
                        f"{unique_ratio * 100:.2f}% unique "
                        "and may represent an identifier."
                    ),
                })

            # Suspicious object types
            if series.dtype == "object":
                clean = series.dropna()

                python_types = clean.map(
                    lambda value: type(value).__name__
                ).unique()

                if len(python_types) > 1:
                    warnings.append({
                        "type": "mixed_types",
                        "column": column,
                        "message": (
                            f"Column '{column}' contains "
                            "mixed value types."
                        ),
                    })

                numeric = pd.to_numeric(
                    clean,
                    errors="coerce",
                )

                numeric_ratio = numeric.notna().mean()

                if 0.2 < numeric_ratio < 0.8:
                    warnings.append({
                        "type": "suspicious_type",
                        "column": column,
                        "message": (
                            f"Column '{column}' contains "
                            "a mixture of numeric and "
                            "non-numeric values."
                        ),
                    })

        # Duplicates
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            duplicate_percent = (
                duplicates / rows * 100
            )

            warnings.append({
                "type": "duplicates",
                "column": None,
                "message": (
                    f"Dataset contains {duplicates} "
                    f"duplicated rows "
                    f"({duplicate_percent:.2f}%)."
                ),
            })

        # Target imbalance
        if (
            target_summary
            and target_summary["task"] == "classification"
        ):
            ratio = target_summary["imbalance_ratio"]

            if ratio < 0.25:
                warnings.append({
                    "type": "target_imbalance",
                    "column": self.target,
                    "message": (
                        f"Target '{self.target}' has "
                        "a strong class imbalance "
                        f"(ratio: {ratio:.2f})."
                    ),
                })

        return warnings