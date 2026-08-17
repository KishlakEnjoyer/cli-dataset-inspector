from pathlib import Path
import shutil
import pandas as pd

from .loader import load_dataset

class Inspector:
    def __init__(self,
        path: str,
        target: str,
        task: str,
        jupyter: bool,
        baseline: bool,
        json: bool,
        output: str
    ):
        self.path = path

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
        """
        Inspecting pipeline
        """
        try:
            df = load_dataset(path=self.path)
            return {
                "shape": self.get_shape(df=df),
                "duplicates": self.get_duplicates(df=df),
                "missing": self.get_missing(df=df),
                "memory": self.get_memory(df=df)
            }
        except Exception:
            self.rollback()
            raise

    def get_shape(self, df: pd.DataFrame):
        return {
            "rows": len(df),
            "columns": len(df.columns),
        }

    def get_duplicates(self, df: pd.DataFrame):
        return df.duplicated().sum()

    def get_missing(self, df: pd.DataFrame):
        return df.isna().sum()

    def get_memory(self, df: pd.DataFrame):
        return df.memory_usage(deep=True).sum()