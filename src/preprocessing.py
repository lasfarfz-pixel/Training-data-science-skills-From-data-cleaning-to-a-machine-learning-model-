import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_duplicates(self) -> pd.DataFrame:
        self.df.drop_duplicates(inplace=True)
        return self.df

    def handle_missing_values(self, strategy='drop', target_col=None) -> pd.DataFrame:
        if strategy == 'drop':
            self.df.dropna(subset=[target_col] if target_col else None, inplace=True)
        return self.df

    def detect_iqr_outliers(self, column: str) -> tuple:
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        return (q1 - 1.5 * iqr), (q3 + 1.5 * iqr)
