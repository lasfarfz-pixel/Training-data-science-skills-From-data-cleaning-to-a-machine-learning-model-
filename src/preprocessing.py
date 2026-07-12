import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    """
    A production-ready data cleaning pipeline for sanitizing data, 
    handling missing values, and identifying statistical outliers.
    """
    def __init__(self, df: pd.DataFrame):
        # Work on a copy to prevent SettingWithCopyWarning
        self.df = df.copy()

    def remove_duplicates(self) -> pd.DataFrame:
        """Removes duplicate rows from the dataset."""
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        dropped_rows = initial_rows - len(self.df)
        logging.info(f"Removed {dropped_rows} duplicate rows.")
        return self.df

    def handle_missing_values(self, strategy: str = 'drop', target_col: str = None) -> pd.DataFrame:
        """
        Handles missing values based on a preferred strategy.
        Strategies: 'drop' or 'median'
        """
        if strategy == 'drop':
            self.df.dropna(subset=[target_col] if target_col else None, inplace=True)
            logging.info(f"Dropped rows with missing values in target column: {target_col}")
        elif strategy == 'median':
            for col in self.df.select_dtypes(include=[np.number]).columns:
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
            logging.info("Imputed missing numeric values with column medians.")
        return self.df

    def detect_iqr_outliers(self, column: str) -> tuple:
        """
        Calculates Upper and Lower bounds using the Interquartile Range (IQR) method.
        Returns a tuple of (lower_bound, upper_bound) and flags outliers.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
            
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        logging.info(f"Detected {len(outliers)} outliers in column '{column}' using IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}].")
        
        return lower_bound, upper_bound
