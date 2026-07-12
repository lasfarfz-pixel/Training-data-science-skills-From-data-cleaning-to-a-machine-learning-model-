import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class InformationTheoryMetrics:
    @staticmethod
    def calculate_probabilities(target: pd.Series) -> np.ndarray:
        return target.value_counts().values / len(target)

    @classmethod
    def shannon_entropy(cls, target: pd.Series) -> float:
        probs = cls.calculate_probabilities(target)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    @classmethod
    def gini_impurity(cls, target: pd.Series) -> float:
        probs = cls.calculate_probabilities(target)
        return float(1.0 - np.sum(probs ** 2))


class RegressionEvaluator:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_metrics_report(self, model_name: str) -> dict:
        return {
            "Model": model_name,
            "MAE": round(mean_absolute_error(self.y_true, self.y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(self.y_true, self.y_pred)), 4),
            "R2_Score": round(r2_score(self.y_true, self.y_pred), 6)
        }
