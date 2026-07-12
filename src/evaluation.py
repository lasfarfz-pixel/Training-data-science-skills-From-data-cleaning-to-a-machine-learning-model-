import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class InformationTheoryMetrics:
    """
    Computes baseline information-theoretic and class impurity metrics
    prior to classification modeling.
    """
    @staticmethod
    def calculate_probabilities(target: pd.Series) -> np.ndarray:
        """Calculates deterministic class probabilities."""
        counts = target.value_counts()
        probabilities = counts.values / len(target)
        return probabilities

    @classmethod
    def shannon_entropy(cls, target: pd.Series) -> float:
        """Computes Shannon Entropy: H(X) = -sum(p_i * log2(p_i))"""
        probs = cls.calculate_probabilities(target)
        # Avoid log(0) by filtering out 0 probabilities
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    @classmethod
    def gini_impurity(cls, target: pd.Series) -> float:
        """Computes Gini Impurity: Gini = 1 - sum(p_i^2)"""
        probs = cls.calculate_probabilities(target)
        return float(1.0 - np.sum(probs ** 2))

    @classmethod
    def baseline_misclassification_error(cls, target: pd.Series) -> float:
        """Computes the maximum-voting floor error: 1 - max(p_i)"""
        probs = cls.calculate_probabilities(target)
        return float(1.0 - np.max(probs))


class RegressionEvaluator:
    """
    Automates the evaluation and benchmarking matrix for regression models.
    """
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_metrics_report(self, model_name: str) -> dict:
        """Generates a structured dictionary of core performance metrics."""
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        r2 = r2_score(self.y_true, self.y_pred)
        
        return {
            "Model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 6)
        }
