"""
Utility functions: model persistence, evaluation metrics, and model comparison.
"""
import os
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger import logger


def save_object(file_path: str, obj) -> None:
    """Serialise any Python object to a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save object: {e}")
        raise


def load_object(file_path: str):
    """Deserialise a pickle file."""
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load object: {e}")
        raise


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of regression evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"R2 Score": round(r2, 4), "MAE": round(mae, 2),
            "MSE": round(mse, 2), "RMSE": round(rmse, 2)}


def compare_models(models: dict, X_train, y_train, X_test, y_test) -> dict:
    """
    Fit each model, evaluate on test set, and return a results dict.
    models: {name: estimator}
    """
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
            results[name] = {"model": model, **metrics}
            logger.info(f"{name} → R²={metrics['R2 Score']}  RMSE={metrics['RMSE']}")
        except Exception as e:
            logger.error(f"Model {name} failed: {e}")
    return results
