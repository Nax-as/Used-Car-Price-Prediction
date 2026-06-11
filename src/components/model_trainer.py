"""
Model Trainer Component
------------------------
Trains, compares, tunes, and persists the best regression model.
All eight candidate models are evaluated; the best by R² score is then
fine-tuned with GridSearchCV before being saved as artifacts/model.pkl.
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor)
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.logger import logger
from src.utils import save_object, compare_models, evaluate_model


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Train and select the best regression model."""

    def __init__(self):
        self.config = ModelTrainerConfig()

    # ------------------------------------------------------------------
    # Candidate models
    # ------------------------------------------------------------------
    @staticmethod
    def _get_models() -> dict:
        return {
            "Linear Regression":        LinearRegression(),
            "Ridge Regression":         Ridge(),
            "Lasso Regression":         Lasso(max_iter=10_000),
            "Decision Tree":            DecisionTreeRegressor(random_state=42),
            "Random Forest":            RandomForestRegressor(n_estimators=100,
                                                              random_state=42,
                                                              n_jobs=-1),
            "Extra Trees":              ExtraTreesRegressor(n_estimators=100,
                                                            random_state=42,
                                                            n_jobs=-1),
            "Gradient Boosting":        GradientBoostingRegressor(random_state=42),
            "XGBoost":                  XGBRegressor(random_state=42,
                                                     verbosity=0,
                                                     n_jobs=-1),
        }

    # ------------------------------------------------------------------
    # Hyperparameter grids (only for top-3 ensemble models)
    # ------------------------------------------------------------------
    PARAM_GRIDS = {
        "Random Forest": {
            "n_estimators":      [100, 200],
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "Extra Trees": {
            "n_estimators":      [100, 200],
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "XGBoost": {
            "n_estimators":  [100, 200],
            "max_depth":     [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample":     [0.8, 1.0],
        },
        "Gradient Boosting": {
            "n_estimators":  [100, 200],
            "max_depth":     [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    }

    # ------------------------------------------------------------------
    def initiate_model_training(self,
                                X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray):
        """
        1. Compare all models on default hyperparameters.
        2. Tune the winner if a grid exists for it.
        3. Persist the best model.

        Returns
        -------
        Tuple[dict, dict]
            (results_table, best_model_metrics)
        """
        logger.info("Starting model training and comparison.")

        # ── Step 1: baseline comparison ───────────────────────────────
        results = compare_models(self._get_models(),
                                 X_train, y_train, X_test, y_test)

        results_df = pd.DataFrame(
            {k: {m: v for m, v in vals.items() if m != "model"}
             for k, vals in results.items()}
        ).T.sort_values("R2 Score", ascending=False)

        print("\n===== Model Comparison (log-scale target) =====")
        print(results_df.to_string())

        best_name = results_df.index[0]
        logger.info(f"Best baseline model: {best_name}")

        # ── Step 2: hyperparameter tuning ─────────────────────────────
        if best_name in self.PARAM_GRIDS:
            logger.info(f"Tuning {best_name} with GridSearchCV …")
            base_model = results[best_name]["model"]
            grid = GridSearchCV(
                base_model, self.PARAM_GRIDS[best_name],
                cv=3, scoring="r2", n_jobs=-1, verbose=1,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")
        else:
            best_model = results[best_name]["model"]

        # ── Step 3: final evaluation ──────────────────────────────────
        y_pred       = best_model.predict(X_test)
        final_metrics = evaluate_model(y_test, y_pred)
        logger.info(f"Final model metrics: {final_metrics}")

        if final_metrics["R2 Score"] < 0.60:
            raise ValueError(
                f"Best model R² ({final_metrics['R2 Score']}) is below threshold."
            )

        save_object(self.config.model_path, best_model)
        logger.info(f"Best model saved to {self.config.model_path}")

        return results_df.to_dict(), final_metrics
