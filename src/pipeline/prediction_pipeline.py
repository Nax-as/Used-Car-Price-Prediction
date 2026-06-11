"""
Prediction Pipeline
-------------------
Accepts raw user inputs from the web form, preprocesses them using the
saved ColumnTransformer, and returns a price prediction (in INR).
"""
import numpy as np
import pandas as pd

from src.utils import load_object
from src.logger import logger


class PredictPipeline:
    """Load artifacts and serve a single prediction."""

    MODEL_PATH       = "artifacts/model.pkl"
    PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

    def __init__(self):
        self.model       = load_object(self.MODEL_PATH)
        self.preprocessor = load_object(self.PREPROCESSOR_PATH)

    def predict(self, features: pd.DataFrame) -> float:
        """
        Parameters
        ----------
        features : pd.DataFrame
            Single-row DataFrame with columns matching the training schema.

        Returns
        -------
        float
            Predicted selling price in INR (reversed log1p).
        """
        try:
            transformed = self.preprocessor.transform(features)
            log_pred    = self.model.predict(transformed)[0]
            price       = np.expm1(log_pred)          # reverse log1p
            return round(float(price), 2)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


class CarData:
    """
    Value object that maps raw HTML form values to a clean DataFrame
    ready for the preprocessor.
    """

    def __init__(self, brand: str, model: str, year: int,
                 km_driven: int, fuel: str, seller_type: str,
                 transmission: str, owner: str,
                 mileage: float, engine: float,
                 max_power: float, seats: int):

        self.brand        = brand
        self.model        = model
        self.year         = int(year)
        self.km_driven    = int(km_driven)
        self.fuel         = fuel
        self.seller_type  = seller_type
        self.transmission = transmission
        self.owner        = owner
        self.mileage      = float(mileage)
        self.engine       = float(engine)
        self.max_power    = float(max_power)
        self.seats        = int(seats)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame matching the preprocessor schema."""
        car_age = 2024 - self.year
        return pd.DataFrame([{
            "km_driven":    self.km_driven,
            "car_age":      car_age,
            "mileage_kmpl": self.mileage,
            "engine_cc":    self.engine,
            "max_power_bhp": self.max_power,
            "seats":        self.seats,
            "brand":        self.brand,
            "model":        self.model,
            "fuel":         self.fuel,
            "seller_type":  self.seller_type,
            "transmission": self.transmission,
            "owner":        self.owner,
        }])
