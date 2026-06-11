"""
Data Transformation Component
------------------------------
Cleans raw data, engineers features, and builds a sklearn ColumnTransformer
pipeline (imputation → encoding → scaling).  The fitted preprocessor is
saved as artifacts/preprocessor.pkl.
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """Clean, engineer features, and build a preprocessing pipeline."""

    def __init__(self):
        self.config = DataTransformationConfig()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform all cleaning & feature engineering steps:
        - Drop duplicates and rows with null selling_price.
        - Parse 'name' into 'brand' and 'model'.
        - Derive 'car_age' from 'year'.
        - Clean 'mileage', 'engine', 'max_power' columns.
        - Drop columns not needed for modelling.
        """
        df = df.copy()
        df.drop_duplicates(inplace=True)
        df.dropna(subset=["selling_price"], inplace=True)

        # ── Brand / Model from 'name' ──────────────────────────────────
        df["brand"] = df["name"].str.split().str[0]
        df["model"] = df["name"].str.split().str[1]

        # ── Car age ───────────────────────────────────────────────────
        current_year = 2024
        df["car_age"] = current_year - df["year"]

        # ── Clean numeric string columns ──────────────────────────────
        def extract_numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(
                series.astype(str).str.extract(r"([\d.]+)", expand=False),
                errors="coerce",
            )

        df["mileage_kmpl"]  = extract_numeric(df["mileage"])
        df["engine_cc"]     = extract_numeric(df["engine"])
        df["max_power_bhp"] = extract_numeric(df["max_power"])

        # ── Drop raw / redundant columns ──────────────────────────────
        drop_cols = ["name", "year", "mileage", "engine", "max_power", "torque"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        return df

    # ------------------------------------------------------------------
    def get_preprocessor(self) -> ColumnTransformer:
        """
        Build a ColumnTransformer with:
          - Numeric pipeline  : median imputation → standard scaling
          - Categorical pipeline: most-frequent imputation → ordinal encoding
        """
        num_features = [
            "km_driven", "car_age", "mileage_kmpl",
            "engine_cc", "max_power_bhp", "seats",
        ]
        cat_features = [
            "brand", "model", "fuel", "seller_type",
            "transmission", "owner",
        ]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ], remainder="drop")

        return preprocessor

    # ------------------------------------------------------------------
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Load train/test CSVs, clean them, fit the preprocessor on training
        data, transform both splits, and persist the preprocessor.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, str]
            (X_train_arr, X_test_arr, y_train, y_test, preprocessor_path)
        """
        logger.info("Starting data transformation.")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            train_df = self.clean_dataframe(train_df)
            test_df  = self.clean_dataframe(test_df)

            TARGET = "selling_price"
            X_train = train_df.drop(columns=[TARGET])
            y_train = np.log1p(train_df[TARGET])   # log-transform target

            X_test  = test_df.drop(columns=[TARGET])
            y_test  = np.log1p(test_df[TARGET])

            preprocessor = self.get_preprocessor()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr  = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_obj_path, preprocessor)
            logger.info("Data transformation complete.")

            return X_train_arr, X_test_arr, y_train.values, y_test.values, \
                   self.config.preprocessor_obj_path

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise
