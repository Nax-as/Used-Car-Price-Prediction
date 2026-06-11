"""
Data Ingestion Component
------------------------
Reads the raw Cardekho CSV, performs a basic train/test split, and
persists the splits to the artifacts directory.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logger


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_path: str = "data/car_details.csv"):
        """
        Load raw data from `source_path`, split into train/test, and save.

        Parameters
        ----------
        source_path : str
            Path to the raw Cardekho CSV file.

        Returns
        -------
        Tuple[str, str]
            Paths to the train and test CSV files.
        """
        logger.info("Starting data ingestion.")
        try:
            df = pd.read_csv(source_path)
            logger.info(f"Dataset loaded: {df.shape}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info("Data ingestion complete.")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
