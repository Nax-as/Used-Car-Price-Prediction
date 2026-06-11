"""
Training Pipeline
-----------------
Orchestrates DataIngestion → DataTransformation → ModelTrainer in one call.

Usage
-----
    python -m src.pipeline.train_pipeline
"""
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger


def run_training_pipeline(data_path: str = "data/car_details.csv"):
    """
    End-to-end training pipeline.

    Parameters
    ----------
    data_path : str
        Path to the raw Cardekho CSV.
    """
    logger.info("======  Training Pipeline Started  ======")

    # 1. Ingest
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(data_path)

    # 2. Transform
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, _ = \
        transformer.initiate_data_transformation(train_path, test_path)

    # 3. Train
    trainer = ModelTrainer()
    results, metrics = trainer.initiate_model_training(
        X_train, X_test, y_train, y_test
    )

    logger.info("======  Training Pipeline Complete  ======")
    logger.info(f"Final metrics: {metrics}")
    print("\nFinal best-model metrics:", metrics)
    return metrics


if __name__ == "__main__":
    run_training_pipeline()
