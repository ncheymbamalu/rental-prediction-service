"""This module provides functionality for building an ML model."""

from pathlib import PosixPath

from src.config import Paths
from src.logger import logger
from src.model import build_model


class ModelBuilderService:
    """
    A class that encapsulates the model building process.

    Attributes:
        model_path (PosixPath): Trained ML model's file path. Defaults to Paths.MODEL.

    Methods:
        __init__: Constructor that initializes the ModelBuilderService.
        build_model: Trains, evaluates, and saves an object of type, 'XGBRegressor', to 
        'model_path', if 'model_path' doesn't exist.
    """

    def __init__(self) -> None:
        """Initializes the ModelBuilderService."""
        self.model_path: PosixPath = Paths.MODEL

    def build_model(self) -> None:
        """Trains, evaluates, and saves an object of type, 'XGBRegressor', to 'model_path', if
        'model_path' doesn't exist.
        """
        if self.model_path.exists():
            logger.info(
                f"'~/{self.model_path.parent.stem}/{self.model_path.name}' exists. \
Skipping the model building process."
            )
        else:
            logger.info("Initiating the model building process.")
            build_model()
