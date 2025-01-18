"""This module provides the functionality for making predictions."""

import pickle

from pathlib import PosixPath

import pandas as pd

from xgboost import XGBRegressor

from src.config import Paths
from src.data import encode_binary_features, encode_neighborhood_ids
from src.logger import logger


class ModelInferenceService:
    """
    A class that encapsulates the model prediction process.

    Attributes:
        model_path (PosixPath): Trained ML model's file path. Defaults to Paths.MODEL.
        model (None | XGBRegressor): Trained ML model. Defaults to None.

    Methods:
        __init__: Constructor that initializes the ModelInferenceService.
        load_model: Loads the trained ML model from 'model_path' or raises
        FileNotFoundError if 'model_path' doesn't exist.
        predict: Makes a prediction using 'model'.
    """

    def __init__(self) -> None:
        """Initializes the ModelInferenceService."""
        self.model_path: PosixPath = Paths.MODEL
        self.model: None | XGBRegressor = None

    def load_model(self) -> None:
        """Loads the trained ML model from 'model_path'

        Raises:
            FileNotFoundError: If 'model_path' doesn't exist.
        """
        logger.info(f"Checking if '~/{self.model_path.parent.stem}/{self.model_path.name}' exists.")
        # if 'model_path' doesn't exist, raise an error
        if not self.model_path.exists():
            raise FileNotFoundError(f"'{self.model_path}' not found!")

        # else, load 'model_path'
        logger.info(
            f"'~/{self.model_path.parent.stem}/{self.model_path.name}' found. \
Loading the trained ML model."
        )
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, record: dict[str, float | int | str]) -> int:
        """Makes a prediction using 'model'.

        Args:
            record (Record): Input data for making a prediction.

        Returns:
            int: Rental prediction.
        """
        logger.info("Generating the prediction...")
        x: pd.DataFrame = (
            pd.DataFrame([record])
            .pipe(encode_binary_features)
            .pipe(encode_neighborhood_ids)
            [self.model.feature_names_in_]
        )
        prediction: float = self.model.predict(x)[0]
        return max(0, int(round(prediction)))
