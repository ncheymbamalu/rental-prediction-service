"""This module provides functionality for managing a ML model."""

import pickle

from pathlib import Path

import pandas as pd

from pydantic import BaseModel
from xgboost import XGBRegressor

from src.config import Paths
from src.data import encode_neighborhood_ids
from src.logger import logger
from src.model import build_model


class Record(BaseModel):
    """Represents an input record"""

    year_built: int
    area: float
    bedrooms: int
    bathrooms: int
    garden_size: float
    balcony: bool
    parking: bool
    furnished: bool
    garage: bool
    storage: bool
    neighborhood_id: int


class ModelService:
    """
    A class that encapsulates the rental prediction service.

    Attributes:
        model (None | XGBRegressor): Trained ML model. Defaults to None.

    Methods:
        __init__: Constructor that initializes the ModelService.
        load_model: Loads the trained ML model from ~/artifacts/model.pkl or starts the
        model building process if it doesn't exist.
        predict: Makes a prediction using the loaded model.
    """

    def __init__(self) -> None:
        """Initializes the ModelService without a trained ML model."""
        self.model: None | XGBRegressor = None

    def load_model(self) -> None:
        """Loads the trained ML model from ~/artifacts/model.pkl or starts the
        model building process if it doesn't exist.
        """
        logger.info(f"Checking if '{Paths.MODEL}' exists.")
        if not Path(Paths.MODEL).exists():
            logger.warning(f"'{Paths.MODEL}' not found. Initiating the model building process.")
            build_model()

        # load ~/artifacts/model.pkl
        with open(Paths.MODEL, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, record: Record) -> int:
        """Makes a prediction using the loaded ML model

        Args:
            record (Record): Input data for making a prediction.

        Returns:
            int: Rental prediction.
        """
        x: pd.DataFrame = pd.DataFrame([record.model_dump()]).pipe(encode_neighborhood_ids)
        prediction: float = self.model.predict(x)[0]
        return max(0, int(round(prediction)))
