"""This module provides functionality for the ML model building process."""

import pickle

from pathlib import PosixPath

import numpy as np
import pandas as pd

from omegaconf import DictConfig
from sklearn.utils import shuffle
from xgboost import XGBRegressor

from src.config import Paths, load_config
from src.data import DATA_CONFIG, encode_neighborhood_ids, preprocess_data
from src.database import read_table
from src.logger import logger

MODEL_CONFIG: DictConfig = load_config().model


def split_data(
    data: pd.DataFrame,
    train_size: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits ML-ready data into train, validation, and test sets.

    Args:
        data (pd.DataFrame): Dataset containing ML-ready features and the target
        train_size (float, optional): Percentage of data reserved for training.
        Defaults to 0.75.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        Train set, validation set, and test set features and targets.
    """
    try:
        logger.info(
            "Splitting the ML-ready features and targets into train, validation, and test sets."
        )
        target: str = DATA_CONFIG.target
        features: list[str] = data.drop(target, axis=1).columns.tolist()
        data = shuffle(data)
        n_records: int = data.shape[0]
        train_split: int = int(round(train_size * n_records))
        val_split: int = train_split + int(round((n_records - train_split) / 2))
        train_data: pd.DataFrame = data.iloc[:train_split, :]
        val_data: pd.DataFrame = data.iloc[train_split:val_split, :]
        test_data: pd.DataFrame = data.iloc[val_split:, :]
        return (
            train_data[features],
            val_data[features],
            test_data[features],
            train_data[target],
            val_data[target],
            test_data[target]
        )
    except Exception as e:
        raise e


def compute_rsquared(
    y: pd.Series | np.ndarray,
    yhat: pd.Series | np.ndarray,
) -> float:
    """Computes the coefficient of determination, R², between y and yhat.

    Args:
        y (pd.Series | np.ndarray): Labels.
        yhat (pd.Series | np.ndarray): Predictions.

    Returns:
        float: R².
    """
    try:
        # compute the R² between y and yhat
        t: pd.Series | np.ndarray = y - y.mean()
        sst: float = t.dot(t)
        e: pd.Series | np.ndarray = y - yhat
        sse: float = e.dot(e)
        r_squared: float = 1 - (sse / sst)
        return round(r_squared, 2)
    except Exception as e:
        raise e


@logger.catch
def build_model() -> None:
    """Trains an object of type, 'XGBRegressor', evaluates it against 'baseline'
    predictions, and saves it to ~/artifacts/model.pkl.
    """
    try:
        # fetch, pre-process, and transform the raw data into ML-ready features and targets
        data: pd.DataFrame = read_table().pipe(preprocess_data).pipe(encode_neighborhood_ids)

        # split the ML-ready data into train, validation, and test sets
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)

        # instantiate an object of type, 'XGBRegressor', that is, the model
        logger.info("Initiating model training and evaluation.")
        model: XGBRegressor = XGBRegressor(**MODEL_CONFIG.hyperparams)

        # fit the model
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        # evaluate the model
        model_metric: float = compute_rsquared(y_test, model.predict(x_test))
        logger.info(f"Model training and evaluation complete. The {model.__class__.__name__} \
produced a test set R² of {model_metric}.")

        # confirm that the model's predictions are better than the 'baseline' predictions
        baseline_metric: float = compute_rsquared(y_test, y_test.mean())
        assert model_metric > baseline_metric

        # save the model to ~/artifacts/model.pkl
        logger.info(f"Saving the {model.__class__.__name__} to '{Paths.MODEL}'.")
        artifacts_dir: PosixPath = Paths.ARTIFACTS_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(Paths.MODEL, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise e
