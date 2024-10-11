"""This module provides functionality for loading, validating, and pre-processing raw data."""

import string

import pandas as pd

from omegaconf import DictConfig, ListConfig

from src.config import load_config
from src.database import aggregate_neighborhood_ids
from src.logger import logger

DATA_CONFIG: DictConfig = load_config().data


def encode_binary_features(data: pd.DataFrame) -> pd.DataFrame:
    """Encodes the binary categorical features.

    Args:
        data (pd.DataFrame): Dataset containing features and the target.

    Returns:
        pd.DataFrame: Dataset containing features and the target, where the binary
        categorical features have been encoded.
    """
    try:
        binary_features: list[str] = [
            col for col in data.select_dtypes(include="object").columns
            if len(data[col].unique()) == 2
        ]
        return (
            pd.get_dummies(data, columns=binary_features, drop_first=True)
            .rename(dict(zip([f"{col}_yes" for col in binary_features], binary_features)), axis=1)
        )
    except Exception as e:
        raise e


def parse_garden_feature(
    data: pd.DataFrame,
    col: str = "garden",
) -> pd.DataFrame:
    """Processes the 'garden' categorical feature, that is, each string entry is
    parsed and numeric digits are extracted.

    Args:
        data (pd.DataFrame): Dataset containing features and the target.
        col (str, optional): Name of the column being parsed. Defaults to "garden".

    Returns:
        pd.DataFrame: Dataset containing features and the target, where the 'garden'
        categorical feature is converted to a numeric feature.
    """
    try:
        digit_filter: dict[int, None] = str.maketrans(
            "", "", string.whitespace + string.punctuation + string.ascii_letters + "²"
        )
        garden_sizes: list[int] = [
            0 if garden_size == "Not present" else int(garden_size.translate(digit_filter))
            for garden_size in data[col]
        ]
        return data.assign(garden_size=garden_sizes)
    except Exception as e:
        raise e


@logger.catch
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Pre-processes the raw data.

    Args:
        data (pd.DataFrame): Dataset containing raw features and the target.

    Returns:
        pd.DataFrame: Dataset that's free of duplicates and nulls and contains
        machine learning-ready features and the target.
    """
    try:
        logger.info(
            "Validating, pre-processing, and transforming the raw data into ML-ready features \
and targets."
        )
        features: ListConfig = DATA_CONFIG.features
        target: str = DATA_CONFIG.target
        output_cols: ListConfig = features + [target]
        data = (
            data
            .pipe(encode_binary_features)
            .pipe(parse_garden_feature)
            [output_cols]
            .drop_duplicates(keep="first")
            .dropna(subset=target)
            .reset_index(drop=True)
        )

        # confirm that 'data' is free of nulls, duplicates, and ...
        # contains only boolean and numeric dtype columns
        assert data.isna().sum().sum() == 0
        assert data.duplicated().sum() == 0
        assert (
            data.select_dtypes(include=["bool", "number"]).columns.tolist() == data.columns.tolist()
        )
        return data
    except Exception as e:
        raise e


def encode_neighborhood_ids(
    data: pd.DataFrame,
    col: str = "neighborhood_id",
) -> pd.DataFrame:
    """Encodes the 'neighborhood_id' feature.

    Args:
        data (pd.DataFrame): Dataset containing features and the target.
        col (str, optional): Name of the feature being encoded. Defaults to "neighborhood_id".

    Returns:
        pd.DataFrame: Dataset containing features and the target, where the 'neighborhood_id'
        feature has been encoded.
    """
    try:
        target: str = DATA_CONFIG.target
        data = data.merge(aggregate_neighborhood_ids(), how="left", on=col).drop(col, axis=1)
        return (
            pd.concat((data.drop(target, axis=1), data[target]), axis=1)
            if target in data.columns else data
        )
    except Exception as e:
        raise e
