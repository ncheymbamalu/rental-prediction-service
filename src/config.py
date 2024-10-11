"""This module sets up the project's configurations."""

from pathlib import Path, PosixPath

from omegaconf import DictConfig, OmegaConf


class Paths:
    """Configuration for the project's primary directories and filepaths.

    Attributes:
        PROJECT_DIR (PosixPath): Project's root directory.
        DATA_DIR (PosixPath): Project's data directory, ~/data/.
        ARTIFACTS_DIR (PosixPath): Project's artifacts directory, ~/artifacts/.
        LOGS_DIR (PosixPath): Project's logs directory, ~/logs/.
        ENV (PosixPath): Project's .env file path, ~/.env.
        CONFIG (PosixPath): Project's configuration file path, ~/config.yaml.
        RAW_DATA (PosixPath): Project's raw data file path, ~/data/raw.parquet.
        MODEL (PosixPath): Project's trained ML model file path, ~/artifacts/model.pkl.
    """

    PROJECT_DIR: PosixPath = Path(__file__).parent.parent.absolute()
    DATA_DIR: PosixPath = PROJECT_DIR / "data"
    ARTIFACTS_DIR: PosixPath = PROJECT_DIR / "artifacts"
    LOGS_DIR: PosixPath = PROJECT_DIR / "logs"
    ENV: PosixPath = PROJECT_DIR / ".env"
    CONFIG: PosixPath = PROJECT_DIR / "config.yaml"
    RAW_DATA: PosixPath = DATA_DIR / "raw.parquet"
    MODEL: PosixPath = ARTIFACTS_DIR / "model.pkl"


def load_config(path: PosixPath = Paths.CONFIG) -> DictConfig:
    """Returns ~/config.yaml as a DictConfig object.

    Args:
        path (PosixPath, optional): Configuration file path, ~/config.yaml.
        Defaults to Paths.CONFIG.

    Returns:
        DictConfig: Dictionary-like object with user-defined key-values pairs.
    """
    return OmegaConf.load(path)
