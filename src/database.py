"""This module provides functionality for interacting with PostgeSQL's 'postgres' database."""

import os

from pathlib import PosixPath

import pandas as pd

from dotenv import load_dotenv
from omegaconf import DictConfig
from sqlalchemy import URL, Connection, create_engine, text

from src.config import Paths, load_config
from src.logger import logger

load_dotenv(Paths.ENV)

DB_CONFIG: DictConfig = load_config().database


def get_db_connection() -> Connection:
    """Returns an object that connects to the 'postgres' database.

    Returns:
        Connection: 'postgres' database connection object.
    """
    try:
        # instantiate an object of type, 'URL', which points to the 'postgres' database
        url: URL = URL.create(
            drivername=DB_CONFIG.drivername,
            username=DB_CONFIG.user,
            host=DB_CONFIG.host,
            database=DB_CONFIG.dbname,
            port=DB_CONFIG.port,
            password=os.getenv("PG_PASSWORD")
        )
        db_connection: Connection = create_engine(url).connect()
        return db_connection
    except Exception as e:
        raise e


def create_schema() -> None:
    """Creates a schema named, 'rentals', under the 'postgres' database."""
    try:
        db_connection: Connection = get_db_connection()
        db_connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DB_CONFIG.schema}"))
        db_connection.commit()
        db_connection.close()
    except Exception as e:
        raise e


def create_table() -> None:
    """Creates a table named, 'raw', under the 'postgres' database's 'rentals' schema."""
    try:
        db_connection: Connection = get_db_connection()
        db_connection.execute(text(
            f"""
            CREATE TABLE IF NOT EXISTS {DB_CONFIG.schema}.{DB_CONFIG.table}
            (
                address TEXT,
                zip TEXT,
                neighborhood TEXT,
                neighborhood_id INTEGER,
                year_built INTEGER,
                area REAL,
                rooms INTEGER,
                bedrooms INTEGER,
                bathrooms REAL,
                balcony TEXT,
                storage TEXT,
                parking TEXT,
                furnished TEXT,
                garage TEXT,
                garden TEXT,
                energy TEXT,
                facilities TEXT,
                rent INTEGER
            )
            """
        ))
        db_connection.commit()
        db_connection.close()
    except Exception as e:
        raise e


@logger.catch
def write_table(path: PosixPath | str = Paths.RAW_DATA) -> None:
    """Writes path to the 'postgres' database's 'rentals.raw' table.

    Args:
        path (PosixPath | str, optional): Raw data's local file path, ~/data/raw.parquet.
        Defaults to Paths.RAW_DATA.
    """
    try:
        # connect to the 'postgres' database
        db_connection: Connection = get_db_connection()

        # write ~/data/raw.parquet to the database's 'rentals.raw' table
        (
            pd.read_parquet(path)
            .replace(float("nan"), None)
            .to_sql(
                con=db_connection,
                schema=DB_CONFIG.schema,
                name=DB_CONFIG.table,
                if_exists="append",
                index=False
            )
        )

        # commit the changes and close the connection to the database
        db_connection.commit()
        db_connection.close()
        logger.info(
            f"Success! '{Paths.RAW_DATA}' has been written to the '{DB_CONFIG.dbname}' database's \
'{DB_CONFIG.schema}.{DB_CONFIG.table}' table."
        )
    except Exception as e:
        raise e


@logger.catch
def read_table() -> pd.DataFrame:
    """Queries the 'postgres' database's 'rentals.raw' table and returns a pd.DataFrame.

    Returns:
        pd.DataFrame: Raw data.
    """
    try:
        logger.info(
            f"Fetching raw data from the '{DB_CONFIG.dbname}' database's \
'{DB_CONFIG.schema}.{DB_CONFIG.table}' table."
        )
        # connect to the 'postgres' database
        db_connection: Connection = get_db_connection()

        # read the database's 'rentals.raw' table as a pd.DataFrame
        script: str = f"SELECT * FROM {DB_CONFIG.schema}.{DB_CONFIG.table}"
        data: pd.DataFrame = pd.DataFrame(db_connection.execute(text(script)))

        # close the connection to the database
        db_connection.close()
        return data
    except Exception as e:
        raise e


def aggregate_neighborhood_ids() -> pd.DataFrame:
    """Returns a pd.DataFrame containing the average area (m²), average number
    of bedrooms, average number of bathrooms, and average garden size (m²) for
    a representative rental in each unique neighborhood ID.
    """
    try:
        # connect to the 'postgres' database and fetch the aggregated data
        db_connection: Connection = get_db_connection()
        query: str = f"""
SELECT
    neighborhood_id,
    CAST(AVG("area") AS FLOAT) AS neighborhood_mean_area,
    CAST(AVG(bedrooms) AS FLOAT) AS neighborhood_mean_bedrooms,
    CAST(AVG(bathrooms) AS FLOAT) AS neighborhood_mean_bathrooms,
    CAST(AVG(CAST(COALESCE(NULLIF(REGEXP_REPLACE(garden, '\D', '', 'g'), ''), '0') AS INTEGER)) \
AS FLOAT) AS neighborhood_mean_garden_size
FROM {DB_CONFIG.dbname}.{DB_CONFIG.schema}.{DB_CONFIG.table}
GROUP BY 1
ORDER BY 1
        """
        data: pd.DataFrame = pd.DataFrame(db_connection.execute(text(query)))
        db_connection.close()
        return data
    except Exception as e:
        raise e


if __name__ == "__main__":
    create_schema()
    create_table()
    write_table()
