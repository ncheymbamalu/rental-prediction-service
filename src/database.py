"""This module provides functionality for interacting with PostgeSQL's 'postgres' database."""

import os

from pathlib import PosixPath

import pandas as pd
import psycopg2

from dotenv import load_dotenv
from omegaconf import DictConfig
from psycopg2.extensions import connection, cursor
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import URL

from src.config import Paths, load_config
from src.logger import logger

load_dotenv(Paths.ENV)

DB_CONFIG: DictConfig = load_config().database


def create_schema() -> None:
    """Creates the 'postgres' database's 'rentals' schema."""
    try:
        conn: connection = psycopg2.connect(
            host=DB_CONFIG.host,
            dbname=DB_CONFIG.dbname,
            user=DB_CONFIG.user,
            port=DB_CONFIG.port,
            password=os.getenv("PG_PASSWORD")
        )
        cur: cursor = conn.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_CONFIG.schema};")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise e


def create_table() -> None:
    """Creates the 'postgres' database's 'rentals.raw' table."""
    try:
        # create the 'URL' object, which points to the 'postgres' database
        url: URL = URL.create(
            drivername=DB_CONFIG.drivername,
            username=DB_CONFIG.user,
            host=DB_CONFIG.host,
            database=DB_CONFIG.dbname,
            port=DB_CONFIG.port,
            password=os.getenv("PG_PASSWORD")
        )

        # create the 'Engine' object, which is required to create a table
        engine: Engine = create_engine(url)

        # create the 'MetaData' object, which stores the database's metadata
        metadata: MetaData = MetaData()

        # create a table named, 'raw', under the 'rentals' schema
        table: Table = Table(
            DB_CONFIG.table,
            metadata,
            Column("address", Text, primary_key=True),
            Column("zip", Text),
            Column("neighborhood", Text),
            Column("neighborhood_id", Integer),
            Column("year_built", Integer),
            Column("area", Float),
            Column("rooms", Integer),
            Column("bedrooms", Integer),
            Column("bathrooms", Integer),
            Column("balcony", Text),
            Column("storage", Text),
            Column("parking", Text),
            Column("furnished", Text),
            Column("garage", Text),
            Column("garden", Text),
            Column("energy", Text),
            Column("facilities", Text),
            Column("rent", Integer),
            schema=DB_CONFIG.schema
        )
        table.create(engine, checkfirst=True)
    except Exception as e:
        raise e


def get_db_connection() -> Connection:
    """Returns an object that connects to the 'postgres' database.

    Returns:
        Connection: 'postgres' database connection object.
    """
    try:
        # create the 'URL' object, which points to the 'postgres' database
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
                if_exists="replace",
                index=False
            )
        )

        # close the connection to the database
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
    a rental in each unique neighborhood ID.
    """
    try:
        # connect to the 'postgres' database
        db_connection: Connection = get_db_connection()

        # fetch the aggregated neighborhood IDs from the database's 'rentals.raw' table
        script: str = f"""
        WITH neighborhood_id_table AS (
            SELECT
                neighborhood_id,
                "area",
                bedrooms,
                bathrooms,
                CAST(COALESCE(NULLIF(REGEXP_REPLACE(garden, '\D', '', 'g'), ''), '0') AS INT) \
AS garden_size
            FROM {DB_CONFIG.dbname}.{DB_CONFIG.schema}.{DB_CONFIG.table}
        )
        SELECT
            neighborhood_id,
            CAST(AVG("area") AS FLOAT) AS neighborhood_mean_area,
            CAST(AVG(bedrooms) AS FLOAT) AS neighborhood_mean_bedrooms,
            CAST(AVG(bathrooms) AS FLOAT) AS neighborhood_mean_bathrooms,
            CAST(AVG(garden_size) AS FLOAT) AS neighborhood_mean_garden_size
        FROM neighborhood_id_table
        GROUP BY neighborhood_id
        ORDER BY neighborhood_id
        """
        data: pd.DataFrame = pd.DataFrame(db_connection.execute(text(script)))
        db_connection.close()
        return data
    except Exception as e:
        raise e


if __name__ == "__main__":
    create_schema()
    create_table()
    write_table()
