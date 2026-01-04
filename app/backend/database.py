import os
import logging
import sqlalchemy
import pandas as pd
from pathlib import Path

# Paths
_TEST_DATASET_PATH = Path("../data/processed/test_dataset.csv")

# Database connection credentials
DB_NAME     = os.getenv("DATABASE_NAME")
DB_USER     = os.getenv("DATABASE_USER")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")
DB_HOST     = os.getenv("DATABASE_HOST")
DB_PORT     = os.getenv("DATABASE_PORT")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Public
TABLE_NAME = "client"

def upload_data(data: pd.DataFrame, table_name: str) -> None:
    """
    Uploads old client data into PostgreSQL server
    """
    # Create database engine
    try:    
        engine = sqlalchemy.create_engine(DB_URL, echo=True)
        data.to_sql(table_name, engine, if_exists="replace", index=False)
    except Exception as err:
        logging.error(err)


def get_record_by_name(client_name: str) -> pd.DataFrame | pd.Series | None:
    """
    Checks if there client with `client_name` name. If so, returns his data
    """
    query = f"SELECT * FROM client WHERE name_surname = '{client_name}';"
    
    # Create database engine
    try:    
        engine = sqlalchemy.create_engine(DB_URL, echo=True)
        result = pd.read_sql(query, engine)
        return result
    except Exception as err:
        logging.error(err)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print(get_record_by_name("Михаил Асташёнок"))