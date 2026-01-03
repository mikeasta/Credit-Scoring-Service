from cgi import print_arguments
import os
import logging
import sqlalchemy
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Paths
_TEST_DATASET_PATH = Path("../data/processed/test_dataset.csv")
_ENVIRONMENT_VARIABLES_PATH = Path("../.env")

# Load environment data
load_dotenv(_ENVIRONMENT_VARIABLES_PATH)

# Database connection credentials
_DB_NAME     = os.environ.get("DATABASE_NAME")
_DB_USER     = os.environ.get("DATABASE_USER")
_DB_PASSWORD = os.environ.get("DATABASE_PASSWORD")
_DB_HOST     = os.environ.get("DATABASE_HOST")
_DB_PORT     = os.environ.get("DATABASE_PORT")

_DB_URL = f"postgresql+psycopg2://{_DB_USER}:{_DB_PASSWORD}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"

# Public
TABLE_NAME = "client"

def upload_data(data: pd.DataFrame, table_name: str) -> None:
    """
    Uploads old client data into PostgreSQL server
    """
    # Create database engine
    try:    
        engine = sqlalchemy.create_engine(_DB_URL, echo=True)
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
        engine = sqlalchemy.create_engine(_DB_URL, echo=True)
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