import os
import logging
import sqlalchemy
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment data
load_dotenv(Path(".env"))

# Database connection credentials
_DB_NAME     = os.environ.get("DATABASE_NAME")
_DB_USER     = os.environ.get("DATABASE_USER")
_DB_PASSWORD = os.environ.get("DATABASE_PASSWORD")
_DB_HOST     = os.environ.get("DATABASE_HOST")
_DB_PORT     = os.environ.get("DATABASE_PORT")

_DB_URL = f"postgresql+psycopg2://{_DB_USER}:{_DB_PASSWORD}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"

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
