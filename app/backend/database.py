import os
import logging
import sqlalchemy
import pandas as pd

# Database connection credentials
DB_NAME     = os.getenv("POSTGRES_DB")
DB_USER     = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST     = os.getenv("POSTGRES_HOST")
DB_PORT     = os.getenv("POSTGRES_PORT")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_record_by_name(client_name: str) -> pd.DataFrame | pd.Series | None:
    """
    Checks if there client with `client_name` name. If so, returns his data
    """
    query = f"SELECT * FROM public.client WHERE name_surname = '{client_name}';"
    
    # Create database engine
    try:    
        engine = sqlalchemy.create_engine(DB_URL, echo=True)
        result = pd.read_sql(query, engine)
        return result
    except Exception as err:
        logging.error(err)
        return pd.DataFrame()  # Return empty DataFrame on error
