import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool

load_dotenv()

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'database': os.getenv('DB_NAME', 'churn_db'),
}

def get_connection_string():
    """Generate PostgreSQL connection string"""
    return URL.create(
        "postgresql+psycopg2",
        username=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host'],
        port=int(DB_CONFIG['port']),
        database=DB_CONFIG['database'],
    )

def get_engine():
    """Create SQLAlchemy engine"""
    return create_engine(get_connection_string(), poolclass=NullPool)

# Paths
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH', 'data/raw/')
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH', 'data/processed/')

def _ensure_directory_path(path_value):
    path = Path(path_value)
    directory = path if path.suffix == '' else path.parent
    directory.mkdir(parents=True, exist_ok=True)
    return path


# Create paths if they don't exist
RAW_DATA_PATH = _ensure_directory_path(RAW_DATA_PATH)
PROCESSED_DATA_PATH = _ensure_directory_path(PROCESSED_DATA_PATH)