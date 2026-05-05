from pathlib import Path
import logging

import pandas as pd
from sqlalchemy import text

from src.config import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / 'data' / 'raw' / 'telco_churn.csv'


class DataIngestion:
    def __init__(self):
        self.engine = get_engine()

    def create_schemas(self):
        """Create raw_data and processed_data schemas."""
        with self.engine.begin() as conn:
            conn.execute(text('CREATE SCHEMA IF NOT EXISTS raw_data;'))
            conn.execute(text('CREATE SCHEMA IF NOT EXISTS processed_data;'))
        logger.info('✅ Schemas created successfully')

    def load_csv_to_db(self, csv_path=DEFAULT_CSV_PATH, table_name='telco_churn', schema='raw_data', if_exists='replace'):
        """Load a CSV file into PostgreSQL."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f'CSV not found at {csv_path}')

        df = pd.read_csv(csv_path)
        logger.info('📖 Loaded %s: %s rows, %s columns', csv_path, df.shape[0], df.shape[1])
        logger.info('Missing values: %s', int(df.isnull().sum().sum()))
        logger.info('Data types:\n%s', df.dtypes)

        with self.engine.begin() as conn:
            df.to_sql(table_name, con=conn, schema=schema, if_exists=if_exists, index=False)

        logger.info('✅ Loaded %s rows to %s.%s', len(df), schema, table_name)
        return df

    def create_data_profile_views(self, table_name='telco_churn', schema='raw_data'):
        """Create simple SQL views for data profiling."""
        with self.engine.begin() as conn:
            conn.execute(text(f'''
                CREATE OR REPLACE VIEW {schema}.vw_churn_distribution AS
                SELECT Churn,
                       COUNT(*) AS count,
                       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
                FROM {schema}.{table_name}
                GROUP BY Churn;
            '''))

            conn.execute(text(f'''
                CREATE OR REPLACE VIEW {schema}.vw_data_quality AS
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{schema}' AND table_name = '{table_name}';
            '''))

        logger.info('✅ SQL views created')

    def explore_data(self, table_name='telco_churn', schema='raw_data'):
        """Quick data exploration query."""
        query = f'SELECT * FROM {schema}.{table_name} LIMIT 10;'
        return pd.read_sql(query, self.engine)


def ingest():
    ingestion = DataIngestion()
    ingestion.create_schemas()
    ingestion.load_csv_to_db(DEFAULT_CSV_PATH, table_name='telco_churn', schema='raw_data')
    ingestion.create_data_profile_views(table_name='telco_churn', schema='raw_data')

    sample = ingestion.explore_data('telco_churn')
    print(sample)


if __name__ == '__main__':
    ingest()