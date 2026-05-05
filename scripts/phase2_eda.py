import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestion


def main():
    ingestion = DataIngestion()
    df = pd.read_sql('SELECT * FROM raw_data.telco_churn', ingestion.engine)

    print(f'Dataset shape: {df.shape}')
    print('\nColumn names and types:')
    print(df.dtypes)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print('\nMissing values:')
    if missing.empty:
        print('None')
    else:
        print(missing)

    print('\nStatistical summary for numeric columns:')
    print(df.select_dtypes(include=['number']).describe())

    churn_dist = df['Churn'].value_counts()
    print('\nChurn distribution:')
    print(churn_dist)
    print('\nChurn percentage:')
    print((100 * churn_dist / len(df)).round(2))

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f'\nCategorical columns ({len(cat_cols)}):')
    print(cat_cols)
    print(f'\nNumerical columns ({len(num_cols)}):')
    print(num_cols)

    if 'Churn' in df.select_dtypes(include=['number']).columns:
        correlations = df.select_dtypes(include=['number']).corr()['Churn'].sort_values(ascending=False)
        print('\nCorrelation with Churn:')
        print(correlations)


if __name__ == '__main__':
    main()
