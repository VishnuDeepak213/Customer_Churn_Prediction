import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    'customerID',
    'gender',
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'Contract',
    'Churn',
]


def validate_dataframe(df: pd.DataFrame) -> dict:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    null_ratio = (df.isna().sum() / max(len(df), 1)).to_dict()
    high_null_columns = [col for col, ratio in null_ratio.items() if ratio > 0.3]

    duplicate_customer_ids = 0
    if 'customerID' in df.columns:
        duplicate_customer_ids = int(df['customerID'].duplicated().sum())

    invalid_target_values = 0
    if 'Churn' in df.columns:
        allowed = {'Yes', 'No'}
        invalid_target_values = int((~df['Churn'].isin(allowed)).sum())

    checks = {
        'required_columns_present': len(missing_columns) == 0,
        'row_count_positive': len(df) > 0,
        'duplicate_customer_ids_zero': duplicate_customer_ids == 0,
        'target_values_valid': invalid_target_values == 0,
        'no_high_null_columns': len(high_null_columns) == 0,
    }

    success = all(checks.values())

    return {
        'success': success,
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'checks': checks,
        'details': {
            'missing_columns': missing_columns,
            'high_null_columns': high_null_columns,
            'duplicate_customer_ids': duplicate_customer_ids,
            'invalid_target_values': invalid_target_values,
        },
    }


def validate_csv(input_path: Path, output_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_path}')

    df = pd.read_csv(input_path)
    report = validate_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate churn dataset and emit JSON report.')
    parser.add_argument('--input', default='data/raw/Telco_Churn.csv')
    parser.add_argument('--output', default='reports/validation_report.json')
    args = parser.parse_args()

    report = validate_csv(Path(args.input), Path(args.output))

    if report['success']:
        print('VALIDATION_PASSED')
    else:
        print('VALIDATION_FAILED')

    print(f"rows={report['rows']} columns={report['columns']}")


if __name__ == '__main__':
    main()
