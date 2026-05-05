import pandas as pd

from src.data_validation import validate_dataframe


def test_validate_dataframe_success():
    df = pd.DataFrame(
        {
            'customerID': ['0001', '0002'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'tenure': [5, 10],
            'MonthlyCharges': [50.0, 75.0],
            'TotalCharges': [250.0, 750.0],
            'Contract': ['Month-to-month', 'One year'],
            'Churn': ['No', 'Yes'],
        }
    )

    report = validate_dataframe(df)
    assert report['success'] is True


def test_validate_dataframe_detects_issues():
    df = pd.DataFrame(
        {
            'customerID': ['0001', '0001'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'tenure': [5, 10],
            'MonthlyCharges': [50.0, 75.0],
            'TotalCharges': [250.0, 750.0],
            'Contract': ['Month-to-month', 'One year'],
            'Churn': ['No', 'Maybe'],
        }
    )

    report = validate_dataframe(df)
    assert report['success'] is False
    assert report['checks']['duplicate_customer_ids_zero'] is False
    assert report['checks']['target_values_valid'] is False
