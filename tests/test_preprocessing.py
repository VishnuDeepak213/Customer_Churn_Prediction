import pandas as pd

from src.preprocessing import TabularPreprocessor


def test_tabular_preprocessor_fit_transform():
    data = pd.DataFrame(
        {
            'customerID': ['a', 'b', 'c'],
            'tenure': [1, 10, 20],
            'MonthlyCharges': [30.0, 50.0, 70.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No'],
        }
    )

    pre = TabularPreprocessor(target_col='Churn')
    X = pre.fit_transform(data)

    assert X.shape[0] == 3
    assert 'tenure' in X.columns
    assert pre.fitted is True


def test_tabular_preprocessor_handles_missing_columns():
    train_df = pd.DataFrame(
        {
            'tenure': [5, 7],
            'MonthlyCharges': [42.1, 58.3],
            'Contract': ['One year', 'Two year'],
            'Churn': ['No', 'Yes'],
        }
    )
    pred_df = pd.DataFrame({'tenure': [9], 'Churn': ['No']})

    pre = TabularPreprocessor(target_col='Churn')
    pre.fit(train_df)
    transformed = pre.transform(pred_df)

    assert transformed.shape[0] == 1
    assert set(pre.feature_columns).issubset(set(transformed.columns))
