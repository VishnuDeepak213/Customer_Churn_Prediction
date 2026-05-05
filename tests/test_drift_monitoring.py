import pandas as pd

from src.drift_monitoring import compute_drift_report


def test_compute_drift_report_no_drift_same_data():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10.0, 20.0, 30.0, 40.0]})
    report = compute_drift_report(df, df.copy(), threshold=0.05)

    assert report['features_checked'] == 2
    assert report['drift_detected'] is False
