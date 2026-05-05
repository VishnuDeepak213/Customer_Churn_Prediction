import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    # Handle common filename case mismatch.
    alt = path.with_name('telco_churn.csv') if path.name == 'Telco_Churn.csv' else path.with_name('Telco_Churn.csv')
    if alt.exists():
        return pd.read_csv(alt)

    raise FileNotFoundError(f'CSV not found: {path}')


def compute_drift_report(baseline_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> dict:
    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
    report = {
      'threshold': threshold,
      'features_checked': 0,
      'features_with_drift': 0,
      'drift_detected': False,
      'details': {}
    }

    for col in numeric_cols:
        if col not in current_df.columns:
            continue

        baseline_vals = baseline_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        current_vals = current_df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(baseline_vals) < 2 or len(current_vals) < 2:
            continue

        stat, p_value = ks_2samp(baseline_vals, current_vals)
        is_drift = bool(p_value < threshold)

        report['features_checked'] += 1
        if is_drift:
            report['features_with_drift'] += 1

        report['details'][col] = {
            'ks_statistic': float(stat),
            'p_value': float(p_value),
            'is_drift': is_drift,
        }

    report['drift_detected'] = report['features_with_drift'] > 0
    return report


def main():
    parser = argparse.ArgumentParser(description='Compute numeric feature drift using KS test.')
    parser.add_argument('--baseline', default='data/raw/Telco_Churn.csv')
    parser.add_argument('--current', default='data/processed/current_batch.csv')
    parser.add_argument('--output', default='reports/drift_report.json')
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    output_path = Path(args.output)

    baseline_df = _read_csv_with_fallback(baseline_path)
    if current_path.exists():
        current_df = _read_csv_with_fallback(current_path)
    else:
        # If no current batch exists yet, reuse baseline for a no-drift initial report.
        current_df = baseline_df.copy()

    report = compute_drift_report(baseline_df, current_df, threshold=args.threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    status = 'DRIFT_DETECTED' if report['drift_detected'] else 'NO_DRIFT'
    print(status)
    print(f"features_checked={report['features_checked']}, features_with_drift={report['features_with_drift']}")


if __name__ == '__main__':
    main()
