import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Trigger retraining when drift report shows drift.')
    parser.add_argument('--drift-report', default='reports/drift_report.json')
    parser.add_argument('--min-drifted-features', type=int, default=1)
    args = parser.parse_args()

    report_path = Path(args.drift_report)
    if not report_path.exists():
        print(f'Drift report not found: {report_path}')
        sys.exit(1)

    report = json.loads(report_path.read_text(encoding='utf-8'))
    drifted = int(report.get('features_with_drift', 0))

    if drifted < args.min_drifted_features:
        print('No retraining needed.')
        return

    print(f'Drift threshold reached ({drifted} features). Starting retraining...')
    result = subprocess.run([sys.executable, 'train_and_save.py'], check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print('Retraining complete.')


if __name__ == '__main__':
    main()
