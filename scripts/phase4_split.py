"""Phase 4 train/test data split script.

Loads Phase 3 processed features and target, splits into train/test sets,
and saves them for model training.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

FEATURES_PATH = DATA_DIR / 'phase3_features.csv'
TARGET_PATH = DATA_DIR / 'phase3_target.csv'


def main():
    print('=' * 80)
    print('PHASE 4: DATA SPLIT')
    print('=' * 80)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f'Missing features file: {FEATURES_PATH}')
    if not TARGET_PATH.exists():
        raise FileNotFoundError(f'Missing target file: {TARGET_PATH}')

    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGET_PATH).iloc[:, 0]

    print(f'Loaded Phase 3 data: X={X.shape}, y={y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f'Train split: X_train={X_train.shape}, X_test={X_test.shape}')
    print(f'Train target distribution:\n{y_train.value_counts()}')
    print(f'Test target distribution:\n{y_test.value_counts()}')

    X_train.to_csv(DATA_DIR / 'phase4_train_features.csv', index=False)
    X_test.to_csv(DATA_DIR / 'phase4_test_features.csv', index=False)
    y_train.to_csv(DATA_DIR / 'phase4_train_target.csv', index=False)
    y_test.to_csv(DATA_DIR / 'phase4_test_target.csv', index=False)

    print(f'\n✅ Saved train/test split CSVs in {DATA_DIR}')


if __name__ == '__main__':
    main()
