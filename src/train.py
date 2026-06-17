import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_data(data_dir):
    """Load all CSVs from data_dir. Returns X (n, 2970) and y (n,) as numpy arrays."""
    csvs = list(Path(data_dir).glob('*.csv'))
    if not csvs:
        raise ValueError(f"No CSV files found in {data_dir}")

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    X = df.drop(columns=['label']).values.astype(np.float32)  # 2970 features per row
    y = df['label'].values                                     # exercise label per row
    return X, y


def train(data_dir, models_dir, **mlp_kwargs):
    X, y = load_data(data_dir)

    # 80/20 split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize to mean=0, std=1 
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Two-hidden-layer neural net: 2970 inputs → 256 → 128 → 5 class outputs
    # mlp_kwargs lets tests pass a tiny network so training finishes in milliseconds
    params = dict(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    params.update(mlp_kwargs)
    model = MLPClassifier(**params)
    model.fit(X_train_s, y_train)

    # Print per-class precision/recall and confusion matrix, and save to report.txt
    y_pred = model.predict(X_test_s)
    report = classification_report(y_test, y_pred)
    matrix = str(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print(report)
    print(matrix)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(models_dir) / 'report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report + '\n')
        f.write(matrix + '\n')

    # Save model + scaler — app.py loads both to classify live webcam frames
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  Path(models_dir) / 'model.pkl')
    joblib.dump(scaler, Path(models_dir) / 'scaler.pkl')
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description='Train exercise classifier')
    parser.add_argument('--data-dir',   default='data/raw')
    parser.add_argument('--models-dir', default='models')
    args = parser.parse_args()
    train(args.data_dir, args.models_dir)


if __name__ == '__main__':
    main()
