"""Treinamento do modelo Random Forest."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, MODELS_DIR, SUCCESS_LABELS
from src.data import build_training_dataset, get_feature_columns, save_processed_dataset


def train_model(random_state: int = 42) -> dict:
    df = build_training_dataset()
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].astype(float)
    y = df["success_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = sorted(SUCCESS_LABELS.keys())
    target_names = [SUCCESS_LABELS[i] for i in labels]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)),
        "classification_report": classification_report(
            y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
        ),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": len(feature_cols),
    }

    importances = sorted(
        zip(feature_cols, model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    metrics["top_features"] = importances[:10]

    dataset_path = ARTIFACTS_DIR / "training_dataset.parquet"
    save_processed_dataset(df, dataset_path)

    model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(model, model_path)

    config = {
        "feature_columns": feature_cols,
        "success_labels": {str(k): v for k, v in SUCCESS_LABELS.items()},
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
    }
    config_path = ARTIFACTS_DIR / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    report_path = ARTIFACTS_DIR / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Relatório de Treinamento ===\n\n")
        f.write(f"Acurácia: {metrics['accuracy']:.3f}\n")
        f.write(f"F1 Macro: {metrics['f1_macro']:.3f}\n\n")
        f.write(metrics["classification_report"])
        f.write("\n\nTop 10 features:\n")
        for name, imp in metrics["top_features"]:
            f.write(f"  {name}: {imp:.4f}\n")

    return metrics


if __name__ == "__main__":
    result = train_model()
    print(f"Acurácia: {result['accuracy']:.3f}")
    print(f"F1 Macro: {result['f1_macro']:.3f}")
    print(result["classification_report"])
