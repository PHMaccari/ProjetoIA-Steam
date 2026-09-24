"""
Pipeline do projeto 2026: classificação de engajamento + regressão de popularidade.

Classificação (A e B): Regressão Logística e KNN; Random Forest como ensemble.
Regressão: baseline da média, linear simples, linear múltipla e polinomial.
K-Means opcional para segmentação (sem variável-alvo).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.config import ARTIFACTS_DIR, MODELS_DIR, SUCCESS_LABELS
from src.data import build_training_dataset, get_feature_columns, save_processed_dataset


def _cls_metrics(y_true, y_pred, labels: list[int], target_names: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        ),
    }


def _reg_metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model(random_state: int = 42) -> dict:
    print("Carregando e preparando o dataset...")
    df = build_training_dataset()
    feature_cols = get_feature_columns(df)
    print(f"Dataset pronto: {len(df)} jogos, {len(feature_cols)} features.")
    labels = sorted(SUCCESS_LABELS.keys())
    target_names = [SUCCESS_LABELS[i] for i in labels]

    X = df[feature_cols].astype(float)
    y_cls = df["engagement_class"]
    y_reg = df["recommendations_total"].astype(float)

    (
        X_train,
        X_test,
        y_cls_train,
        y_cls_test,
        y_reg_train,
        y_reg_test,
    ) = train_test_split(
        X,
        y_cls,
        y_reg,
        test_size=0.2,
        random_state=random_state,
        stratify=y_cls,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    print("Treinando classificação — Regressão Logística...")
    pipe_log = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    search_log = GridSearchCV(
        pipe_log,
        {"clf__C": [0.1, 1.0, 10.0]},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    search_log.fit(X_train, y_cls_train)
    pred_log = search_log.predict(X_test)
    metrics_log = _cls_metrics(y_cls_test, pred_log, labels, target_names)
    metrics_log["best_params"] = search_log.best_params_

    print("Treinando classificação — KNN...")
    pipe_knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    search_knn = GridSearchCV(
        pipe_knn,
        {"clf__n_neighbors": [5, 15, 31]},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    search_knn.fit(X_train, y_cls_train)
    pred_knn = search_knn.predict(X_test)
    metrics_knn = _cls_metrics(y_cls_test, pred_knn, labels, target_names)
    metrics_knn["best_params"] = search_knn.best_params_

    print("Treinando classificação — Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_cls_train)
    pred_rf = rf.predict(X_test)
    metrics_rf = _cls_metrics(y_cls_test, pred_rf, labels, target_names)
    importances = sorted(
        zip(feature_cols, rf.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    cls_scores = {
        "logistic_regression": metrics_log["f1_macro"],
        "knn": metrics_knn["f1_macro"],
        "random_forest": metrics_rf["f1_macro"],
    }
    best_cls_name = max(cls_scores, key=cls_scores.get)

    print("Treinando regressão (baseline, linear, polinomial, RF)...")
    y_log_train = np.log1p(y_reg_train)

    dummy = DummyRegressor(strategy="median")
    dummy.fit(X_train, y_reg_train)
    pred_dummy = np.clip(dummy.predict(X_test), 0, None)
    metrics_dummy = _reg_metrics(y_reg_test, pred_dummy)

    simple = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    simple.fit(X_train[["price_usd"]], y_log_train)
    pred_simple = np.clip(np.expm1(simple.predict(X_test[["price_usd"]])), 0, None)
    metrics_simple = _reg_metrics(y_reg_test, pred_simple)

    multiple = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    multiple.fit(X_train, y_log_train)
    pred_multiple = np.clip(np.expm1(multiple.predict(X_test)), 0, None)
    metrics_multiple = _reg_metrics(y_reg_test, pred_multiple)

    polynomial = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )
    polynomial.fit(X_train, y_log_train)
    pred_poly = np.clip(np.expm1(polynomial.predict(X_test)), 0, None)
    metrics_poly = _reg_metrics(y_reg_test, pred_poly)

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_log_train)
    pred_rf_reg = np.clip(np.expm1(rf_reg.predict(X_test)), 0, None)
    metrics_rf_reg = _reg_metrics(y_reg_test, pred_rf_reg)

    reg_scores = {
        "dummy_median": metrics_dummy["mae"],
        "linear_simple": metrics_simple["mae"],
        "linear_multiple": metrics_multiple["mae"],
        "polynomial": metrics_poly["mae"],
        "random_forest_regressor": metrics_rf_reg["mae"],
    }
    best_reg_name = min(reg_scores, key=reg_scores.get)

    print("Ajustando K-Means...")
    scaler_km = StandardScaler()
    X_train_scaled = scaler_km.fit_transform(X_train)
    kmeans = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    km_labels = kmeans.fit_predict(X_train_scaled)
    silhouette = float(silhouette_score(X_train_scaled, km_labels, sample_size=5000, random_state=random_state))

    print("Salvando modelos e relatório...")
    save_processed_dataset(df, ARTIFACTS_DIR / "training_dataset.parquet")
    joblib.dump(search_log.best_estimator_, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(search_knn.best_estimator_, MODELS_DIR / "knn.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    joblib.dump(multiple, MODELS_DIR / "linear_multiple.joblib")
    joblib.dump(polynomial, MODELS_DIR / "polynomial.joblib")
    joblib.dump(rf_reg, MODELS_DIR / "random_forest_regressor.joblib")

    metrics = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_count": len(feature_cols),
        "classification": {
            "logistic_regression": {k: v for k, v in metrics_log.items() if k != "report"},
            "knn": {k: v for k, v in metrics_knn.items() if k != "report"},
            "random_forest": {k: v for k, v in metrics_rf.items() if k != "report"},
            "best_model": best_cls_name,
        },
        "regression": {
            "dummy_median": metrics_dummy,
            "linear_simple": metrics_simple,
            "linear_multiple": metrics_multiple,
            "polynomial": metrics_poly,
            "random_forest_regressor": metrics_rf_reg,
            "best_model": best_reg_name,
            "target_transform": "log1p no treino; métricas na escala original",
        },
        "kmeans": {"k": 4, "silhouette": silhouette},
        "top_features": importances[:10],
        "accuracy": metrics_rf["accuracy"],
        "f1_macro": metrics_rf["f1_macro"],
    }

    config = {
        "feature_columns": feature_cols,
        "success_labels": {str(k): v for k, v in SUCCESS_LABELS.items()},
        "metrics": metrics,
    }
    with open(ARTIFACTS_DIR / "model_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)

    report_path = ARTIFACTS_DIR / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("=== Relatório — Perfil de engajamento e popularidade Steam ===\n\n")
        fh.write(f"Treino: {metrics['train_size']} | Teste: {metrics['test_size']} | Features: {metrics['feature_count']}\n")
        fh.write("Alvo classificação: engagement_class (tercis de recommendations)\n")
        fh.write("Alvo regressão: recommendations_total (contínuo)\n")
        fh.write("Recommendations NÃO entram como feature.\n\n")

        fh.write("--- Classificação ---\n")
        fh.write(f"Melhor modelo (F1 macro): {best_cls_name}\n\n")
        fh.write("Modelo A — Regressão Logística\n")
        fh.write(f"Params: {metrics_log['best_params']}\n")
        fh.write(f"Accuracy: {metrics_log['accuracy']:.3f} | F1 macro: {metrics_log['f1_macro']:.3f}\n")
        fh.write(metrics_log["report"] + "\n")
        fh.write("Modelo B — KNN\n")
        fh.write(f"Params: {metrics_knn['best_params']}\n")
        fh.write(f"Accuracy: {metrics_knn['accuracy']:.3f} | F1 macro: {metrics_knn['f1_macro']:.3f}\n")
        fh.write(metrics_knn["report"] + "\n")
        fh.write("Ensemble — Random Forest\n")
        fh.write(f"Accuracy: {metrics_rf['accuracy']:.3f} | F1 macro: {metrics_rf['f1_macro']:.3f}\n")
        fh.write(metrics_rf["report"] + "\n")

        fh.write("--- Regressão (MAE / RMSE / R² na escala original) ---\n")
        fh.write(f"Melhor modelo (menor MAE): {best_reg_name}\n")
        for name, vals in (
            ("Baseline mediana", metrics_dummy),
            ("Linear simples (price_usd)", metrics_simple),
            ("Linear múltipla", metrics_multiple),
            ("Polinomial grau 2 + Ridge", metrics_poly),
            ("Random Forest Regressor", metrics_rf_reg),
        ):
            fh.write(
                f"  {name}: MAE={vals['mae']:.1f} | RMSE={vals['rmse']:.1f} | R²={vals['r2']:.3f}\n"
            )
        fh.write("\n--- K-Means (k=4, treino) ---\n")
        fh.write(f"Silhouette: {silhouette:.3f}\n\n")
        fh.write("Top 10 features (Random Forest classificador):\n")
        for name, imp in importances[:10]:
            fh.write(f"  {name}: {imp:.4f}\n")

    metrics["classification_report"] = (
        f"Melhor classificador: {best_cls_name}\n"
        + metrics_log["report"]
        + "\n"
        + metrics_knn["report"]
        + "\n"
        + metrics_rf["report"]
    )
    return metrics


if __name__ == "__main__":
    result = train_model()
    print("Melhor classificador:", result["classification"]["best_model"])
    print("Melhor regressor:", result["regression"]["best_model"])
    print(f"RF Accuracy: {result['accuracy']:.3f} | RF F1: {result['f1_macro']:.3f}")
    print(result["classification_report"])
