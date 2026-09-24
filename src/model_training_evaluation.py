"""Entrena candidatos, elige modelo con CV temporal y umbral con validación.

Ejecutar desde la raíz: python -m src.model_training_evaluation
La clase positiva interna es riesgo de pago fuera de plazo (1 - Pago_atiempo).
"""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, fbeta_score, precision_score,
    recall_score, f1_score, roc_auc_score, accuracy_score, ConfusionMatrixDisplay,
    PrecisionRecallDisplay, RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.fe_engineering import ROOT, FEATURES, load_data, temporal_split, make_pipeline


def evaluate(y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    result = {
        "accuracy": accuracy_score(y, prediction),
        "precision_riesgo": precision_score(y, prediction, zero_division=0),
        "recall_riesgo": recall_score(y, prediction, zero_division=0),
        "f1_riesgo": f1_score(y, prediction, zero_division=0),
        "f2_riesgo": fbeta_score(y, prediction, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) == 2 else None,
        "average_precision": average_precision_score(y, probability) if np.sum(y) else None,
        "tasa_alertas": float(prediction.mean()),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "n": len(y), "n_riesgo": int(np.sum(y)),
    }
    return result


def choose_threshold(y, probability):
    # El test no participa. F2 da más peso a recall que a precision.
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [fbeta_score(y, probability >= t, beta=2, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def train_model(output_dir=None):
    output = Path(output_dir or ROOT / "src")
    output.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / "Base_de_datos.csv"
    data = load_data(data_path)
    train, valid, test = temporal_split(data)
    X_train, y_train = train[FEATURES], 1 - train.Pago_atiempo
    X_valid, y_valid = valid[FEATURES], 1 - valid.Pago_atiempo
    X_test, y_test = test[FEATURES], 1 - test.Pago_atiempo
    cv = TimeSeriesSplit(n_splits=3)
    for a, b in cv.split(X_train):
        if y_train.iloc[a].nunique() < 2 or y_train.iloc[b].nunique() < 2:
            raise ValueError("Un fold temporal no contiene ambas clases.")
    candidates = {
        "RegresionLogistica": (
            LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=42),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=1),
            {"model__max_depth": [5, 10], "model__min_samples_leaf": [10, 30]},
        ),
    }
    searches, comparison = {}, []
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(make_pipeline(model), grid, scoring="average_precision", cv=cv,
                              n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        searches[name] = search
        comparison.append({"model": name, "cv_average_precision": float(search.best_score_),
                           "parameters": search.best_params_})
        print(name, search.best_score_, search.best_params_, flush=True)
    winner = max(comparison, key=lambda row: row["cv_average_precision"])["model"]
    pipeline = searches[winner].best_estimator_
    validation_probability = pipeline.predict_proba(X_valid)[:, 1]
    threshold = choose_threshold(y_valid, validation_probability)
    probability = pipeline.predict_proba(X_test)[:, 1]
    baseline = make_pipeline(DummyClassifier(strategy="prior"))
    baseline.fit(X_train, y_train)
    baseline_probability = baseline.predict_proba(X_test)[:, 1]
    split_info = {}
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        split_info[name] = {"n": len(part), "n_riesgo": int((part.Pago_atiempo == 0).sum()),
                            "start": str(part.fecha_prestamo.min()), "end": str(part.fecha_prestamo.max())}
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib"]},
        "positive_class": "riesgo = 1 - Pago_atiempo; interpretación pendiente de diccionario",
        "features": FEATURES, "seed": 42, "split": split_info,
        "candidates": comparison, "selected_model": winner,
        "selection": "Average precision media en tres folds temporales del entrenamiento",
        "threshold_selection": "Mayor F2 en validación; sin reajuste posterior del modelo",
        "validation": evaluate(y_valid, validation_probability, threshold),
        "test": evaluate(y_test, probability, threshold),
        "baseline_test": evaluate(y_test, baseline_probability, 0.5),
    }
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "metadata": metadata}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    train[FEATURES].to_csv(output / "reference_data.csv", index=False)
    test.to_csv(output / "monitoring_sample.csv", index=False)
    pd.DataFrame({"row_index": test.index, "fecha_prestamo": test.fecha_prestamo,
                  "riesgo_real": y_test, "score_riesgo": probability,
                  "alerta": (probability >= threshold).astype(int)}).to_csv(output / "test_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ConfusionMatrixDisplay.from_predictions(y_test, probability >= threshold, labels=[0, 1],
        display_labels=["A tiempo", "Fuera de plazo"], ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Test: matriz de confusión (umbral {threshold:.2f})")
    axes[0].set_xlabel("Predicción"); axes[0].set_ylabel("Clase real")
    RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1], name=winner)
    axes[1].plot([0, 1], [0, 1], "--", color="grey"); axes[1].set_title("Test: curva ROC")
    PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[2], name=winner)
    axes[2].axhline(y_test.mean(), linestyle="--", color="grey", label="Prevalencia test")
    axes[2].legend(fontsize=8); axes[2].set_title("Test: precision y recall del riesgo")
    fig.tight_layout(); fig.savefig(output / "evaluation.png", dpi=150); plt.close(fig)
    print(json.dumps(metadata["test"], indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    train_model()
