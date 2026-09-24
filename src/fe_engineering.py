"""Limpieza e ingeniería compartidas por entrenamiento, API y aplicación.

Las reglas fijas no aprenden del dataset. Medianas y escalas se ajustan
exclusivamente dentro del Pipeline, usando los datos de entrenamiento.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "tipo_credito", "capital_prestado", "plazo_meses", "edad_cliente",
    "tipo_laboral", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
]
CATEGORICAL = ["tipo_credito", "tipo_laboral"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
DERIVED = ["cuota_ingreso", "capital_ingreso", "otros_prestamos_ingreso"]


def load_data(path=None):
    """Lee la fuente sin modificar el archivo original."""
    path = Path(path or ROOT / "Base_de_datos.csv")
    data = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
    required = FEATURES + ["fecha_prestamo", "Pago_atiempo"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    data["fecha_prestamo"] = pd.to_datetime(data["fecha_prestamo"], errors="raise")
    if data["fecha_prestamo"].isna().any():
        raise ValueError("Hay fechas vacías; revisar antes de separar los datos.")
    if data["Pago_atiempo"].isna().any() or not data["Pago_atiempo"].isin([0, 1]).all():
        raise ValueError("Pago_atiempo debe contener solamente 0 o 1, sin nulos.")
    return data


def temporal_split(data):
    """60/20/20 aproximado, sin repartir una misma fecha-hora entre conjuntos."""
    data = data.drop_duplicates().sort_values("fecha_prestamo", kind="stable")
    cut1 = data["fecha_prestamo"].iloc[int(len(data) * 0.6)]
    cut2 = data["fecha_prestamo"].iloc[int(len(data) * 0.8)]
    train = data.loc[data.fecha_prestamo < cut1].copy()
    valid = data.loc[(data.fecha_prestamo >= cut1) & (data.fecha_prestamo < cut2)].copy()
    test = data.loc[data.fecha_prestamo >= cut2].copy()
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        if len(part) == 0 or part.Pago_atiempo.nunique() != 2:
            raise ValueError(f"{name} debe tener registros de las dos clases.")
    return train, valid, test


class CreditFeatures(TransformerMixin, BaseEstimator):
    """Transformador sencillo, sin estadísticas aprendidas ni acceso al objetivo."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = set(FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan variables para predecir: {sorted(missing)}")
        data = X[FEATURES].copy()
        for col in NUMERIC:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            data.loc[data[col] < 0, col] = np.nan
        # Rango operativo asumido: requiere confirmación del negocio.
        data.loc[~data.edad_cliente.between(18, 100), "edad_cliente"] = np.nan
        for col in ["capital_prestado", "plazo_meses", "salario_cliente", "cuota_pactada"]:
            data.loc[data[col] == 0, col] = np.nan
        data["tipo_credito"] = data["tipo_credito"].map(
            lambda value: str(int(value)) if pd.notna(value) else "Desconocido"
        )
        data["tipo_laboral"] = data["tipo_laboral"].fillna("Desconocido").astype(str).str.strip()
        income = data["salario_cliente"]
        data["cuota_ingreso"] = data["cuota_pactada"] / income
        data["capital_ingreso"] = data["capital_prestado"] / income
        data["otros_prestamos_ingreso"] = data["total_otros_prestamos"] / income
        return data.replace([np.inf, -np.inf], np.nan)


def make_pipeline(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", RobustScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC + DERIVED),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("features", CreditFeatures()), ("preprocess", preprocess), ("model", model)])
