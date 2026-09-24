"""Monitoreo offline de distribución (PSI), faltantes y rendimiento con etiquetas.

Sin --current usa el test histórico: NO representa monitoreo de producción.
Ejemplo: python -m src.model_monitoring --current nuevos_creditos.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.fe_engineering import ROOT, FEATURES, CATEGORICAL, CreditFeatures
from src.model_training_evaluation import evaluate


def psi(reference, current, categorical=False):
    """Bins definidos SOLO con referencia, incluyendo faltantes y categorías nuevas."""
    if categorical:
        a = reference.fillna("<FALTANTE>").astype(str)
        b = current.fillna("<FALTANTE>").astype(str)
    else:
        a = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan)
        observed = a.dropna()
        if observed.empty:
            a = a.notna().astype(str); b = b.notna().astype(str)
        elif observed.nunique() == 1:
            value = observed.iloc[0]
            def constant_bins(s):
                return pd.Series(np.select([s.isna(), s < value, s > value],
                                           ["missing", "lower", "higher"], default="equal"), index=s.index)
            a, b = constant_bins(a), constant_bins(b)
        else:
            cuts = np.unique(observed.quantile(np.linspace(0, 1, 11)).to_numpy())
            edges = np.r_[-np.inf, cuts[1:-1], np.inf]
            a = pd.cut(a, bins=edges, labels=False).fillna(-1).astype(str)
            b = pd.cut(b, bins=edges, labels=False).fillna(-1).astype(str)
    categories = sorted(set(a) | set(b))
    # Pseudoconteo pequeño para evitar log(0); se normaliza después de sumar.
    p = a.value_counts().reindex(categories, fill_value=0).astype(float) + 0.5
    q = b.value_counts().reindex(categories, fill_value=0).astype(float) + 0.5
    p, q = p / p.sum(), q / q.sum()
    return float(((q - p) * np.log(q / p)).sum())


def drift_report(reference, current):
    if reference.empty or current.empty:
        raise ValueError("Referencia y lote deben contener filas.")
    a, b = CreditFeatures().transform(reference), CreditFeatures().transform(current)
    rows = []
    for column in a.columns:
        value = psi(a[column], b[column], column in CATEGORICAL)
        rows.append({"variable": column, "psi": value,
                     "estado": "revisar" if value >= 0.2 else "observar" if value >= 0.1 else "estable",
                     "faltantes_referencia": float(a[column].isna().mean()),
                     "faltantes_actual": float(b[column].isna().mean())})
    return rows


def monitor(current_path=None, output_path=None):
    path = Path(current_path or ROOT / "src/monitoring_sample.csv")
    reference = pd.read_csv(ROOT / "src/reference_data.csv")
    current = pd.read_csv(path)
    rows = drift_report(reference, current)
    result = {"mode": "lote_proporcionado" if current_path else "demostracion_test_historico",
              "current_file": path.name, "n_reference": len(reference), "n_current": len(current),
              "thresholds": "PSI >= 0.1 observar; >= 0.2 revisar. Heurísticas, no pruebas de hipótesis.",
              "variables": rows, "performance": None}
    if "Pago_atiempo" in current:
        labeled = current.loc[current.Pago_atiempo.notna()].copy()
        if not labeled.Pago_atiempo.isin([0, 1]).all():
            raise ValueError("Las etiquetas deben ser 0 o 1.")
        if len(labeled):
            artifact = joblib.load(ROOT / "src/model.joblib")
            probability = artifact["pipeline"].predict_proba(labeled[FEATURES])[:, 1]
            result["performance"] = evaluate(1 - labeled.Pago_atiempo, probability, artifact["threshold"])
            result["labeled_fraction"] = len(labeled) / len(current)
    target = Path(output_path or ROOT / "src/monitoring_report.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Monitoreo: {target.name}; {sum(r['estado'] == 'revisar' for r in rows)} variables para revisar.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    monitor(args.current, args.output)
