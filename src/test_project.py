"""Pruebas de riesgos reales: fugas, entradas inválidas, serialización y drift."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.fe_engineering import ROOT, FEATURES, CreditFeatures, load_data, temporal_split
from src.model_deploy import app, CreditInput, predict_credit, load_model
from src.model_monitoring import psi, drift_report
from src.model_training_evaluation import evaluate, choose_threshold


@pytest.fixture
def example():
    return {"tipo_credito": 4, "capital_prestado": 2000000, "plazo_meses": 12,
            "edad_cliente": 40, "tipo_laboral": "Empleado", "salario_cliente": 3000000,
            "total_otros_prestamos": 1000000, "cuota_pactada": 200000}


def test_temporal_boundaries():
    train, valid, test = temporal_split(load_data())
    assert train.fecha_prestamo.max() < valid.fecha_prestamo.min()
    assert valid.fecha_prestamo.max() < test.fecha_prestamo.min()
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(valid) + len(test) == len(load_data().drop_duplicates())


def test_target_never_enters_transformer(example):
    a = pd.DataFrame([example | {"Pago_atiempo": 0, "puntaje": -999}])
    b = pd.DataFrame([example | {"Pago_atiempo": 1, "puntaje": 999}])
    pd.testing.assert_frame_equal(CreditFeatures().transform(a), CreditFeatures().transform(b))


def test_zero_income_and_invalid_age(example):
    frame = pd.DataFrame([example | {"salario_cliente": 0, "edad_cliente": 123}])
    clean = CreditFeatures().transform(frame)
    assert clean.edad_cliente.isna().all()
    assert clean.cuota_ingreso.isna().all()
    p = load_model()["pipeline"].predict_proba(frame)
    assert np.isfinite(p).all()


def test_unknown_credit_category(example):
    result = predict_credit(CreditInput(**(example | {"tipo_credito": 999})))
    assert 0 <= result["score_riesgo"] <= 1


def test_imputer_learned_only_train():
    train, _, _ = temporal_split(load_data())
    clean = CreditFeatures().transform(train)
    imputer = load_model()["pipeline"].named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    from src.fe_engineering import NUMERIC, DERIVED
    np.testing.assert_allclose(imputer.statistics_, clean[NUMERIC + DERIVED].median().values)


def test_serialization_roundtrip(example, tmp_path):
    artifact = load_model()
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    restored = joblib.load(path)
    frame = pd.DataFrame([example])
    np.testing.assert_allclose(artifact["pipeline"].predict_proba(frame), restored["pipeline"].predict_proba(frame))


def test_api_matches_python(example):
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=example)
        assert response.status_code == 200
        assert response.json() == predict_credit(CreditInput(**example))


@pytest.mark.parametrize("change", [{"capital_prestado": -1}, {"plazo_meses": 0},
    {"edad_cliente": 123}, {"tipo_laboral": "otro"}, {"Pago_atiempo": 1}, {"salario_cliente": 0}])
def test_api_rejects_invalid_input(example, change):
    with TestClient(app) as client:
        assert client.post("/predict", json=example | change).status_code == 422


def test_api_missing_field(example):
    example.pop("capital_prestado")
    with TestClient(app) as client:
        assert client.post("/predict", json=example).status_code == 422


def test_api_optional_missing(example):
    result = predict_credit(CreditInput(**(example | {"edad_cliente": None, "salario_cliente": None})))
    assert np.isfinite(result["score_riesgo"])


def test_drift_same_and_shifted():
    reference = pd.Series(np.arange(1000, dtype=float))
    assert psi(reference, reference) == pytest.approx(0)
    assert psi(reference, reference + 10000) > 0.2
    assert psi(pd.Series([0]*100), pd.Series([1]*100)) > 0.2
    assert psi(pd.Series(["A"]*100), pd.Series(["B"]*100), True) > 0.2
    assert psi(pd.Series([np.nan]*100), pd.Series([5]*100)) > 0.2


def test_empty_drift_fails():
    with pytest.raises(ValueError):
        drift_report(pd.DataFrame(columns=FEATURES), pd.DataFrame(columns=FEATURES))


def test_metrics_orientation():
    result = evaluate(np.array([0, 0, 1, 1]), np.array([.1, .8, .7, .2]), .5)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["recall_riesgo"] == .5
    assert choose_threshold(np.array([0, 1]), np.array([.1, .9])) > .1


def test_saved_metrics_match_predictions():
    info = json.loads((ROOT / "src/metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(ROOT / "src/test_predictions.csv")
    metrics = evaluate(predictions.riesgo_real, predictions.score_riesgo, info["test"]["threshold"])
    assert metrics["confusion_matrix"] == info["test"]["confusion_matrix"]
    assert metrics["average_precision"] == pytest.approx(info["test"]["average_precision"])


def test_streamlit_form():
    from streamlit.testing.v1 import AppTest
    application = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
    assert not application.exception
    application.button[0].click().run()
    assert not application.exception
    assert len(application.json) >= 1
