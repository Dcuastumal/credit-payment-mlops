"""API: python -m uvicorn src.model_deploy:app --host 127.0.0.1 --port 8000."""
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.fe_engineering import ROOT


class CreditInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tipo_credito: int = Field(gt=0)
    capital_prestado: float = Field(gt=0)
    plazo_meses: int = Field(gt=0)
    edad_cliente: int | None = Field(default=None, ge=18, le=100)
    tipo_laboral: Literal["Empleado", "Independiente"]
    salario_cliente: float | None = Field(default=None, gt=0)
    total_otros_prestamos: float = Field(ge=0)
    cuota_pactada: float = Field(gt=0)


@lru_cache(maxsize=1)
def load_model():
    path = ROOT / "src/model.joblib"
    if not path.exists():
        raise FileNotFoundError("Ejecuta python -m src.model_training_evaluation antes de predecir.")
    return joblib.load(path)


def predict_credit(credit: CreditInput):
    artifact = load_model()
    frame = pd.DataFrame([credit.model_dump()])
    probability = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    alert = probability >= artifact["threshold"]
    return {"score_riesgo": probability, "umbral": artifact["threshold"],
            "alerta_revision": bool(alert), "Pago_atiempo_predicho": int(not alert),
            "modelo": artifact["metadata"]["selected_model"],
            "nota": "Score no calibrado. Uso académico; no es una decisión de aprobación de crédito."}


app = FastAPI(title="Credit Payment MLOps", version="1.2.0",
              description="Prototipo académico para estimar riesgo de pago fuera de plazo.")


@app.get("/health")
def health():
    try:
        artifact = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "model": artifact["metadata"]["selected_model"]}


@app.post("/predict")
def predict(credit: CreditInput):
    try:
        return predict_credit(credit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
