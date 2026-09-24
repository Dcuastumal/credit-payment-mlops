"""Interfaz: python -m streamlit run src/app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from src.model_deploy import CreditInput, predict_credit

st.set_page_config(page_title="Credit Payment", page_icon="📊", layout="wide")
st.title("Credit Payment · Proyecto integrador")
st.caption("Predicción de pago, evaluación del modelo y monitoreo de datos")
st.info("Demostración académica. El score no está calibrado y no debe utilizarse para aprobar o rechazar créditos.")
predict_tab, metrics_tab, drift_tab = st.tabs(["Predicción", "Resultados", "Monitoreo"])

with predict_tab:
    st.write("Introduce los datos de una solicitud. Usa las mismas unidades monetarias de la base original.")
    with st.form("credit"):
        left, right = st.columns(2)
        with left:
            credit_type = st.selectbox("Tipo de crédito (código)", [4, 9, 10, 6, 7, 68])
            capital = st.number_input("Capital solicitado", min_value=1.0, value=2000000.0, step=100000.0)
            months = st.number_input("Plazo en meses", min_value=1, value=12, step=1)
            age = st.number_input("Edad", min_value=18, max_value=100, value=40, step=1)
        with right:
            job = st.selectbox("Tipo laboral", ["Empleado", "Independiente"])
            salary = st.number_input("Ingreso declarado", min_value=1.0, value=3000000.0, step=100000.0)
            others = st.number_input("Monto de otros préstamos", min_value=0.0, value=1000000.0, step=100000.0)
            payment = st.number_input("Cuota pactada", min_value=1.0, value=200000.0, step=10000.0)
        submitted = st.form_submit_button("Evaluar solicitud")
    if submitted:
        try:
            result = predict_credit(CreditInput(tipo_credito=credit_type, capital_prestado=capital,
                plazo_meses=months, edad_cliente=age, tipo_laboral=job, salario_cliente=salary,
                total_otros_prestamos=others, cuota_pactada=payment))
            st.metric("Score de riesgo (0–1; no probabilidad calibrada)", f"{result['score_riesgo']:.3f}")
            st.write(f"Umbral de revisión: {result['umbral']:.2f}")
            if result["alerta_revision"]:
                st.warning("Señal para revisión: score igual o superior al umbral.")
            else:
                st.success("Score inferior al umbral. Esto no garantiza el pago a tiempo.")
            st.json(result)
        except FileNotFoundError as exc:
            st.error(str(exc))

with metrics_tab:
    path = ROOT / "src/metrics.json"
    if path.exists():
        info = json.loads(path.read_text(encoding="utf-8"))
        st.subheader(info["selected_model"])
        st.caption("Resultados en test temporal. Clase positiva: pago fuera de plazo.")
        a, b, c = st.columns(3)
        a.metric("Recall del riesgo", f"{info['test']['recall_riesgo']:.1%}")
        b.metric("Precision del riesgo", f"{info['test']['precision_riesgo']:.1%}")
        c.metric("Average precision", f"{info['test']['average_precision']:.3f}")
        st.dataframe(pd.DataFrame(info["candidates"]), hide_index=True)
        st.image(str(ROOT / "src/evaluation.png"))
        st.write("La elección del modelo se hizo con entrenamiento; el umbral se eligió en validación.")
        st.json(info["split"])
    else:
        st.warning("Ejecuta el entrenamiento para generar los resultados.")

with drift_tab:
    path = ROOT / "src/monitoring_report.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        st.caption(f"Modo: {report['mode']} · Lote: {report['current_file']}")
        st.write("PSI compara distribuciones. Una alerta requiere investigación; no demuestra que el modelo falle.")
        st.dataframe(pd.DataFrame(report["variables"]), hide_index=True)
        st.caption(report["thresholds"])
    else:
        st.warning("Ejecuta python -m src.model_monitoring para generar el reporte.")
