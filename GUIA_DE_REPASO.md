# Guía de repaso y defensa

La meta es que puedas ejecutar, explicar y modificar el proyecto. Lee una etapa, ejecuta su archivo y responde las preguntas antes de pasar a la siguiente. No memorices métricas sin entender qué significan.

## Sesión 1 · Entender el problema y cargar los datos

**Archivo:** `src/Cargar_datos.ipynb`.

1. Explica qué representa una fila: asumimos que es un crédito, no necesariamente un cliente único.
2. Identifica `Pago_atiempo` y diferencia variable objetivo de variables predictoras.
3. Ejecuta la lectura y comprueba 10.763 filas y 23 columnas.
4. Localiza los faltantes y los tipos de datos.
5. Comprueba que el CSV se vuelve a leer sin pérdida de valores, considerando las columnas de tipos mixtos.

**Pregunta de defensa:** ¿Por qué no empezaste directamente a entrenar?

**Respuesta orientativa:** Primero debía verificar qué información había, qué significaba el objetivo y qué problemas de calidad podían afectar el resultado. El nombre de una columna no sustituye un diccionario de datos.

**Ejercicio:** encuentra la columna con más faltantes y calcula su porcentaje sin copiar el resultado del README.

## Sesión 2 · EDA y separación temporal

**Archivo:** `src/comprension_eda.ipynb`.

1. Explica por qué reservamos el periodo más reciente para test.
2. Identifica qué gráficos son univariables, bivariables y multivariables.
3. Interpreta el desbalance del objetivo.
4. Compara media y mediana de ingresos y explica qué hacen los valores extremos.
5. Interpreta una correlación sin afirmar causalidad.

**Pregunta:** ¿Por qué no hiciste un split aleatorio?

**Respuesta:** El objetivo habla de usuarios nuevos. Una separación temporal se parece más a entrenar con el pasado y evaluar con solicitudes posteriores. Aun así, falta conocer la maduración de etiquetas y no tenemos ID de cliente para controlar repeticiones.

**Ejercicio:** describe una asociación visible en un gráfico e incluye el tamaño del grupo. Después explica por qué esa asociación no demuestra una causa.

## Sesión 3 · Ingeniería de características

**Archivo:** `src/fe_engineering.py`.

Lee en este orden: `FEATURES`, `temporal_split`, `CreditFeatures.transform`, `make_pipeline`.

| Componente | Qué hace | Por qué importa |
|---|---|---|
| Reglas fijas | Marcan valores fuera del contrato operativo como faltantes | Evitan divisiones por cero y valores no utilizables |
| Cocientes | Relacionan cuota, capital y préstamos con el ingreso | Añaden información relativa a los montos absolutos |
| SimpleImputer | Aprende medianas y añade indicadores de faltantes | Usa estadísticas de entrenamiento, no de test |
| RobustScaler | Centra y escala usando mediana y rango intercuartílico | Reduce sensibilidad de la escala a extremos; no elimina outliers |
| OneHotEncoder | Convierte categorías en columnas | No supone un orden numérico entre tipos de crédito |
| Pipeline | Encadena transformaciones y modelo | Mantiene el mismo proceso en entrenamiento y predicción |

**Pregunta:** ¿Qué es una fuga de información?

**Respuesta:** Usar datos que no estarían disponibles al predecir, como información posterior al resultado, o aprender estadísticas del test. El pipeline controla la segunda; para la primera hace falta conocer la procedencia temporal de cada campo.

**Ejercicio:** calcula a mano `cuota_ingreso` para ingreso 3.000.000 y cuota 200.000. Debe dar aproximadamente 0,0667. No lo llames porcentaje mensual hasta confirmar la periodicidad del ingreso.

## Sesión 4 · Modelos y métricas

**Archivo:** `src/model_training_evaluation.py`.

1. Lee cómo se define `y = 1 - Pago_atiempo`.
2. Compara un modelo constante, regresión logística y Random Forest.
3. Identifica los hiperparámetros que se prueban.
4. Explica `TimeSeriesSplit` y `GridSearchCV`.
5. Distingue elección del modelo de elección del umbral.
6. Abre `src/evaluation.png` y explica sus tres gráficos.

**Matriz de confusión del test:**

| Concepto | Cantidad | Interpretación |
|---|---:|---|
| TP | 18 | Se alertó y el pago fue fuera de plazo |
| FP | 334 | Se alertó, pero el pago fue a tiempo |
| FN | 51 | No se alertó y el pago fue fuera de plazo |
| TN | 1.750 | No se alertó y el pago fue a tiempo |

- Recall = 18 / (18 + 51) ≈ 26,1 %: de los atrasos observados, cuántos detectamos.
- Precision = 18 / (18 + 334) ≈ 5,1 %: de las alertas emitidas, cuántas fueron correctas.
- Accuracy = (18 + 1.750) / 2.153 ≈ 82,1 %: proporción total de aciertos.
- F1 combina precision y recall con el mismo peso.
- F2 da más peso a recall. No significa que el costo de FN sea exactamente dos veces el de FP.
- ROC-AUC resume la capacidad de ordenar positivos por encima de negativos.
- Average precision resume precision a distintos niveles de recall y es útil con desbalance.

**Pregunta:** ¿Cuál fue el mejor modelo?

**Respuesta:** Random Forest fue el mejor de los dos candidatos por average precision en la validación cruzada temporal de entrenamiento. Ser el mejor de los probados no significa ser suficientemente bueno: el test muestra una capacidad limitada y muchas falsas alertas.

**Pregunta:** ¿Por qué el modelo base tiene mayor accuracy?

**Respuesta:** La mayoría paga a tiempo. Predecir siempre esa clase acierta mucho, pero no encuentra ningún atraso. Por eso accuracy no fue la métrica de selección.

**Pregunta:** ¿Por qué el umbral no es 0,5?

**Respuesta:** Se eligió 0,53 al maximizar F2 en validación. No se eligió mirando el test ni se asegura que sea el umbral adecuado para costos reales.

**Ejercicio:** explica qué ocurriría al bajar el umbral. En general aumentan o se mantienen las alertas y el recall; la precision no tiene por qué cambiar de forma monótona.

## Sesión 5 · API y Streamlit

**Archivos:** `src/model_deploy.py` y `src/app.py`.

1. Arranca la API con el comando del README.
2. Abre `/docs`, prueba `/health` y después `/predict` con `src/example_request.json`.
3. Envía un capital negativo: debe devolver HTTP 422.
4. Arranca Streamlit y evalúa la misma solicitud.
5. Comprueba que ambas interfaces usan `predict_credit` y el mismo artefacto.

**Pregunta:** ¿Qué guardaste en `model.joblib`?

**Respuesta:** El pipeline ya ajustado, el umbral y metadatos. Así no vuelvo a calcular medianas ni escalas con la solicitud nueva.

**Pregunta:** ¿El score de 0,6 significa 60 % de probabilidad de atraso?

**Respuesta:** No podemos afirmarlo. Es un score producido por un modelo no calibrado con pesos de clase. Para interpretarlo como probabilidad hace falta calibración y evaluación adicional.

## Sesión 6 · Monitoreo y MLOps

**Archivos:** `src/model_monitoring.py`, `run_pipeline.py`, `Jenkinsfile` y `Dockerfile`.

1. Ejecuta el monitoreo por defecto y explica que compara train con test histórico.
2. Localiza las variables con PSI alto en `monitoring_report.json`.
3. Diferencia drift de datos de degradación de rendimiento.
4. Explica por qué hacen falta etiquetas maduras para medir rendimiento real.
5. Ejecuta las pruebas y revisa la cobertura sin confundirla con corrección total.

**Pregunta:** ¿Qué aporta Docker?

**Respuesta:** Describe un entorno empaquetado con Python, dependencias, código y modelo. El Dockerfile no es la imagen: debemos construirla y ejecutarla para comprobarla.

**Pregunta:** ¿Qué aporta Jenkins?

**Respuesta:** Puede repetir instalación, ejecución y pruebas ante cambios en el repositorio. El archivo incluido necesita un agente y un job configurados; no demuestra por sí solo un despliegue real.

**Pregunta:** ¿Reentrenarías automáticamente al detectar drift?

**Respuesta:** Primero investigaría cambios de origen, faltantes, unidades y población. Drift no significa necesariamente que el modelo haya empeorado, y reentrenar con datos incorrectos puede empeorarlo.

## Sesión 7 · Git y cierre

Trabaja en `developer`, valida en `certification` e integra lo aprobado en `master`. La carpeta descargada no sustituye el historial del repositorio. Usa commits reales y no crees etiquetas de avances sobre contenidos que no les corresponden.

Revisa con el docente los nombres exactos de archivos, la política de versiones y si permite los archivos complementarios de despliegue. Jenkins institucional no fue accesible para esta preparación.

Antes de entregar, confirma acceso público al repositorio, ejecución local, construcción de Docker y el estado real de SonarCloud si vas a presentar el extra credit.

## Preguntas de negocio y dónde buscar el código

| Pregunta | Archivo / función |
|---|---|
| ¿Qué datos recibimos y qué falta? | `Cargar_datos.ipynb`: resumen de calidad |
| ¿Cuál es el desbalance? | `comprension_eda.ipynb`: distribución del objetivo |
| ¿Qué asociaciones hay en entrenamiento? | EDA: ocupación, montos y Spearman |
| ¿Cómo prevenimos aprender del futuro? | `temporal_split` y pipeline dentro de GridSearchCV |
| ¿Qué modelo gana entre los candidatos? | `train_model`: comparación por AP temporal |
| ¿Cuándo emitimos una alerta? | `choose_threshold` y `predict_credit` |
| ¿Cuántos atrasos detectamos? | `evaluate`: recall y matriz de confusión |
| ¿Cuánto trabajo generarían las alertas? | `evaluate`: tasa de alertas y precision |
| ¿Cambió la población? | `drift_report` y `psi` |
| ¿Se mantiene el desempeño? | `monitor`: métricas con etiquetas disponibles |

## Glosario breve

| Término | Significado |
|---|---|
| Feature | Variable que recibe el modelo |
| Target | Resultado que se intenta predecir |
| Fit | Aprender parámetros o estadísticas de entrenamiento |
| Transform | Aplicar reglas o parámetros ya aprendidos |
| Inferencia | Producir una predicción para nuevos datos |
| Hiperparámetro | Configuración elegida antes del ajuste |
| Fold | Partición usada en validación cruzada |
| Serialización | Guardar un objeto entrenado para recuperarlo |
| Data drift | Cambio en la distribución de las entradas |
| CI | Integración continua con verificaciones automáticas |
| Cobertura | Proporción de líneas ejecutadas por las pruebas; no garantía de ausencia de errores |
| Reproducibilidad | Poder repetir el proceso con datos, código y entorno identificados |

## Cómo presentar el trabajo

Explica con tus palabras lo que ejecutaste y comprendiste. Sigue las reglas de tu curso sobre asistencia con IA; este paquete no prueba autoría individual ni sustituye la comprensión. Cambia y mejora decisiones solo después de entender sus efectos y registra las modificaciones con commits.
