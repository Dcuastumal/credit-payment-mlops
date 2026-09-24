# Validación del paquete

## Ejecutado

Entorno: Linux, Python 3.12.14. Dependencias principales en `requirements.txt`; `pip check` no encontró dependencias rotas en el entorno utilizado.

| Verificación | Resultado |
|---|---|
| Fuente Excel y equivalencia CSV | 10.763 filas, 23 columnas; comprobación de valores y faltantes aprobada |
| Notebooks | Ambos ejecutados secuencialmente, con espacio de variables independiente por notebook |
| Formato de notebooks | Validado con nbformat |
| Salidas | Tablas guardadas y seis gráficos de EDA embebidos |
| Inspección visual | Se revisaron los seis gráficos y la figura de evaluación; etiquetas legibles y sin recortes materiales |
| Entrenamiento | Regresión logística y Random Forest, tres folds temporales; modelo y métricas guardados |
| Monitoreo | Reporte train/test generado; dos variables con PSI >= 0,2 |
| Pruebas | 20 aprobadas, 0 fallidas |
| Cobertura de código fuente | 214 de 313 líneas: 68,37 %; se excluye el archivo de pruebas |
| API | TestClient: /health, predicción válida, consistencia Python/API y rechazo de entradas inválidas |
| Streamlit | AppTest: carga de la aplicación y envío de formulario sin excepciones |
| Serialización | Recuperación del artefacto conserva las predicciones |

La cobertura reportada mide la suite pytest, no la ejecución previa de entrenamiento y notebooks. Por eso no cuenta todo el código ejecutado durante el pipeline. No se afirma cobertura completa ni ausencia universal de errores.

Comandos utilizados, desde la raíz:

```bash
python run_pipeline.py --in-process
python -m pytest src/test_project.py -q --cov=src --cov-report=term-missing --cov-report=xml
python -m pip check
```

La suite emitió dos advertencias de deprecación de Starlette/AnyIO sobre la integración de su cliente HTTP. No fueron fallos de pruebas. No se ocultaron advertencias para presentar una ejecución aparentemente limpia.

## Límites comprobados del entorno

- El inicio de un kernel Jupyter por TCP falló por restricciones del entorno (`Operation not permitted`). Se ejecutaron realmente todas las celdas con la alternativa `--in-process`; no se simularon salidas. La vía normal de kernel queda para comprobar en VS Code.
- La UI se probó con AppTest, no con una sesión de navegador real.
- La API se probó con TestClient, no en un servidor público desplegado.
- `set_up.bat` se revisó como código, pero no se ejecutó en Windows.
- Docker no está instalado: se entrega el Dockerfile, no una imagen construida ni evidencia de ejecución de un contenedor.
- No se ejecutaron Jenkins ni SonarCloud. Se entregan sus configuraciones y las instrucciones del README; las credenciales y la configuración externa deben completarse en las cuentas correspondientes.
- El ZIP no modifica el repositorio remoto ni fabrica ramas, tags o commits de avances.

## Estado por avance

| Avance | Entregado en el paquete | Pendiente externo |
|---|---|---|
| 1 | Fuente, CSV, notebooks ejecutados, dependencias, entorno y documentación | Incorporación al repositorio y versiones reales |
| 2 | Ingeniería, modelos, CV, evaluación, modelo serializado | Confirmar diccionario y disponibilidad temporal para uso no académico |
| 3 | Monitoreo offline, Streamlit y README | Datos reales nuevos si se quiere monitoreo de producción |
| 4 | API probada, Dockerfile y comandos | Construir y probar imagen Docker en equipo con Docker |
| Extra credit | Configuración Sonar y reporte de cobertura | Conectar SonarCloud, ejecutar análisis y revisar quality gate |

## Resultado del modelo

Con umbral 0,53: 18 TP, 334 FP, 51 FN y 1.750 TN en 2.153 filas de test. ROC-AUC 0,5827 y AP 0,0614. Estos resultados son evidencia de rendimiento limitado, no de un modelo listo para decisiones reales de crédito.
