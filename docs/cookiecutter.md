# Organización con Cookiecutter Data Science

El repositorio adopta la separación de datos, notebooks, código instalable, referencias, modelos e informes de Cookiecutter Data Science. Es una adaptación del caso existente: se conserva `src/loan_renewal` para mantener una sola implementación importable y evitar duplicar lógica en notebooks, API o carpetas por unidad.

| Ruta | Contenido y política |
|---|---|
| `data/raw/` | CSV original, inmutable y privado; Git conserva únicamente punteros DVC y archivos de estructura |
| `data/interim/` | Resultados intermedios locales si una etapa necesita materializarlos |
| `data/processed/` | Datasets derivados locales; no implica preajustar features antes del split |
| `data/external/` | Fuentes externas identificadas, cuando existan |
| `models/` | Bundle completo de transformaciones, estimador, umbral y manifiesto; excluido de Git |
| `notebooks/01_renovacion_prestamo.ipynb` | Único notebook ejecutable del flujo, sin salidas versionadas |
| `references/original/` | Notebook fuente de `docs/`, diccionario original; privados y sin modificar |
| `reports/training/` | Métricas calculadas, comparación, manifiesto y evidencia por etapa |
| `mlruns/` | SQLite y artefactos locales de MLflow, excluidos de Git |
| `pipelines/` | Scripts numerados y orquestadores de experimentación |
| `api/` | Servicio FastAPI y esquemas de entrada/salida |
| `reports/figures/` | Figuras generadas para análisis e informes |
| `docs/` | Documentación mantenida: auditoría, contrato de datos |
| `src/loan_renewal/` | Ingestión, features, modelado, evaluación, inferencia, API y orquestación compartidos |
| `tests/` | Pruebas de contrato, aislamiento temporal, serialización, CLI y servicio |
| `scripts/` | Herramientas de ejecución de notebook y verificación de servicio |
| `.devcontainer/` | Configuración de Codespaces |

## Rutas y reproducción

La configuración central define las rutas del proyecto. Entrenamiento y orquestación utilizan `models/` para el bundle y `reports/training/` para evidencia; la base compartida de MLflow es `mlruns/mlflow.db`. Docker monta `models/` en modo de solo lectura. Notebook, CLI y API importan el mismo paquete.

`dvc.yaml` declara un stage que ejecuta el pipeline con `--track`: el bundle es una salida DVC y `reports/training/metrics.json` registra las métricas. La base SQLite y los artefactos de MLflow cambian con cada experimento y permanecen fuera de las salidas DVC. DVC conserva el modelo y la evidencia del run; MLflow requiere un respaldo propio. El CSV se identifica mediante su puntero DVC. Deben conservarse caché o remoto autorizado para recuperar sus bytes.

No se obliga a generar matrices transformadas intermedias. Imputadores, escaladores, categorías y clustering se ajustan dentro del pipeline de train/folds y se persisten con el estimador. La estructura de carpetas no modifica el protocolo temporal ni permite usar validación o test durante `fit`.

## Comandos habituales

| Comando | Resultado |
|---|---|
| `make install` | Preparar el entorno y dependencias del proyecto |
| `make test` | Ejecutar las pruebas automatizadas |
| `make pipeline` | Ejecutar etapas con entrenamiento y tracking |
| `make notebook` | Ejecutar el notebook en un kernel nuevo y fallar ante errores |
| `make api` | Iniciar FastAPI con el bundle local |
| `make docker` | Construir e iniciar el servicio con Compose |
| `make dvc` | Reproducir el stage de DVC |

Los commits checkpoint deben agrupar cambios verificables: estructura y rutas; notebook e integraciones; pruebas y evidencia. Revisar los archivos preparados antes de cada push para conservar fuera de Git datos, modelos, originales con salidas, cachés y credenciales.

La plantilla base se generó con CCDS 2.3.0. Parámetros y commit de referencia: `cookiecutter-generation.json`.
