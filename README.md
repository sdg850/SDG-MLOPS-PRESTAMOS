# SDG-MLOPS-PRESTAMOS

Proyecto educativo de MLOps para estimar la **propensión comercial de un cliente a aceptar la renovación de un préstamo**. La variable objetivo es `FLAG_VENTA`: `1` indica renovación observada y `0` ausencia de venta. El score no representa riesgo de impago ni determina elegibilidad crediticia.

## 1. Funcionalidad del proyecto

El flujo permite:

1. Cargar el CSV original, validar columnas, tipos, etiqueta y particiones temporales.
2. Aprender imputación, transformaciones, codificación de categorías y clustering dentro del entrenamiento o de cada fold.
3. Comparar modelos base y estrategias de balanceo; ejecutar GridSearchCV o RandomizedSearchCV cuando se solicita.
4. Seleccionar modelo y umbral con validación y calcular las métricas finales del candidato seleccionado.
5. Guardar un bundle con preprocesamiento, estimador, umbral y manifiesto; comprobar su paridad tras recargarlo.
6. Registrar experimentos en MLflow, comparar runs compatibles y crear versiones en Model Registry.
7. Ejecutar inferencia con variables originales mediante CSV o una API FastAPI, también disponible en Docker.
8. Reproducir el entrenamiento con DVC y conservar la identidad de datos, código y salidas.

El notebook llama a las mismas funciones que los comandos. Los scripts de `pipelines/` organizan la ejecución; la lógica compartida permanece en `src/loan_renewal/`.

## 2. Estructura de carpetas

| Ruta | Contenido y responsabilidad |
|---|---|
| `api/` | Servicio HTTP: `app.py` define la aplicación y `schemas.py` sus contratos de entrada y salida. |
| `pipelines/` | Scripts numerados de experimentación y orquestadores. No contienen copias del modelo. |
| `src/loan_renewal/` | Núcleo reutilizable: datos, features, modelos, evaluación, inferencia, tracking y orquestación. |
| `data/raw/` | CSV original privado y su puntero DVC. La fuente no se modifica durante el entrenamiento. |
| `data/interim/` | Espacio reservado para datos intermedios; el flujo actual no exige materializarlos. |
| `data/processed/` | Espacio reservado para datasets derivados. |
| `data/external/` | Espacio reservado para fuentes externas. |
| `models/` | Bundle utilizado por la API: `model.joblib`. |
| `models/experiments/{base,grid,random}/` | Bundles independientes de los experimentos. Se crean al ejecutarlos. |
| `mlruns/` | Estado privado de MLflow: base SQLite `mlflow.db` y directorio `mlflow-artifacts/`. |
| `notebooks/` | Notebook refactorizado `01_renovacion_prestamo.ipynb`, versionado sin salidas. |
| `reports/training/` | Métricas, manifiesto, comparación, contrato, perfil de datos y estado de las etapas. |
| `reports/experiments/` | Evidencia por experimento, comparación de runs, mejor run y registro del candidato. |
| `reports/figures/` | Espacio para gráficos exportados. |
| `reports/notebook.executed.ipynb` | Copia local del notebook con sus salidas, generada por `make notebook`. |
| `references/original/` | Notebook original, diccionario y presentación; permanecen locales. |
| `docs/` | Diccionario, auditoría metodológica, correspondencia con el curso y evidencia de validación. |
| `scripts/` | Ejecución del notebook, comprobación del servicio y migración del tracking anterior. |
| `tests/` | Pruebas de contratos, particiones, persistencia, CLI, API, MLflow y migración. |
| `.devcontainer/` | Configuración del entorno de desarrollo de Codespaces. |
| `.vscode/` | Intérprete del proyecto y preferencias del explorador. |

Los directorios de salidas se completan al ejecutar los comandos

| Archivo raíz | Función |
|---|---|
| `pyproject.toml` | Dependencias, extras e instalación de `loan-renewal`, `loan_renewal` y `api`. |
| `Makefile` | Atajos de instalación, pruebas, entrenamiento, notebook, API, Docker y DVC. |
| `Dockerfile` / `compose.yaml` | Construcción y ejecución del servicio con el modelo montado como solo lectura. |
| `dvc.yaml` / `dvc.lock` | Etapa reproducible y huellas de sus dependencias y salidas. |
| `.gitignore` / `.dvcignore` / `.dockerignore` | Exclusiones de datos privados, metadatos generados y cachés según cada herramienta. |

## 3. Requisitos e instalación

- Python **3.12** recomendado; el paquete admite Python `>=3.11,<3.14`.
- Git y `make`; los comandos de esta guía se ejecutan en Bash desde la raíz del repositorio.
- Docker Engine y Docker Compose para la alternativa en contenedor.
- Acceso autorizado al CSV original: sus registros no están publicados en Git.

Comenzar con un clon nuevo:

```bash
make install
source .venv/bin/activate
python -m loan_renewal check-env --include-dev
make test
```

`make install` crea `.venv`, instala `.[dev,api,mlops]` y registra el kernel **Loan renewal (.venv)**. En un sistema que no incluya soporte para entornos virtuales, instala primero el paquete correspondiente a Python 3.12, por ejemplo `python3.12-venv` en Ubuntu.

Alternativa manual a `make install`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,api,mlops]'
python -m ipykernel install --user --name loan-renewal --display-name "Loan renewal (.venv)"
```

SMOTE y XGBoost son opcionales: `python -m pip install -e '.[smote,xgboost]'` agrega sus dependencias al entorno ya preparado.

## 4. Preparar los datos

Coloca el archivo autorizado en:

```text
data/raw/Dataset Renovacion_prestamo.csv
```

Comprueba la presencia del archivo antes de entrenar:

```bash
make check-data
```

El CSV usa `;` como separador. El entrenamiento requiere las 22 columnas originales, incluidas `FLAG_VENTA`, `MES` y `CLIENTE`; estas tres columnas no se usan como predictores. La inferencia utiliza las 19 variables originales detalladas en el [diccionario](docs/diccionario.md). Las claves requeridas ausentes generan error; los nulos admitidos se procesan con los imputadores persistidos y las categorías nuevas con el encoder aprendido.

Puedes indicar otra fuente con `--data "/ruta/archivo.csv"`. DVC utiliza la ruta declarada en `dvc.yaml`; pasar `--data` a otro comando no modifica esa declaración. No se generan datos sintéticos para sustituir una fuente ausente ni para informar métricas del caso.

## 5. Entrenamiento y notebook

### Ruta principal

```bash
make pipeline
```

Equivale a `python pipelines/run_training.py --tune --track`. Ejecuta ingestión, contrato de features, entrenamiento y predicciones finales, evaluación y tracking. Incluye GridSearchCV temporal y registra el resultado en MLflow.

Para una ejecución sin búsqueda de hiperparámetros, conservando los candidatos base y el tracking:

```bash
python pipelines/run_training.py --track
```

También está disponible la CLI:

```bash
python -m loan_renewal train --data "data/raw/Dataset Renovacion_prestamo.csv" --output models --reports reports/training --tune --track
```

| Salida principal | Contenido |
|---|---|
| `models/model.joblib` | Pipeline completo, umbral y manifiesto. |
| `reports/training/metrics.json` | Métricas calculadas de validación y test del seleccionado. |
| `reports/training/comparison.csv` | Comparación de candidatos mediante validación. |
| `reports/training/manifest.json` | Hashes, particiones, configuración, versiones y trazabilidad del código. |
| `reports/training/stage_report.json` | Estado y duración de las etapas; conserva evidencia si una etapa falla. |
| `reports/training/tracking.json` | Identidad del run y URI exacta del modelo registrado en Tracking. |

### Notebook

Abre `notebooks/01_renovacion_prestamo.ipynb` y selecciona **Loan renewal (.venv)**. Para ejecutarlo en un kernel nuevo y detenerse ante errores:

```bash
make notebook
```

La copia ejecutada se guarda en `reports/notebook.executed.ipynb`. El notebook vuelve a entrenar y registra un run; es una ruta alternativa para recorrer el caso, no un paso obligatorio después de `make pipeline`.

## 6. Experimentos y registro de versiones

Ejecuta los scripts en este orden, o utiliza el orquestador completo:

| Script | Funcionalidad |
|---|---|
| `pipelines/01_base_pipeline.py` | Ejecutar candidatos base y registrar su resultado en MLflow. |
| `pipelines/02_gridsearch_mlflow.py` | Agregar GridSearchCV de Random Forest con folds temporales internos. |
| `pipelines/03_randomsearch_mlflow.py` | Ejecutar la alternativa RandomizedSearchCV. |
| `pipelines/04_comparar_runs.py` | Comparar runs finalizados con igual dataset y protocolo. |
| `pipelines/05_mejor_modelo.py` | Guardar el run seleccionado por validación. |
| `pipelines/05_evaluar_ganador.py` | Evaluar una vez el ganador global en test, sin ajustar el modelo ni el umbral. |
| `pipelines/06_model_registry.py` | Crear una versión en Registry y asignarle el alias `candidato`. |
| `pipelines/run_pipeline.py` | Ejecutar base, grid, random, comparación, evaluación final y registro. |
| `pipelines/run_training.py` | Ejecutar las etapas de un entrenamiento; es la entrada utilizada por Make y DVC. |

```bash
python pipelines/run_pipeline.py
```

Esta secuencia sigue **entrenar → comparar → evaluar ganador → registrar**. Base, grid y random usan entrenamiento/validación y dejan `test: null` en sus informes; solo el ganador global se evalúa en septiembre. La comparación se limita a los runs identificados en el directorio de informes actual.

Esta secuencia entrena tres experimentos. Sus modelos quedan en `models/experiments/` y su evidencia en `reports/experiments/`. Los scripts admiten `--help`, `--data`, `--output`, `--reports`, `--experiment` y `--tracking-uri`; la implementación actual utiliza SQLite local para Tracking y Registry.

Para seleccionar, evaluar y registrar a partir de experimentos ya entrenados:

```bash
python pipelines/04_comparar_runs.py
python pipelines/05_mejor_modelo.py
python pipelines/05_evaluar_ganador.py
python pipelines/06_model_registry.py
```

La evaluación guarda `reports/experiments/final_evaluation.json`, vinculado al run, URI exacta del modelo, hash del dataset, protocolo y umbral. `experiments evaluate` reutiliza ese comprobante sin volver a predecir; solo el ganador recibe métricas `test_report_*` en MLflow. El registro rechaza comprobantes ausentes o que no correspondan al ganador. Después de evaluar, `all`, los entrenamientos individuales y `best` se bloquean en ese directorio para no volver a seleccionar tras observar test; `evaluate` y `registry` permiten continuar.

Un bloqueo concurrente protege la evaluación. No borrar el comprobante ni el archivo `.lock` para reintentar una evaluación en curso. Una nueva ruta de informes no convierte un test ya observado en independiente: para una evaluación prospectiva se necesitan campañas no usadas.

`make pipeline`, `make notebook` y el comando `train` conservan su recorrido de entrenamiento individual y evaluación de un único ganador; no es necesario ejecutarlos después del recorrido completo de experimentos.

Estos comandos requieren resultados previos de los experimentos. La selección exige igualdad de hash de datos y protocolo y usa métricas de validación. El registro conserva la URI de origen, el umbral y la paridad de recarga.

El modelo registrado se llama `LoanRenewal-Propensity`. Cada nuevo registro crea una versión; volver a registrar el mismo artefacto también puede incrementarla. El alias `candidato` apunta a una versión concreta. Sus metadatos están en `mlruns/mlflow.db`; no se crean carpetas manuales `version-1`, `version-2`.

**Registrar o cambiar un alias no actualiza automáticamente la API.** El servicio carga `models/model.joblib` al arrancar; los experimentos generan bundles en sus propios directorios. La elección del artefacto de serving y el reinicio del servicio son acciones independientes.

## 7. Consultar MLflow

Después de generar el primer run, inicia la interfaz en otra terminal con `.venv` activa y desde la raíz:

```bash
make mlflow
```

Si el puerto está ocupado, usa `make mlflow MLFLOW_PORT=5001`. Abre `http://127.0.0.1:5000` (o el puerto elegido). En Codespaces, usa el puerto **5000** de la pestaña **Ports**. El experimento predeterminado es `loan-renewal`; permite consultar parámetros, métricas de validación, métricas finales, artefactos y versiones registradas.

El historial anterior de `reports/tracking/` se migró a `mlruns/` conservando SQLite, registros y artefactos. `scripts/migrate_tracking.py` existe para instalaciones antiguas: requiere que no haya tracking activo, crea un respaldo y rechaza sobrescribir un destino con estado existente. No se necesita en una instalación nueva.

## 8. Inferencia por CSV y API

### CSV

Con el bundle ya entrenado, prepara `entrada_raw.csv` separado por `;` con las 19 variables predictoras originales:

```bash
python -m loan_renewal predict --model models/model.joblib --input entrada_raw.csv --output reports/predicciones.csv
```

No se construyen columnas `_LOG`, dummies o clusters a mano: el artefacto aplica sus transformaciones y el umbral guardado.

### FastAPI local

```bash
make api
```

Si el puerto está ocupado, usa `make api API_PORT=8001` y comprueba el servicio con `python scripts/check_service.py --url http://127.0.0.1:8001`. Equivale a ejecutar `uvicorn api.app:app --host 127.0.0.1 --port 8000` con el puerto predeterminado. Abre `http://127.0.0.1:8000/docs` o el puerto **8000** de Codespaces.

| Endpoint | Funcionalidad |
|---|---|
| `GET /` | Información del servicio. |
| `GET /health` | Readiness: `200` con modelo disponible, `503` si no pudo cargarse. |
| `POST /predecir` | Recibe las variables raw y devuelve `score_propension`, `flag_venta`, `umbral` y `modelo`. |
| `GET /docs` | Documentación interactiva de los contratos HTTP. |

Desde otra terminal, con `.venv` activa, comprueba el servicio y su paridad con el bundle local:

```bash
python scripts/check_service.py
```

La comprobación utiliza entradas sintéticas y verifica readiness, paridad offline/HTTP y respuesta `422` para un campo requerido ausente. Comprueba el funcionamiento del servicio, no el rendimiento predictivo del caso.

## 9. API con Docker

El Codespace necesita Docker Engine y Compose. La configuración `.devcontainer` incorpora Docker-in-Docker y una imagen de desarrollo que retira la fuente APT obsoleta de Yarn (no utilizada); en un Codespace creado antes de este cambio ejecuta **Codespaces: Rebuild Container** para aplicarla. Comprueba `docker info` y `docker compose version`.

Entrena primero para disponer de `models/model.joblib`. Utiliza esta ruta como alternativa al servidor local; ambos usan el puerto 8000.

```bash
make docker
docker compose ps
python scripts/check_service.py
```

`make docker` construye la imagen y espera al healthcheck. Compose monta `models/` en modo de solo lectura y expone `127.0.0.1:8000`. Los modelos y datos privados no forman parte de las capas de la imagen.

```bash
docker compose logs --tail=100 api
docker compose down
```

Para cargar un nuevo bundle, reinicia o recrea el servicio después de colocarlo en la ruta configurada. El arranque no vuelve a entrenar el modelo.

## 10. DVC y reproducción

Con el CSV original disponible en `data/raw/`:

```bash
python -m loan_renewal version-data --dvc
make dvc
dvc status
```

El primer comando calcula la identidad de la fuente y ejecuta `dvc add`. `make dvc` reproduce la etapa `training` declarada en `dvc.yaml`, utilizando el Python de `.venv`. Si dependencias y salidas no cambiaron, DVC omite el entrenamiento.

DVC declara el bundle y la evidencia de entrenamiento como salidas, y `reports/training/metrics.json` como métricas sin caché. La base y los artefactos mutables de MLflow se conservan aparte en `mlruns/` y requieren respaldo propio.

Git guarda código, punteros y `dvc.lock`; los bytes de datos y modelos permanecen en caché local o en un remoto que se configure por separado. **No hay un remoto DVC configurado por este proyecto**: un clon con un puntero no garantiza recuperar el CSV mediante `dvc pull`. No se ejecuta `dvc push` automáticamente.

## 11. Protocolo y límites de interpretación

| Partición | Meses de 2015 | Uso |
|---|---|---|
| Train | Enero–junio | Aprender transformaciones y modelos; búsqueda con folds temporales internos. |
| Validación | Julio–agosto | Seleccionar candidato y umbral. |
| Test | Septiembre | Reportar el rendimiento final del candidato seleccionado. |

Imputación, escalado, categorías, clustering y balanceo se ajustan exclusivamente con el train correspondiente. La selección usa F1 de la clase positiva y Average Precision como desempate; ROC-AUC y AP se calculan con scores. La accuracy se interpreta junto al desbalance.

El notebook histórico ya exploró todo el CSV. Repetir estas ejecuciones reproduce el protocolo histórico, pero no crea un test prospectivo independiente ni autoriza ajustar candidatos según septiembre. Los scores no están calibrados; faltan costes de campaña, confirmación de disponibilidad temporal de variables y evaluación del uso de atributos demográficos antes de un uso operativo. SMOTE sobre features one-hot requiere revisar su pertinencia.

## 12. Comandos y documentación de referencia

| Comando | Resultado |
|---|---|
| `make install` | Crear entorno, instalar dependencias y registrar kernel. |
| `make test` | Ejecutar las pruebas automatizadas. |
| `make check-data` | Comprobar la existencia del CSV y obtener su hash. |
| `make mlflow` | Iniciar la interfaz de MLflow en el puerto 5000. |
| `make pipeline` | Entrenar por etapas con GridSearch y tracking. |
| `make notebook` | Ejecutar el notebook completo y guardar una copia local con salidas. |
| `make api` | Iniciar FastAPI con el bundle local. |
| `make docker` | Construir y esperar al servicio Docker saludable. |
| `make dvc` | Reproducir o reutilizar las salidas de la etapa DVC. |


| Documento | Contenido |
|---|---|
| [Diccionario](docs/diccionario.md) | Variables y contrato del dataset. |
| [Auditoría de refactorización](docs/refactorizacion.md) | Hallazgos del notebook original y cambios metodológicos. |
| [Estructura Cookiecutter](docs/cookiecutter.md) | Organización y adaptaciones de la  implementación. |
| [Validación en Codespaces](docs/validacion.md) | Evidencia fechada de comandos, 39 pruebas, notebook, API, MLflow, Registry y requisitos de Docker/DVC. |
| [Fuentes originales](docs/fuentes.json) | Hashes de los archivos inspeccionados. |

Datos, originales con salidas, modelos, tracking y resultados completos se mantienen fuera de Git.
