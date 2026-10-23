# Validación en Codespaces — 22 de octubre de 2026

Separación de responsabilidades en la rama refactor/cookiecutter. Checkpoint de implementación: 7bbaf64.

| Carpeta | Responsabilidad verificada |
|---|---|
| pipelines/ | Scripts 01–06, run_pipeline.py y run_training.py; reutilizan el núcleo |
| api/ | FastAPI en app.py y contratos HTTP en schemas.py |
| mlruns/ | Historial SQLite y artefactos privados de MLflow |
| src/loan_renewal/ | Datos, features, modelos, evaluación, inferencia y lógica compartida |

| Comprobación | Resultado |
|---|---|
| Pruebas | 34 aprobadas, sin omisiones; incluyen API y MLflow reales, scripts desde otro directorio y migración del historial |
| Notebook | 18 celdas ejecutadas en Python 3.12 sin errores; usa api.app y registra en mlruns |
| Wheel | Paquete construido e instalado en un destino aislado; imports de api y loan_renewal verificados desde fuera del repositorio |
| Historial MLflow | Copia de respaldo antes de moverlo; conteos de filas conservados y rutas absolutas actualizadas |
| Registry existente | LoanRenewal-Propensity, versión 1 y alias candidato recargados desde la nueva ubicación con predicciones idénticas |
| Docker/API | Nueva imagen con api.app:app saludable; readiness 200, paridad offline/HTTP y campo ausente 422 |
| DVC | Reproducción desde pipelines/run_training.py completada; segundo repro omitido sin cambios; estado actualizado |
| Fuentes originales | Los cuatro hashes coinciden con docs/fuentes.json |

## Metadatos de instalación

Se verificaron las copias generadas src/sdg_loan_renewal.egg-info y sdg_loan_renewal.egg-info. Los metadatos instalados están disponibles en .venv/lib/python3.12/site-packages/sdg_loan_renewal-0.1.0.dist-info. Se retiraron las copias redundantes del árbol de trabajo y se verificaron en un proceso nuevo los imports, la versión y el comando loan-renewal check-env --include-dev desde otro directorio.

Setuptools puede regenerar egg-info al construir o reinstalar. Esos archivos se excluyen de Git y del explorador. Los metadatos activos de la distribución instalada se conservan.

## Persistencia

La migración mantiene SQLite y el historial existente; la estructura interna de experimentos y Registry la gestiona MLflow. No se crean directorios de versiones ficticios ni se cambia a otro backend. scripts/migrate_tracking.py rechaza un destino con estado existente y conserva una copia antes de modificar rutas.

El notebook original, diccionario y presentación permanecen en references/original; el CSV en data/raw. Los modelos, logs y resultados completos permanecen fuera de Git. DVC conserva caché local y metadatos; no se configuró un remoto ni se publicaron datos. La validación técnica no acredita calidad comercial ni aprobación productiva.


## Comandos — 23 de octubre de 2026

Validación desde una copia aislada del proyecto, con instalación nueva mediante `make install`, Python 3.12 y datos sintéticos de 72 registros. No se usaron esas métricas para evaluar el caso comercial ni se sustituyeron los runs existentes.

| Comprobación | Resultado |
|---|---|
| Instalación y entorno | `make install`, `pip check` y `check-env --include-dev` completados |
| Pruebas | `make test`: 39 aprobadas sin omisiones |
| Entrenamiento | `make pipeline`, `run_training.py --track` y CLI `train --tune --track` completados |
| Notebook | `make notebook`: 18 celdas ejecutadas sin errores |
| Experimentos | Los siete scripts numerados y `run_pipeline.py` completados; evaluación final única y reutilización del comprobante |
| Inferencia | CLI `predict` genera CSV; API HTTP pasa readiness, paridad con bundle y campo ausente 422 |
| MLflow UI | `make mlflow MLFLOW_PORT=15001` responde HTTP 200; un worker para reducir recursos |
| DVC | `version-data --dvc`, `make dvc`, `dvc status` y segundo `make dvc` sin cambios completados |
| Docker | Codespace reconstruido con Docker-in-Docker; `make docker`, `compose ps`, logs y `down` completados. Contenedor saludable; readiness 200, paridad offline/HTTP y campo ausente 422 |

`make check-data` comprueba la presencia del CSV antes de los recorridos de entrenamiento, notebook y DVC. `make api API_PORT=8001` y `make mlflow MLFLOW_PORT=5001` permiten elegir puertos libres. No se detienen servicios ajenos para liberar puertos.

La reconstrucción detectó una fuente APT de Yarn obsoleta en la imagen base. `.devcontainer/Dockerfile` la retira antes de instalar Features, sin desactivar la verificación de firmas. Se comprobó que MLflow rechazaba el hostname reenviado con HTTP 403; `make mlflow` permite el hostname exacto del Codespace y mantiene el rechazo de hosts ajenos.
